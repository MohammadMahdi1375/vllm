# SPDX-License-Identifier: Apache-2.0
"""ReTrace suffix alignment and gated fusion using proposal-indexed states.

The target tensor contains the final normalized states whose logits scored each
proposal. It must come from verification of the actual proposed trajectory.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ReTraceMemory:
    draft: torch.Tensor  # [N, K, draft_hidden]
    target: torch.Tensor  # [N, K, target_hidden]
    valid: torch.Tensor  # [N, K], bool

    def detached(self, *, dtype=torch.float16, device=None):
        """Store one round in the paper's FP16 representation format."""
        return type(self)(
            self.draft.detach().to(device=device, dtype=dtype),
            self.target.detach().to(device=device, dtype=dtype),
            self.valid.detach().to(device=device),
        )


def carry_suffix(
    draft: torch.Tensor,
    target_scores: torch.Tensor,
    accepted: torch.Tensor,
    attempted: torch.Tensor | None = None,
) -> ReTraceMemory:
    """Drop the accepted prefix AND first rejection, shift, detach, and pad.

    ``accepted`` counts accepted proposals, excluding correction/bonus tokens.
    ``attempted`` excludes scheduler padding and unverified proposals.
    No target causal shift is performed here: target_scores is already aligned.
    """
    if draft.ndim != 3 or target_scores.shape[:2] != draft.shape[:2]:
        raise ValueError("Expected matching [batch, proposals, hidden] tensors")
    n, k, h = draft.shape
    if k == 0 or accepted.shape != (n,):
        raise ValueError("Need at least one proposal and accepted.shape == (batch,)")
    if attempted is None:
        attempted = torch.full_like(accepted, k)
    if attempted.shape != accepted.shape:
        raise ValueError("attempted and accepted must have identical shapes")
    j = torch.arange(k, device=draft.device)[None, :]
    source = j + accepted.to(device=draft.device, dtype=torch.long)[:, None] + 1
    valid = (source < attempted[:, None]) & (source < k) & (accepted[:, None] >= 0)
    source = source.clamp(0, k - 1)
    m = draft.detach().gather(1, source[..., None].expand(n, k, h))
    u = target_scores.detach().gather(
        1, source[..., None].expand(n, k, target_scores.shape[-1])
    )
    return ReTraceMemory(
        torch.where(valid[..., None], m, torch.zeros_like(m)),
        torch.where(valid[..., None], u, torch.zeros_like(u)),
        valid,
    )


class ReTraceConditioner(nn.Module):
    """Target correction followed by zero-initialized gated residual fusion."""

    def __init__(self, hidden_size: int, target_hidden_size: int | None = None):
        super().__init__()
        target_hidden_size = target_hidden_size or hidden_size
        self.correction = nn.Linear(
            hidden_size + target_hidden_size, hidden_size, bias=False
        )
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.value.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        memory: ReTraceMemory,
        beta: float = 1.0,
    ) -> torch.Tensor:
        if embeddings.shape != memory.draft.shape:
            raise ValueError(
                "Embeddings and aligned draft memory must have identical shapes"
            )
        m = memory.draft.detach().to(embeddings.dtype)
        u = memory.target.detach().to(embeddings.dtype)
        r = m + beta * self.correction(torch.cat((m, u), dim=-1))
        residual = self.value(r) * torch.sigmoid(
            self.gate(torch.cat((embeddings, r), dim=-1))
        )
        # where, rather than multiply: padded NaNs must never contaminate inputs.
        residual = torch.where(
            memory.valid[..., None], residual, torch.zeros_like(residual)
        )
        return embeddings + residual


def condition_query_block(
    conditioner: ReTraceConditioner,
    embeddings: torch.Tensor,
    memory: ReTraceMemory | None,
    beta: float = 1.0,
) -> torch.Tensor:
    """Map proposal-indexed memory to the slots that predict those proposals.

    Slot zero is the verified anchor and always stays untouched.
    """
    if memory is None:
        return embeddings
    start = 1
    if embeddings.shape[1] - start != memory.draft.shape[1]:
        raise ValueError("Memory width does not match the configured proposal slots")
    conditioned = conditioner(embeddings[:, start:], memory, beta)
    return torch.cat((embeddings[:, :start], conditioned), dim=1)


@dataclass
class _Proposal:
    hidden: torch.Tensor
    positions: torch.Tensor
    tokens: torch.Tensor


class ReTraceRequestCache:
    """One-round, request-keyed state with trajectory and position checks.

    Batch indices are never identities. Every begin_step replaces the preceding
    generation, dropping unscheduled/finished/preempted requests automatically.
    The caller supplies target states already aligned through logits_indices.
    """

    def __init__(self, proposals: int, hidden_size: int, storage_dtype=None):
        self.k = proposals
        self.hidden_size = hidden_size
        self.storage_dtype = storage_dtype
        self._previous: dict[str, _Proposal] = {}
        self._pending: dict[str, tuple[ReTraceMemory, torch.Tensor]] = {}
        self._captured: tuple[torch.Tensor, torch.Tensor] | None = None
        self.req_ids: list[str] = []

    def begin_step(
        self,
        req_ids: list[str],
        target_scores: torch.Tensor,
        score_positions: torch.Tensor,
        verified_tokens: torch.Tensor,
        num_draft_tokens: list[int],
        accepted: torch.Tensor,
    ) -> None:
        previous, self._previous = self._previous, {}
        self._pending = {}
        self._captured = None
        self.req_ids = list(req_ids)
        if len(set(req_ids)) != len(req_ids) or len(num_draft_tokens) != len(req_ids):
            raise ValueError("Need unique request ids and matching draft counts")
        offset = 0
        for i, (req_id, count) in enumerate(zip(req_ids, num_draft_tokens)):
            end = offset + count
            old = previous.get(req_id)
            if old is not None and 0 < count <= self.k:
                same_path = (
                    (old.tokens[:count] == verified_tokens[offset:end]).all()
                    & (old.positions[:count] == score_positions[offset:end]).all()
                    & (accepted[i] >= 0)
                    & (accepted[i] <= count)
                )
                u = target_scores.new_zeros(1, self.k, target_scores.shape[-1])
                u[0, :count] = target_scores[offset:end]
                memory = carry_suffix(
                    old.hidden[None],
                    u,
                    accepted[i : i + 1],
                    torch.full_like(accepted[i : i + 1], count),
                )
                if self.storage_dtype is not None:
                    memory.draft = memory.draft.to(self.storage_dtype)
                    memory.target = memory.target.to(self.storage_dtype)
                source = (
                    (
                        torch.arange(self.k, device=old.positions.device)
                        + accepted[i]
                        + 1
                    )
                    .clamp(0, self.k - 1)
                    .long()
                )
                positions = old.positions[source]
                memory.valid &= same_path
                self._pending[req_id] = memory, positions
            offset = end
        if offset != verified_tokens.numel() or target_scores.shape[0] != offset:
            raise ValueError("Verification rows must equal the sum of draft counts")

    def inputs(
        self, proposal_positions: torch.Tensor, dtype: torch.dtype
    ) -> ReTraceMemory:
        n, k = proposal_positions.shape
        if k != self.k or n < len(self.req_ids):
            raise ValueError("Unexpected query padding or proposal width")
        device = proposal_positions.device
        m = torch.zeros(n, k, self.hidden_size, device=device, dtype=self.storage_dtype or dtype)
        u = torch.zeros_like(m)
        valid = torch.zeros(n, k, device=device, dtype=torch.bool)
        for i, req_id in enumerate(self.req_ids):
            pending = self._pending.pop(req_id, None)
            if pending is not None:
                memory, positions = pending
                m[i] = memory.draft[0]
                u[i] = memory.target[0]
                valid[i] = memory.valid[0] & (positions == proposal_positions[i])
        return ReTraceMemory(m, u, valid)

    def capture(self, hidden: torch.Tensor, positions: torch.Tensor) -> None:
        n = len(self.req_ids)
        if hidden.shape[0] < n or hidden.shape[1:] != (self.k, self.hidden_size):
            raise ValueError("Unexpected draft hidden-state layout")
        retained = hidden[:n].detach().to(self.storage_dtype or hidden.dtype).clone()
        self._captured = retained, positions[:n].detach().clone()

    def finish_step(self, tokens: torch.Tensor) -> None:
        self._previous = {}
        if self._captured is None:
            return
        hidden, positions = self._captured
        if tokens.shape != (len(self.req_ids), self.k):
            raise ValueError("Scheduled proposal width changed during ReTrace drafting")
        self._previous = {
            req_id: _Proposal(hidden[i], positions[i], tokens[i].detach().clone())
            for i, req_id in enumerate(self.req_ids)
        }
        self._captured = None
