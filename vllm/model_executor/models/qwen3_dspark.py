# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 DSpark draft model for semi-autoregressive drafting.

DSpark drafts a whole block in one parallel pass (DFlash-style: context-KV
precompute + a non-causal query-block forward) and then injects intra-block
dependency with a lightweight sequential Markov head.

The parallel backbone is a standard Qwen3 decoder stack reused from the
DFlash Qwen3 draft (see qwen3_dflash.py). DSpark adds:
  * ``markov_head``: low-rank V x r / r x V transition bias added to the base
    logits, sampled left-to-right by the speculator (the sequential stage).
  * ``confidence_head``: per-position acceptance-probability estimate.

DSparkMarkovHead and DSparkConfidenceHead are shared with the DSV4-style DSpark model.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
)

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


class DSparkMarkovHead(nn.Module):
    """Sequential DSpark Markov head.

    Mirrors the MarkovHead variants used by Speculators training:
      * vanilla: first-order previous-token Markov bias
      * gated: hidden-state-gated previous-token bias
      * rnn: recurrent state across positions in a speculative block
    """

    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
        hidden_size: int | None = None,
        head_type: str = "vanilla",
    ) -> None:
        super().__init__()

        if markov_rank <= 0:
            raise ValueError(
                f"markov_rank must be > 0, got {markov_rank}"
            )

        if head_type not in ("vanilla", "gated", "rnn"):
            raise ValueError(
                f"Unsupported markov_head_type: {head_type!r}"
            )

        if head_type in ("gated", "rnn") and hidden_size is None:
            raise ValueError(
                f"hidden_size is required for markov_head_type={head_type!r}"
            )

        self.head_type = head_type
        self.markov_rank = markov_rank
        self.hidden_size = hidden_size

        # Same previous-token embedding as Speculators.
        self.markov_w1 = nn.Embedding(
            vocab_size,
            markov_rank,
        )

        # Rank -> draft-vocabulary logit bias.
        # Preserve vLLM's existing ParallelLMHead implementation.
        self.markov_w2 = ParallelLMHead(
            draft_vocab_size,
            markov_rank,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "markov_w2"),
            disable_tp=True,
        )

        # Exact extra modules created by Speculators MarkovHead.
        if head_type == "gated":
            assert hidden_size is not None
            self.gate_proj = nn.Linear(
                hidden_size + markov_rank,
                markov_rank,
            )

        elif head_type == "rnn":
            assert hidden_size is not None

            # Speculators:
            #
            # joint_proj(
            #   [state(r), prev_emb(r), hidden(hidden_size)]
            # )
            # -> [gate_raw(r), cand_raw(r), out_raw(r)]
            self.joint_proj = nn.Linear(
                2 * markov_rank + hidden_size,
                3 * markov_rank,
            )

    def embed(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """r-dimensional embedding of the previous token."""
        return self.markov_w1(token_ids.long())

    def bias(
        self,
        markov_embed: torch.Tensor,
        logits_processor: LogitsProcessor,
    ) -> torch.Tensor:
        """Project a Markov representation to draft-vocabulary bias."""
        return logits_processor(
            self.markov_w2,
            markov_embed,
        )

    def step(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        state: torch.Tensor | None,
        logits_processor: LogitsProcessor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Run one sequential Markov-head position.

        This is the inference equivalent of Speculators
        MarkovHead.block_bias().

        Returns:
            logits_bias:
                Bias to add to the current position's backbone logits.

            prev_emb:
                Previous-token embedding. Returned because this is also
                the representation used by the trained confidence head.

            state:
                Updated recurrent state for RNN mode.
        """

        prev_emb = self.embed(token_ids)

        # Exact training behavior:
        # prev_emb = prev_emb.to(self.markov_w2.weight.dtype)
        prev_emb = prev_emb.to(
            self.markov_w2.weight.dtype
        )

        # ----------------------------------------------------
        # vanilla
        # ----------------------------------------------------
        if self.head_type == "vanilla":
            logits_bias = self.bias(
                prev_emb,
                logits_processor,
            )
            return logits_bias, prev_emb, state

        hidden_states = hidden_states.to(
            prev_emb.dtype
        )

        # ----------------------------------------------------
        # gated
        #
        # gate = sigmoid(gate_proj([hidden, prev_emb]))
        # bias = W2(gate * prev_emb)
        # ----------------------------------------------------
        if self.head_type == "gated":
            gate = torch.sigmoid(
                self.gate_proj(
                    torch.cat(
                        [
                            hidden_states,
                            prev_emb,
                        ],
                        dim=-1,
                    )
                )
            )

            logits_bias = self.bias(
                gate * prev_emb,
                logits_processor,
            )

            return logits_bias, prev_emb, state

        # ----------------------------------------------------
        # rnn
        #
        # This exactly matches:
        #
        # state = zeros(...)
        #
        # z = cat([state, prev_emb, hidden])
        #
        # gate_raw, cand_raw, out_raw =
        #     joint_proj(z).chunk(3, dim=-1)
        #
        # gate = sigmoid(gate_raw)
        #
        # state =
        #     gate * state
        #     + (1-gate) * tanh(cand_raw)
        #
        # bias = W2(tanh(out_raw))
        # ----------------------------------------------------

        if state is None:
            state = prev_emb.new_zeros(
                prev_emb.shape[0],
                self.markov_rank,
            )
        else:
            state = state.to(prev_emb.dtype)

        z = torch.cat(
            [
                state,
                prev_emb,
                hidden_states,
            ],
            dim=-1,
        )

        gate_raw, cand_raw, out_raw = (
            self.joint_proj(z).chunk(
                3,
                dim=-1,
            )
        )

        gate = torch.sigmoid(gate_raw)

        state = (
            gate * state
            + (1.0 - gate)
            * torch.tanh(cand_raw)
        )

        markov_output = torch.tanh(out_raw)

        logits_bias = self.bias(
            markov_output,
            logits_processor,
        )

        return logits_bias, prev_emb, state

    def apply_bias_gathered(
        self,
        markov_embed: torch.Tensor,
        logits: torch.Tensor,
        values: torch.Tensor,
        index: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Existing gathered W2 path used by vLLM."""

        weight = self.markov_w2.weight[index]

        corrected = values.unsqueeze(-1)

        corrected.baddbmm_(
            weight,
            markov_embed.unsqueeze(-1),
            beta=1.0,
            alpha=scale,
        )

        return logits.scatter_(
            1,
            index,
            corrected.squeeze(-1),
        )


class DSparkConfidenceHead(nn.Module):
    """DSpark acceptance-confidence head."""

    def __init__(
        self,
        input_dim: int,
        prefix: str,
        bias: bool = False,
        with_markov: bool = True,
    ) -> None:
        super().__init__()
        self.with_markov = with_markov
        self.proj = ReplicatedLinear(
            input_dim,
            1,
            bias=bias,
            return_bias=False,
            params_dtype=torch.float32,
            prefix=maybe_prefix(prefix, "proj"),
        )

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        x = (
            torch.cat([hidden, markov_embed], dim=-1) if self.with_markov else hidden
        ).float()
        return self.proj(x).squeeze(-1)


class Qwen3DSparkModel(DFlashQwen3Model):
    """DFlash Qwen3 backbone + DSpark Markov / confidence heads."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        config = self.config
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
            quant_config=self.quant_config,
            hidden_size=config.hidden_size,
            head_type=getattr(
                config,
                "markov_head_type",
                "vanilla",
            ),
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if getattr(config, "enable_confidence_head", False):
            with_markov = getattr(config, "confidence_head_with_markov", False)
            input_dim = config.hidden_size
            if with_markov:
                input_dim += config.markov_rank
            self.confidence_head = DSparkConfidenceHead(
                input_dim,
                prefix=maybe_prefix(prefix, "confidence_head"),
                bias=True,
                with_markov=with_markov,
            )


class Qwen3DSparkForCausalLM(DFlashQwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = Qwen3DSparkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        self.target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != self.target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Draft-vocab logits without the d2t scatter: the speculator adds the
        # Markov bias in draft space, then remaps via map_draft_to_target.
        return self.logits_processor(self.lm_head, hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        # Map draft-vocab ids to target ids (identity for full-vocab drafts).
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def markov_step(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Run one sequential DSpark Markov-head step."""

        return self.model.markov_head.step(
            token_ids=token_ids,
            hidden_states=hidden_states,
            state=state,
            logits_processor=self.logits_processor,
        )

    def apply_markov_bias_gathered(
        self,
        markov_embed: torch.Tensor,
        logits: torch.Tensor,
        values: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.markov_head.apply_bias_gathered(
            markov_embed,
            logits,
            values,
            index,
            self.logits_processor.scale,
        )

    def compute_confidence(
        self, head_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token."""
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_embed_tokens = False
        includes_lm_head = False
        includes_draft_id_mapping = False
        includes_confidence_head = False
        for name, loaded_weight in weights:
            # t2d is training-only; the draft remaps via d2t at sampling time.
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            if "lm_head" in name:
                includes_lm_head = True
            if "confidence_head" in name:
                includes_confidence_head = True
            model_weights[name] = loaded_weight
            # Sets has_own_embed_tokens / has_own_lm_head so load_dspark_model
            # knows whether to keep these or alias the target's.
            process_eagle_weight(self, name)

        # mask_embedding is an unused placeholder param; DSpark masks via the vocab row.
        # embed_tokens / lm_head are optional; when omitted they are shared from
        # the target by load_dspark_model, so skip the unloaded params here.
        uses_expanded_input_vocab = self.config.vocab_size > self.target_vocab_size
        uses_reduced_vocab = self.config.draft_vocab_size != self.target_vocab_size
        if uses_expanded_input_vocab and not includes_embed_tokens:
            raise ValueError(
                "Qwen3 DSpark checkpoints whose input vocab_size is larger than "
                "the target vocabulary must include embed_tokens weights."
            )
        if uses_reduced_vocab and not includes_lm_head:
            raise ValueError(
                "Reduced-vocabulary Qwen3 DSpark checkpoints must include "
                "lm_head weights; the full target lm_head cannot be shared."
            )
        if uses_reduced_vocab and not includes_draft_id_mapping:
            raise ValueError(
                "Reduced-vocabulary Qwen3 DSpark checkpoints must include a "
                "d2t mapping so sampled draft ids can be converted to target ids."
            )

        orig_to_new_substr = {"mask_embedding": None}
        if not includes_embed_tokens:
            orig_to_new_substr["embed_tokens"] = None
        if not includes_lm_head:
            orig_to_new_substr["lm_head"] = None
        if not includes_draft_id_mapping:
            orig_to_new_substr["draft_id_to_target_id"] = None
        if self.model.confidence_head is None or not includes_confidence_head:
            self.model.confidence_head = None
            orig_to_new_substr["confidence_head"] = None
        mapper = WeightsMapper(orig_to_new_substr=orig_to_new_substr)
        loader = AutoWeightsLoader(self)
        loader.load_weights(model_weights.items(), mapper=mapper)
        self.model._build_fused_kv_buffers()
