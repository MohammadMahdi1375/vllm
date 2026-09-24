# SPDX-License-Identifier: Apache-2.0
"""DFlash prefix-selector model; serving currently requires the Ascend v1 proposer."""

from speculators.models.dflash_prefix.selector_v2 import make_prefix_head, unary_topk

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model


class DFlashPrefixQwen3Model(DFlashQwen3Model):
    def __init__(self, *, vllm_config, start_layer_id=0, prefix=""):
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        cfg = self.config.dflash_config
        self.prefix_head = make_prefix_head(
            self.config.vocab_size,
            self.config.hidden_size,
            rank=cfg["prefix_rank"],
            top_k=cfg["prefix_top_k"],
            max_proposals=cfg["prefix_block_size"] - 1,
            gate_init=cfg.get("prefix_gate_init", 0.1),
            correction_scale=cfg.get("prefix_inference_gate_scale", 1.0),
            kind=cfg.get("prefix_selector_kind", "legacy"),
            heads=cfg.get("prefix_attention_heads", 4),
            layers=cfg.get("prefix_attention_layers", 2),
            history_scale=cfg.get("prefix_history_scale", 1.0),
        )


class DFlashPrefixQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DFlashPrefixQwen3Model

    def __init__(self, *, vllm_config, prefix=""):
        if getattr(vllm_config, "use_v2_model_runner", False):
            raise ValueError("DFlashPrefix currently needs VLLM_USE_V2_MODEL_RUNNER=0")
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("DFlashPrefix currently requires tensor_parallel_size=1")
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        # Fail closed if a mislabeled plain DFlash checkpoint is supplied.
        required = {
            "prefix_head." + name
            for name, _ in self.model.prefix_head.named_parameters()
        }
        seen = set()

        def checked_weights():
            for name, value in weights:
                if name in required:
                    seen.add(name)
                yield name, value

        result = super().load_weights(checked_weights())
        if missing := required - seen:
            raise ValueError(
                f"Checkpoint is missing prefix-selector weights: {sorted(missing)}"
            )
        return result

    def compute_candidates(self, hidden_states):
        logits = super().compute_logits(hidden_states)
        if self.config.dflash_config.get("prefix_selector_kind") == "local_prefix_v2":
            values, ids = unary_topk(logits, self.config.dflash_config["prefix_top_k"])
            return ids, values
        values, ids = logits.topk(self.config.dflash_config["prefix_top_k"], dim=-1)
        return ids, values

    def prepare_prefix_tables(self, hidden, candidates, unary, anchors):
        head = self.model.prefix_head
        kwargs = {}
        if getattr(head, "requires_token_embeddings", False):
            kwargs["embedding_weight"] = self.model.embed_tokens.weight
        return head.prepare_tables(hidden, candidates, unary, anchors, **kwargs)

    def compute_unary_token_ids(self, hidden_states):
        """Same-checkpoint ablation: skip both top-k and the prefix head."""
        return super().compute_logits(hidden_states).argmax(dim=-1)

    def compute_logits(self, hidden_states):
        raise RuntimeError(
            "DFlashPrefix requires its prefix-aware proposer. Plain DFlash "
            "sampling would silently ignore the trained selector. Install the "
            "Ascend adapter and use the v1 runner with greedy draft sampling."
        )
