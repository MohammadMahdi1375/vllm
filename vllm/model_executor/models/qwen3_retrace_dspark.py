# SPDX-License-Identifier: Apache-2.0
"""ReTrace conditioning on the complete, unmodified Qwen3 DSpARK model."""

from .qwen3_dspark import Qwen3DSparkForCausalLM
from .retrace_dspark import ReTraceConditioner, condition_query_block


class ReTraceDSparkQwen3ForCausalLM(Qwen3DSparkForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = self.config
        if (
            getattr(config, "retrace_backbone", None) != "dspark"
            or getattr(config, "retrace_format_version", None) != 1
            or not config.sample_from_anchor
            or config.markov_head_type != "vanilla"
            or config.markov_rank <= 0
            or not config.enable_confidence_head
            or not config.confidence_head_with_markov
        ):
            raise ValueError("Expected the retrace_dspark vanilla Markov architecture")
        self.model.retrace = ReTraceConditioner(config.hidden_size)

    def forward(
        self,
        input_ids,
        positions,
        inputs_embeds=None,
        retrace_memory=None,
        retrace_runtime=False,
    ):
        if self.config.retrace_enabled:
            if not retrace_runtime and not getattr(self, "retrace_profile_run", False):
                raise RuntimeError("ReTrace-on-DSpARK requires its Ascend proposer")
            if retrace_memory is not None:
                if inputs_embeds is None:
                    inputs_embeds = self.embed_input_ids(input_ids)
                inputs_embeds = condition_query_block(
                    self.model.retrace,
                    inputs_embeds.reshape(
                        -1, self.config.block_size, self.config.hidden_size
                    ),
                    retrace_memory,
                    self.config.retrace_beta_max,
                ).reshape(-1, self.config.hidden_size)
        return self.model(input_ids, positions, inputs_embeds)

    def load_weights(self, weights):
        weights = list(weights)
        names = {name for name, _ in weights}
        required = {
            "retrace.correction.weight",
            "retrace.value.weight",
            "retrace.gate.weight",
            "markov_head.markov_w1.weight",
            "markov_head.markov_w2.weight",
            "confidence_head.proj.weight",
            "confidence_head.proj.bias",
        }
        if not required.issubset(names):
            raise ValueError(
                f"ReTrace-on-DSpARK checkpoint is missing {sorted(required - names)}"
            )
        return super().load_weights(weights)
