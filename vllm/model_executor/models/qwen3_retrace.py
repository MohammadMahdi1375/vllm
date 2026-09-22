# SPDX-License-Identifier: Apache-2.0
"""Independent ReTrace architecture using the unmodified DFlash backbone."""

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .retrace import ReTraceConditioner, condition_query_block


class ReTraceQwen3Model(DFlashQwen3Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.retrace = ReTraceConditioner(self.config.hidden_size)


class ReTraceQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = ReTraceQwen3Model

    def forward(
        self,
        input_ids,
        positions,
        inputs_embeds=None,
        retrace_memory=None,
        retrace_runtime=False,
    ):
        if getattr(self.config, "retrace_enabled", True):
            if not retrace_runtime and not getattr(self, "retrace_profile_run", False):
                raise RuntimeError(
                    "ReTrace requires its Ascend proposer; rejected-trajectory memory was not supplied by this runtime"
                )
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
            f"retrace.{part}.weight" for part in ("correction", "value", "gate")
        }
        if not required.issubset(names):
            raise ValueError(
                "Independent ReTrace checkpoint is missing conditioning weights"
            )
        if any("markov" in name or "confidence_head" in name for name in names):
            raise ValueError(
                "Legacy DSpARK+ReTrace weights need their original architecture"
            )
        return super().load_weights(weights)
