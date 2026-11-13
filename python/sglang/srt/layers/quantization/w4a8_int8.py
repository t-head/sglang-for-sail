from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from sglang.srt.layers.quantization.unquant import (
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.utils import is_sail, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.models.utils import WeightsMapper

logger = logging.getLogger(__name__)

_is_sail = is_sail()


class W4A8Int8Config(QuantizationConfig):
    """Config class for W4A8 Int8 Quantization.

    - Weight: static, per-channel, symmetric, packed INT4
    - Activation: dynamic, per-token, symmetric, INT8
    - Compute: INT8 (weights are dequantized to INT8 in-kernel)
    """

    def __init__(self, quant_config: Dict[str, Any] = {}):
        super().__init__()
        self.quant_description = quant_config
        self.is_dynamic = quant_config.get("is_dynamic", False)
        ignore = cast(List[str], quant_config.get("ignore", []))
        self.ignore = ignore if ignore is not None else []
        packed_modules_mapping = quant_config.get("packed_modules_mapping", {})
        self.packed_modules_mapping = (
            packed_modules_mapping if packed_modules_mapping is not None else {}
        )
        # W4 quantization parameters
        self.weight_bits = 4
        self.pack_factor = 8 // self.weight_bits

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_name(cls) -> str:
        return "w4a8_int8"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> W4A8Int8Config:
        return cls(config)

    def apply_weight_name_mapper(self, hf_to_sglang_mapper: WeightsMapper):
        """Map checkpoint module names used by the quantization ignore list."""
        if self.ignore:
            self.ignore = list(
                dict.fromkeys(hf_to_sglang_mapper.apply_list(self.ignore))
            )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8LinearMethod

        if isinstance(layer, LinearBase):
            if should_ignore_layer(
                prefix, ignore=self.ignore, fused_mapping=self.packed_modules_mapping
            ):
                return UnquantizedLinearMethod()
            # Dense layers of a non-ignored W4A8-Int8 checkpoint carry
            # per-channel INT8 weights; serve them with the W8A8-Int8
            # linear method (same as mixed_precision_w4).
            return W8A8Int8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            if should_ignore_layer(
                prefix, ignore=self.ignore, fused_mapping=self.packed_modules_mapping
            ):
                return UnquantizedFusedMoEMethod()
            return W4A8Int8MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class W4A8Int8MoEMethod(FusedMoEMethodBase):
    """MoE method for W4A8-Int8 quantization.

    Weights are stored as packed INT4 with per-channel scales and are
    dequantized to INT8 in-kernel; activations are dynamically quantized to
    INT8 per-token. Only the PPU DeepGEMM runner backend is supported.
    """

    def __init__(self, quant_config: W4A8Int8Config):
        assert _is_sail, "W4A8Int8 MoE only supported on PPU now"
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        # INT4 packed weights for w13 (gate_up_proj) - column parallel.
        # Stored as uint8 before deep_gemm preprocessing.
        w13_weight = torch.nn.Parameter(
            data=torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # INT4 packed weights for w2 (down_proj) - row parallel.
        # Stored as uint8 before deep_gemm preprocessing.
        w2_weight = torch.nn.Parameter(
            data=torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from deep_gemm import preprocess_a8w4_weight

        # Repack the raw packed uint4 (uint8 storage) weights into the layout
        # expected by the m_grouped_gemm_int8_int4_bf16_nt_* kernels.
        layer.w13_weight = Parameter(
            preprocess_a8w4_weight(layer.w13_weight), requires_grad=False
        )
        layer.w2_weight = Parameter(
            preprocess_a8w4_weight(layer.w2_weight), requires_grad=False
        )
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        # W4A8-Int8 is only served by the PPU DeepGEMM runner backend.
        assert deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM, (
            "W4A8Int8 MoE requires JIT DeepGEMM"
        )
        import sglang.srt.layers.moe.moe_runner.ppu_deepgemm_moe  # noqa: F401 – triggers @register_fused_func

        self.runner = MoeRunner(MoeRunnerBackend.DEEP_GEMM, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> torch.Tensor:
        quant_info = DeepGemmMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_int4_w4a8=True,
            per_channel_quant=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            block_shape=None,
        )
        return self.runner.run(dispatch_output, quant_info)
