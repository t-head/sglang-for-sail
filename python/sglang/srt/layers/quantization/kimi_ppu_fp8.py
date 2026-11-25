"""Online channel-wise FP8 for BF16 projections in the Kimi text decoder."""

import logging

import torch

from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
from sglang.srt.layers.quantization.fp8_utils import gemm_nt_f8f8bf16
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_device_sm


class KimiPPUDenseFP8Method(UnquantizedLinearMethod):
    channelwise_ppu_fp8 = True

    # A tile holds at most 1M weights (or one output channel for wider layers).
    _reload_tile_elements = 1 << 20

    def _reload_rows(self, weight):
        return max(1, self._reload_tile_elements // weight.shape[1])

    def stage_reload_on_host(self, layer):
        packed = layer.weight.detach()
        host = torch.empty(packed.shape, dtype=torch.bfloat16, device="cpu")
        scales = layer.ppu_channel_scale.detach().cpu().view(-1, 1)
        rows = self._reload_rows(packed)
        for start in range(0, packed.shape[0], rows):
            tile = slice(start, start + rows)
            host[tile].copy_(packed[tile].cpu().float().mul_(scales[tile]))
        layer._ppu_reload_fp8_weight = packed
        layer.weight.data = host

    def finish_reload_from_host(self, layer, target_device):
        packed = layer._ppu_reload_fp8_weight
        rows = self._reload_rows(packed)
        for start in range(0, packed.shape[0], rows):
            tile = slice(start, start + rows)
            if packed.device.type == "cpu":
                # Offload graphs replay copies from these original host addresses.
                q, scale = sglang_per_token_quant_fp8(
                    layer.weight[tile].to(target_device)
                )
                packed[tile].copy_(q)
                layer.ppu_channel_scale[tile].copy_(scale)
            else:
                sglang_per_token_quant_fp8(
                    layer.weight[tile].to(packed.device),
                    output_q=packed[tile],
                    output_s=layer.ppu_channel_scale[tile],
                )
        layer.weight.data = packed
        del layer._ppu_reload_fp8_weight

    def process_weights_after_loading(self, layer):
        if layer.weight.dtype == torch.float8_e4m3fn:
            return
        if hasattr(layer, "_ppu_reload_fp8_weight"):
            sglang_per_token_quant_fp8(
                layer.weight.data.contiguous(),
                output_q=layer._ppu_reload_fp8_weight,
                output_s=layer.ppu_channel_scale,
            )
            layer.weight.data = layer._ppu_reload_fp8_weight
            del layer._ppu_reload_fp8_weight
            return
        weight, scale = sglang_per_token_quant_fp8(layer.weight.data.contiguous())
        layer.weight.data = weight
        layer.register_buffer("ppu_channel_scale", scale)

    def restore_weights_before_loading(self, layer):
        # The checkpoint loaders write BF16 shards into the existing Parameter.
        if layer.weight.dtype == torch.float8_e4m3fn:
            # A captured GEMM references both allocations. Retain its FP8
            # destination while the checkpoint loader writes BF16 shards.
            layer._ppu_reload_fp8_weight = layer.weight.detach()
            layer.weight.data = (
                layer.weight.data.float() * layer.ppu_channel_scale
            ).to(torch.bfloat16)

    def apply(self, layer, x, bias=None):
        output = x.new_empty((*x.shape[:-1], layer.weight.shape[0]))
        return self.apply_into(layer, x, output, bias)

    def apply_into(self, layer, x, output, bias=None):
        if x.numel() == 0:
            return output
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        q, scale = sglang_per_token_quant_fp8(flat)
        shape = (flat.shape[0], layer.weight.shape[0])
        target = (
            output.view(shape) if output.is_contiguous() else output.new_empty(shape)
        )
        gemm_nt_f8f8bf16(q, scale, layer.weight, layer.ppu_channel_scale, target)
        if bias is not None:
            target.add_(bias)
        if not output.is_contiguous():
            output.copy_(target.view_as(output))
        return output


def enable_kimi_dense_fp8(layers):
    if get_device_sm() < 89:
        raise ValueError("Kimi channel-wise W8A8 requires PPU FP8 support")
    projections = {
        "gate_up_proj",
        "down_proj",
        "routed_expert_up_proj",
        "qkv_proj",
        "fused_qkvg_proj",
        "fused_qkvgbfa_proj",
        "fused_qkvbfg_a_proj",
        "fused_qkv_a_proj_with_mqa",
        "q_proj",
        "q_a_proj",
        "kv_a_proj_with_mqa",
        "o_proj",
    }
    parallel = get_parallel()
    replicated_q = set()
    if parallel.dcp_enabled and parallel.dcp_replicate_q_proj:
        from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

        # DCP gathers the standalone MLA Q weight for a BF16 linear.
        # With Q-LoRA it gathers q_b_proj, already outside the selection above.
        replicated_q = {
            attn.q_proj
            for attn in layers.modules()
            if isinstance(attn, DeepseekV2AttentionMLA) and not attn.has_q_b_proj
        }
    count = 0
    for name, layer in layers.named_modules():
        if name.rsplit(".", 1)[-1] not in projections or ".indexer." in name:
            continue
        if layer in replicated_q:
            continue
        if type(getattr(layer, "quant_method", None)) is not UnquantizedLinearMethod:
            continue
        weight = layer.weight
        if (
            weight.dtype == torch.bfloat16
            and weight.shape[0] >= 64
            and weight.shape[1] % 16 == 0
        ):
            layer.quant_method = KimiPPUDenseFP8Method()
            count += 1
    logging.getLogger(__name__).info(
        "Enabled channel-wise W8A8 on %d Kimi projections", count
    )
