"""Check native versus windowed FlashMLA on PPU DCP."""

import argparse

import torch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--bf16-kv", action="store_true")
    args = parser.parse_args()
    from sglang.kernels.ops.quantization.dsa_quant_k_cache import quantize_k_cache
    from sglang.srt.hardware_backend.ppu.attention.flash_mla import (
        DcpWindowedScheduler,
        flash_mla_with_kvcache,
        get_mla_metadata,
    )
    from sglang.srt.utils import is_sail

    if not is_sail():
        parser.error("this probe requires a SAIL PPU")
    if args.rows < 1:
        parser.error("--rows must be positive")
    scheduler = DcpWindowedScheduler()
    torch.manual_seed(41)
    rows, topk, page = args.rows, 2048, 64
    fp8 = not args.bf16_kv
    key = (rows, fp8, topk)
    kv = torch.randn(topk // page, page, 1, 576, device="cuda", dtype=torch.bfloat16)
    if not args.bf16_kv:
        kv = quantize_k_cache(kv, pow2_scale=True)
    q = torch.randn(rows, 1, 128, 576, device="cuda", dtype=torch.bfloat16)
    q[:, :, 96:].zero_()
    lengths = torch.empty(rows, dtype=torch.int32, device="cuda")
    cache_seqlens = torch.full_like(lengths, topk)
    row = torch.arange(topk, dtype=torch.int32, device="cuda").view(1, 1, topk)
    indices = torch.empty((rows, 1, topk), dtype=torch.int32, device="cuda")
    block_table = torch.empty((rows, 0), dtype=torch.int32, device="cuda")

    def update(seed, skewed, empty=False):
        rng = torch.Generator(device="cuda").manual_seed(seed)
        lengths.random_(
            0 if skewed else 160, topk + 1 if skewed else 401, generator=rng
        )
        n = min(rows, 6)
        lengths[:n].copy_(lengths.new_tensor([0, 1, 63, 64, 65, topk][:n]))
        if empty:
            lengths.zero_()
        indices.copy_(torch.where(row < lengths[:, None, None], row, -1))

    def attention(sched):
        return flash_mla_with_kvcache(
            q,
            kv,
            block_table,
            cache_seqlens,
            512,
            sched,
            softmax_scale=192**-0.5,
            indices=indices,
            topk_length=lengths,
            is_fp8_kvcache=fp8,
            kv_format="V32",
        )

    def native():
        return attention(get_mla_metadata()[0])

    update(1, False)
    oracle = get_mla_metadata()[0]
    assert scheduler.prepare(oracle, lengths, key, topk)
    attention(oracle)
    scheduler.observe(oracle, rows, key)

    def check_out(actual, expected):
        torch.testing.assert_close(actual[0], expected[0], atol=0.02, rtol=0.02)
        torch.testing.assert_close(actual[1], expected[1], atol=0.01, rtol=0.01)

    def windowed():
        sched = get_mla_metadata()[0]
        observe = scheduler.prepare(sched, lengths, key, topk)
        assert not observe and sched.have_initialized
        meta, splits = sched.tile_scheduler_metadata, sched.num_splits
        output = attention(sched)
        assert meta.data_ptr() == sched.tile_scheduler_metadata.data_ptr()
        assert splits.data_ptr() == sched.num_splits.data_ptr()
        return output

    for seed, skewed, empty in ((1, False, False), (2, True, False), (3, False, True)):
        update(seed, skewed, empty)
        check_out(windowed(), native())

    update(4, False)

    def capture(fn):
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fn()
        return graph, output

    native_graph, native_output = capture(native)
    window_graph, window_output = capture(windowed)
    for seed, skewed, empty in ((4, False, False), (5, True, False), (6, False, True)):
        update(seed, skewed, empty)
        native_graph.replay()
        window_graph.replay()
        check_out(window_output, native_output)

    update(7, False)
    for name, graph in (("native+MLA", native_graph), ("windowed+MLA", window_graph)):
        for _ in range(20):
            graph.replay()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        print(f"{name}: {start.elapsed_time(end) * 10:.2f} us/replay")


if __name__ == "__main__":
    main()
