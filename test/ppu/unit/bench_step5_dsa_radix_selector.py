"""Cost of the SAIL JIT Step5 DSA radix region selector.

Only meaningful on a PPU host where the JIT radix selector module builds;
the script reports SKIP otherwise.

    python test/ppu/unit/bench_step5_dsa_radix_selector.py
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.step5.selection_ppu import (
    _jit_step5_radix_module,
)

TOPK = 512
REGION_BLOCK = 8
PAGE_SIZE = 64
LIVE_GAP = 8
# One scores copy per replay iteration keeps L2 cold; above this budget the
# working set already exceeds L2 many times over, so the copy is skipped.
CLONE_BUDGET_BYTES = 512 << 20
REPLAY_ITERS = 100


def _available() -> bool:
    """Whether the JIT radix selector module can build on this machine."""
    try:
        _jit_step5_radix_module()
    except Exception:
        return False
    return True


@marker.parametrize("rows", [1, 16, 64, 512])
@marker.parametrize("width", [1024, 8192, 65536, 131080], [65536])
@marker.benchmark("stage", ["sail_jit_radix"])
def benchmark(rows: int, width: int, stage: str):
    """Rank the live prefix of ``rows`` score rows of ``width`` regions."""
    device = torch.device("cuda")
    live = width - LIVE_GAP
    scores = torch.randn((rows, 1, width), dtype=torch.float32, device=device)
    positions = torch.full(
        (rows,), live * REGION_BLOCK, dtype=torch.int64, device=device
    )
    pages = torch.zeros(1, dtype=torch.int32, device=device)
    chunks = (width + 16383) // 16384
    candidates = torch.empty(
        (rows, 1, max(chunks, 1) * TOPK), dtype=torch.int32, device=device
    )
    regions = torch.empty((rows, 1, TOPK + 1), dtype=torch.int32, device=device)
    starts = torch.empty_like(regions)
    clone = scores.numel() * scores.element_size() * REPLAY_ITERS < CLONE_BUDGET_BYTES
    return marker.do_bench(
        _jit_step5_radix_module().radix_select,
        input_args=(
            scores,
            positions,
            pages,
            candidates,
            regions,
            starts,
            REGION_BLOCK,
            PAGE_SIZE,
            TOPK,
            False,
        ),
        graph_clone_args=(0,) if clone else None,
        replay_iters=REPLAY_ITERS,
        memory_args=(0,),
        memory_output=None,
    )


if __name__ == "__main__":
    if not torch.cuda.is_available() or not _available():
        raise SystemExit("SKIP: the JIT radix selector is not available")
    benchmark.run()
