# SPDX-License-Identifier: Apache-2.0
"""Compare four scoring variants; time wipe, per-layer K pack and scoring, not allocation."""

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
import triton

from sglang.test.kernels.step5_scoring import (
    VARIANTS,
    compare_scores,
    make_case,
    make_production_runner,
    make_runner,
    selection_snapshot,
)


def _capture(run):
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    return graph


def _measure_interleaved(graphs, *, repeats, cold, flush, seed):
    samples = {name: [] for name in graphs}
    rng = random.Random(seed)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        names = list(graphs)
        rng.shuffle(names)
        for name in names:
            if cold:
                flush.add_(1)
            else:
                graphs[name].replay()
            begin.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000)
    return samples


def _bench_shape(*, rows, context, args, flush):
    width = triton.cdiv(triton.cdiv(context, 8), 256) * 256
    case = make_case(lengths=(rows,), prefixes=(context - rows,), width=width, seed=args.seed)
    results = []
    for mode in args.modes.split(","):
        fp8 = mode == "fp8"
        for group in map(int, args.groups.split(",")):
            runners = {name: make_runner(case, name, fp8=fp8, group_rows=group,
                                         block_regions=args.block_regions, grid_tiles=args.grid_tiles)
                       for name in VARIANTS}
            baseline = runners["baseline"]["run"]().clone()
            qualifications = {}
            for name, runner in runners.items():
                actual = runner["run"]()
                qualifications[name] = compare_scores(actual, baseline)
            graphs = {name: _capture(runner["run"]) for name, runner in runners.items()}
            pack_graphs = {name: _capture(runners[name]["pack"])
                           for name in ("compact", "compact_tile")}
            for cache in args.cache_modes.split(","):
                samples = _measure_interleaved(
                    graphs, repeats=args.repeats, cold=cache == "cold", flush=flush, seed=args.seed,
                )
                pack_samples = _measure_interleaved(
                    pack_graphs, repeats=args.repeats, cold=cache == "cold", flush=flush, seed=args.seed,
                )
                medians = {name: statistics.median(values) for name, values in samples.items()}
                eligible = [name for name in VARIANTS if qualifications[name]["bitwise_equal"]
                            and qualifications[name]["top512_equal"]]
                winner = min(eligible, key=medians.get)
                print(f"\nrows={rows} context={context} width={width} mode={mode} group={group} cache={cache}", flush=True)
                for name in VARIANTS:
                    q = qualifications[name]
                    print(f"  {name:12s} {medians[name]:9.2f} us  "
                          f"speedup={medians['baseline']/medians[name]:.3f}x  "
                          f"exact={q['bitwise_equal']} top512={q['top512_equal']} "
                          f"mismatches={q['mismatches']} max_abs={q['max_abs']:.6g}", flush=True)
                print(f"  合格最优: {winner}; pack中位数(us): "
                      f"{ {name: round(statistics.median(values), 2) for name, values in pack_samples.items()} }", flush=True)
                results.append(dict(rows=rows, context=context, width=width, mode=mode, group_rows=group,
                                    cache=cache, configs={name: runner["config"] for name, runner in runners.items()},
                                    qualifications=qualifications, medians_us=medians, samples_us=samples,
                                    pack_samples_us=pack_samples, winner=winner))
            del graphs, pack_graphs, runners, baseline
    return results


def _qualify_production_graph(case, runners, prefix):
    """Graph replay must refresh scratch after history, owner and K updates."""
    saved = {name: case[name].clone() for name in ("positions", "owners", "keys")}
    compact = runners["compact"]
    graph = _capture(compact["run"])
    workspace = compact["workspace"]
    columns = torch.arange(513, device=workspace.packed.device)
    try:
        for history, active, zero_keys in (
            (0, True, True), (prefix, False, False), (prefix, True, False),
        ):
            case["positions"].copy_(saved["positions"] - prefix + history)
            if active:
                case["owners"].copy_(saved["owners"])
            else:
                case["owners"].zero_()
            if zero_keys:
                case["keys"].zero_()
            else:
                case["keys"].copy_(saved["keys"])
            graph.replay()
            expected_scores = runners["baseline"]["score"]()
            if not torch.equal(compact["out"].view(torch.int32), expected_scores.view(torch.int32)):
                raise RuntimeError("生产 graph 重放的 score 未刷新")
            expected = selection_snapshot(runners["baseline"])
            valid = columns[None, :] < workspace.counts[:, None]
            if bool((valid & (workspace.packed < 0)).any()):
                raise RuntimeError("生产 graph 重放的有效 packed 输出中存在负项")
            selected = torch.where(valid, workspace.packed, -1).sort(dim=1).values
            actual = (selected, workspace.counts, workspace.lengths)
            if not all(torch.equal(a, b) for a, b in zip(actual, expected)):
                raise RuntimeError("生产 graph 重放的 selector 输出未刷新")
    finally:
        for name, value in saved.items():
            case[name].copy_(value)
    return True


def _bench_production_shape(*, rows, context, args, flush):
    """Time public-entry A/B for scoring alone and the scoring/selector/pack chain."""
    from sglang.srt.environ import envs

    width = triton.cdiv(triton.cdiv(context, 8), 256) * 256
    case = make_case(
        lengths=(rows,), prefixes=(context - rows,), width=width, seed=args.seed,
    )
    results = []
    for mode in args.modes.split(","):
        with (
            envs.SGLANG_SAIL_STEP5_SCORING_COMPACT.override(True),
            envs.SGLANG_SAIL_STEP5_FP8_INDEXER.override(mode == "fp8"),
        ):
            runners = {
                name: make_production_runner(case, compact=compact)
                for name, compact in (("baseline", False), ("compact", True))
            }
            scratch = runners["compact"]["workspace"].compact_scoring
            if scratch is None:
                raise RuntimeError("测试 shape 未启用 compact，不能作为优化性能证据")
            if any(r["workspace"].ppu_selector is None for r in runners.values()):
                raise RuntimeError("真实 radix selector 不可用，不能作为生产链路对照")
            baseline = runners["baseline"]["score"]().clone()
            qualification = compare_scores(runners["compact"]["score"](), baseline)
            expected = selection_snapshot(runners["baseline"])
            actual = selection_snapshot(runners["compact"])
            qualification["selector_equal"] = all(
                torch.equal(a, b) for a, b in zip(actual, expected)
            )
            selector_valid = True
            for runner in runners.values():
                workspace = runner["workspace"]
                columns = torch.arange(513, device=workspace.packed.device)
                valid = columns[None, :] < workspace.counts[:, None]
                selector_valid &= not bool((valid & (workspace.packed < 0)).any())
            qualification["selector_valid"] = selector_valid
            if not all(qualification[k] for k in (
                "bitwise_equal", "top512_equal", "selector_equal", "selector_valid",
            )):
                raise RuntimeError(f"生产入口对齐失败: {qualification}")
            qualification["graph_refresh_equal"] = _qualify_production_graph(
                case, runners, context - rows,
            )
            for stage, entry in (("score", "score"), ("score_selector", "run")):
                graphs = {name: _capture(runner[entry]) for name, runner in runners.items()}
                for cache in args.cache_modes.split(","):
                    samples = _measure_interleaved(
                        graphs, repeats=args.repeats, cold=cache == "cold",
                        flush=flush, seed=args.seed,
                    )
                    medians = {name: statistics.median(v) for name, v in samples.items()}
                    speedup = medians["baseline"] / medians["compact"]
                    print(
                        f"production rows={rows} context={context} mode={mode} "
                        f"stage={stage} cache={cache} baseline={medians['baseline']:.2f}us "
                        f"compact={medians['compact']:.2f}us speedup={speedup:.3f}x "
                        f"exact={qualification['bitwise_equal']} "
                        f"selector={qualification['selector_equal']}", flush=True,
                    )
                    results.append(dict(
                        rows=rows, context=context, width=width, mode=mode, stage=stage,
                        cache=cache, group_rows=scratch.group_rows,
                        compact_bytes=scratch.keys.numel() * scratch.keys.element_size(),
                        qualification=qualification, medians_us=medians,
                        samples_us=samples, speedup=speedup,
                    ))
                del graphs
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", default="8192,16384,32768,65536,76479")
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--tails", action="store_true")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--groups", default="8")
    parser.add_argument("--block-regions", type=int, default=32)
    parser.add_argument("--grid-tiles", type=int, default=32)
    parser.add_argument("--modes", default="fp8,bf16")
    parser.add_argument("--cache-modes", default="warm,cold")
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(mode not in ("fp8", "bf16") for mode in args.modes.split(",")):
        parser.error("modes 必须是 fp8/bf16")
    if any(mode not in ("warm", "cold") for mode in args.cache_modes.split(",")):
        parser.error("cache-modes 必须是 warm/cold")
    if args.repeats < 3:
        parser.error("至少需要 3 次重复")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    metadata = dict(device=props.name, capability=torch.cuda.get_device_capability(),
                    torch=torch.__version__, triton=triton.__version__, sms=props.multi_processor_count,
                    seed=args.seed, cache_flush_bytes=128 * 1024 * 1024,
                    scope="预量化 Q；计时包含 wipe + 每次 K pack + scoring；不含分配/selector",
                    inputs="synthetic；physical pages 随机打散，page 内连续；不是服务现场 dump",
                    statistics="随机交错顺序；CUDA graph 单次调用；中位数；cold flush 不计入时间")
    if args.production:
        metadata["scope"] = "公共入口；每层 wipe + pack + scoring，另测包含真实 radix selector/address fold；不含分配或整模型"
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    flush = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    shapes = [(args.rows, context) for context in map(int, args.contexts.split(","))]
    if args.tails:
        shapes += [(2751, 76479), (5016, 76479)]
    if any(rows <= 0 or context < rows for rows, context in shapes):
        parser.error("context 必须不小于正数 rows")
    results = []
    bench_shape = _bench_production_shape if args.production else _bench_shape
    for rows, context in shapes:
        results.extend(bench_shape(rows=rows, context=context, args=args, flush=flush))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(metadata=metadata, results=results), indent=2,
                                         ensure_ascii=False) + "\n")
        print(f"结果已保存: {args.output}", flush=True)


if __name__ == "__main__":
    main()
