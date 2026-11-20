"""CPU planner/stream guards; no SWA pool or GPU work."""

import types
import unittest
from unittest.mock import patch

from sglang.srt.model_executor.swa_page_clear_warmup import (
    _check_schedule_stream,
    _private_swa_inputs,
    plan_swa_page_clear_warmup,
    prepare_swa_allocator_startup,
    prepare_swa_page_clear_startup,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def settings(**changes):
    result = dict(
        enabled=True,
        is_ppu=True,
        is_swa_allocator=True,
        swa_req_ring=False,
        page_size=256,
        mapping_dtype="torch.int64",
        representative_dtype="torch.int64",
        debug_mode=False,
    )
    return {**result, **changes}


def fake_torch(*, device="cuda:0", current_device=0, stream=17, capture=False):
    def parse(value):
        return (
            value
            if not isinstance(value, str)
            else types.SimpleNamespace(
                type=value.split(":")[0],
                index=int(value.split(":")[1]) if ":" in value else None,
            )
        )

    return types.SimpleNamespace(
        device=parse,
        cuda=types.SimpleNamespace(
            current_device=lambda: current_device,
            current_stream=lambda dev: types.SimpleNamespace(cuda_stream=stream),
            is_current_stream_capturing=lambda: capture,
        ),
    ), types.SimpleNamespace(device=parse(device), cuda_stream=17)


class TestSwaPageClearWarmup(CustomTestCase):
    def test_page_derived_strides_and_tail_mask(self):
        for page in (1, 64, 256):
            cases = plan_swa_page_clear_warmup(**settings(page_size=page))
            self.assertEqual(
                [case["index_stride"] for case in cases], list(dict.fromkeys((1, page)))
            )
            self.assertTrue(
                all(case["num_pages"] == case["block_pages"] + 1 for case in cases)
            )

    def test_debug_flag_uses_actual_page(self):
        for page in (1, 64, 256):
            for debug in (False, True):
                self.assertTrue(
                    all(
                        case["check_page_mappings"] == (page > 1 and debug)
                        for case in plan_swa_page_clear_warmup(
                            **settings(page_size=page, debug_mode=debug)
                        )
                    )
                )

    def test_disabled_nonppu_nonswa_ring_skip_before_torch(self):
        class ForbiddenTorch:
            def __getattr__(self, name):
                raise AssertionError("Torch accessed on ineligible path: " + name)

        for change in (
            dict(enabled=False),
            dict(is_ppu=False),
            dict(is_swa_allocator=False),
            dict(swa_req_ring=True),
        ):
            with patch.dict("sys.modules", {"torch": ForbiddenTorch()}):
                self.assertIsNone(
                    prepare_swa_page_clear_startup(
                        device=None, schedule_stream=None, **settings(**change)
                    )
                )

    def test_bad_page_dtype_or_flag_rejected(self):
        for change in (
            dict(page_size=0),
            dict(page_size=True),
            dict(mapping_dtype="torch.int32"),
            dict(representative_dtype="torch.float32"),
            dict(debug_mode=1),
        ):
            with self.assertRaises(ValueError):
                plan_swa_page_clear_warmup(**settings(**change))

    def test_request_rows_and_radix_rows_keep_their_actual_dtype(self):
        for dtype in ("torch.int32", "torch.int64"):
            cases = plan_swa_page_clear_warmup(**settings(representative_dtype=dtype))
            self.assertTrue(
                all(case["representative_dtype"] == dtype for case in cases)
            )

    def test_private_inputs_on_cpu(self):
        import torch

        for page in (1, 64, 256):
            for dtype in ("torch.int32", "torch.int64"):
                for stride in dict.fromkeys((1, page)):
                    with self.subTest(page=page, dtype=dtype, stride=stride):
                        representatives, mapping = _private_swa_inputs(
                            torch,
                            device="cpu",
                            page_size=page,
                            num_pages=3,
                            index_stride=stride,
                            representative_dtype=dtype,
                        )
                        self.assertEqual(str(representatives.dtype), dtype)
                        self.assertEqual(representatives.stride(0), stride)
                        self.assertEqual(
                            representatives.tolist(), [page, 2 * page, 3 * page]
                        )
                        self.assertEqual(mapping.dtype, torch.int64)
                        self.assertEqual(mapping[:page].tolist(), [0] * page)
                        self.assertEqual(
                            mapping[page : 4 * page].tolist(),
                            list(range(5 * page, 8 * page)),
                        )
                        self.assertEqual(
                            mapping[4 * page : -1].tolist(), [-7] * (3 * page)
                        )
                        self.assertEqual(mapping[-1].item(), -1)

    def test_allocator_metadata_only_and_dtype_validation_before_gpu(self):
        class Allocator:
            swa_req_ring = False
            page_size = 256
            full_to_swa_index_mapping = types.SimpleNamespace(dtype="torch.int64")
            swa_attn_allocator = types.SimpleNamespace(debug_mode=False)

        name = "sglang.srt.mem_cache.allocator.swa"
        allocator_module = types.ModuleType(name)
        allocator_module.SWATokenToKVPoolAllocator = Allocator
        target = (
            "sglang.srt.model_executor.swa_page_clear_warmup."
            "prepare_swa_page_clear_startup"
        )
        with (
            patch.dict("sys.modules", {name: allocator_module}),
            patch(target) as prepare,
        ):
            allocator = Allocator()
            options = dict(device="cuda:3", schedule_stream=object())
            prepare_swa_allocator_startup(
                allocator,
                **options,
                representative_dtypes=("torch.int32", "torch.int64", "torch.int32"),
            )
            self.assertEqual(
                [
                    call.kwargs["representative_dtype"]
                    for call in prepare.call_args_list
                ],
                ["torch.int32", "torch.int64"],
            )
            for call in prepare.call_args_list:
                self.assertNotIn("mapping", call.kwargs)
                self.assertEqual(call.kwargs["mapping_dtype"], "torch.int64")
                self.assertIs(
                    call.kwargs["schedule_stream"], options["schedule_stream"]
                )
            prepare.reset_mock()
            for candidate in (object(),):
                prepare_swa_allocator_startup(
                    candidate, **options, representative_dtypes=()
                )
            allocator.swa_req_ring = True
            prepare_swa_allocator_startup(
                allocator, **options, representative_dtypes=()
            )
            prepare.assert_not_called()
            allocator.swa_req_ring = False
            for dtypes in ((), ("torch.int32", "torch.float32")):
                with self.assertRaises(ValueError):
                    prepare_swa_allocator_startup(
                        allocator, **options, representative_dtypes=dtypes
                    )
                prepare.assert_not_called()
            allocator.page_size = 1
            allocator.swa_attn_allocator = object()
            prepare_swa_allocator_startup(
                allocator, **options, representative_dtypes=("torch.int32",)
            )
            self.assertFalse(prepare.call_args.kwargs["debug_mode"])

    def test_correct_schedule_stream(self):
        torch, schedule = fake_torch()
        _check_schedule_stream(torch, schedule.device, schedule)

    def test_wrong_device_stream_and_capture_rejected(self):
        for change in (
            dict(device="cpu"),
            dict(current_device=1),
            dict(stream=19),
            dict(capture=True),
        ):
            torch, schedule = fake_torch(**change)
            with self.assertRaises((ValueError, RuntimeError)):
                _check_schedule_stream(torch, schedule.device, schedule)


if __name__ == "__main__":
    unittest.main()
