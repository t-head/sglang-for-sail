"""SAIL scheduler ABI and expanded weighted-DSA query-row contracts."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa import kpool_plan
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend


def backend(fp4=False):
    return SimpleNamespace(
        device_capability=(8, 9),
        num_q_heads=8,  # Main attention heads must not be used for the indexer.
        num_indexer_heads=32,
        indexer_head_dim=128,
        token_to_kv_pool=SimpleNamespace(use_fp4_indexer=fp4),
        _paged_mqa_schedule_metadata_extra=lambda: (
            1,
            32,
            128,
            torch.int8 if fp4 else torch.float8_e4m3fn,
        ),
    )


@pytest.mark.parametrize("fp4", [False, True])
def test_typed_metadata_and_expanded_rows(fp4):
    b = backend(fp4)
    lens = torch.tensor([[64, 65], [128, 129]], dtype=torch.int32)
    with (
        patch("sglang.srt.layers.attention.dsa_backend._is_sail", True),
        patch(
            "sglang.srt.layers.attention.dsa_backend.resolve_paged_mqa_logits_metadata_fn"
        ) as resolver,
        patch(
            "sglang.srt.layers.attention.dsa_backend.resolve_num_sms", return_value=39
        ),
        envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
    ):
        DeepseekSparseAttnBackend._build_paged_mqa_schedule_metadata(b, lens)
    args, kwargs = resolver.return_value.call_args
    assert args[0].shape == (4, 1)
    torch.testing.assert_close(args[0][:, 0], lens.flatten())
    assert kwargs["metadata_extra"] == (
        1,
        32,
        128,
        torch.int8 if fp4 else torch.float8_e4m3fn,
    )
    if fp4:
        torch.testing.assert_close(
            kwargs["indices"], torch.arange(4, dtype=torch.int32)
        )
    else:
        assert "indices" not in kwargs


def test_indexer_heads_and_dtype():
    with envs.SGLANG_SAIL_BF16_INDEXER.override(False):
        for fp4 in [False, True]:
            assert DeepseekSparseAttnBackend._paged_mqa_schedule_metadata_extra(
                backend(fp4)
            ) == (1, 32, 128, torch.int8 if fp4 else torch.float8_e4m3fn)


def test_pool_schedule_uses_backend_builder():
    seen = []

    def builder(lens):
        seen.append(lens)
        return lens

    with patch.object(kpool_plan, "is_cuda", return_value=True):
        actual = kpool_plan._compute_pool_schedule_metadata(
            torch.tensor([0, 4]), index_page_size=64, schedule_builder=builder
        )
    assert actual is seen[0]
    torch.testing.assert_close(actual, torch.tensor([[1], [4]]))


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("fp4", torch.int8),
        ("int8", torch.int8),
        ("bf16", torch.bfloat16),
        ("fp8", torch.float8_e4m3fn),
    ],
)
def test_fallback_metadata_uses_native_logical_format(mode, expected):
    from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer

    indexer = SimpleNamespace(
        use_fp4=mode == "fp4",
        use_int8=mode == "int8",
        bf16_indexer=mode == "bf16",
        n_heads=32,
        head_dim=128,
    )
    assert Indexer._sail_paged_mqa_metadata_extra(indexer) == (1, 32, 128, expected)
