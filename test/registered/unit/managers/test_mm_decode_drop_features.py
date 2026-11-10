"""PD decode drops multimodal feature tensors before dispatch.

Decode receives prompt KV from prefill and never runs the multimodal encoder,
so with SGLANG_DISAGG_DECODE_DROP_MM_FEATURES the tokenizer ships only item
metadata. These tests pin that the metadata the decode scheduler reads
(hash/pad_value, offsets, model_specific_data) survives, that the wire payload
shrinks to metadata size, and that the switch only applies to decode.
"""

from array import array
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.mm_utils import (  # noqa: E402
    MultiModalityDataPaddingPatternMultimodalTokens,
    drop_mm_features_for_decode,
)
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.multimodal.transport.cuda_ipc import (  # noqa: E402
    CudaIpcTensorTransportProxy,
)
from sglang.srt.utils.msgpack_utils import dec_hook, enc_hook, ext_hook  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

IMAGE_TOKEN = 7


def _image_item(offset, *, hash_value=None):
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=hash_value,
        offsets=[offset],
        feature=torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8),
        model_specific_data={"image_grid_thw": torch.tensor([[1, 4, 4]])},
    )


def test_drop_keeps_metadata_and_precomputed_hash():
    item = _image_item((1, 2), hash_value=0xABCD)
    item.set_pad_value()
    pad_value = item.pad_value

    drop_mm_features_for_decode([item])

    assert item.feature is None
    assert item.precomputed_embeddings is None
    assert item.hash == 0xABCD
    assert item.pad_value == pad_value
    assert item.offsets == [(1, 2)]
    assert torch.equal(
        item.model_specific_data["image_grid_thw"], torch.tensor([[1, 4, 4]])
    )


def test_drop_derives_pad_value_from_feature_first():
    item = _image_item((1, 2))
    reference = MultimodalDataItem(
        modality=Modality.IMAGE, feature=item.feature.clone()
    )
    reference.set_pad_value()

    drop_mm_features_for_decode([item])

    assert item.feature is None
    assert item.hash == reference.hash
    assert item.pad_value == reference.pad_value


def test_drop_leaves_transport_proxies_alone():
    proxy = object.__new__(CudaIpcTensorTransportProxy)
    item = MultimodalDataItem(modality=Modality.IMAGE, hash=1, feature=proxy)

    drop_mm_features_for_decode([item])

    assert item.feature is proxy


def test_decode_scheduler_builds_inputs_from_metadata_only():
    items = [
        _image_item((1, 2), hash_value=0x1111),
        _image_item((4, 5), hash_value=0x2222),
    ]
    input_ids = [0, IMAGE_TOKEN, IMAGE_TOKEN, 0, IMAGE_TOKEN, IMAGE_TOKEN]
    full = MultimodalProcessorOutput(
        mm_items=items, input_ids=input_ids, im_token_id=IMAGE_TOKEN
    )
    encoder = msgspec.msgpack.Encoder(enc_hook=enc_hook)
    full_size = len(encoder.encode(full))

    drop_mm_features_for_decode(items)
    wire = encoder.encode(full)
    received = msgspec.msgpack.Decoder(
        MultimodalProcessorOutput, dec_hook=dec_hook, ext_hook=ext_hook
    ).decode(wire)

    assert len(wire) * 10 < full_size
    mm_inputs = MultimodalInputs.from_processor_output(received)
    pads = [item.pad_value for item in items]
    assert [item.pad_value for item in mm_inputs.mm_items] == pads
    assert all(item.feature is None for item in mm_inputs.mm_items)
    padded = MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
        array("q", received.input_ids), mm_inputs
    )
    assert padded == array("q", [0, pads[0], pads[0], 0, pads[1], pads[1]])


@pytest.mark.parametrize(
    "mode, enabled, expected",
    [
        ("decode", True, True),
        ("decode", False, False),
        ("prefill", True, False),
        ("null", True, False),
    ],
)
def test_switch_applies_to_decode_only(mode, enabled, expected):
    disagg = SimpleNamespace(disaggregation_mode=mode, language_only=False)
    tm = object.__new__(TokenizerManager)
    with (
        envs.SGLANG_DISAGG_DECODE_DROP_MM_FEATURES.override(enabled),
        patch("sglang.srt.managers.tokenizer_manager.get_disagg", return_value=disagg),
    ):
        tm.init_disaggregation(start_pd_bootstrap_service=False)

    assert tm._drop_decode_mm_features is expected
