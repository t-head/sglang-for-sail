"""Vectorized multimodal token ops must match the per-token implementations.

pad_input_tokens (scheduler) and Kimi-K3 placeholder expansion (tokenizer event
loop) run once per request over the whole prompt; both used per-token Python
work that took ~0.1 s at ~1M tokens. These tests pin bit-identical output
against the previous implementations.
"""

import random
from array import array
from collections import defaultdict

import numpy as np
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.mm_utils import (  # noqa: E402
    MultiModalityDataPaddingPatternMultimodalTokens,
)
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.multimodal.processors.kimi_k3 import (  # noqa: E402
    _expand_k3_image_prompt_token_ids,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

IMAGE_TOKEN = 163606
AUDIO_TOKEN = 163700


def _reference_pad(input_ids, mm_inputs):
    """Previous pad_input_tokens body, verbatim apart from the early return."""
    input_ids_tensor = torch.as_tensor(input_ids)
    items_by_modality = defaultdict(list)
    for item in mm_inputs.mm_items:
        items_by_modality[item.modality].append(item)
    token_id_map = {
        Modality.IMAGE: mm_inputs.im_token_id,
        Modality.AUDIO: mm_inputs.audio_token_id,
        Modality.VIDEO: mm_inputs.video_token_id,
    }
    for modality, items in items_by_modality.items():
        token_id = token_id_map.get(modality)
        if not items or token_id is None:
            continue
        for i, item in enumerate(items):
            for offset in items[i].offsets:
                input_ids_tensor[offset[0] : offset[1] + 1] = item.pad_value
    return input_ids_tensor.tolist()


def _random_mm_inputs(rng, length, *, audio_token_id=None):
    items = []
    for modality in (Modality.IMAGE, Modality.AUDIO):
        for _ in range(rng.randint(0, 4)):
            start = rng.randrange(length)
            end = min(length - 1, start + rng.randrange(1, 50))
            item = MultimodalDataItem(
                modality=modality, offsets=[(start, end)], hash=rng.getrandbits(60)
            )
            item.set_pad_value()
            items.append(item)
    rng.shuffle(items)  # overlapping spans across modalities: order matters
    return MultimodalInputs(
        mm_items=items, im_token_id=IMAGE_TOKEN, audio_token_id=audio_token_id
    )


@pytest.mark.parametrize("seed", range(30))
def test_pad_input_tokens_matches_reference(seed):
    rng = random.Random(seed)
    ids = [rng.randrange(150000) for _ in range(rng.randrange(1, 400))]
    mm_inputs = _random_mm_inputs(
        rng, len(ids), audio_token_id=AUDIO_TOKEN if seed % 2 else None
    )
    pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    input_ids = array("q", ids)
    before = input_ids[:]
    result = pattern.pad_input_tokens(input_ids, mm_inputs)
    assert isinstance(result, array) and result.typecode == "q"
    if mm_inputs.mm_items:
        assert list(result) == _reference_pad(ids, mm_inputs)
    else:
        assert result is input_ids  # unchanged early return
    assert input_ids == before  # caller's ids are not modified


@pytest.mark.parametrize("input_ids", [array("i", [1, 2]), [1, 2]])
def test_pad_input_tokens_requires_int64_array(input_ids):
    with pytest.raises(AssertionError, match="input_ids must be array"):
        MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
            input_ids, MultimodalInputs(mm_items=[])
        )


class _RecordingTokenizer:
    def __init__(self, wrap=list, empty_end=False):
        self.calls = []
        self.wrap = wrap
        self.empty_end = empty_end

    def encode(self, text, allowed_special=None):
        self.calls.append(text)
        if self.empty_end and text == "<|media_end|>":
            return self.wrap([])
        return self.wrap([sum(map(ord, text)) % 1000, len(text)])


def _reference_expand(input_ids, image_token_id, counts, sizes, tokenizer):
    """Previous _expand_k3_image_prompt_token_ids body (per-token loop)."""
    input_ids = np.asarray(input_ids, dtype=np.int64)
    output = []
    image_index = 0
    for token_id in input_ids:
        if token_id != image_token_id:
            output.append(int(token_id))
            continue
        width, height = sizes[image_index]
        output.extend(
            tokenizer.encode(
                f"<|media_begin|>image {width}x{height}<|media_content|>",
                allowed_special="all",
            )
        )
        output.extend([image_token_id] * counts[image_index])
        output.extend(tokenizer.encode("<|media_end|>", allowed_special="all"))
        image_index += 1
    return torch.tensor(output, dtype=torch.long).unsqueeze(0)


@pytest.mark.parametrize("seed", range(40))
def test_k3_expansion_matches_reference(seed):
    rng = random.Random(seed)
    ids = [rng.randrange(150000) for _ in range(rng.randrange(0, 300))]
    num_images = rng.randrange(0, 6)
    for _ in range(num_images):
        ids.insert(rng.randrange(len(ids) + 1), IMAGE_TOKEN)
    if seed % 5 == 0 and num_images:  # placeholders at both ends
        ids = [IMAGE_TOKEN] + [t for t in ids if t != IMAGE_TOKEN] + [IMAGE_TOKEN]
        num_images = 2
    counts = [rng.randrange(0, 40) for _ in range(num_images)]
    sizes = [
        (rng.randrange(1, 4000), rng.randrange(1, 4000)) for _ in range(num_images)
    ]
    ref_tok, new_tok = _RecordingTokenizer(), _RecordingTokenizer()
    expected = _reference_expand(ids, IMAGE_TOKEN, counts, sizes, ref_tok)

    for input_ids in (ids, np.asarray(ids), torch.tensor(ids, dtype=torch.long)):
        new_tok.calls.clear()
        result = _expand_k3_image_prompt_token_ids(
            input_ids, IMAGE_TOKEN, counts, sizes, new_tok
        )
        assert result.dtype == torch.long
        assert result.shape == expected.shape
        assert torch.equal(result, expected)
        assert new_tok.calls == ref_tok.calls


@pytest.mark.parametrize(
    "ids, counts",
    [
        ([IMAGE_TOKEN], [3]),
        ([4, IMAGE_TOKEN, 6], [-1]),  # old loop silently emitted no image tokens
        ([IMAGE_TOKEN, IMAGE_TOKEN, 5, IMAGE_TOKEN], [2, 0, 4]),
        ([7, 8, IMAGE_TOKEN, IMAGE_TOKEN], [1, 2]),
        ([], []),
        ([1, 2, 3], []),
    ],
)
@pytest.mark.parametrize(
    "wrap, empty_end",
    [
        (list, False),
        (tuple, False),
        (lambda v: np.asarray(v, dtype=np.int64), False),
        (list, True),
    ],
)
def test_k3_expansion_edge_cases(ids, counts, wrap, empty_end):
    sizes = [(10 * i + 1, 20 * i + 1) for i in range(len(counts))]
    ref_tok = _RecordingTokenizer(wrap, empty_end)
    new_tok = _RecordingTokenizer(wrap, empty_end)
    expected = _reference_expand(ids, IMAGE_TOKEN, counts, sizes, ref_tok)
    result = _expand_k3_image_prompt_token_ids(ids, IMAGE_TOKEN, counts, sizes, new_tok)
    assert result.dtype == torch.long
    assert torch.equal(result, expected)
    assert new_tok.calls == ref_tok.calls


def test_k3_expansion_rejects_placeholder_mismatch():
    with pytest.raises(ValueError, match="placeholder"):
        _expand_k3_image_prompt_token_ids(
            [1, IMAGE_TOKEN, 2],
            IMAGE_TOKEN,
            [3, 4],
            [(1, 1), (2, 2)],
            _RecordingTokenizer(),
        )
