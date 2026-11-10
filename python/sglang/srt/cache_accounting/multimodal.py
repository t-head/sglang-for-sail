# SPDX-License-Identifier: Apache-2.0
"""Strong media identities prepared before feature transport or PD disposal.

This module uses NumPy for CPU validation without importing the GPU serving tree.
Feature hashing is synchronous and should run in the tokenizer's worker thread;
freezing snapshots final tokens and validates spans with already prepared digests.
"""

import hashlib
import math
import struct
from array import array
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum

import numpy as np

_DIGEST_KEY = "cache_accounting_digest"
_ARTIFACT_KEY = "cache_accounting_artifact_key"
_TRANSPORT_KEYS = frozenset(
    {
        "_sglang_defer_cuda_ipc_feature_reconstruction",
        "_sglang_borrow_cuda_ipc_feature",
        "_sglang_cuda_ipc_feature_copy_event",
        "_sglang_retained_cuda_ipc_feature_proxy",
    }
)


@dataclass(frozen=True)
class CacheArtifactIdentity:
    """A processor-owned artifact identity that public JSON cannot construct."""

    artifact_key: str


def _digest_bytes(value: str) -> bytes:
    if isinstance(value, str) and value.startswith("sha256:"):
        value = value[7:]
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdefABCDEF" for c in value)
    ):
        raise ValueError("A complete SHA-256 media identity is required")
    return bytes.fromhex(value)


def _field(hasher, tag: bytes, payload) -> None:
    hasher.update(tag)
    hasher.update(struct.pack(">Q", len(payload)))
    hasher.update(payload)


def _unwrap_feature(value):
    """Read known engine transport payloads, never their process-local handles."""
    cls = type(value)
    if cls.__module__.startswith("sglang.srt."):
        if cls.__name__ == "ShmPointerMMData":
            # materialize() closes/unlinks this shared-memory transport. The
            # scheduler still owns the payload after accounting has read it.
            tensor = value.tensor
            if tensor is None:
                raise ValueError("Shared-memory media tensor is unavailable")
            return tensor
        if hasattr(value, "reconstruct_on_target_device"):
            # Reconstruction acknowledges a pool consumer and can recycle its
            # slot. Accounting must never consume the model worker's lease.
            tensor = getattr(value, "reconstruct_tensor", None)
            if tensor is None:
                raise ValueError("Media identity must be prepared before CUDA IPC")
            return tensor
    return value


def _is_torch_tensor(value) -> bool:
    module = type(value).__module__
    if module.split(".")[0] != "torch" and not module.startswith("sglang.srt."):
        return False
    import torch

    return isinstance(value, torch.Tensor)


def _hash_producer_shared_memory(hasher, value) -> bool:
    """Read a producer SHM view without consuming its transport ownership."""
    cls = type(value)
    if not (
        cls.__module__.startswith("sglang.srt.")
        and cls.__name__ == "ShmPointerMMData"
        and value.tensor is None
    ):
        return False
    from multiprocessing import shared_memory

    dtype = str(value.dtype)
    element_sizes = {
        "torch.bool": 1,
        "torch.uint8": 1,
        "torch.int8": 1,
        "torch.int16": 2,
        "torch.uint16": 2,
        "torch.float16": 2,
        "torch.bfloat16": 2,
        "torch.int32": 4,
        "torch.uint32": 4,
        "torch.float32": 4,
        "torch.int64": 8,
        "torch.uint64": 8,
        "torch.float64": 8,
        "torch.complex64": 8,
        "torch.complex128": 16,
        "torch.float8_e4m3fn": 1,
        "torch.float8_e4m3fnuz": 1,
        "torch.float8_e5m2": 1,
        "torch.float8_e5m2fnuz": 1,
    }
    shape = tuple(value.shape)
    if dtype not in element_sizes or any(
        type(dim) is not int or dim < 0 for dim in shape
    ):
        raise ValueError("Unsupported shared-memory media layout")
    size = math.prod(shape) * element_sizes[dtype]
    handle = shared_memory.SharedMemory(name=value.shm_name, create=False)
    try:
        if size > handle.size:
            raise ValueError("Shared-memory media layout exceeds its buffer")
        # Match a normal torch tensor's canonical encoding. Do not convert to
        # NumPy or make a feature-sized copy just to hash these shared bytes.
        hasher.update(b"torch[")
        _hash_value(hasher, dtype)
        _hash_value(hasher, shape)
        payload = handle.buf[:size].toreadonly()
        try:
            _field(hasher, b"data", payload)
        finally:
            payload.release()
        hasher.update(b"]")
    finally:
        # Only close this temporary reader. Never unlink or mutate the owner.
        handle.close()
    return True


def _hash_value(hasher, value) -> None:
    if _hash_producer_shared_memory(hasher, value):
        return
    value = _unwrap_feature(value)
    if isinstance(value, Enum):
        _field(hasher, b"enum", type(value).__qualname__.encode())
        _hash_value(hasher, value.value)
    elif is_dataclass(value) and not isinstance(value, type):
        _field(
            hasher,
            b"dataclass",
            (type(value).__module__ + "." + type(value).__qualname__).encode(),
        )
        _hash_value(
            hasher, {field.name: getattr(value, field.name) for field in fields(value)}
        )
    elif value is None:
        hasher.update(b"none;")
    elif type(value) is bool:
        hasher.update(b"true;" if value else b"false;")
    elif type(value) is int:
        _field(hasher, b"int", str(value).encode())
    elif type(value) is float:
        _field(hasher, b"float", struct.pack(">d", value))
    elif isinstance(value, str):
        _field(hasher, b"str", value.encode())
    elif isinstance(value, (bytes, bytearray, memoryview)):
        _field(hasher, b"bytes", memoryview(value).cast("B"))
    elif isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Media metadata keys must be strings")
        hasher.update(b"mapping[")
        for key in sorted(value):
            _hash_value(hasher, key)
            _hash_value(hasher, value[key])
        hasher.update(b"]")
    elif isinstance(value, (tuple, list)):
        hasher.update(b"tuple[" if isinstance(value, tuple) else b"list[")
        for part in value:
            _hash_value(hasher, part)
        hasher.update(b"]")
    elif type(value).__module__.split(".")[0] == "numpy":
        import numpy as np

        snapshot = np.asarray(value)
        if snapshot.dtype.hasobject:
            raise ValueError("Object arrays cannot identify media features")
        shape = tuple(snapshot.shape)
        snapshot = np.ascontiguousarray(snapshot)
        hasher.update(b"numpy[")
        _hash_value(hasher, snapshot.dtype.str)
        _hash_value(hasher, shape)
        _field(hasher, b"data", memoryview(snapshot.reshape(-1).view(np.uint8)))
        hasher.update(b"]")
    elif _is_torch_tensor(value):
        import torch

        snapshot = value.detach().to("cpu").contiguous()
        hasher.update(b"torch[")
        _hash_value(hasher, str(snapshot.dtype))
        _hash_value(hasher, tuple(snapshot.shape))
        raw = snapshot.reshape(-1).view(torch.uint8).numpy()
        _field(hasher, b"data", memoryview(raw).cast("B"))
        hasher.update(b"]")
    elif type(value).__module__.split(".")[0] == "PIL":
        from PIL import Image

        if not isinstance(value, Image.Image):
            raise ValueError("Unsupported PIL media feature")
        hasher.update(b"pil[")
        _hash_value(hasher, value.mode)
        _hash_value(hasher, tuple(value.size))
        _hash_value(
            hasher, value.palette.tobytes() if value.palette is not None else None
        )
        _hash_value(hasher, value.info.get("transparency"))
        _field(hasher, b"data", value.tobytes())
        hasher.update(b"]")
    else:
        raise ValueError("Unsupported multimodal identity value")


def _prepare_media_digests(mm_inputs) -> None:
    """Attach full digests using trusted artifacts or actual processor features.

    ``cache_accounting_artifact_key`` is internal processor-owned metadata,
    never a user-supplied mm_hashes/content_hash assertion. An artifact key
    already binds content and preprocessing settings. Other items are hashed
    from actual features/embeddings and stable model metadata, including shape
    and dtype. Neither the 64-bit item hash nor its 30-bit pad projection is
    used. Calling again recomputes rather than trusting a preexisting digest.
    """
    items = getattr(mm_inputs, "mm_items", None)
    if not items:
        raise ValueError("Multimodal inputs do not contain media items")
    # A failed recomputation must never leave an externally supplied or stale
    # digest available for the Scheduler to accept as confirmed media identity.
    for item in items:
        metadata = getattr(item, "model_specific_data", None)
        if isinstance(metadata, dict):
            metadata.pop(_DIGEST_KEY, None)
    prepared = []
    for item in items:
        metadata = getattr(item, "model_specific_data", None)
        if not isinstance(metadata, dict):
            raise ValueError("Multimodal item metadata is unavailable")
        hasher = hashlib.sha256(b"sglang:cache:media:v1\x00")
        _hash_value(hasher, getattr(item, "modality", None))
        artifact = metadata.get(_ARTIFACT_KEY)
        if isinstance(artifact, CacheArtifactIdentity):
            _field(hasher, b"artifact", _digest_bytes(artifact.artifact_key))
        else:
            feature = getattr(item, "feature", None)
            embedding = getattr(item, "precomputed_embeddings", None)
            if feature is None and embedding is None:
                raise ValueError(
                    "Media features and strong artifact identity are absent"
                )
            if feature is not None and embedding is not None:
                raise ValueError("Ambiguous media feature and embedding payloads")
            _field(hasher, b"kind", b"feature" if feature is not None else b"embedding")
            _hash_value(hasher, feature if feature is not None else embedding)
            stable_metadata = {
                key: value
                for key, value in metadata.items()
                if not key.startswith("cache_accounting_")
                and key not in _TRANSPORT_KEYS
            }
            _hash_value(hasher, stable_metadata)
        prepared.append((metadata, hasher.hexdigest()))
    # Do not leave a partly prepared set when a later item cannot be identified.
    for metadata, digest in prepared:
        metadata[_DIGEST_KEY] = digest


def prepare_media_digests(mm_inputs) -> None:
    """Prepare strong identities, reporting unsupported payloads as ValueError."""
    try:
        _prepare_media_digests(mm_inputs)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Multimodal media identity could not be prepared") from exc


def freeze_cache_input_metadata(input_ids, mm_inputs) -> dict:
    """Copy final token positions and prepared media spans; never hash features.

    Scheduler errors are represented explicitly so an unsupported or incomplete
    media shape can use physical accounting instead of aborting inference.
    Offsets on MultimodalDataItem are inclusive; the wire intervals are half-open.
    """
    try:
        # Preserve the owned snapshot and array's integer conversion semantics.
        # View that copy, never the caller's live input, for native validation.
        ids = array("q", input_ids)
        ids_view = np.frombuffer(ids, dtype="q")
        ids_view.flags.writeable = False
        items = getattr(mm_inputs, "mm_items", None)
        if not items:
            raise ValueError("Missing media items")
        span_records = []
        pad_values = set()
        for item in items:
            metadata = getattr(item, "model_specific_data", None)
            digest = _digest_bytes(metadata[_DIGEST_KEY]).hex()
            offsets = getattr(item, "offsets", None)
            if not offsets:
                raise ValueError("Missing final media offsets")
            padding = getattr(item, "pad_values", None) or (
                getattr(item, "pad_value", None),
            )
            if any(type(pad) is not int for pad in padding):
                raise ValueError("Missing final media pad value")
            if any(not -(1 << 63) <= pad < (1 << 63) for pad in padding):
                raise ValueError("Final media offsets cover non-media tokens")
            pad_values.update(padding)
            for start, inclusive_end in offsets:
                if (
                    type(start) is not int
                    or type(inclusive_end) is not int
                    or not 0 <= start <= inclusive_end < len(ids)
                ):
                    raise ValueError("Invalid final media offsets")
                span_records.append((start, inclusive_end + 1, digest, padding))
        span_records.sort(key=lambda span: (span[0], span[1]))
        previous_end = 0
        for start, end, _, _ in span_records:
            if start < previous_end:
                raise ValueError("Overlapping media intervals")
            previous_end = end

        # Validate all spans together: per-span NumPy reductions and per-gap
        # membership checks become expensive with thousands of short spans.
        # Keep padding patterns attached while sorting. Each span restarts its
        # pattern, matching MultimodalDataItem.padding_sequence().
        starts = np.fromiter(
            (span[0] for span in span_records), dtype=np.intp, count=len(span_records)
        )
        ends = np.fromiter(
            (span[1] for span in span_records), dtype=np.intp, count=len(span_records)
        )
        lengths = ends - starts
        pattern_lengths = np.fromiter(
            (len(span[3]) for span in span_records),
            dtype=np.intp,
            count=len(span_records),
        )
        pattern_values = np.fromiter(
            (pad for span in span_records for pad in span[3]), dtype="q"
        )
        if np.all(pattern_lengths == 1):
            expected_pads = np.repeat(pattern_values, lengths)
        else:
            # Index the flattened patterns in NumPy rather than expanding
            # media tokens in Python. Offsets are relative to each span.
            indices = np.arange(lengths.sum(), dtype=np.intp)
            indices -= np.repeat(np.cumsum(lengths) - lengths, lengths)
            indices %= np.repeat(pattern_lengths, lengths)
            indices += np.repeat(
                np.cumsum(pattern_lengths) - pattern_lengths, lengths
            )
            expected_pads = pattern_values[indices]
        coverage_delta = np.zeros(len(ids) + 1, dtype=np.int8)
        # Starts and ends are each unique after rejecting overlaps. An end
        # may equal the next start: accumulate both so coverage remains one.
        coverage_delta[starts] += 1
        coverage_delta[ends] -= 1
        covered = np.cumsum(coverage_delta[:-1], dtype=np.int8).astype(bool)
        if np.any(ids_view[covered] != expected_pads):
            raise ValueError("Final media offsets cover non-media tokens")

        # Media IDs may lie outside uint32. Only uncovered positions are text,
        # and even a uint32 pad ID is forbidden there. Check membership once.
        text = ids_view[~covered]
        if text.size and (text.min() < 0 or text.max() > 0xFFFFFFFF):
            raise ValueError("Media token is not covered by final offsets")
        text_pad_values = np.asarray(
            [pad for pad in pad_values if 0 <= pad <= 0xFFFFFFFF], dtype="q"
        )
        if text_pad_values.size and np.any(np.isin(text, text_pad_values)):
            raise ValueError("Media token is not covered by final offsets")
        spans = [(start, end, digest) for start, end, digest, _ in span_records]
        return {"input_ids": ids, "media_spans": spans}
    except Exception:
        return {"error": "invalid_multimodal_metadata"}
