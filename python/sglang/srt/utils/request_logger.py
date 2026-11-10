# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

import dataclasses
import hashlib
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from sglang.srt.environ import envs
from sglang.srt.sampling.watermarking.core import redact_watermark_secrets
from sglang.srt.utils.log_utils import create_log_targets, log_json

if TYPE_CHECKING:
    import fastapi

    from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput

logger = logging.getLogger(__name__)

# (field names, min chars): string values of these fields, at any depth, are
# logged as their length plus a hash instead of verbatim.
_Abbreviation = Tuple[FrozenSet[str], int]

_DEFAULT_WHITELISTED_HEADERS = ["x-smg-routing-key"]
WHITELISTED_HEADERS = _DEFAULT_WHITELISTED_HEADERS + [
    h.lower() for h in envs.SGLANG_LOG_REQUEST_HEADERS.get()
]


def _extract_whitelisted_headers(
    request: Optional[fastapi.Request],
) -> Optional[Dict[str, str]]:
    if request is None:
        return None
    return {h: v for h in WHITELISTED_HEADERS if (v := request.headers.get(h))}


class RequestLogger:
    def __init__(
        self,
        log_requests: bool,
        log_requests_level: int,
        log_requests_format: str,
        log_requests_target: Optional[List[str]],
    ):
        self.log_requests = log_requests
        self.log_requests_level = log_requests_level
        self.log_requests_format = log_requests_format
        self.log_requests_target = log_requests_target

        self.metadata: Tuple[Optional[int], Optional[Set[str]], Optional[Set[str]]] = (
            self._compute_metadata()
        )
        self.abbreviation = self._compute_abbreviation()
        self.targets = self._setup_targets()

        self.log_exceeded_ms = envs.SGLANG_LOG_REQUEST_EXCEEDED_MS.get()

    def _setup_targets(self) -> List[logging.Logger]:
        return create_log_targets(
            targets=self.log_requests_target, name_prefix=__name__
        )

    def configure(
        self,
        log_requests: Optional[bool] = None,
        log_requests_level: Optional[int] = None,
        log_requests_format: Optional[str] = None,
        log_requests_target: Optional[List[str]] = None,
    ) -> None:
        if log_requests is not None:
            self.log_requests = log_requests
        if log_requests_level is not None:
            self.log_requests_level = log_requests_level
        if log_requests_format is not None:
            self.log_requests_format = log_requests_format
        if log_requests_target is not None:
            self.log_requests_target = log_requests_target

        self.metadata = self._compute_metadata()
        self.abbreviation = self._compute_abbreviation()
        self.targets = self._setup_targets()

    def log_received_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        tokenizer: Any = None,
        request: Optional[fastapi.Request] = None,
    ) -> None:
        if not self.log_requests:
            return

        max_length, skip_names, _ = self.metadata
        headers = _extract_whitelisted_headers(request)
        if self.log_requests_format == "json":
            log_data = {
                "rid": obj.rid,
                "obj": _transform_data_for_logging(
                    obj, max_length, skip_names, self.abbreviation
                ),
            }
            if headers:
                log_data["headers"] = headers
            log_json(self.targets, "request.received", log_data)
        else:
            headers_str = f", headers={headers}" if headers else ""
            obj_str = _dataclass_to_string_truncated(
                obj, max_length, skip_names=skip_names, abbreviation=self.abbreviation
            )
            self._log(f"Receive: obj={obj_str}{headers_str}")

        # FIXME: This is a temporary fix to get the text from the input ids.
        # We should remove this once we have a proper way.
        if (
            self.log_requests_level >= 2
            and obj.text is None
            and obj.input_ids is not None
            and tokenizer is not None
        ):
            if obj.input_ids and isinstance(obj.input_ids[0], list):
                # Prefill node warmup while PD disaggregated.
                decoded = [
                    tokenizer.decode(_input_ids, skip_special_tokens=False)
                    for _input_ids in obj.input_ids
                ]
            else:
                decoded = tokenizer.decode(obj.input_ids, skip_special_tokens=False)
            obj.text = decoded

    def log_openai_received_request(
        self,
        obj: Any,
        request: Optional[fastapi.Request] = None,
    ) -> None:
        """Log the raw OpenAI request payload before request adaptation/tokenization."""
        max_length, _, _ = self.metadata
        max_length = max_length if max_length is not None else 2048
        headers = _extract_whitelisted_headers(request)

        if hasattr(obj, "model_dump"):
            obj_to_log = obj.model_dump(exclude_none=True)
        else:
            obj_to_log = obj

        if self.log_requests_format == "json":
            log_data = {
                "obj": _transform_data_for_logging(obj_to_log, max_length=max_length),
            }
            if headers:
                log_data["headers"] = headers
            log_json(self.targets, "request.received.openai", log_data)
        else:
            headers_str = f", headers={headers}" if headers else ""
            self._log(
                f"Receive OpenAI: obj={_dataclass_to_string_truncated(obj_to_log, max_length)}{headers_str}"
            )

    def log_finished_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        out: Any,
        request: Optional[fastapi.Request] = None,
    ) -> None:
        if not self.log_requests:
            return

        e2e_latency_ms = out["meta_info"].get("e2e_latency", 0) * 1000
        if self.log_exceeded_ms > 0 and e2e_latency_ms < self.log_exceeded_ms:
            return

        max_length, skip_names, out_skip_names = self.metadata
        headers = _extract_whitelisted_headers(request)
        if self.log_requests_format == "json":
            log_data = {
                "rid": obj.rid,
                "obj": _transform_data_for_logging(
                    obj, max_length, skip_names, self.abbreviation
                ),
            }
            if headers:
                log_data["headers"] = headers
            log_data["out"] = _transform_data_for_logging(
                out, max_length, out_skip_names, self.abbreviation
            )
            log_json(self.targets, "request.finished", log_data)
        else:
            obj_str = _dataclass_to_string_truncated(
                obj, max_length, skip_names=skip_names, abbreviation=self.abbreviation
            )
            out_str = f", out={_dataclass_to_string_truncated(out, max_length, skip_names=out_skip_names, abbreviation=self.abbreviation)}"
            headers_str = f", headers={headers}" if headers else ""
            self._log(f"Finish: obj={obj_str}{headers_str}{out_str}")

    def _compute_metadata(
        self,
    ) -> Tuple[Optional[int], Optional[Set[str]], Optional[Set[str]]]:
        max_length: Optional[int] = None
        skip_names: Optional[Set[str]] = None
        out_skip_names: Optional[Set[str]] = None
        if self.log_requests:
            if self.log_requests_level == 0:
                max_length = 1 << 30
                skip_names = {
                    "text",
                    "input_ids",
                    "input_embeds",
                    "image_data",
                    "audio_data",
                    "video_data",
                    "lora_path",
                    "sampling_params",
                }
                out_skip_names = {"text", "output_ids", "embedding"}
            elif self.log_requests_level == 1:
                max_length = 1 << 30
                skip_names = {
                    "text",
                    "input_ids",
                    "input_embeds",
                    "image_data",
                    "audio_data",
                    "video_data",
                    "lora_path",
                }
                out_skip_names = {"text", "output_ids", "embedding"}
            elif self.log_requests_level == 2:
                max_length = 2048
            elif self.log_requests_level == 3:
                max_length = 1 << 30
            else:
                raise ValueError(
                    f"Invalid --log-requests-level: {self.log_requests_level=}"
                )
        return max_length, skip_names, out_skip_names

    def _compute_abbreviation(self) -> Optional[_Abbreviation]:
        # Level 0 drops sampling_params altogether; level 3 is verbatim by definition.
        if not self.log_requests or self.log_requests_level not in (1, 2):
            return None
        fields = frozenset(envs.SGLANG_LOG_REQUEST_ABBREVIATE_FIELDS.get())
        if not fields:
            return None
        return fields, max(0, envs.SGLANG_LOG_REQUEST_ABBREVIATE_MIN_CHARS.get())

    def _log(self, msg: str) -> None:
        for target in self.targets:
            target.info(msg)


def _abbreviated(key: Any, value: Any, abbreviation: Optional[_Abbreviation]):
    """The short form of ``value`` if ``key`` is to be abbreviated, else None."""
    if abbreviation is None or not isinstance(value, str):
        return None
    fields, min_chars = abbreviation
    if key not in fields or len(value) <= min_chars:
        return None
    digest = hashlib.blake2b(
        value.encode("utf-8", "surrogatepass"), digest_size=8
    ).hexdigest()
    return f"<{key}: {len(value)} chars, blake2b={digest}>"


# TODO unify this w/ `_transform_data_for_logging` if we find performance enough
def _dataclass_to_string_truncated(
    data: Any,
    max_length: int = 2048,
    skip_names: Optional[Set[str]] = None,
    abbreviation: Optional[_Abbreviation] = None,
) -> str:
    if skip_names is None:
        skip_names = set()
    if isinstance(data, str):
        if len(data) > max_length:
            half_length = max_length // 2
            return f"{repr(data[:half_length])} ... {repr(data[-half_length:])}"
        else:
            return f"{repr(data)}"
    elif isinstance(data, (list, tuple)):
        if len(data) > max_length:
            half_length = max_length // 2
            return (
                str(redact_watermark_secrets(data[:half_length]))
                + " ... "
                + str(redact_watermark_secrets(data[-half_length:]))
            )
        else:
            return str(redact_watermark_secrets(data))
    elif isinstance(data, dict):
        data = redact_watermark_secrets(data)
        return (
            "{"
            + ", ".join(
                f"'{k}': {_abbreviated(k, v, abbreviation) or _dataclass_to_string_truncated(v, max_length, abbreviation=abbreviation)}"
                for k, v in data.items()
                if k not in skip_names
            )
            + "}"
        )
    elif dataclasses.is_dataclass(data):
        fields = dataclasses.fields(data)
        values = redact_watermark_secrets(
            {f.name: getattr(data, f.name) for f in fields}
        )
        return (
            f"{data.__class__.__name__}("
            + ", ".join(
                f"{f.name}={_abbreviated(f.name, values[f.name], abbreviation) or _dataclass_to_string_truncated(values[f.name], max_length, abbreviation=abbreviation)}"
                for f in fields
                if f.name not in skip_names
            )
            + ")"
        )
    else:
        return str(data)


def _transform_data_for_logging(
    data: Any,
    max_length: int = 2048,
    skip_names: Optional[Set[str]] = None,
    abbreviation: Optional[_Abbreviation] = None,
) -> Any:
    if skip_names is None:
        skip_names = set()
    if isinstance(data, str):
        if len(data) > max_length:
            half_length = max_length // 2
            return data[:half_length] + "..." + data[-half_length:]
        return data
    elif isinstance(data, (list, tuple)):
        if len(data) > max_length:
            half_length = max_length // 2
            return list(data[:half_length]) + ["..."] + list(data[-half_length:])
        return [
            _transform_data_for_logging(v, max_length, abbreviation=abbreviation)
            for v in data
        ]
    elif isinstance(data, dict):
        data = redact_watermark_secrets(data)
        return {
            k: _abbreviated(k, v, abbreviation)
            or _transform_data_for_logging(v, max_length, abbreviation=abbreviation)
            for k, v in data.items()
            if k not in skip_names
        }
    elif dataclasses.is_dataclass(data):
        fields = dataclasses.fields(data)
        values = redact_watermark_secrets(
            {f.name: getattr(data, f.name) for f in fields}
        )
        return {
            f.name: _abbreviated(f.name, values[f.name], abbreviation)
            or _transform_data_for_logging(
                values[f.name], max_length, abbreviation=abbreviation
            )
            for f in fields
            if f.name not in skip_names
        }
    elif isinstance(data, (int, float, bool, type(None))):
        return data
    else:
        return str(data)
