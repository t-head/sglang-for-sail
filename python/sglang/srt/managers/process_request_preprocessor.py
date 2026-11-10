# Copyright 2023-2026 SGLang Team
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
"""Out-of-process request preprocessing for the HTTP frontend.

A tokenizer worker's asyncio loop both admits new requests and streams the
outputs of every in-flight request. Admission runs CPU-heavy Python: message
dumps and copies, chat-template rendering and tokenization of the full prompt.
The default ``thread`` mode moves that work onto a helper thread, but the thread
still holds the GIL for most of its runtime, so a 300k-token prompt stalls the
loop -- and with it the next token of every other request this worker streams
-- by hundreds of milliseconds.

``SGLANG_REQUEST_PREPROCESSOR_MODE=process`` runs the same conversion in
dedicated child processes. Each child rebuilds the tokenizer, chat template and
serving handler from the worker's config, so the loop only pays for handing the
request over and taking the result back.

Fallbacks keep requests working rather than failing them: handlers that do not
opt in (``supports_process_preprocessing``), a conversion that reads more of the
HTTP request than its headers, a pool that fails to start or breaks, and a child
whose tokenizer/template fingerprint differs from the parent's all use the
in-process thread path. Children replay the config overrides the parent had
recorded when the pool started (e.g. parsers resolved from the chat template);
later overrides are not propagated.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import multiprocessing
import os
import pickle
import signal
import threading
import time
import traceback
from array import array
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)

# Request fields too large to deep-copy for change detection. Conversion only
# reads them (it renders dumps/copies), so the child reports them back only if
# the attribute was rebound or resized.
_IDENTITY_TRACKED_FIELDS = frozenset({"messages"})
# Token id lists at least this long cross the process boundary as int64 arrays:
# unpickling a list costs ~40ns per id on the receiving side, an array ~nothing.
_PACK_MIN_IDS = 256
_MAX_POOL_RESTARTS = 3
_PROBE_TEXT = (
    'Hello, world! 你好，世界。 <tool_call> {"k": [1, 2.5, null]}\n'
    "\tdef f(x):\n        return x ** 2  # comment\n"
)


class NeedsInProcessConversion(BaseException):
    """The conversion read request state the child does not have.

    A ``BaseException`` so broad ``except Exception`` handlers inside the
    conversion cannot swallow it and continue with partial data.
    """


class _RemoteHTTPException(Exception):
    """Picklable carrier for an ``HTTPException`` raised during conversion."""

    def __init__(self, status_code: int, detail: Any, headers: Any):
        super().__init__(status_code, detail, headers)
        self.status_code = status_code
        self.detail = detail
        self.headers = headers


class _RemoteError(Exception):
    """Picklable carrier for a conversion error that cannot be pickled itself."""

    def __init__(self, type_name: str, message: str, is_value_error: bool):
        super().__init__(type_name, message, is_value_error)
        self.type_name = type_name
        self.message = message
        self.is_value_error = is_value_error


class _RemoteTraceback(Exception):
    def __init__(self, tb: str):
        super().__init__(tb)
        self.tb = tb

    def __str__(self) -> str:
        return self.tb


# ----------------------------------------------------------------------------
# Helpers shared by both sides
# ----------------------------------------------------------------------------


def snapshot_headers(raw_request: Any) -> Optional[List[Tuple[bytes, bytes]]]:
    """The request headers as raw (name, value) byte pairs, or None without a request."""
    if raw_request is None:
        return None
    headers = getattr(raw_request, "headers", None)
    if headers is None:
        return []
    raw = getattr(headers, "raw", None)
    if raw is not None:
        return list(raw)
    items = headers.items() if hasattr(headers, "items") else headers
    return [(_header_bytes(str(k).lower()), _header_bytes(str(v))) for k, v in items]


def _header_bytes(value: str) -> bytes:
    try:
        return value.encode("latin-1")
    except UnicodeEncodeError:
        return value.encode("utf-8")


def pack_token_ids(ids: Any) -> Tuple[str, Any]:
    if isinstance(ids, list):
        if ids and isinstance(ids[0], list):
            return ("batch", [pack_token_ids(x) for x in ids])
        if len(ids) >= _PACK_MIN_IDS:
            try:
                return ("array", array("q", ids))
            except (TypeError, OverflowError):
                pass
    return ("raw", ids)


def unpack_token_ids(packed: Tuple[str, Any]) -> Any:
    kind, value = packed
    if kind == "array":
        return value.tolist()
    if kind == "batch":
        return [unpack_token_ids(x) for x in value]
    return value


def compute_preprocessing_fingerprint(tokenizer: Any, template_manager: Any) -> Dict:
    """What request conversion depends on, cheap enough to compare at startup."""
    fingerprint: Dict[str, Any] = {"tokenizer_class": type(tokenizer).__qualname__}
    try:
        fingerprint["vocab_size"] = len(tokenizer)
    except Exception:
        fingerprint["vocab_size"] = None
    special_tokens = list(getattr(tokenizer, "all_special_tokens", None) or [])[:32]
    try:
        fingerprint["probe_ids"] = list(
            tokenizer.encode(_PROBE_TEXT + "".join(special_tokens))
        )
    except Exception as e:
        fingerprint["probe_ids"] = f"<encode failed: {type(e).__name__}>"
    template = getattr(tokenizer, "chat_template", None)
    fingerprint["chat_template_sha1"] = hashlib.sha1(
        json.dumps(template, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    for attr in (
        "chat_template_name",
        "completion_template_name",
        "jinja_template_content_format",
    ):
        fingerprint[attr] = getattr(template_manager, attr, None)
    return fingerprint


def _values_equal(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:
        return False


def _safe_len(value: Any) -> Optional[int]:
    try:
        return len(value)
    except TypeError:
        return None


def _snapshot_request_state(request: Any):
    fields = getattr(request, "__dict__", None)
    if fields is None or not hasattr(request, "model_fields_set"):
        return None
    tracked = {}
    for name, value in fields.items():
        if name in _IDENTITY_TRACKED_FIELDS:
            tracked[name] = (id(value), _safe_len(value))
        else:
            tracked[name] = copy.deepcopy(value)
    extra = copy.deepcopy(getattr(request, "__pydantic_extra__", None) or {})
    return tracked, set(request.model_fields_set), extra


def _collect_request_changes(request: Any, snapshot) -> Dict[str, Any]:
    """Fields the conversion assigned or mutated, to replay on the parent's copy."""
    tracked, fields_set, extra = snapshot
    newly_set = set(request.model_fields_set) - fields_set
    changes: Dict[str, Any] = {}
    for name, value in request.__dict__.items():
        if name not in tracked or name in newly_set:
            changes[name] = value
        elif name in _IDENTITY_TRACKED_FIELDS:
            if tracked[name] != (id(value), _safe_len(value)):
                changes[name] = value
        elif not _values_equal(tracked[name], value):
            changes[name] = value
    for name, value in (getattr(request, "__pydantic_extra__", None) or {}).items():
        if name not in extra or not _values_equal(extra[name], value):
            changes[name] = value
    return changes


def _portable_exception(exc: BaseException) -> BaseException:
    try:
        from starlette.exceptions import HTTPException as StarletteHTTPException
    except ImportError:  # pragma: no cover - starlette ships with fastapi
        StarletteHTTPException = None
    if StarletteHTTPException is not None and isinstance(exc, StarletteHTTPException):
        return _RemoteHTTPException(
            exc.status_code, exc.detail, getattr(exc, "headers", None)
        )
    try:
        clone = pickle.loads(pickle.dumps(exc))
        if type(clone) is type(exc) and str(clone) == str(exc):
            return exc
    except Exception:
        pass
    return _RemoteError(
        f"{type(exc).__module__}.{type(exc).__qualname__}",
        str(exc),
        isinstance(exc, ValueError),
    )


def _local_exception(exc: BaseException) -> BaseException:
    if isinstance(exc, _RemoteHTTPException):
        from fastapi import HTTPException

        return HTTPException(
            status_code=exc.status_code, detail=exc.detail, headers=exc.headers
        )
    if isinstance(exc, _RemoteError):
        if exc.is_value_error:
            return ValueError(exc.message)
        return RuntimeError(f"{exc.type_name}: {exc.message}")
    return exc


# ----------------------------------------------------------------------------
# Child process side
# ----------------------------------------------------------------------------


class _HeadersOnlyRequest:
    """Stands in for the HTTP request inside a child: conversion reads headers only."""

    __slots__ = ("headers",)

    def __init__(self, raw_headers: List[Tuple[bytes, bytes]]):
        from starlette.datastructures import Headers

        self.headers = Headers(raw=raw_headers)

    def __getattr__(self, name: str):
        raise NeedsInProcessConversion(f"request conversion read raw_request.{name}")


class _ChildState:
    def __init__(self, server_args, tokenizer_manager_cls, config_overrides):
        from sglang.srt.parser.template_manager import TemplateManager
        from sglang.srt.runtime_context import (
            get_context,
            get_model,
            get_serving,
            publish,
        )

        publish(server_args, role="tokenizer")
        context = get_context()
        for source, fields in config_overrides:
            try:
                context.override(source, **fields)
            except Exception as e:
                logger.warning(
                    "Request preprocessor could not replay config override %s=%r: %r",
                    source,
                    fields,
                    e,
                )

        self.tokenizer_manager = tokenizer_manager_cls.create_for_request_preprocessing(
            server_args
        )
        self.template_manager = TemplateManager()
        self.template_manager.initialize_templates(
            tokenizer_manager=self.tokenizer_manager,
            model_path=get_model().model_path,
            chat_template=get_serving().chat_template,
            completion_template=get_serving().completion_template,
        )
        self._handlers: Dict[type, Any] = {}

    def handler(self, handler_cls: type):
        handler = self._handlers.get(handler_cls)
        if handler is None:
            handler = handler_cls(self.tokenizer_manager, self.template_manager)
            self._handlers[handler_cls] = handler
        return handler


_CHILD_STATE: Optional[_ChildState] = None


def _exit_when_parent_dies(parent_pid: int) -> None:
    # PR_SET_PDEATHSIG fires when the *thread* that spawned us exits, and pool
    # workers may be spawned from short-lived threads; poll the parent instead.
    def watch():
        while True:
            time.sleep(1.0)
            if os.getppid() != parent_pid:
                os._exit(0)

    threading.Thread(target=watch, name="parent-watchdog", daemon=True).start()


def _child_init(
    server_args, tokenizer_manager_cls, config_overrides, parent_pid, log_prefix
):
    global _CHILD_STATE
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _exit_when_parent_dies(parent_pid)
    try:
        import setproctitle

        setproctitle.setproctitle("sglang::request_preprocessor")
    except Exception:
        pass

    from sglang.srt.plugins import load_plugins
    from sglang.srt.utils import configure_logger

    configure_logger(server_args, prefix=log_prefix)
    load_plugins()
    _CHILD_STATE = _ChildState(server_args, tokenizer_manager_cls, config_overrides)


def _child_warmup(handler_classes: Sequence[type]):
    state = _CHILD_STATE
    for handler_cls in handler_classes:
        state.handler(handler_cls)
    fingerprint = compute_preprocessing_fingerprint(
        state.tokenizer_manager.tokenizer, state.template_manager
    )
    return fingerprint, os.getpid()


def _child_convert(handler_cls: type, request: Any, raw_headers):
    handler = _CHILD_STATE.handler(handler_cls)
    raw_request = None if raw_headers is None else _HeadersOnlyRequest(raw_headers)
    snapshot = _snapshot_request_state(request)
    try:
        adapted, processed = handler._convert_to_internal_request(request, raw_request)
    except NeedsInProcessConversion as e:
        return ("fallback", str(e))
    except Exception as e:
        return ("error", _portable_exception(e), traceback.format_exc())

    if processed is request and snapshot is not None:
        processed_payload = ("changes", _collect_request_changes(request, snapshot))
    else:
        processed_payload = ("object", processed)
    packed_ids = None
    if getattr(adapted, "input_ids", None) is not None:
        packed_ids = pack_token_ids(adapted.input_ids)
        adapted.input_ids = None
    return ("ok", adapted, packed_ids, processed_payload)


# ----------------------------------------------------------------------------
# Parent (tokenizer worker) side
# ----------------------------------------------------------------------------


class ProcessRequestPreprocessor:
    """Runs serving handlers' ``_convert_to_internal_request`` in child processes."""

    def __init__(
        self,
        *,
        tokenizer_manager: TokenizerManager,
        template_manager: Any,
        handler_classes: Sequence[type],
        num_processes: int,
    ):
        from sglang.srt.runtime_context import get_context

        self._num_processes = max(1, int(num_processes))
        self._handler_classes = tuple(
            cls
            for cls in handler_classes
            if getattr(cls, "supports_process_preprocessing", False)
        )
        self._expected_fingerprint = compute_preprocessing_fingerprint(
            tokenizer_manager.tokenizer, template_manager
        )
        self._initargs = (
            tokenizer_manager.server_args,
            type(tokenizer_manager),
            get_context().overrides_log(),
            os.getpid(),
            f" RequestPreprocessor(parent={os.getpid()})",
        )
        # Fail in the caller, not in a child, when the config cannot cross over.
        pickle.dumps(self._initargs)

        self._lock = threading.Lock()
        self._pool: Optional[ProcessPoolExecutor] = None
        self._ready = False
        self._restarts = 0
        self._disabled_reason: Optional[str] = None
        self._warned: set = set()
        self._start_pool()

    # -- lifecycle -------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready and self._disabled_reason is None

    def accepts(self, handler: Any) -> bool:
        return self.ready and type(handler) in self._handler_classes

    def _start_pool(self) -> None:
        pool = ProcessPoolExecutor(
            max_workers=self._num_processes,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_child_init,
            initargs=self._initargs,
        )
        with self._lock:
            self._pool = pool
            self._ready = False
        started = time.monotonic()
        pending = [self._num_processes]

        def on_warmup_done(future: Future) -> None:
            if pool is not self._pool:
                return
            try:
                fingerprint, pid = future.result()
            except BaseException as e:
                self._disable(f"a preprocessor process failed to start: {e!r}")
                return
            if fingerprint != self._expected_fingerprint:
                mismatched = sorted(
                    k
                    for k in set(fingerprint) | set(self._expected_fingerprint)
                    if fingerprint.get(k) != self._expected_fingerprint.get(k)
                )
                self._disable(
                    f"process {pid} built a different tokenizer/template than the "
                    f"tokenizer worker (mismatched: {mismatched})"
                )
                return
            with self._lock:
                pending[0] -= 1
                if pending[0] or pool is not self._pool:
                    return
                self._ready = True
            logger.info(
                "Request preprocessor: %d process(es) ready in %.1fs; chat template "
                "rendering and tokenization now run outside the tokenizer worker.",
                self._num_processes,
                time.monotonic() - started,
            )

        # Submitting from this (long-lived) thread spawns the workers now, so
        # their startup overlaps server startup instead of the first requests.
        for _ in range(self._num_processes):
            pool.submit(_child_warmup, self._handler_classes).add_done_callback(
                on_warmup_done
            )

    def _disable(self, reason: str) -> None:
        with self._lock:
            if self._disabled_reason is not None:
                return
            self._disabled_reason = reason
            pool, self._pool = self._pool, None
            self._ready = False
        logger.error(
            "Request preprocessor disabled, falling back to the in-process thread: %s",
            reason,
        )
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def _handle_broken_pool(
        self, pool: ProcessPoolExecutor, exc: BaseException
    ) -> None:
        with self._lock:
            if pool is not self._pool or self._disabled_reason is not None:
                return
            self._restarts += 1
            restarts = self._restarts
        if restarts > _MAX_POOL_RESTARTS:
            self._disable(
                f"the process pool broke {restarts} times; last error: {exc!r}"
            )
            return
        logger.error(
            "Request preprocessor pool broke (%r); restarting it (%d/%d). Requests use "
            "the in-process thread until it is ready again.",
            exc,
            restarts,
            _MAX_POOL_RESTARTS,
        )
        pool.shutdown(wait=False, cancel_futures=True)
        self._start_pool()

    def shutdown(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
            self._ready = False
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def _warn_once(self, key: Any, msg: str, *args: Any) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(msg, *args)

    # -- requests ----------------------------------------------------------------

    async def convert(
        self, handler: Any, request: Any, raw_request: Any
    ) -> Optional[Tuple[Any, Any]]:
        """``handler._convert_to_internal_request(request, raw_request)`` in a child.

        Returns None when the caller should run the conversion in-process instead.
        Conversion errors are re-raised here with the same type the in-process
        path would raise (ValueError, HTTPException, ...).
        """
        pool = self._pool
        if pool is None or not self.accepts(handler):
            if pool is not None and type(handler) in self._handler_classes:
                self._warn_once(
                    "not-ready",
                    "Request preprocessor processes are not ready yet; converting "
                    "on the in-process thread until they are.",
                )
            return None
        try:
            headers = snapshot_headers(raw_request)
            result = await asyncio.get_running_loop().run_in_executor(
                pool, _child_convert, type(handler), request, headers
            )
        except BrokenProcessPool as e:
            self._handle_broken_pool(pool, e)
            return None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._warn_once(
                ("convert-error", type(e)),
                "Out-of-process request conversion failed (%r); using the "
                "in-process thread for this request.",
                e,
            )
            return None

        status = result[0]
        if status == "fallback":
            self._warn_once(
                ("fallback", result[1]),
                "Request conversion needs the in-process thread: %s",
                result[1],
            )
            return None
        if status == "error":
            _, exc, remote_tb = result
            raise _local_exception(exc) from _RemoteTraceback(remote_tb)

        _, adapted, packed_ids, (kind, value) = result
        if packed_ids is not None:
            adapted.input_ids = unpack_token_ids(packed_ids)
        if kind == "changes":
            for name, field_value in value.items():
                setattr(request, name, field_value)
            processed = request
        else:
            processed = value
        return adapted, processed
