# SPDX-License-Identifier: Apache-2.0
"""Prompt-cache accounting in 256-token blocks, isolated by user identity in Redis.

The caller supplies a model/tokenizer/template/adapter namespace
and calls ``account`` once, at the first successful output event. Redis operations
are not retried automatically; errors leave accounting unconfirmed so the caller
can fall back to the engine's actual cache usage.
"""

import asyncio
import hashlib
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Sequence
from uuid import uuid4

logger = logging.getLogger(__name__)

CACHE_BLOCK_SIZE = 256

_KEY_PREFIX = "sglang:cache:v2"
_DOMAIN = b"sglang:cache:v2\x00"
_MAX_SAFE_INTEGER = (1 << 53) - 1
_TTLS = (300, 3600)


class CacheAccountingError(RuntimeError):
    """Accounting did not return a confirmed result; do not invent usage."""


@dataclass(frozen=True)
class CacheUsage:
    ttl_seconds: int
    read_tokens: int = 0
    write_tokens: int = 0
    generation: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.ttl_seconds) is not int or self.ttl_seconds not in _TTLS:
            raise ValueError("ttl_seconds must be 300 or 3600")
        for name in ("read_tokens", "write_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.generation is not None:
            _validate_generation(self.generation)

    def to_dict(self) -> dict[str, int]:
        return {
            "ttl_seconds": self.ttl_seconds,
            "read_tokens": self.read_tokens,
            "write_tokens": self.write_tokens,
        }


@dataclass(frozen=True)
class PageCachePlan:
    page_keys: tuple[str, ...]
    prompt_tokens: int
    page_size: int
    ttl_seconds: int
    mode: str = "implicit"


@dataclass(frozen=True)
class CacheMediaSpan:
    """One media interval in the final expanded input, with a strong identity."""

    start: int
    end: int
    digest: str


def _validate_media_spans(
    media_spans: Sequence[CacheMediaSpan], prompt_tokens: int
) -> tuple[CacheMediaSpan, ...]:
    spans = tuple(media_spans)
    previous_end = 0
    for span in spans:
        if (
            not isinstance(span, CacheMediaSpan)
            or type(span.start) is not int
            or type(span.end) is not int
            or not previous_end <= span.start < span.end <= prompt_tokens
            or not isinstance(span.digest, str)
            or len(span.digest) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in span.digest)
        ):
            raise ValueError("Invalid multimodal cache span")
        previous_end = span.end
    return spans


def _positive_int(name: str, value: int, maximum: int = _MAX_SAFE_INTEGER) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")


def _validate_generation(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("generation must contain 32 lowercase hexadecimal characters")


def _text(name: str, value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    encoded = value.encode("utf-8")
    return struct.pack(">Q", len(encoded)) + encoded


def build_page_cache_plan(
    input_ids: Sequence[int],
    cache_user_id: str,
    namespace: str,
    page_size: int,
    ttl_seconds: int,
    media_spans: Sequence[CacheMediaSpan] = (),
    mode: str = "implicit",
) -> PageCachePlan:
    """Hash complete input pages with a cumulative, fixed-uint32 token encoding.

    Production callers pass ``CACHE_BLOCK_SIZE`` as ``page_size``; the low-level
    parameter remains available for testing. Partial final blocks are excluded
    from read/write counts, independently of the physical cache page size.
    ``namespace`` isolates deployments and input interpretation. TTL is selected
    by the first writer of a prefix, independently of its page identity. Branches
    sharing the first complete page inherit that root's TTL until its generation
    expires; the next writer can then select another TTL. Accounting adds no
    page-count limit. Each first-page prefix defines a Redis Cluster hash slot
    instead of pinning all prefixes for one user identity.
    ``mode`` controls whether misses may be written and does not change identity.
    """
    _positive_int("page_size", page_size)
    if type(ttl_seconds) is not int or ttl_seconds not in _TTLS:
        raise ValueError("ttl_seconds must be 300 (5m) or 3600 (1h)")
    if mode not in ("implicit", "explicit"):
        raise ValueError("mode must be implicit or explicit")
    identity = _text("cache_user_id", cache_user_id) + _text("namespace", namespace)
    prompt_tokens = len(input_ids)
    if prompt_tokens > _MAX_SAFE_INTEGER:
        raise ValueError("prompt_tokens exceeds Redis Lua integer precision")
    page_count = prompt_tokens // page_size
    spans = _validate_media_spans(media_spans, prompt_tokens)

    previous = hashlib.sha256(
        _DOMAIN + identity + struct.pack(">Q", page_size)
    ).digest()
    digests = []
    # Validate tail tokens too: a plan must describe a valid token sequence.
    page_hash = None
    span_index = 0
    media_seen = False
    typed_page = False
    for index, token in enumerate(input_ids):
        while span_index < len(spans) and index >= spans[span_index].end:
            span_index += 1
        span = spans[span_index] if span_index < len(spans) else None
        is_media = span is not None and span.start <= index < span.end
        if type(token) is not int or (not is_media and not 0 <= token <= 0xFFFFFFFF):
            raise ValueError("input_ids must contain unsigned 32-bit integer tokens")
        if index >= page_count * page_size:
            continue
        if index % page_size == 0:
            # Preserve existing text-only keys, including complete text blocks
            # before the first media item. Typed media pages cannot alias a
            # crafted sequence of ordinary uint32 text tokens.
            typed_page = media_seen or (
                span is not None and span.start < index + page_size
            )
            domain = b"media-page\x00" if typed_page else b"page\x00"
            page_hash = hashlib.sha256(_DOMAIN + domain + previous)
        if is_media:
            media_seen = True
            if index == span.start:
                page_hash.update(
                    b"m"
                    + struct.pack(">Q", span.end - span.start)
                    + bytes.fromhex(span.digest)
                )
            else:
                page_hash.update(b"c")
        elif typed_page:
            page_hash.update(b"t" + struct.pack(">I", token))
        else:
            page_hash.update(struct.pack(">I", token))
        if (index + 1) % page_size == 0:
            previous = page_hash.digest()
            digests.append(previous.hex())

    if digests:
        tag = digests[0]
        page_keys = tuple(f"{_KEY_PREFIX}:{{{tag}}}:page:{d}" for d in digests)
    else:
        page_keys = ()
    return PageCachePlan(
        page_keys=page_keys,
        prompt_tokens=prompt_tokens,
        page_size=page_size,
        ttl_seconds=ttl_seconds,
        mode=mode,
    )


# All keys are declared in KEYS, and share the first page's hash tag. Validate
# every argument and key type BEFORE writing: Redis scripts are atomic but do
# not roll back preceding writes when a later command fails.
_ACCOUNT_LUA = r"""
local ttl = tonumber(ARGV[1])
local page_size = tonumber(ARGV[2])
local prompt_tokens = tonumber(ARGV[3])
local generation = ARGV[4]
local mode = ARGV[5]
local pages = #KEYS
local function integer(n, minimum)
    return n and n >= minimum and n <= 9007199254740991 and n == math.floor(n)
end
if #ARGV ~= 5 or (ttl ~= 300 and ttl ~= 3600)
    or (mode ~= 'implicit' and mode ~= 'explicit')
    or not integer(page_size, 1) or not integer(prompt_tokens, 0)
    or not integer(pages, 1)
    or not generation or #generation ~= 32
    or not string.match(generation, '^[0-9a-f]+$')
    or math.floor(prompt_tokens / page_size) ~= pages then
    return redis.error_reply('invalid cache accounting arguments')
end
local values = {}
for i = 1, pages do
    local kind = redis.call('TYPE', KEYS[i]).ok
    if kind ~= 'none' and kind ~= 'string' then
        return redis.error_reply('invalid cache key type')
    end
    local value = redis.call('GET', KEYS[i])
    if value then
        local locked_ttl, token = string.match(value, '^(%d+):([0-9a-f]+)$')
        if (locked_ttl ~= '300' and locked_ttl ~= '3600')
            or not token or #token ~= 32 then
            return redis.error_reply('invalid cache lock value')
        end
    end
    values[i] = value
end
-- The root outlives every descendant touched by this prefix. A missing root
-- invalidates any old suffix, including otherwise unexpired page remnants.
if mode == 'explicit' and not values[1] then
    return {0, 0, ttl, false}
end
if values[1] then
    local locked_ttl, token = string.match(values[1], '^(%d+):([0-9a-f]+)$')
    ttl = tonumber(locked_ttl)
    generation = token
end
local lock = tostring(ttl) .. ':' .. generation
local hits = 0
for i = 1, pages do
    if values[i] ~= lock then
        break
    end
    hits = hits + 1
end
local read = hits * page_size
local written = 0
local touched_pages = hits
if mode == 'implicit' then
    written = (pages - hits) * page_size
    touched_pages = pages
end
local durations = {}
for i = 1, touched_pages do
    durations[i] = ttl * 1000
    if values[i] == lock then
        durations[i] = math.max(durations[i], redis.call('PTTL', KEYS[i]))
    end
    durations[1] = math.max(durations[1], durations[i])
end
for i = 1, touched_pages do
    if mode == 'explicit' then
        -- Read-only requests refresh the hit prefix without creating pages.
        redis.call('PEXPIRE', KEYS[i], durations[i])
    else
        -- Existing pages after a hole are written again, under the same lock.
        redis.call('SET', KEYS[i], lock, 'PX', durations[i])
    end
end
return {read, written, ttl, generation}
"""


_RENEW_LUA = r"""
local ttl = tonumber(ARGV[1])
local generation = ARGV[2]
local mode = ARGV[3]
if #ARGV ~= 3 or (ttl ~= 300 and ttl ~= 3600) or #KEYS < 1
    or (mode ~= 'implicit' and mode ~= 'explicit')
    or not generation or #generation ~= 32
    or not string.match(generation, '^[0-9a-f]+$') then
    return redis.error_reply('invalid cache renewal arguments')
end
local lock = tostring(ttl) .. ':' .. generation
local values = {}
for i = 1, #KEYS do
    local kind = redis.call('TYPE', KEYS[i]).ok
    if kind ~= 'none' and kind ~= 'string' then
        return redis.error_reply('invalid cache key type')
    end
    local value = redis.call('GET', KEYS[i])
    if value then
        local locked_ttl, token = string.match(value, '^(%d+):([0-9a-f]+)$')
        if (locked_ttl ~= '300' and locked_ttl ~= '3600')
            or not token or #token ~= 32 then
            return redis.error_reply('invalid cache lock value')
        end
    end
    values[i] = value
end
-- An explicit request cannot resurrect an expired root or its descendants.
if mode == 'explicit' and values[1] ~= lock then
    return 0
end
for i = 1, #KEYS do
    -- A later writer may have established a new generation during a long
    -- request. Its pages and TTL must not be replaced by stale maintenance.
    if values[i] and values[i] ~= lock then
        return 0
    end
end
local durations = {}
local renewed = 0
for i = 1, #KEYS do
    if mode == 'implicit' or values[i] == lock then
        durations[i] = math.max(ttl * 1000, redis.call('PTTL', KEYS[i]))
        durations[1] = math.max(durations[1], durations[i])
    end
end
for i = 1, #KEYS do
    if durations[i] then
        if mode == 'explicit' then
            redis.call('PEXPIRE', KEYS[i], durations[i])
        else
            -- Restore missing pages without making another accounting decision.
            redis.call('SET', KEYS[i], lock, 'PX', durations[i])
        end
        renewed = renewed + 1
    end
end
return renewed
"""


def _validate_plan(plan: PageCachePlan) -> None:
    """Check caller-constructed plans before accessing Redis."""
    _positive_int("page_size", plan.page_size)
    if (
        type(plan.prompt_tokens) is not int
        or not 0 <= plan.prompt_tokens <= _MAX_SAFE_INTEGER
    ):
        raise ValueError(
            f"prompt_tokens must be an integer between 0 and {_MAX_SAFE_INTEGER}"
        )
    if type(plan.ttl_seconds) is not int or plan.ttl_seconds not in _TTLS:
        raise ValueError("ttl_seconds must be 300 or 3600")
    if plan.mode not in ("implicit", "explicit"):
        raise ValueError("mode must be implicit or explicit")
    if not isinstance(plan.page_keys, tuple):
        raise ValueError("page_keys must be a tuple")
    pages = len(plan.page_keys)
    if pages != plan.prompt_tokens // plan.page_size:
        raise ValueError("page_keys must cover exactly the complete prompt pages")
    if not pages:
        return
    first = plan.page_keys[0]
    prefix = f"{_KEY_PREFIX}:{{"
    if not isinstance(first, str) or not first.startswith(prefix):
        raise ValueError("invalid page key")
    tag, separator, _ = first[len(prefix) :].partition("}:page:")
    if not separator or len(tag) != 64 or any(c not in "0123456789abcdef" for c in tag):
        raise ValueError("invalid page hash tag")
    base = f"{_KEY_PREFIX}:{{{tag}}}:page:"
    if len(set(plan.page_keys)) != pages:
        raise ValueError("page keys must be unique")
    for key in plan.page_keys:
        if not isinstance(key, str) or not key.startswith(base):
            raise ValueError("all page keys must share one Redis Cluster hash tag")
        digest = key[len(base) :]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid page key digest")


class RedisPageCacheStore:
    """Lazy async Redis boundary; an error never becomes read/write success.

    ``account`` and its Redis client do not replay operations after an uncertain
    result. Cluster clients may follow server-side MOVED/ASK redirects.
    Cache keys require a dedicated Redis eviction/retention policy;
    eviction is equivalent to expiration for accounting. Memory exhaustion can
    still cause Redis runtime errors, so Redis failure remains explicitly
    unconfirmed rather than being billed as a cache miss.

    The default client targets a standalone Redis endpoint. Cluster mode uses
    the same key layout and routes each script to the prefix's primary node.
    """

    def __init__(
        self,
        redis_url: str,
        timeout_seconds: float = 1.0,
        username: str | None = None,
        password: str | None = None,
        metrics: Any = None,
        *,
        cluster: bool = False,
    ):
        if not isinstance(redis_url, str) or not redis_url:
            raise ValueError("redis_url must be nonempty")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        if type(cluster) is not bool:
            raise ValueError("cluster must be a boolean")
        self._redis_url = redis_url
        self._timeout_seconds = timeout_seconds
        self._username = username
        self._password = password
        self._metrics = metrics
        self._cluster = cluster
        self._redis: Any = None

    def _client(self):
        if self._redis is None:
            from redis.asyncio.retry import Retry
            from redis.backoff import NoBackoff

            kwargs = dict(
                socket_timeout=self._timeout_seconds,
                socket_connect_timeout=self._timeout_seconds,
                decode_responses=False,
                retry=Retry(NoBackoff(), 0),
                username=self._username,
                password=self._password,
            )
            if self._cluster:
                from redis.asyncio.cluster import RedisCluster

                self._redis = RedisCluster.from_url(
                    self._redis_url,
                    read_from_replicas=False,
                    dynamic_startup_nodes=False,
                    cluster_error_retry_attempts=0,
                    **kwargs,
                )
            else:
                from redis import asyncio as redis_async

                self._redis = redis_async.from_url(
                    self._redis_url, retry_on_timeout=False, **kwargs
                )
        return self._redis

    async def account(
        self, plan: PageCachePlan, *, rid: str | None = None
    ) -> CacheUsage:
        _validate_plan(plan)
        if not plan.page_keys:
            return CacheUsage(ttl_seconds=plan.ttl_seconds)
        generation = uuid4().hex
        request_id = rid if rid is not None else "-"
        started = time.perf_counter()
        timeout_ms = self._timeout_seconds * 1000
        try:
            client = self._client()
            logger.info(
                "Cache Redis EVAL start rid=%s page_count=%d ttl=%d mode=%s "
                "timeout_ms=%.3f",
                request_id,
                len(plan.page_keys),
                plan.ttl_seconds,
                plan.mode,
                timeout_ms,
            )
            result = await asyncio.wait_for(
                client.eval(
                    _ACCOUNT_LUA,
                    len(plan.page_keys),
                    *plan.page_keys,
                    plan.ttl_seconds,
                    plan.page_size,
                    plan.prompt_tokens,
                    generation,
                    plan.mode,
                ),
                timeout=self._timeout_seconds,
            )
            if (
                not isinstance(result, (list, tuple))
                or len(result) != 4
                or any(type(value) is not int or value < 0 for value in result[:2])
                or type(result[2]) is not int
                or result[2] not in _TTLS
                or any(value % plan.page_size for value in result[:2])
            ):
                raise ValueError("Invalid cache Redis result")
            complete_tokens = len(plan.page_keys) * plan.page_size
            if plan.mode == "explicit":
                if result[1] != 0 or result[0] > complete_tokens:
                    raise ValueError("Invalid read-only cache Redis result")
            elif sum(result[:2]) != complete_tokens:
                raise ValueError("Invalid cache Redis result")
            generation = result[3]
            if isinstance(generation, bytes):
                generation = generation.decode("ascii")
            if generation is None:
                if plan.mode != "explicit" or result[0] != 0:
                    raise ValueError("Invalid cache Redis generation")
            else:
                _validate_generation(generation)
            usage = CacheUsage(
                ttl_seconds=result[2],
                read_tokens=result[0],
                write_tokens=result[1],
                generation=generation,
            )
        except asyncio.CancelledError:
            logger.debug(
                "Cache Redis EVAL cancelled rid=%s page_count=%d ttl=%d",
                request_id,
                len(plan.page_keys),
                plan.ttl_seconds,
            )
            raise
        except Exception as exc:
            is_timeout = isinstance(exc, (TimeoutError, asyncio.TimeoutError))
            if not is_timeout:
                try:
                    from redis.exceptions import TimeoutError as RedisTimeoutError
                except ImportError:
                    pass
                else:
                    is_timeout = isinstance(exc, RedisTimeoutError)
            result = "timeout" if is_timeout else "error"
            elapsed = time.perf_counter() - started
            if self._metrics is not None:
                self._metrics.observe_redis(plan.ttl_seconds, result, elapsed)
            logger.warning(
                "Cache Redis EVAL failed rid=%s result=%s error_type=%s "
                "page_count=%d ttl=%d elapsed_ms=%.3f timeout_ms=%.3f",
                request_id,
                result,
                type(exc).__name__,
                len(plan.page_keys),
                plan.ttl_seconds,
                elapsed * 1000,
                timeout_ms,
            )
            raise CacheAccountingError("Redis cache accounting is unconfirmed") from exc
        elapsed = time.perf_counter() - started
        if self._metrics is not None:
            self._metrics.observe_redis(usage.ttl_seconds, "success", elapsed)
        logger.info(
            "Cache Redis EVAL complete rid=%s result=success page_count=%d ttl=%d "
            "prompt=%d read=%d write=%d elapsed_ms=%.3f "
            "timeout_ms=%.3f",
            request_id,
            len(plan.page_keys),
            usage.ttl_seconds,
            plan.prompt_tokens,
            usage.read_tokens,
            usage.write_tokens,
            elapsed * 1000,
            timeout_ms,
        )
        return usage

    async def renew(
        self, plan: PageCachePlan, *, generation: str, rid: str | None = None
    ) -> None:
        """Extend confirmed pages; only implicit requests may restore missing pages."""
        _validate_plan(plan)
        _validate_generation(generation)
        if not plan.page_keys:
            return
        request_id = rid if rid is not None else "-"
        started = time.perf_counter()
        timeout_ms = self._timeout_seconds * 1000
        try:
            client = self._client()
            logger.info(
                "Cache Redis renew start rid=%s page_count=%d ttl=%d mode=%s "
                "timeout_ms=%.3f",
                request_id,
                len(plan.page_keys),
                plan.ttl_seconds,
                plan.mode,
                timeout_ms,
            )
            result = await asyncio.wait_for(
                client.eval(
                    _RENEW_LUA,
                    len(plan.page_keys),
                    *plan.page_keys,
                    plan.ttl_seconds,
                    generation,
                    plan.mode,
                ),
                timeout=self._timeout_seconds,
            )
            if (
                type(result) is not int
                or not 0 <= result <= len(plan.page_keys)
                or (plan.mode == "implicit" and result not in (0, len(plan.page_keys)))
            ):
                raise ValueError("Invalid cache Redis renewal result")
        except asyncio.CancelledError:
            logger.debug(
                "Cache Redis renew cancelled rid=%s page_count=%d ttl=%d "
                "elapsed_ms=%.3f",
                request_id,
                len(plan.page_keys),
                plan.ttl_seconds,
                (time.perf_counter() - started) * 1000,
            )
            raise
        except Exception as exc:
            is_timeout = isinstance(exc, (TimeoutError, asyncio.TimeoutError))
            if not is_timeout:
                try:
                    from redis.exceptions import TimeoutError as RedisTimeoutError
                except ImportError:
                    pass
                else:
                    is_timeout = isinstance(exc, RedisTimeoutError)
            result = "timeout" if is_timeout else "error"
            elapsed = time.perf_counter() - started
            if self._metrics is not None:
                self._metrics.observe_redis(plan.ttl_seconds, result, elapsed)
            logger.warning(
                "Cache Redis renew failed rid=%s result=%s error_type=%s "
                "page_count=%d ttl=%d elapsed_ms=%.3f timeout_ms=%.3f",
                request_id,
                result,
                type(exc).__name__,
                len(plan.page_keys),
                plan.ttl_seconds,
                elapsed * 1000,
                timeout_ms,
            )
            raise CacheAccountingError("Redis cache renewal is unconfirmed") from exc
        elapsed = time.perf_counter() - started
        # Expiration or a newer generation may leave no pages eligible to renew.
        outcome = "success" if result else "skipped"
        if self._metrics is not None:
            self._metrics.observe_redis(plan.ttl_seconds, outcome, elapsed)
        logger.info(
            "Cache Redis renew complete rid=%s result=%s page_count=%d ttl=%d "
            "elapsed_ms=%.3f timeout_ms=%.3f",
            request_id,
            outcome,
            len(plan.page_keys),
            plan.ttl_seconds,
            elapsed * 1000,
            timeout_ms,
        )

    async def aclose(self) -> None:
        client, self._redis = self._redis, None
        if client is not None:
            await client.aclose()
