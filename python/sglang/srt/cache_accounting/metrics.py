# SPDX-License-Identifier: Apache-2.0
"""Opt-in cache metrics using SGLang's Prometheus multiprocess directory.

Prometheus is imported only when a collector is enabled, after the launcher has
configured PROMETHEUS_MULTIPROC_DIR. Metric families are shared per registry so
reinitializing a tokenizer or changing its role does not register duplicates.
"""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _families(registry):
    from prometheus_client import Counter, Histogram

    labels = ("role", "ttl")
    usage_labels = (*labels, "source")
    return {
        "redis_operations": Counter(
            "sglang:cache_redis_operations_total",
            "Completed cache Redis operations by outcome; cancellations excluded.",
            (*labels, "result"),
            registry=registry,
        ),
        "redis_duration": Histogram(
            "sglang:cache_redis_operation_duration_seconds",
            "Cache Redis operation duration, including connection establishment.",
            (*labels, "result"),
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
            registry=registry,
        ),
        "requests": Counter(
            "sglang:cache_accounting_requests_total",
            "Completed per-request cache accounting decisions.",
            usage_labels,
            registry=registry,
        ),
        "fallbacks": Counter(
            "sglang:cache_accounting_fallbacks_total",
            "Requests using physical cache usage after unconfirmed cache accounting.",
            labels,
            registry=registry,
        ),
        **{
            field: Counter(
                f"sglang:cache_{name}_tokens_total",
                f"Input tokens classified as {name} by cache accounting.",
                usage_labels,
                registry=registry,
            )
            for field, name in (
                ("read_tokens", "read"),
                ("write_tokens", "write"),
            )
        },
    }


class CacheAccountingMetrics:
    """Bounded labels; one Redis observation and one usage observation per event."""

    def __init__(self, *, role, registry=None):
        if role not in ("standalone", "decode"):
            raise ValueError("Cache metrics require the standalone or decode role")
        # Keep this import lazy, as in SGLang's existing metrics collectors.
        if registry is None:
            from prometheus_client import REGISTRY

            registry = REGISTRY
        self.role = role
        self._metrics = _families(registry)

    def _labels(self, ttl_seconds):
        if ttl_seconds not in (300, 3600):
            raise ValueError("Cache metrics TTL must be 300 or 3600")
        return {"role": self.role, "ttl": "5m" if ttl_seconds == 300 else "1h"}

    def observe_redis(self, ttl_seconds, result, elapsed_seconds):
        if result not in ("success", "skipped", "timeout", "error"):
            raise ValueError("Unsupported cache Redis metric result")
        labels = {**self._labels(ttl_seconds), "result": result}
        try:
            self._metrics["redis_operations"].labels(**labels).inc()
            self._metrics["redis_duration"].labels(**labels).observe(elapsed_seconds)
        except Exception as exc:
            # A metrics export failure must not change a confirmed cache decision.
            logger.warning(
                "Cache metrics observation failed error_type=%s", type(exc).__name__
            )

    def observe_usage(self, ttl_seconds, usage, source):
        if source not in (
            "redis",
            "physical",
            "no_complete_block",
            "missing_identity",
            "multimodal_unavailable",
        ):
            raise ValueError("Unsupported cache accounting metric source")
        if usage.ttl_seconds != ttl_seconds:
            raise ValueError("Cache usage TTL differs from metric TTL")
        base_labels = self._labels(ttl_seconds)
        labels = {**base_labels, "source": source}
        try:
            self._metrics["requests"].labels(**labels).inc()
            for field in (
                "read_tokens",
                "write_tokens",
            ):
                self._metrics[field].labels(**labels).inc(getattr(usage, field))
            if source in ("physical", "multimodal_unavailable"):
                self._metrics["fallbacks"].labels(**base_labels).inc()
        except Exception as exc:
            logger.warning(
                "Cache metrics observation failed error_type=%s", type(exc).__name__
            )
