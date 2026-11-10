from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, final

from sglang.srt.entrypoints.openai.protocol import (
    CacheUsage,
    CompletionTokensDetails,
    PromptTokensDetails,
    UsageInfo,
)


@final
class UsageProcessor:
    """Stateless helpers that turn raw token counts into a UsageInfo."""

    @staticmethod
    def _cache_total(values: Iterable[Any]) -> Optional[CacheUsage]:
        snapshots = [
            CacheUsage.model_validate(value) for value in values if value is not None
        ]
        if not snapshots:
            return None
        ttl_seconds = snapshots[0].ttl_seconds
        if any(snapshot.ttl_seconds != ttl_seconds for snapshot in snapshots):
            raise ValueError("Cache usage snapshots must use the same ttl_seconds.")
        return CacheUsage(
            ttl_seconds=ttl_seconds,
            read_tokens=sum(snapshot.read_tokens for snapshot in snapshots),
            write_tokens=sum(snapshot.write_tokens for snapshot in snapshots),
        )

    @staticmethod
    def calculate_response_usage(
        responses: List[Dict[str, Any]],
        n_choices: int = 1,
        enable_cache_report: bool = False,
        image_tokens: int = 0,
        audio_tokens: int = 0,
        video_tokens: int = 0,
    ) -> UsageInfo:
        completion_tokens = sum(
            r["meta_info"].get("completion_tokens", 0) for r in responses
        )
        prompt_tokens = sum(
            responses[i]["meta_info"].get("prompt_tokens", 0)
            for i in range(0, len(responses), n_choices)
        )

        # some API don't have reasoning_tokens semantics
        reasoning_tokens = sum(
            r["meta_info"].get("reasoning_tokens", 0) for r in responses
        )

        cached_details = None
        if enable_cache_report:
            cached_total = sum(
                responses[i]["meta_info"].get("cached_tokens", 0)
                for i in range(0, len(responses), n_choices)
            )
            cached_details = PromptTokensDetails(cached_tokens=cached_total)

        cache_usage = UsageProcessor._cache_total(
            responses[i]["meta_info"].get("cache_usage")
            for i in range(0, len(responses), n_choices)
        )

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=prompt_tokens,
            reasoning_tokens=reasoning_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_details,
            image_tokens=image_tokens,
            audio_tokens=audio_tokens,
            video_tokens=video_tokens,
            cache_usage=cache_usage,
        )

    @staticmethod
    def calculate_streaming_usage(
        prompt_tokens: Mapping[int, int],
        reasoning_tokens: Mapping[int, int],
        completion_tokens: Mapping[int, int],
        cached_tokens: Mapping[int, int],
        n_choices: int,
        enable_cache_report: bool = False,
        image_tokens: int = 0,
        audio_tokens: int = 0,
        video_tokens: int = 0,
        *,
        cache_usage: Optional[Mapping[int, Any]] = None,
    ) -> UsageInfo:
        # index % n_choices == 0 marks the first choice of a prompt
        total_prompt_tokens = sum(
            tok for idx, tok in prompt_tokens.items() if idx % n_choices == 0
        )
        total_reasoning_tokens = sum(reasoning_tokens.values())
        total_completion_tokens = sum(completion_tokens.values())

        cached_details = (
            PromptTokensDetails(
                cached_tokens=sum(
                    tok for idx, tok in cached_tokens.items() if idx % n_choices == 0
                )
            )
            if enable_cache_report
            else None
        )

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=total_prompt_tokens,
            reasoning_tokens=total_reasoning_tokens,
            completion_tokens=total_completion_tokens,
            cached_tokens=cached_details,
            image_tokens=image_tokens,
            audio_tokens=audio_tokens,
            video_tokens=video_tokens,
            cache_usage=UsageProcessor._cache_total(
                snapshot
                for index, snapshot in (cache_usage or {}).items()
                if index % n_choices == 0
            ),
        )

    @staticmethod
    def calculate_token_usage(
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: Optional[int] = 0,
        cached_tokens: Optional[PromptTokensDetails] = None,
        image_tokens: int = 0,
        audio_tokens: int = 0,
        video_tokens: int = 0,
        *,
        cache_usage: Optional[CacheUsage | Dict[str, int]] = None,
    ) -> UsageInfo:
        """Calculate token usage information"""
        # `cached_tokens` is already a PromptTokensDetails (or None) carrying the
        # cached count. Attach multimodal counts to the same object, creating one
        # only when there is something to report so plain-text requests keep
        # prompt_tokens_details=None (backward compatible).
        details = cached_tokens
        if cache_usage is not None:
            cache_usage = CacheUsage.model_validate(cache_usage)
            cache_tokens = cache_usage.read_tokens + cache_usage.write_tokens
            if cache_tokens > prompt_tokens:
                raise ValueError(
                    "Cache read and write tokens must not exceed prompt_tokens "
                    f"({cache_tokens} > {prompt_tokens})."
                )
            details = (
                details.model_copy() if details is not None else PromptTokensDetails()
            )
            details.cached_tokens = cache_usage.read_tokens
            details.cache_write_tokens = cache_usage.write_tokens
        if image_tokens or audio_tokens or video_tokens:
            if details is None:
                details = PromptTokensDetails()
            if image_tokens:
                details.image_tokens = image_tokens
            if audio_tokens:
                details.audio_tokens = audio_tokens
            if video_tokens:
                details.video_tokens = video_tokens

        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=details,
            reasoning_tokens=reasoning_tokens,
            completion_tokens_details=(
                CompletionTokensDetails(reasoning_tokens=reasoning_tokens)
                if reasoning_tokens
                else None
            ),
            cache_usage=cache_usage,
        )
