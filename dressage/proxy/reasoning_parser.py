"""Reasoning parsing helpers for proxy responses."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal

from dressage.profiling import async_profile_span, profile_set, profile_span

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReasoningParseResult:
    reasoning_content: str | None
    text: str


ReasoningParser = Callable[[str], ReasoningParseResult]

_LOCAL_REASONING_TYPES = {"qwen3", "qwen3_5"}
_QWEN_COMPLETION_REPAIR_TYPES = {"qwen3", "qwen3_5", "qwen3-thinking"}


def _supports_profile_arg(func: Any) -> bool:
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "profile"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def canonicalize_reasoning_content(value: Any) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def parse_qwen3_reasoning(text: str) -> ReasoningParseResult:
    """Split Qwen3-style thinking text from visible content."""

    end_tag = "</think>"
    if end_tag not in text:
        return ReasoningParseResult(reasoning_content=None, text=text)

    reasoning_text, visible_text = text.split(end_tag, 1)
    if "<think>" in reasoning_text:
        reasoning_text = reasoning_text.rsplit("<think>", 1)[1]
    visible_text = visible_text.lstrip("\n")
    return ReasoningParseResult(
        reasoning_content=canonicalize_reasoning_content(reasoning_text),
        text=visible_text,
    )


def _should_repair_qwen_completion(
    text: str,
    model_reasoning_type: str | None,
) -> bool:
    if model_reasoning_type is None:
        return False
    if model_reasoning_type.lower() not in _QWEN_COMPLETION_REPAIR_TYPES:
        return False
    end_index = text.find("</think>")
    if end_index < 0:
        return False
    return "<think>" not in text[:end_index]


def _repair_qwen_completion_for_sglang_api(
    text: str,
    model_reasoning_type: str | None,
) -> str:
    if _should_repair_qwen_completion(text, model_reasoning_type):
        return "<think>" + text
    return text


def get_local_reasoning_parser(model_reasoning_type: str | None) -> ReasoningParser | None:
    if model_reasoning_type is None:
        return None
    if model_reasoning_type in _LOCAL_REASONING_TYPES:
        return parse_qwen3_reasoning
    return None


class ProxyReasoningParser:
    """Parse raw SGLang text into reasoning content and visible text."""

    def __init__(
        self,
        sglang_client: Any,
        *,
        model_reasoning_type: str | None,
        backend: Literal["local", "sglang_api", "hybrid"],
    ):
        self._sglang_client = sglang_client
        self._model_reasoning_type = model_reasoning_type
        self._backend = backend

    @staticmethod
    def local_model_supported(model_reasoning_type: str | None) -> bool:
        return model_reasoning_type in _LOCAL_REASONING_TYPES

    def _parse_locally(
        self,
        raw_text: str,
        *,
        profile: dict[str, Any] | None = None,
    ) -> ReasoningParseResult:
        with profile_span(profile, "reasoning.parse.local_s"):
            local_parser = get_local_reasoning_parser(self._model_reasoning_type)
            if local_parser is None:
                raise ValueError(
                    "local reasoning parser only supports qwen3/qwen3_5, "
                    f"got {self._model_reasoning_type!r}"
                )
            return local_parser(raw_text)

    @staticmethod
    def _normalize_sglang_result(parsed: dict[str, Any] | None) -> ReasoningParseResult | None:
        if not isinstance(parsed, dict) or "text" not in parsed:
            return None

        visible_text = parsed.get("text")
        if visible_text is None:
            visible_text = ""
        elif not isinstance(visible_text, str):
            visible_text = str(visible_text)

        reasoning_content = parsed.get("reasoning_text", parsed.get("reasoning_content"))
        if reasoning_content is not None and not isinstance(reasoning_content, str):
            reasoning_content = str(reasoning_content)

        return ReasoningParseResult(
            reasoning_content=canonicalize_reasoning_content(reasoning_content),
            text=visible_text,
        )

    async def _parse_with_sglang_api(
        self,
        raw_text: str,
        *,
        routing_key: str | None,
        profile: dict[str, Any] | None = None,
    ) -> ReasoningParseResult | None:
        if self._model_reasoning_type is None:
            return None
        with profile_span(profile, "reasoning.parse.repair_s"):
            parser_text = _repair_qwen_completion_for_sglang_api(
                raw_text,
                self._model_reasoning_type,
            )
        try:
            async with async_profile_span(profile, "reasoning.parse.api_http_s"):
                kwargs: dict[str, Any] = {
                    "reasoning_parser": self._model_reasoning_type,
                    "routing_key": routing_key,
                }
                if _supports_profile_arg(self._sglang_client.separate_reasoning):
                    kwargs["profile"] = profile
                parsed = await self._sglang_client.separate_reasoning(
                    parser_text,
                    **kwargs,
                )
        except Exception:
            return None
        with profile_span(profile, "reasoning.parse.normalize_s"):
            result = self._normalize_sglang_result(parsed)
        if (
            result is not None
            and result.reasoning_content is None
            and "</think>" in result.text
        ):
            logger.warning(
                "SGLang reasoning parser returned empty reasoning content for "
                "text that still contains </think>; parser=%r",
                self._model_reasoning_type,
            )
        return result

    async def parse(
        self,
        raw_text: str,
        *,
        routing_key: str | None = None,
        profile: dict[str, Any] | None = None,
    ) -> ReasoningParseResult:
        profile_set(profile, "reasoning.parse.backend.local", float(self._backend == "local"))
        profile_set(
            profile,
            "reasoning.parse.backend.sglang_api",
            float(self._backend == "sglang_api"),
        )
        profile_set(profile, "reasoning.parse.backend.hybrid", float(self._backend == "hybrid"))
        if self._model_reasoning_type is None:
            return ReasoningParseResult(reasoning_content=None, text=raw_text)

        if self._backend == "local":
            return self._parse_locally(raw_text, profile=profile)

        if self._backend == "sglang_api":
            return (
                await self._parse_with_sglang_api(
                    raw_text, routing_key=routing_key, profile=profile
                )
            ) or ReasoningParseResult(reasoning_content=None, text=raw_text)

        parsed = await self._parse_with_sglang_api(
            raw_text, routing_key=routing_key, profile=profile
        )
        if parsed is not None:
            return parsed
        local_parser = get_local_reasoning_parser(self._model_reasoning_type)
        if local_parser is None:
            return ReasoningParseResult(reasoning_content=None, text=raw_text)
        with profile_span(profile, "reasoning.parse.hybrid_local_fallback_s"):
            return local_parser(raw_text)
