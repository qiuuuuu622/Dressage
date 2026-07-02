"""FastAPI proxy that converts OpenAI chat calls into SGLang ``/generate``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

from dressage.config import (
    DEFAULT_TRAJECTORY_BUILD_MODEL,
    sglang_router_url as default_sglang_router_url,
    trajectory_build_defaults,
)
from dressage.profiling import (
    async_profile_span,
    enabled as profiling_enabled,
    profile_add,
    profile_public,
    profile_span,
    profile_set,
)

from .generation_controller import (
    GenerationController,
    GenerationPreempted,
    GenerationStaleEpoch,
    ProxyShuttingDown,
)
from .last_step import (
    PromptAssistantMaskBuilder,
    create_default_mask_template_registry,
)
from .reasoning_parser import ProxyReasoningParser, canonicalize_reasoning_content
from .session_manager import Session, SessionFinalizedError, SessionManager
from .sglang_client import SGLangRouterClient
from .tool_call_parser import (
    ModelToolCallParserRegistry,
    ProxyToolCallParser,
    ToolCallParser,
    create_default_tool_call_parser_registry,
    parse_hermes_tool_calls,
)
from .trajectory_store import TrajectoryStore

logger = logging.getLogger(__name__)

_INPUT_TOKEN_VERSION = "-1"
_DEFAULT_TOOL_CALL_PARSER = object()
_NON_REAL_TOKEN_VERSIONS = {"", "-1", "unknown", "none"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _real_token_version(value: Any) -> str | None:
    if value is None:
        return None
    version = str(value)
    if version.strip().lower() in _NON_REAL_TOKEN_VERSIONS:
        return None
    return version


def _session_real_versions(session: Session) -> set[str]:
    versions: set[str] = set()
    for step in session.steps:
        values: list[Any] = []
        values.extend(step.response_versions)
        values.extend([step.response_version, step.request_version])
        for value in values:
            version = _real_token_version(value)
            if version is not None:
                versions.add(version)
    return versions


def _raise_if_cross_version_trajectory(
    *,
    session: Session,
    candidate_versions: list[Any],
    partial_rollout: bool,
) -> None:
    if partial_rollout:
        return
    previous_versions = _session_real_versions(session)
    if not previous_versions:
        return
    new_versions = {
        version
        for value in candidate_versions
        if (version := _real_token_version(value)) is not None
    }
    if not new_versions or new_versions.issubset(previous_versions):
        return
    logger.warning(
        "reject non-partial rollout: error=trajectory_version_changed "
        "session_id=%s instance_id=%s previous_versions=%s new_versions=%s",
        session.session_id,
        session.instance_id,
        sorted(previous_versions),
        sorted(new_versions),
    )
    raise HTTPException(
        status_code=502,
        detail={
            "error": "trajectory_version_changed",
            "message": (
                "SGLang weight version changed during a non-partial rollout "
                "trajectory; rejecting the trajectory instead of continuing it."
            ),
            "previous_versions": sorted(previous_versions),
            "new_versions": sorted(new_versions),
            "session_id": session.session_id,
            "instance_id": session.instance_id,
        },
    )


def _raise_if_stale_rollout_epoch(
    *,
    session: Session,
    current_epoch: int,
    partial_rollout: bool,
) -> None:
    if partial_rollout or not session.steps or session.rollout_epoch is None:
        return
    if session.rollout_epoch == current_epoch:
        return
    logger.warning(
        "reject non-partial rollout: error=trajectory_version_changed "
        "session_id=%s instance_id=%s session_epoch=%s current_epoch=%s",
        session.session_id,
        session.instance_id,
        session.rollout_epoch,
        current_epoch,
    )
    raise HTTPException(
        status_code=502,
        detail={
            "error": "trajectory_version_changed",
            "message": (
                "Dressage rollout epoch changed during a non-partial rollout "
                "trajectory; rejecting the trajectory before sending it to SGLang."
            ),
            "session_epoch": session.rollout_epoch,
            "current_epoch": current_epoch,
            "session_id": session.session_id,
            "instance_id": session.instance_id,
        },
    )


def _tools_sanity_probes() -> list[list[dict[str, Any]]]:
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"q":"x"}'},
    }
    return [
        [{"role": "user", "content": "x"}],
        [{"role": "system", "content": "s"}, {"role": "user", "content": "x"}],
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": None, "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": "1", "content": "r"},
        ],
    ]


def _tools_none_equals_empty(
    mask_builder: PromptAssistantMaskBuilder,
) -> bool:
    for probe in _tools_sanity_probes():
        normalized_probe = mask_builder.normalize_template_messages(probe)
        none_ids = mask_builder.tokenize_messages(
            normalized_probe,
            tools=None,
            add_generation_prompt=False,
        )
        empty_ids = mask_builder.tokenize_messages(
            normalized_probe,
            tools=[],
            add_generation_prompt=False,
        )
        if none_ids != empty_ids:
            return False
    return True


def _canonicalize_tools(
    tools: list[dict[str, Any]] | None,
    *,
    none_equals_empty: bool,
) -> str | None:
    if tools is None:
        return _canonical_json([]) if none_equals_empty else None
    return _canonical_json(tools)


def _tools_changed(
    previous_tools: list[dict[str, Any]] | None,
    current_tools: list[dict[str, Any]] | None,
    *,
    none_equals_empty: bool,
) -> bool:
    return _canonicalize_tools(
        previous_tools, none_equals_empty=none_equals_empty
    ) != _canonicalize_tools(current_tools, none_equals_empty=none_equals_empty)


def _boundary_reasons(
    *,
    rewrite_detected: bool,
    tools_changed: bool,
    message_prefix_mismatch: bool = False,
) -> list[str]:
    reasons: list[str] = []
    if rewrite_detected:
        reasons.append("history_rewrite")
    if message_prefix_mismatch:
        reasons.append("message_prefix_mismatch")
    if tools_changed:
        reasons.append("tools_changed")
    return reasons


def _build_sampling_params(
    body: dict[str, Any],
    default_max_tokens: int,
    rollout_temperature: float,
) -> dict[str, Any]:
    temperature = body.get("temperature")
    if temperature is None:
        temperature = rollout_temperature
    return {
        "max_new_tokens": body.get("max_tokens") or default_max_tokens,
        "temperature": temperature,
        "top_p": body.get("top_p", 1.0),
        "top_k": body.get("top_k", -1),
        "skip_special_tokens": False,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }


def _context_overflow_payload(
    *,
    phase: str,
    context_window: int,
    input_tokens: int,
    output_tokens: int,
    max_tokens: int,
    session_id: str | None,
    turn_id: str | None,
    last_proxy_step_recorded: bool,
) -> dict[str, Any]:
    return {
        "error": "context_overflow",
        "message": "Dressage proxy context window overflow.",
        "details": {
            "phase": phase,
            "context_window": context_window,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "max_tokens": max_tokens,
            "session_id": session_id,
            "turn_id": turn_id,
            "last_proxy_step_recorded": last_proxy_step_recorded,
        },
    }


def _runtime_ids_from_request(
    request: Request, body: dict[str, Any]
) -> tuple[str | None, str | None, str | None]:
    session_id = (
        request.headers.get("X-Session-Id")
        or request.headers.get("X-SMG-Routing-Key")
        or body.get("session_id")
    )
    instance_id = request.headers.get("X-Instance-Id") or body.get("instance_id")
    turn_id = request.headers.get("X-Turn-Id") or body.get("turn_id")
    return session_id, instance_id, turn_id


def _assistant_message(
    content: str | None,
    tool_calls: list[dict] | None,
    *,
    reasoning_content: str | None = None,
) -> dict[str, Any]:
    normalized_content = None if content is None else str(content)
    if tool_calls and (normalized_content is None or not normalized_content.strip()):
        normalized_content = None

    normalized_reasoning_content = canonicalize_reasoning_content(reasoning_content)

    message: dict[str, Any] = {"role": "assistant", "content": normalized_content}
    if normalized_reasoning_content is not None:
        message["reasoning_content"] = normalized_reasoning_content
    if tool_calls:
        message["tool_calls"] = tool_calls
    if normalized_content is None and not tool_calls:
        message["content"] = ""
    return message


def _strip_public_stop_markers(content: str | None) -> str | None:
    if content is None:
        return None
    cleaned = str(content)
    for marker in ("<|im_end|>", "<|endoftext|>"):
        while cleaned.rstrip().endswith(marker):
            cleaned = cleaned.rstrip()
            cleaned = cleaned[: -len(marker)]
    return cleaned if cleaned.strip() else None


def _trajectory_id_from_body(body: dict[str, Any]) -> str | None:
    return body.get("trajectory_id") or body.get("session_id")


def _ordered_turn_ids(steps: list[Any]) -> list[str]:
    turn_ids: list[str] = []
    seen: set[str] = set()
    for step in steps:
        if step.turn_id not in seen:
            seen.add(step.turn_id)
            turn_ids.append(step.turn_id)
    return turn_ids


def _split_session_into_segments(session: Session) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current_steps: list[Any] = []
    current_segment_reasons: list[str] = ["initial"]

    def flush_segment() -> None:
        nonlocal current_steps, current_segment_reasons
        if not current_steps:
            return
        segments.append(
            {
                "steps": current_steps,
                "segment_reason": current_segment_reasons[0],
                "segment_reasons": list(current_segment_reasons),
                "turn_ids": _ordered_turn_ids(current_steps),
            }
        )
        current_steps = []
        current_segment_reasons = ["initial"]

    for step in session.steps:
        if step.segment_boundary_before and current_steps:
            flush_segment()
            current_segment_reasons = list(step.segment_reasons_before or ["initial"])
        current_steps.append(step)

    flush_segment()
    return segments


def create_app(
    *,
    sglang_router_url: str | None = None,
    tokenizer_path: str | None = None,
    trajectory_store: TrajectoryStore | None = None,
    session_manager: SessionManager | None = None,
    tokenizer: Any | None = None,
    sglang_client: SGLangRouterClient | None = None,
    tool_call_parser: ToolCallParser | object = _DEFAULT_TOOL_CALL_PARSER,
    model_tool_call_type: str | None = None,
    tool_call_parse_backend: Literal["local", "sglang_api", "hybrid"] = "sglang_api",
    tool_call_parser_registry: ModelToolCallParserRegistry | None = None,
    model_reasoning_type: str | None = None,
    reasoning_parse_backend: Literal["local", "sglang_api", "hybrid"] = "sglang_api",
    model_mask_type: str | None = None,
    default_max_tokens: int = 4096,
    api_key: str = "no-auth",
    trajectory_build_mode: Literal["last_step", "concat"] = "concat",
    trajectory_build_model: str = DEFAULT_TRAJECTORY_BUILD_MODEL,
    tito_model: str | None = None,
    record_token_versions: bool = False,
    mask_nonlast_version_tokens: bool = False,
    rollout_temperature: float = 1.0,
    context_window: int | None = None,
    dynamic_max_tokens: bool = True,
    use_rollout_routing_replay: bool = False,
    partial_rollout: bool = False,
) -> FastAPI:
    """Create the Dressage proxy FastAPI app."""

    if context_window is not None and context_window <= 0:
        raise ValueError("context_window must be greater than 0 when provided")
    if trajectory_build_mode not in {"last_step", "concat"}:
        raise ValueError(
            "trajectory_build_mode must be 'last_step' or 'concat', "
            f"got {trajectory_build_mode!r}"
        )
    if tool_call_parse_backend not in {"local", "sglang_api", "hybrid"}:
        raise ValueError(
            "tool_call_parse_backend must be 'local', 'sglang_api', or 'hybrid', "
            f"got {tool_call_parse_backend!r}"
        )
    if reasoning_parse_backend not in {"local", "sglang_api", "hybrid"}:
        raise ValueError(
            "reasoning_parse_backend must be 'local', 'sglang_api', or 'hybrid', "
            f"got {reasoning_parse_backend!r}"
        )
    if (
        model_reasoning_type is not None
        and reasoning_parse_backend == "local"
        and not ProxyReasoningParser.local_model_supported(model_reasoning_type)
    ):
        raise ValueError(
            "local reasoning parser only supports qwen3/qwen3_5, "
            f"got {model_reasoning_type!r}"
        )

    build_defaults = trajectory_build_defaults(
        trajectory_build_mode=trajectory_build_mode,
        trajectory_build_model=trajectory_build_model,
    )
    if model_mask_type is None:
        model_mask_type = build_defaults.model_mask_type
    if model_tool_call_type is None:
        model_tool_call_type = build_defaults.model_tool_call_type
    if model_reasoning_type is None:
        model_reasoning_type = build_defaults.model_reasoning_type
    if tito_model is None:
        tito_model = build_defaults.tito_model
    if sglang_router_url is None:
        sglang_router_url = default_sglang_router_url()
    if (
        model_reasoning_type is not None
        and reasoning_parse_backend == "local"
        and not ProxyReasoningParser.local_model_supported(model_reasoning_type)
    ):
        raise ValueError(
            "local reasoning parser only supports qwen3/qwen3_5, "
            f"got {model_reasoning_type!r}"
        )

    if tokenizer is None:
        if tokenizer_path is None:
            raise ValueError("tokenizer_path is required when tokenizer is not provided")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    trajectory_store = trajectory_store or TrajectoryStore()
    session_manager = session_manager or SessionManager()
    sglang_client = sglang_client or SGLangRouterClient(
        sglang_router_url,
        return_routed_experts=use_rollout_routing_replay,
    )
    generation_controller = GenerationController(
        sglang_client,
        partial_rollout=partial_rollout,
    )
    tito_tokenizer = None
    effective_model_mask_type = model_mask_type
    if trajectory_build_mode == "concat":
        if tito_model is None:
            raise ValueError("concat mode requires tito_model='qwen3_5'")
        if tito_model != "qwen3_5":
            raise ValueError(f"Unsupported TITO model type: {tito_model!r}")
        from .tito import create_tito_tokenizer, load_fixed_template

        tokenizer.chat_template = load_fixed_template(tito_model)
        tito_tokenizer = create_tito_tokenizer(tokenizer, model_type=tito_model)
        effective_model_mask_type = None

    mask_template_registry = create_default_mask_template_registry()
    tool_call_registry = tool_call_parser_registry or create_default_tool_call_parser_registry()
    mask_builder = PromptAssistantMaskBuilder(
        tokenizer,
        model_mask_type=effective_model_mask_type,
        registry=mask_template_registry,
    )
    legacy_local_parser: ToolCallParser | None
    if tool_call_parser is _DEFAULT_TOOL_CALL_PARSER:
        legacy_local_parser = (
            parse_hermes_tool_calls if model_tool_call_type is None else None
        )
    else:
        legacy_local_parser = tool_call_parser
    proxy_tool_call_parser = ProxyToolCallParser(
        sglang_client,
        model_tool_call_type=model_tool_call_type,
        backend=tool_call_parse_backend,
        registry=tool_call_registry,
        legacy_local_parser=legacy_local_parser,
    )
    proxy_reasoning_parser = ProxyReasoningParser(
        sglang_client,
        model_reasoning_type=model_reasoning_type,
        backend=reasoning_parse_backend,
    )
    none_equals_empty_tools = _tools_none_equals_empty(mask_builder)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await generation_controller.shutdown(timeout_seconds=5.0)
            await sglang_client.close()

    app = FastAPI(title="Dressage Proxy", lifespan=lifespan)

    def _check_auth(request: Request) -> None:
        if api_key == "no-auth":
            return
        auth_header = request.headers.get("Authorization", "")
        expected = f"Bearer {api_key}"
        if auth_header != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    def _build_segment_record(
        *,
        session: Session,
        trajectory_id: str,
        segment: dict[str, Any],
        segment_index: int,
        segment_count: int,
        instance_id: str,
        label: Any | None = None,
    ) -> dict[str, Any]:
        steps = segment["steps"]
        base_step = steps[-1]
        turn_ids = segment["turn_ids"]
        messages = base_step.messages_snapshot
        tools = base_step.tools
        alignment = mask_builder.build_segment_alignment(base_step, tools)
        extra_info = {
            "alignment_method": "last_step_all_logprobs+assistant_prompt_mask",
            "trajectory_build_mode": "last_step",
            "mask_template_equivalent": alignment["mask_template_equivalent"],
            "prompt_assistant_token_count": alignment["prompt_assistant_token_count"],
            "output_token_count": len(base_step.response_token_ids),
            "num_steps": len(steps),
            "num_turns": len(turn_ids),
            "turn_ids": turn_ids,
            "timestamp": str(time.time()),
            "history_rewritten": session.history_rewritten,
            "segment_reason": segment["segment_reason"],
            "segment_reasons": segment["segment_reasons"],
            "trajectory_num_segments": segment_count,
        }
        if base_step.profile:
            extra_info["dressage_profile"] = dict(base_step.profile)
        if alignment["mask_fallback_reason"] is not None:
            extra_info["mask_fallback_reason"] = alignment["mask_fallback_reason"]
        if mask_nonlast_version_tokens:
            extra_info["mask_nonlast_version_tokens"] = True

        response_version = (
            base_step.response_version
            or (base_step.response_versions[-1] if base_step.response_versions else None)
            or base_step.request_version
            or "unknown"
        )
        response_mask = list(alignment["response_mask"])
        response_logprobs = list(alignment["response_logprobs"])
        full_versions = None
        if record_token_versions:
            full_versions = [
                response_version if int(mask_value) == 1 else _INPUT_TOKEN_VERSION
                for mask_value in response_mask
            ]

        record = {
            "uid": str(uuid.uuid4()),
            "trajectory_id": trajectory_id,
            "turn_id": turn_ids[-1],
            "instance_id": instance_id,
            "segment_index": segment_index,
            "segment_count": segment_count,
            "messages": messages,
            "tools": tools,
            "tokens": alignment["tokens"],
            "full_logprobs": response_logprobs,
            "full_loss_mask": response_mask,
            "aligned_response_length": sum(response_mask),
            "label": label,
            "finish_reason": base_step.finish_reason,
            "extra_info": extra_info,
        }
        if full_versions is not None:
            record["full_versions"] = full_versions
        if base_step.response_routed_experts_chunks:
            record["routed_experts_chunks"] = base_step.response_routed_experts_chunks
        if base_step.response_routed_experts is not None:
            record["routed_experts"] = base_step.response_routed_experts
        return record

    def _normalize_logprobs_to_length(
        values: list[float],
        token_count: int,
    ) -> tuple[list[float], bool]:
        invalid = len(values) != token_count
        normalized: list[float] = []
        for index in range(token_count):
            if index >= len(values):
                normalized.append(0.0)
                continue
            value = values[index]
            try:
                normalized.append(float(value))
            except (TypeError, ValueError):
                normalized.append(0.0)
                invalid = True
        return normalized, invalid

    def _build_concat_segment_record(
        *,
        session: Session,
        trajectory_id: str,
        segment: dict[str, Any],
        segment_index: int,
        segment_count: int,
        instance_id: str,
        label: Any | None = None,
    ) -> dict[str, Any]:
        steps = segment["steps"]
        base_step = steps[-1]
        turn_ids = segment["turn_ids"]
        messages = base_step.messages_snapshot
        tools = base_step.tools

        tokens: list[int] = []
        response_logprobs: list[float] = []
        response_mask: list[int] = []
        full_versions: list[str] = []
        context_token_count = 0
        output_token_count = 0
        concat_logprobs_invalid = False
        concat_incremental_tokenization_failed = False

        for step in steps:
            tokens.extend(step.concat_token_ids)
            response_logprobs.extend(step.concat_response_logprobs)
            response_mask.extend(step.concat_response_mask)
            if record_token_versions:
                full_versions.extend(step.concat_versions)
            context_token_count += step.concat_context_token_count
            output_token_count += step.concat_output_token_count
            concat_logprobs_invalid = (
                concat_logprobs_invalid or step.concat_logprobs_invalid
            )
            concat_incremental_tokenization_failed = (
                concat_incremental_tokenization_failed
                or step.concat_incremental_tokenization_failed
            )

        extra_info = {
            "alignment_method": "tito_concat",
            "trajectory_build_mode": "concat",
            "context_token_count": context_token_count,
            "context_delta_token_count": context_token_count,
            "output_token_count": output_token_count,
            "num_steps": len(steps),
            "num_turns": len(turn_ids),
            "turn_ids": turn_ids,
            "timestamp": str(time.time()),
            "history_rewritten": session.history_rewritten,
            "segment_reason": segment["segment_reason"],
            "segment_reasons": segment["segment_reasons"],
            "trajectory_num_segments": segment_count,
        }
        if base_step.profile:
            extra_info["dressage_profile"] = dict(base_step.profile)
        if concat_logprobs_invalid:
            extra_info["concat_logprobs_invalid"] = True
        if concat_incremental_tokenization_failed:
            extra_info["concat_incremental_tokenization_failed"] = True
        if mask_nonlast_version_tokens:
            extra_info["mask_nonlast_version_tokens"] = True

        if not (len(tokens) == len(response_logprobs) == len(response_mask)):
            raise RuntimeError(
                "concat segment arrays are not aligned: "
                f"tokens={len(tokens)}, "
                f"full_logprobs={len(response_logprobs)}, "
                f"full_loss_mask={len(response_mask)}"
            )
        if record_token_versions and len(full_versions) != len(tokens):
            raise RuntimeError(
                "concat segment arrays are not aligned: "
                f"tokens={len(tokens)}, "
                f"full_logprobs={len(response_logprobs)}, "
                f"full_loss_mask={len(response_mask)}, "
                f"full_versions={len(full_versions)}"
            )

        record = {
            "uid": str(uuid.uuid4()),
            "trajectory_id": trajectory_id,
            "turn_id": turn_ids[-1],
            "instance_id": instance_id,
            "segment_index": segment_index,
            "segment_count": segment_count,
            "messages": messages,
            "tools": tools,
            "tokens": tokens,
            "full_logprobs": response_logprobs,
            "full_loss_mask": response_mask,
            "aligned_response_length": sum(response_mask),
            "label": label,
            "finish_reason": base_step.finish_reason,
            "extra_info": extra_info,
        }
        if record_token_versions:
            record["full_versions"] = full_versions

        routed_experts_parts: list[dict[str, Any]] = []
        accumulated_prefix_len = 0
        for step_index, step in enumerate(steps):
            if step.response_routed_experts_chunks or step.response_routed_experts is not None:
                part = {
                    "prefix_token_count": accumulated_prefix_len,
                    "concat_token_count": len(step.concat_token_ids),
                    "is_first_step": step_index == 0,
                }
                if step.response_routed_experts_chunks:
                    part["chunks"] = step.response_routed_experts_chunks
                if step.response_routed_experts is not None:
                    part["data"] = step.response_routed_experts
                routed_experts_parts.append(part)
            accumulated_prefix_len += len(step.concat_token_ids)
        if routed_experts_parts:
            record["routed_experts_parts"] = routed_experts_parts

        return record

    def _concat_prefix_token_ids(session: Session) -> list[int]:
        start_index = 0
        for index, step in enumerate(session.steps):
            if step.segment_boundary_before:
                start_index = index

        tokens: list[int] = []
        for step in session.steps[start_index:]:
            tokens.extend(step.concat_token_ids)
        return tokens

    def _full_prompt_token_ids(
        *,
        normalized_request_messages: list[dict],
        tools: list[dict[str, Any]] | None,
    ) -> list[int]:
        return mask_builder.tokenize_messages(
            normalized_request_messages,
            tools,
            add_generation_prompt=True,
        )

    def _build_prompt_tokens(
        *,
        session: Session,
        previous_step: Any | None,
        segment_boundary_before: bool,
        normalized_request_messages: list[dict],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if (
            trajectory_build_mode != "concat"
            or previous_step is None
            or segment_boundary_before
        ):
            input_ids = _full_prompt_token_ids(
                normalized_request_messages=normalized_request_messages,
                tools=tools,
            )
            return {
                "input_ids": input_ids,
                "context_delta_ids": list(input_ids),
                "used_tito_for_prompt": False,
                "concat_incremental_tokenization_failed": False,
            }

        if tito_tokenizer is None:
            raise RuntimeError("TITO tokenizer is not initialized for concat mode.")

        prefix_tokens = _concat_prefix_token_ids(session)
        try:
            merged_tokens = tito_tokenizer.merge_tokens(
                old_messages=previous_step.messages_snapshot,
                new_messages=normalized_request_messages,
                pretokenized_token_ids=prefix_tokens,
                tools=tools,
            )
        except Exception:
            logger.exception(
                "TITO online prompt tokenization failed; falling back to full tokenizer."
            )
            input_ids = _full_prompt_token_ids(
                normalized_request_messages=normalized_request_messages,
                tools=tools,
            )
            return {
                "input_ids": input_ids,
                "context_delta_ids": list(input_ids),
                "used_tito_for_prompt": False,
                "concat_incremental_tokenization_failed": True,
            }

        return {
            "input_ids": list(merged_tokens),
            "context_delta_ids": list(merged_tokens[len(prefix_tokens) :]),
            "used_tito_for_prompt": True,
            "concat_incremental_tokenization_failed": False,
        }

    def _build_concat_step_payload(
        *,
        context_delta_ids: list[int],
        response_token_ids: list[int],
        response_logprobs: list[float],
        response_versions: list[str],
        context_version: str,
        concat_incremental_tokenization_failed: bool,
    ) -> dict[str, Any]:
        context_ids = list(context_delta_ids)
        output_logprobs, invalid = _normalize_logprobs_to_length(
            list(response_logprobs),
            len(response_token_ids),
        )
        normalized_response_versions = [str(value) for value in response_versions]
        if len(normalized_response_versions) != len(response_token_ids):
            version = normalized_response_versions[-1] if normalized_response_versions else "unknown"
            normalized_response_versions = [str(version)] * len(response_token_ids)
        token_ids = context_ids + list(response_token_ids)
        return {
            "concat_token_ids": token_ids,
            "concat_response_logprobs": [0.0] * len(context_ids) + output_logprobs,
            "concat_response_mask": [0] * len(context_ids)
            + [1] * len(response_token_ids),
            "concat_versions": [_INPUT_TOKEN_VERSION] * len(context_ids)
            + normalized_response_versions,
            "concat_context_token_count": len(context_ids),
            "concat_output_token_count": len(response_token_ids),
            "concat_logprobs_invalid": invalid,
            "concat_incremental_tokenization_failed": (
                concat_incremental_tokenization_failed
            ),
        }

    def _completion_usage_tokens(
        *,
        response_token_ids: list[int],
        raw_text: str,
    ) -> int:
        if response_token_ids:
            return len(response_token_ids)
        if not raw_text:
            return 0

        def _warn_estimated(method: str) -> None:
            logger.warning(
                "SGLang response missing output_ids; estimated completion_tokens "
                "for public usage using %s.",
                method,
            )

        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            try:
                token_ids = encode(raw_text, add_special_tokens=False)
                token_count = len(token_ids)
                _warn_estimated("tokenizer.encode")
                return token_count
            except TypeError:
                try:
                    token_ids = encode(raw_text)
                    token_count = len(token_ids)
                    _warn_estimated("tokenizer.encode")
                    return token_count
                except Exception:
                    pass
            except Exception:
                pass

        if callable(tokenizer):
            try:
                encoded = tokenizer(raw_text, add_special_tokens=False)
                input_ids = (
                    encoded.get("input_ids")
                    if isinstance(encoded, dict)
                    else getattr(encoded, "input_ids", None)
                )
                if input_ids is not None:
                    token_count = len(input_ids)
                    _warn_estimated("callable tokenizer")
                    return token_count
            except Exception:
                pass

        _warn_estimated("raw text length")
        return len(raw_text)

    def _openai_response(
        *,
        model: str,
        content: str | None,
        reasoning_content: str | None,
        tool_calls: list[dict] | None,
        finish_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
        response_id: str | None = None,
        profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        message = _assistant_message(
            content, tool_calls, reasoning_content=reasoning_content
        )
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if profile:
            usage["profile"] = profile_public(profile)
        return {
            "id": response_id or f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish_reason}
            ],
            "usage": usage,
        }

    async def _pseudo_stream_chunks(
        response_id: str,
        model: str,
        content: str | None,
        reasoning_content: str | None,
        tool_calls: list[dict] | None,
        finish_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
        include_usage: bool,
        profile: dict[str, Any] | None = None,
    ):
        created = int(time.time())
        stream_started = time.perf_counter() if profiling_enabled() else 0.0
        chunk_count = 0

        def _chunk(delta: dict[str, Any], reason: str | None = None) -> str:
            nonlocal chunk_count
            chunk_count += 1
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
            }
            if include_usage:
                data["usage"] = None
            return f"data: {json.dumps(data)}\n\n"

        def _usage_chunk() -> str:
            if profile is not None and profiling_enabled():
                profile_set(
                    profile,
                    "chat.stream_emit_s",
                    time.perf_counter() - stream_started,
                )
                profile_set(profile, "chat.stream_chunk_count", float(chunk_count))
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            if profile:
                data["usage"]["profile"] = profile_public(profile)
            return f"data: {json.dumps(data)}\n\n"

        yield _chunk({"role": "assistant"})
        if reasoning_content:
            chunk_size = 8
            for start in range(0, len(reasoning_content), chunk_size):
                yield _chunk(
                    {"reasoning_content": reasoning_content[start : start + chunk_size]}
                )
                await asyncio.sleep(0)
        if content:
            chunk_size = 8
            for start in range(0, len(content), chunk_size):
                yield _chunk({"content": content[start : start + chunk_size]})
                await asyncio.sleep(0)
        if tool_calls:
            for tool_call in tool_calls:
                yield _chunk(
                    {
                        "tool_calls": [
                            {
                                "index": tool_call["index"],
                                "id": tool_call["id"],
                                "type": "function",
                                "function": tool_call["function"],
                            }
                        ]
                    }
                )
                await asyncio.sleep(0)
        yield _chunk({}, finish_reason)
        if include_usage:
            yield _usage_chunk()
        yield "data: [DONE]\n\n"

    @app.get("/v1/models")
    async def list_models(request: Request):
        _check_auth(request)
        try:
            return JSONResponse(await sglang_client.list_models())
        except Exception:
            logger.exception("Failed to fetch models from SGLang router.")
            return JSONResponse(
                {
                    "object": "list",
                    "data": [
                        {"id": "proxy-model", "object": "model", "owned_by": "proxy"}
                    ],
                }
            )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        chat_profile: dict[str, Any] = {}
        chat_started = time.perf_counter() if profiling_enabled() else 0.0
        _check_auth(request)
        body = await request.json()
        messages: list[dict] = body.get("messages", [])
        model = body.get("model", "proxy-model")
        stream = bool(body.get("stream", False))
        tools = body.get("tools")
        stream_options = body.get("stream_options")
        if not isinstance(stream_options, dict):
            stream_options = {}
        # Dressage defaults usage on for opencode compatibility; set
        # stream_options.include_usage=false to opt out.
        include_usage = stream_options.get("include_usage", True) is not False

        session_id, instance_id, turn_id = _runtime_ids_from_request(request, body)
        try:
            session, _ = session_manager.get_or_create_session(
                session_id, messages, instance_id=instance_id
            )
        except SessionFinalizedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        session_id = session.session_id
        lock_wait_started = time.perf_counter() if profiling_enabled() else 0.0
        async with session.request_lock:
            profile_add(
                chat_profile,
                "chat.session_lock_wait_s",
                time.perf_counter() - lock_wait_started,
            )
            try:
                session_manager.ensure_session_active(session_id, session)
            except SessionFinalizedError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

            instance_id = instance_id or session.instance_id
            try:
                effective_turn_id = session_manager.resolve_turn_id(
                    session_id=session_id, requested_turn_id=turn_id
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            previous_step = session.latest_step
            request_rollout_epoch = generation_controller.current_epoch
            _raise_if_stale_rollout_epoch(
                session=session,
                current_epoch=request_rollout_epoch,
                partial_rollout=partial_rollout,
            )
            append_only = (
                previous_step is None
                or session_manager.is_append_only_continuation(
                    previous_step.messages_snapshot, messages
                )
            )
            rewrite_detected = (
                previous_step is not None
                and previous_step.turn_id == effective_turn_id
                and not append_only
            )
            message_prefix_mismatch = (
                trajectory_build_mode == "concat"
                and previous_step is not None
                and not append_only
                and not rewrite_detected
            )
            tools_changed = previous_step is not None and _tools_changed(
                previous_step.tools,
                tools,
                none_equals_empty=none_equals_empty_tools,
            )
            segment_reasons_before = _boundary_reasons(
                rewrite_detected=rewrite_detected,
                tools_changed=tools_changed,
                message_prefix_mismatch=message_prefix_mismatch,
            )
            segment_boundary_before = previous_step is not None and bool(
                segment_reasons_before
            )
            rewrite_reason = None
            if rewrite_detected:
                rewrite_reason = "Turn history rewritten between steps."
                session_manager.mark_history_rewritten(session_id, rewrite_reason)

            normalized_request_messages = mask_builder.normalize_template_messages(messages)
            prompt_payload = _build_prompt_tokens(
                session=session,
                previous_step=previous_step,
                segment_boundary_before=segment_boundary_before,
                normalized_request_messages=normalized_request_messages,
                tools=tools,
            )
            input_ids = prompt_payload["input_ids"]
            concat_incremental_tokenization_failed = bool(
                prompt_payload["concat_incremental_tokenization_failed"]
            )
            if (
                concat_incremental_tokenization_failed
                and previous_step is not None
                and not segment_boundary_before
            ):
                segment_reasons_before = list(segment_reasons_before) + [
                    "concat_incremental_tokenization_failed"
                ]
                segment_boundary_before = True
            request_logprob_start_len = -1 if trajectory_build_mode == "concat" else 0
            prompt_tokens = len(input_ids)
            sampling_params = _build_sampling_params(
                body,
                default_max_tokens,
                rollout_temperature,
            )
            max_tokens = int(sampling_params.get("max_new_tokens") or 0)

            if context_window is not None and prompt_tokens >= context_window:
                return JSONResponse(
                    _context_overflow_payload(
                        phase="input",
                        context_window=context_window,
                        input_tokens=prompt_tokens,
                        output_tokens=0,
                        max_tokens=max_tokens,
                        session_id=session_id,
                        turn_id=effective_turn_id,
                        last_proxy_step_recorded=False,
                    ),
                    status_code=413,
                )

            if dynamic_max_tokens and context_window is not None:
                sampling_params["max_new_tokens"] = min(
                    max_tokens,
                    context_window - prompt_tokens,
                )
                max_tokens = int(sampling_params["max_new_tokens"])

            _max_steps = int(os.environ.get("DRESSAGE_PROXY_MAX_STEPS_PER_SESSION", "0"))
            if _max_steps > 0 and len(session.steps) >= _max_steps:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Session {session_id} reached proxy step limit "
                        f"({_max_steps}). Refusing further inference."
                    ),
                )

            expected_version = request.headers.get("X-Dressage-Expected-Version")
            _raise_if_cross_version_trajectory(
                session=session,
                candidate_versions=[
                    expected_version,
                    generation_controller.current_version,
                ],
                partial_rollout=partial_rollout,
            )
            try:
                async with async_profile_span(chat_profile, "chat.generation_s"):
                    router_response = await generation_controller.generate_preemptible(
                        input_ids=input_ids,
                        sampling_params=sampling_params,
                        session_id=session_id,
                        instance_id=instance_id,
                        turn_id=effective_turn_id,
                        routing_key=session_id,
                        expected_version=expected_version,
                        expected_epoch=(
                            request_rollout_epoch
                            if (not partial_rollout and session.steps)
                            else None
                        ),
                        logprob_start_len=request_logprob_start_len,
                        context_window=context_window,
                    )
            except GenerationStaleEpoch as exc:
                logger.warning(
                    "reject non-partial rollout: error=trajectory_version_changed "
                    "session_id=%s instance_id=%s session_epoch=%s current_epoch=%s",
                    session_id,
                    instance_id,
                    exc.expected_epoch,
                    exc.current_epoch,
                )
                raise HTTPException(
                    status_code=502,
                    detail={
                        "error": "trajectory_version_changed",
                        "message": str(exc),
                        "session_epoch": exc.expected_epoch,
                        "current_epoch": exc.current_epoch,
                        "session_id": session_id,
                        "instance_id": instance_id,
                    },
                ) from exc
            except GenerationPreempted as exc:
                logger.warning(
                    "reject non-partial rollout: error=generation_preempted "
                    "session_id=%s instance_id=%s message=%s",
                    session_id,
                    instance_id,
                    str(exc),
                )
                raise HTTPException(
                    status_code=502,
                    detail={
                        "error": "generation_preempted",
                        "message": str(exc),
                        "session_id": session_id,
                        "instance_id": instance_id,
                    },
                ) from exc
            except ProxyShuttingDown as exc:
                raise HTTPException(
                    status_code=503,
                    detail={"error": "proxy_shutting_down"},
                ) from exc
            except httpx.RequestError as exc:
                logger.info("SGLang upstream request failed: %r", exc)
                raise HTTPException(
                    status_code=503,
                    detail={"error": "sglang_upstream_unavailable", "detail": str(exc)},
                ) from exc

            raw_text = router_response.text
            response_token_ids = router_response.output_ids
            response_logprobs = router_response.output_token_logprobs
            response_versions = list(router_response.output_versions)
            context_overflow = router_response.meta_info.get("context_overflow")
            output_overflow = (
                isinstance(context_overflow, dict) and context_window is not None
            )
            response_version = (
                response_versions[-1]
                if response_versions
                else router_response.weight_version
                or expected_version
                or generation_controller.current_version
                or "unknown"
            )
            if len(response_versions) != len(response_token_ids):
                response_versions = [str(response_version)] * len(response_token_ids)
            request_version = expected_version or generation_controller.current_version or response_version
            _raise_if_cross_version_trajectory(
                session=session,
                candidate_versions=[*response_versions, response_version, request_version],
                partial_rollout=partial_rollout,
            )
            if session.rollout_epoch is None:
                session.rollout_epoch = router_response.rollout_epoch
            prompt_versions = [_INPUT_TOKEN_VERSION] * len(input_ids)
            all_versions = prompt_versions + response_versions
            public_completion_tokens = _completion_usage_tokens(
                response_token_ids=response_token_ids,
                raw_text=raw_text,
            )
            finish_reason = (
                "length"
                if output_overflow or router_response.finish_reason == "length"
                else "stop"
            )

            if output_overflow:
                content = ""
                tool_calls = None
                reasoning_content = None
            else:
                profile_set(chat_profile, "chat.raw_text_chars", float(len(raw_text)))
                async with async_profile_span(chat_profile, "chat.reasoning_parse_s"):
                    reasoning_result = await proxy_reasoning_parser.parse(
                        raw_text,
                        routing_key=session_id,
                        profile=chat_profile,
                    )
                reasoning_content = reasoning_result.reasoning_content
                profile_set(
                    chat_profile,
                    "chat.visible_text_chars",
                    float(len(reasoning_result.text)),
                )
                async with async_profile_span(chat_profile, "chat.tool_parse_s"):
                    content, tool_calls = await proxy_tool_call_parser.parse(
                        reasoning_result.text,
                        tools,
                        routing_key=session_id,
                        profile=chat_profile,
                    )
                content = _strip_public_stop_markers(content)
                if tool_calls:
                    finish_reason = "tool_calls"

            full_messages = messages + [
                _assistant_message(
                    content,
                    tool_calls,
                    reasoning_content=reasoning_content,
                )
            ]
            recorded_response_token_ids = [] if output_overflow else response_token_ids
            recorded_response_logprobs = [] if output_overflow else response_logprobs
            recorded_response_versions = [] if output_overflow else response_versions
            recorded_output_token_texts = (
                [] if output_overflow else list(router_response.output_token_texts)
            )
            recorded_all_token_ids = (
                list(input_ids)
                if output_overflow
                else list(router_response.all_token_ids)
            )
            recorded_all_logprobs = (
                list(router_response.input_token_logprobs_raw)
                if output_overflow
                else list(router_response.all_logprobs)
            )
            recorded_all_versions = (
                prompt_versions if output_overflow else all_versions
            )
            recorded_raw_text = "" if output_overflow else raw_text
            concat_payload: dict[str, Any] = {}
            if trajectory_build_mode == "concat":
                snapshot_token_ids = list(input_ids) + list(recorded_response_token_ids)
                concat_payload = _build_concat_step_payload(
                    context_delta_ids=prompt_payload["context_delta_ids"],
                    response_token_ids=recorded_response_token_ids,
                    response_logprobs=recorded_response_logprobs,
                    response_versions=recorded_response_versions,
                    context_version=str(request_version),
                    concat_incremental_tokenization_failed=(
                        concat_incremental_tokenization_failed
                    ),
                )
            else:
                normalized_full_messages = mask_builder.normalize_template_messages(
                    full_messages
                )
                snapshot_token_ids = mask_builder.tokenize_messages(
                    normalized_full_messages,
                    tools,
                    add_generation_prompt=False,
                )

            generation_profile = router_response.meta_info.get("dressage_profile")
            if isinstance(generation_profile, dict) and profiling_enabled():
                for key, value in generation_profile.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        chat_profile[str(key)] = float(value)

            with profile_span(chat_profile, "chat.record_step_s"):
                session_manager.record_step(
                    session_id=session_id,
                    turn_id=effective_turn_id,
                    request_messages=messages,
                    normalized_request_messages=normalized_request_messages,
                    prompt_token_ids=input_ids,
                    prompt_token_logprobs=list(router_response.input_token_logprobs_raw),
                    snapshot_token_ids=snapshot_token_ids,
                    response_token_ids=recorded_response_token_ids,
                    response_logprobs=recorded_response_logprobs,
                    response_versions=recorded_response_versions,
                    all_token_ids=recorded_all_token_ids,
                    all_logprobs=recorded_all_logprobs,
                    all_versions=recorded_all_versions,
                    prompt_versions=prompt_versions,
                    input_token_texts=list(router_response.input_token_texts),
                    output_token_texts=recorded_output_token_texts,
                    messages=full_messages,
                    raw_response_text=recorded_raw_text,
                    all_logprobs_invalid=router_response.all_logprobs_invalid,
                    **concat_payload,
                    response_routed_experts=router_response.routed_experts,
                    response_routed_experts_chunks=router_response.routed_experts_chunks,
                    profile=router_response.meta_info.get("dressage_profile") or {},
                    tools=tools,
                    segment_boundary_before=segment_boundary_before,
                    rewrite_reason=rewrite_reason,
                    segment_reason_before=(
                        segment_reasons_before[0] if segment_reasons_before else None
                    ),
                    segment_reasons_before=segment_reasons_before,
                    finish_reason=finish_reason,
                    request_version=str(request_version),
                    response_version=str(response_version),
                )

            if output_overflow:
                details = dict(context_overflow)
                details.update(
                    {
                        "phase": "input_output",
                        "context_window": context_window,
                        "input_tokens": prompt_tokens,
                        "output_tokens": len(response_token_ids),
                        "total_tokens": prompt_tokens + len(response_token_ids),
                        "max_tokens": max_tokens,
                        "session_id": session_id,
                        "turn_id": effective_turn_id,
                        "last_proxy_step_recorded": True,
                    }
                )
                return JSONResponse(
                    {
                        "error": "context_overflow",
                        "message": "Dressage proxy context window overflow.",
                        "details": details,
                    },
                    status_code=413,
                )

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        profile_set(chat_profile, "chat.prompt_tokens", float(prompt_tokens))
        profile_set(chat_profile, "chat.completion_tokens", float(public_completion_tokens))
        profile_set(chat_profile, "chat.tool_call_count", float(len(tool_calls or [])))
        profile_add(chat_profile, "chat.total_s", time.perf_counter() - chat_started)
        if stream:
            return StreamingResponse(
                _pseudo_stream_chunks(
                    response_id,
                    model,
                    content,
                    reasoning_content,
                    tool_calls,
                    finish_reason,
                    prompt_tokens,
                    public_completion_tokens,
                    include_usage,
                    chat_profile,
                ),
                media_type="text/event-stream",
            )

        return JSONResponse(
            _openai_response(
                model=model,
                content=content,
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=public_completion_tokens,
                response_id=response_id,
                profile=chat_profile,
            )
        )

    @app.post("/session/finalize")
    async def finalize_session(request: Request):
        _check_auth(request)
        body = await request.json()
        session_id = body["session_id"]
        instance_id = body.get("instance_id")
        label = body.get("label")
        if "trajectory_build_mode" in body or "trajectory_build_modes" in body:
            raise HTTPException(
                status_code=400,
                detail="trajectory_build_mode is configured at proxy startup",
            )

        session = session_manager.get_session(session_id)
        if session is None:
            finalized = session_manager.get_finalized_session_metadata(session_id)
            if finalized is not None:
                finalized["mode"] = "already_finalized"
                return finalized
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        async with session.request_lock:
            session = session_manager.finalize_session(session_id)
            if session is None:
                finalized = session_manager.get_finalized_session_metadata(session_id)
                if finalized is not None:
                    finalized["mode"] = "already_finalized"
                    return finalized
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            if not session.steps:
                raise HTTPException(status_code=400, detail="Session has no turns")

            effective_instance_id = instance_id or session.instance_id
            trajectory_id = session.session_id
            segments = _split_session_into_segments(session)
            segment_count = len(segments)
            build_segment_record = (
                _build_concat_segment_record
                if trajectory_build_mode == "concat"
                else _build_segment_record
            )
            for segment_index, segment in enumerate(segments):
                trajectory_store.write_dict(
                    build_segment_record(
                        session=session,
                        trajectory_id=trajectory_id,
                        segment=segment,
                        segment_index=segment_index,
                        segment_count=segment_count,
                        instance_id=effective_instance_id,
                        label=label,
                    )
                )

            finalized_payload = {
                "success": True,
                "session_id": session_id,
                "trajectory_id": trajectory_id,
                "instance_id": effective_instance_id,
                "num_steps": len(session.steps),
                "num_turns": len(session.turn_ids),
                "num_segments": segment_count,
                "history_rewritten": session.history_rewritten,
                "mode": "finalized",
                "trajectory_build_mode": trajectory_build_mode,
                "trajectory_build_model": trajectory_build_model,
                "record_token_versions": record_token_versions,
                "mask_nonlast_version_tokens": mask_nonlast_version_tokens,
            }
            session_manager.set_finalized_session_metadata(session_id, finalized_payload)

        return finalized_payload

    @app.post("/trajectory/read")
    async def trajectory_read(request: Request):
        _check_auth(request)
        body = await request.json()
        trajectory_id = _trajectory_id_from_body(body)
        instance_id = body.get("instance_id")
        max_groups = body.get("max_groups")
        drain = bool(body.get("drain", False))

        if trajectory_id:
            data = (
                trajectory_store.pop_trajectory(
                    trajectory_id, instance_id=instance_id
                )
                if drain
                else trajectory_store.read_trajectory(
                    trajectory_id, instance_id=instance_id
                )
            )
            return {
                "success": bool(data),
                "mode": "trajectory",
                "data": data,
                "meta_info": trajectory_store.stats(),
                "drained": drain,
            }

        data = trajectory_store.read_batch(max_groups=max_groups)
        return {
            "success": bool(data),
            "mode": "batch",
            "data": data,
            "meta_info": trajectory_store.stats(),
        }

    @app.get("/trajectory/stats")
    async def trajectory_stats(request: Request):
        _check_auth(request)
        return trajectory_store.stats()

    @app.post("/v1/rollout/pause")
    async def pause_rollout(request: Request):
        _check_auth(request)
        body = await request.json()
        return await generation_controller.pause(
            session_id=body.get("session_id"),
            instance_id=body.get("instance_id"),
            reason=str(body.get("reason") or "weight_update"),
            mode=str(body.get("mode") or "preempt"),
            timeout_seconds=body.get("timeout_seconds"),
        )

    @app.post("/v1/rollout/resume")
    async def resume_rollout(request: Request):
        _check_auth(request)
        body = await request.json()
        result = await generation_controller.resume(
            version=None if body.get("version") is None else str(body.get("version")),
            reason=str(body.get("reason") or "weight_update"),
        )
        if result.get("status") in {"backend_not_ready", "shutting_down"}:
            raise HTTPException(status_code=503, detail=result)
        return result

    @app.get("/v1/rollout/pause_state")
    async def pause_state(request: Request):
        _check_auth(request)
        return generation_controller.state()

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "active_sessions": session_manager.active_count(),
            "store": trajectory_store.stats(),
            "rollout_pause": generation_controller.state(),
            "config": {
                "sglang_router_url": sglang_router_url,
                "trajectory_build_mode": trajectory_build_mode,
                "trajectory_build_model": trajectory_build_model,
                "record_token_versions": record_token_versions,
                "mask_nonlast_version_tokens": mask_nonlast_version_tokens,
                "rollout_temperature": rollout_temperature,
                "context_window": context_window,
                "dynamic_max_tokens": dynamic_max_tokens,
                "use_rollout_routing_replay": use_rollout_routing_replay,
                "partial_rollout": partial_rollout,
            },
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dressage Proxy Server")
    parser.add_argument(
        "--sglang-router-url",
        default=None,
        help="SGLang Router base URL, e.g. http://localhost:30000",
    )
    parser.add_argument("--tokenizer-path", required=True, help="HF tokenizer path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--default-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--context-window",
        type=_positive_int,
        default=None,
        help="Total input+output context window used for proxy-side overflow checks.",
    )
    parser.add_argument(
        "--no-dynamic-max-tokens",
        action="store_false",
        dest="dynamic_max_tokens",
        help="Disable clamping max_tokens to the remaining context window.",
    )
    parser.add_argument(
        "--rollout-temperature",
        type=float,
        default=1.0,
        help="Fallback sampling temperature when the chat completion body omits temperature.",
    )
    parser.add_argument(
        "--use-rollout-routing-replay",
        action="store_true",
        default=False,
        help="Request routed expert IDs from SGLang for rollout routing replay.",
    )
    parser.add_argument("--model-mask-type")
    parser.add_argument("--model-tool-call-type")
    parser.add_argument(
        "--tool-call-parse-backend",
        choices=("local", "sglang_api", "hybrid"),
        default="sglang_api",
    )
    parser.add_argument(
        "--model-reasoning-type",
        help="SGLang reasoning parser name, e.g. qwen3.",
    )
    parser.add_argument(
        "--reasoning-parse-backend",
        choices=("local", "sglang_api", "hybrid"),
        default="sglang_api",
    )
    parser.add_argument("--api-key", default="no-auth")
    parser.add_argument("--min-group-size", type=int, default=1)
    parser.add_argument("--session-timeout", type=float, default=3200.0)
    parser.add_argument("--group-timeout", type=float, default=300.0)
    parser.add_argument(
        "--trajectory-build-mode",
        choices=("last_step", "concat"),
        default="concat",
        help="Trajectory build strategy, fixed for the proxy lifetime.",
    )
    parser.add_argument(
        "--trajectory-build-model",
        default=DEFAULT_TRAJECTORY_BUILD_MODEL,
        help="Trajectory model defaults to infer mask/parser/TITO settings.",
    )
    parser.add_argument(
        "--tito-model",
        choices=("qwen3_5",),
        default=None,
        help="TITO model type, required when --trajectory-build-mode=concat.",
    )
    parser.add_argument(
        "--record-token-versions",
        action="store_true",
        help="Persist token-level model weight versions in trajectory payloads.",
    )
    parser.add_argument(
        "--mask-nonlast-version-tokens",
        action="store_true",
        help=(
            "Mask trainable tokens from non-last model weight versions when "
            "--record-token-versions is enabled."
        ),
    )
    parser.add_argument(
        "--dressage-partial-rollout",
        action="store_true",
        help="Allow interrupted SGLang generations to resume from partial output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    app = create_app(
        sglang_router_url=args.sglang_router_url,
        tokenizer_path=args.tokenizer_path,
        trajectory_store=TrajectoryStore(
            min_group_size=args.min_group_size, group_timeout=args.group_timeout
        ),
        session_manager=SessionManager(session_timeout=args.session_timeout),
        model_tool_call_type=args.model_tool_call_type,
        tool_call_parse_backend=args.tool_call_parse_backend,
        model_reasoning_type=args.model_reasoning_type,
        reasoning_parse_backend=args.reasoning_parse_backend,
        model_mask_type=args.model_mask_type,
        default_max_tokens=args.default_max_tokens,
        api_key=args.api_key,
        trajectory_build_mode=args.trajectory_build_mode,
        trajectory_build_model=args.trajectory_build_model,
        tito_model=args.tito_model,
        record_token_versions=args.record_token_versions,
        mask_nonlast_version_tokens=args.mask_nonlast_version_tokens,
        rollout_temperature=args.rollout_temperature,
        context_window=args.context_window,
        dynamic_max_tokens=args.dynamic_max_tokens,
        use_rollout_routing_replay=args.use_rollout_routing_replay,
        partial_rollout=args.dressage_partial_rollout,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
