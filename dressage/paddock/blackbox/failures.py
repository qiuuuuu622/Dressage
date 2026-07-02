"""Semantic blackbox agent failure parsing and metadata helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

AGENT_FAILED_STATES = {"desynced", "error", "failed"}
SEMANTIC_HTTP_ERRORS = {
    "backend_timeout",
    "context_overflow",
    "max_steps_exceeded",
}
HARVESTABLE_AGENT_ERRORS = {
    "context_overflow",
    "max_steps_exceeded",
}
SEMANTIC_HTTP_ERROR_LABELS = {
    "backend_timeout": "backend timeout",
    "context_overflow": "context overflow",
    "max_steps_exceeded": "max steps exceeded",
}
METADATA_TEXT_LIMIT = 4096
GENERATION_PREEMPTED_MARKER = "Dressage proxy generation_preempted"


class BlackboxAgentFailure(RuntimeError):
    """Raised when the blackbox server reports a semantic agent failure."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        state: str | None = None,
        agent_response: str = "",
        details: dict[str, Any] | None = None,
        http_status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.state = state
        self.agent_response = agent_response
        self.details = details or {}
        self.http_status_code = http_status_code


def agent_response_text(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("response", "content", "message", "text", "output"):
            if key in payload and payload[key] is not None:
                return str(payload[key])
    return "" if payload is None else str(payload)


def failure_from_payload_state(
    payload: Any,
    *,
    agent_response: str,
) -> BlackboxAgentFailure | None:
    """Reject structured failed/desynced payloads without inspecting text content."""

    state = _agent_payload_state(payload)
    state_normalized = state.lower() if state is not None else None
    if state_normalized not in AGENT_FAILED_STATES:
        return None

    return BlackboxAgentFailure(
        f"blackbox agent returned failed state {state!r}",
        kind="agent_failed_state",
        state=state,
        agent_response=agent_response,
    )


def failure_from_call_agent_exception(
    exc: BaseException,
) -> BlackboxAgentFailure | None:
    """Map blackbox-server semantic HTTP errors to explicit rollout errors."""

    if not isinstance(exc, httpx.HTTPStatusError):
        return None

    payload = _http_error_json(exc)
    error_code = payload.get("error")
    if error_code not in SEMANTIC_HTTP_ERRORS:
        return None

    details = payload.get("details")
    if not isinstance(details, dict):
        details = {}
    message = payload.get("message") or f"blackbox agent {error_code}"
    error_label = SEMANTIC_HTTP_ERROR_LABELS.get(
        str(error_code),
        str(error_code),
    )
    status_suffix = f" (HTTP {exc.response.status_code})"
    return BlackboxAgentFailure(
        f"blackbox agent {error_label}{status_suffix}: {message}",
        kind=str(error_code),
        state=str(details["state"]) if "state" in details else None,
        details={str(key): _json_safe(value) for key, value in details.items()},
        http_status_code=exc.response.status_code,
    )


def expected_abort_from_call_agent_exception(exc: BaseException) -> str | None:
    """Return the expected abort kind for retry-only blackbox call failures."""

    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    if exc.response.status_code != 502:
        return None
    if "/v1/sessions/" not in exc.request.url.path or not exc.request.url.path.endswith(
        "/messages"
    ):
        return None

    payload = _http_error_json(exc)
    if payload.get("error") != "backend_error":
        return None

    message = payload.get("message")
    if message is not None and GENERATION_PREEMPTED_MARKER in str(message):
        return "generation_preempted"
    return "backend_error"


def record_agent_failure_metadata(
    metadata: dict[str, Any],
    failure: BlackboxAgentFailure,
) -> None:
    metadata["blackbox_agent_error_kind"] = failure.kind
    if failure.state is not None:
        metadata["blackbox_agent_state"] = failure.state
    if failure.http_status_code is not None:
        metadata["blackbox_agent_http_status_code"] = failure.http_status_code
    if failure.details:
        metadata["blackbox_agent_error_details"] = _json_safe(failure.details)
    if failure.agent_response:
        text, truncated = _truncated_metadata_text(failure.agent_response)
        metadata["blackbox_agent_response"] = text
        metadata["blackbox_agent_response_truncated"] = truncated


def record_agent_early_stop_metadata(
    metadata: dict[str, Any],
    failure: BlackboxAgentFailure,
) -> None:
    metadata["blackbox_agent_early_stop"] = True
    metadata["blackbox_agent_early_stop_kind"] = failure.kind


def record_blackbox_abort_for_retry(
    metadata: dict[str, Any],
    session_id: str,
    exc: BaseException,
) -> None:
    """Append blackbox-specific abort context to sample metadata."""

    history = metadata.get("blackbox_failure_history")
    if not isinstance(history, list):
        history = []
        metadata["blackbox_failure_history"] = history

    error_text = str(exc)
    history.append(
        {
            "session_id": session_id,
            "error_type": type(exc).__name__,
            "error": error_text,
            "retry_count": metadata.get("dressage_retry_count", 0),
        }
    )
    metadata["blackbox_error"] = error_text


def _agent_payload_state(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    state = payload.get("state") or payload.get("status")
    return None if state is None else str(state)


def _http_error_json(exc: httpx.HTTPStatusError) -> dict[str, Any]:
    try:
        payload = exc.response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _truncated_metadata_text(value: Any) -> tuple[str, bool]:
    text = "" if value is None else str(value)
    if len(text) <= METADATA_TEXT_LIMIT:
        return text, False
    return text[:METADATA_TEXT_LIMIT], True


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)
