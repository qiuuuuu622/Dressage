from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from blackbox_server.core.models import DEFAULT_PROXY_MAX_STEPS


LOGGER = logging.getLogger(__name__)
_HEALTH_TOKEN_FIELD = "health_token"

DRESSAGE_ROLLOUT_INVALIDATED_ERRORS = {
    "generation_preempted",
    "trajectory_version_changed",
}


def rollout_proxy_health_matches(
    response: httpx.Response,
    expected_token: str | None,
) -> bool:
    if response.status_code != 200 or not expected_token:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    return isinstance(payload, dict) and payload.get(_HEALTH_TOKEN_FIELD) == expected_token


@dataclass
class _TurnScope:
    turn_id: str
    backend_session_id: str | None = None
    step_counter: int = 0
    inflight_requests: int = 0
    context_overflow_error: dict[str, Any] | None = None
    rollout_invalidated_error: dict[str, Any] | None = None
    max_steps_error: dict[str, Any] | None = None
    llm_request_count: int = 0
    llm_plain_http_s: float = 0.0
    llm_stream_http_s: float = 0.0
    llm_request_bytes: int = 0
    llm_response_bytes: int = 0
    llm_error_count: int = 0
    step_records: list[dict[str, Any]] = field(default_factory=list)
    drained: asyncio.Event = field(default_factory=asyncio.Event)
    max_steps_exceeded: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.drained.set()


@dataclass(frozen=True)
class _TurnSnapshot:
    session_id: str
    turn_id: str | None
    backend_session_id: str | None
    step: int
    scope: _TurnScope | None = None
    max_steps_exceeded: bool = False


@dataclass
class _LLMRequestContext:
    proxy: "RolloutLLMProxy"
    snapshot: _TurnSnapshot | None
    request_bytes: int
    streaming: bool
    request_profile: dict[str, Any] | None = None
    status_code: int | None = None
    response_bytes: int = 0
    response_profile: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False
    _started_at: float = 0.0
    _closed: bool = False

    async def __aenter__(self) -> "_LLMRequestContext":
        self._started_at = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        await self.close(is_error=exc_type is not None)
        return False

    async def close(self, *, is_error: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        scope = self.snapshot.scope if self.snapshot is not None else None
        if scope is None:
            return
        if self.is_error or is_error:
            await self.proxy._record_llm_error(scope)
        await self.proxy._record_llm_request(
            scope,
            step=self.snapshot.step if self.snapshot is not None else None,
            seconds=time.perf_counter() - self._started_at,
            request_bytes=self.request_bytes,
            response_bytes=self.response_bytes,
            status_code=self.status_code,
            streaming=self.streaming,
            request_profile=self.request_profile,
            response_profile=self.response_profile,
        )
        await self.proxy._mark_request_finished(scope)


class RolloutLLMProxy:
    def __init__(
        self,
        *,
        upstream_origin: str,
        router_api_path: str,
        bound_session_id: str,
        bound_instance_id: str,
        sticky_header_name: str,
        max_steps: int | None = DEFAULT_PROXY_MAX_STEPS,
        default_temperature: float | None = None,
    ) -> None:
        self.upstream_origin = upstream_origin.rstrip("/")
        self.router_api_path = router_api_path.rstrip("/") or "/"
        self.bound_session_id = bound_session_id
        self.bound_instance_id = bound_instance_id
        self.sticky_header_name = sticky_header_name
        self.max_steps = max_steps
        self.default_temperature = default_temperature
        self._client: httpx.AsyncClient | None = None
        self._turn_scope: _TurnScope | None = None
        self._scope_lock = asyncio.Lock()
        self._paused = False
        self._pause_reason: str | None = None
        self._current_version: str | None = None
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._pause_state_changed = asyncio.Event()
        self._pause_started_at: float | None = None
        self._total_paused_seconds = 0.0
        self._health_token = uuid4().hex
        self._app = self._build_app()

    @property
    def app(self) -> FastAPI:
        return self._app

    @property
    def health_token(self) -> str:
        return self._health_token

    async def open_turn(self, turn_id: str, backend_session_id: str | None = None) -> None:
        async with self._scope_lock:
            if self._turn_scope is not None:
                raise RuntimeError(f"turn scope is already active for {self._turn_scope.turn_id}")
            self._turn_scope = _TurnScope(turn_id=turn_id, backend_session_id=backend_session_id)

    async def update_turn_backend_session(self, backend_session_id: str) -> None:
        async with self._scope_lock:
            if self._turn_scope is None:
                return
            self._turn_scope.backend_session_id = backend_session_id

    async def drain_turn(self, timeout: float | None = None) -> None:
        async with self._scope_lock:
            scope = self._turn_scope
        if scope is None:
            return
        await self._wait_event_excluding_pause(scope.drained, timeout=timeout)

    @property
    def total_paused_seconds(self) -> float:
        return self._total_paused_seconds

    def pause_state(self) -> dict[str, Any]:
        return {
            "paused": self._paused,
            "pause_reason": self._pause_reason,
            "version": self._current_version,
            "http_inflight_requests": self._http_inflight_count(),
            "total_paused_seconds": self._total_paused_seconds,
        }

    async def pause(
        self,
        *,
        reason: str = "weight_update",
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        async with self._scope_lock:
            already_paused = self._paused
            self._paused = True
            self._pause_reason = reason
            if self._pause_started_at is None:
                self._pause_started_at = loop.time()
            self._resume_event.clear()
            self._notify_pause_state_changed_locked()

        result = await self._control_post(
            "/rollout/pause",
            {
                "session_id": self.bound_session_id,
                "instance_id": self.bound_instance_id,
                "reason": reason,
                "mode": "preempt",
                "timeout_seconds": timeout_seconds,
            },
        )
        result.setdefault("status", "already_paused" if already_paused else "paused")
        result.setdefault("reason", reason)
        result["http_inflight_requests"] = self._http_inflight_count()
        return result

    async def resume(
        self,
        *,
        version: str | None = None,
        reason: str = "weight_update",
    ) -> dict[str, Any]:
        result = await self._control_post(
            "/rollout/resume",
            {
                "session_id": self.bound_session_id,
                "instance_id": self.bound_instance_id,
                "reason": reason,
                "version": version,
            },
        )
        loop = asyncio.get_running_loop()
        async with self._scope_lock:
            if version is not None:
                self._current_version = str(version)
            if self._pause_started_at is not None:
                self._total_paused_seconds += loop.time() - self._pause_started_at
                self._pause_started_at = None
            was_paused = self._paused
            self._paused = False
            self._pause_reason = None
            self._resume_event.set()
            self._notify_pause_state_changed_locked()
        result.setdefault("status", "resumed" if was_paused else "already_running")
        result.setdefault("reason", reason)
        result.setdefault("version", self._current_version)
        result["http_inflight_requests"] = self._http_inflight_count()
        return result

    async def clear_turn(self) -> None:
        async with self._scope_lock:
            self._turn_scope = None

    async def turn_profile(self) -> dict[str, Any]:
        async with self._scope_lock:
            scope = self._turn_scope
            if scope is None:
                return {}
            records = [dict(record) for record in scope.step_records]

            profile: dict[str, Any] = {
                "proxy.llm_request_count": float(scope.llm_request_count),
                "proxy.llm_plain_http_s": float(scope.llm_plain_http_s),
                "proxy.llm_stream_http_s": float(scope.llm_stream_http_s),
                "proxy.llm_http_s": float(scope.llm_plain_http_s + scope.llm_stream_http_s),
                "proxy.llm_request_bytes": float(scope.llm_request_bytes),
                "proxy.llm_response_bytes": float(scope.llm_response_bytes),
                "proxy.llm_error_count": float(scope.llm_error_count),
            }
            durations = [
                float(record["http_s"])
                for record in records
                if isinstance(record.get("http_s"), (int, float))
            ]
            if durations:
                durations_sorted = sorted(durations)
                profile.update(
                    {
                        "proxy.llm_step_mean_s": sum(durations) / len(durations),
                        "proxy.llm_step_p50_s": _percentile(durations_sorted, 0.50),
                        "proxy.llm_step_p95_s": _percentile(durations_sorted, 0.95),
                        "proxy.llm_step_max_s": durations_sorted[-1],
                    }
                )
            profile.update(_sum_step_fields(records))
            if records:
                profile["proxy.step_timeline_json"] = json.dumps(
                    records[:128],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            return profile

    async def consume_context_overflow_error(self) -> dict[str, Any] | None:
        async with self._scope_lock:
            scope = self._turn_scope
            if scope is None or scope.context_overflow_error is None:
                return None
            payload = dict(scope.context_overflow_error)
            scope.context_overflow_error = None
            return payload

    async def consume_rollout_invalidated_error(self) -> dict[str, Any] | None:
        async with self._scope_lock:
            scope = self._turn_scope
            if scope is None or scope.rollout_invalidated_error is None:
                return None
            payload = dict(scope.rollout_invalidated_error)
            scope.rollout_invalidated_error = None
            return payload

    async def wait_for_max_steps_error(
        self,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        async with self._scope_lock:
            scope = self._turn_scope
            if scope is None:
                return None
            if scope.max_steps_error is not None:
                return dict(scope.max_steps_error)
            event = scope.max_steps_exceeded

        try:
            if timeout is None:
                await event.wait()
            else:
                await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

        async with self._scope_lock:
            if self._turn_scope is not scope or scope.max_steps_error is None:
                return None
            return dict(scope.max_steps_error)

    async def consume_max_steps_error(self) -> dict[str, Any] | None:
        async with self._scope_lock:
            scope = self._turn_scope
            if scope is None or scope.max_steps_error is None:
                return None
            payload = dict(scope.max_steps_error)
            scope.max_steps_error = None
            scope.max_steps_exceeded.clear()
            return payload

    def _http_inflight_count(self) -> int:
        scope = self._turn_scope
        return 0 if scope is None else scope.inflight_requests

    def _notify_pause_state_changed_locked(self) -> None:
        event = self._pause_state_changed
        self._pause_state_changed = asyncio.Event()
        event.set()

    async def _wait_event_excluding_pause(
        self,
        event: asyncio.Event,
        *,
        timeout: float | None,
    ) -> None:
        if timeout is None:
            await event.wait()
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not event.is_set():
            async with self._scope_lock:
                paused = self._paused
                resume_event = self._resume_event
                state_changed = self._pause_state_changed
            if paused:
                pause_started = loop.time()
                event_task = asyncio.create_task(event.wait())
                resume_task = asyncio.create_task(resume_event.wait())
                try:
                    done, pending = await asyncio.wait(
                        {event_task, resume_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    if event_task in done and event_task.result():
                        return
                finally:
                    for task in (event_task, resume_task):
                        if not task.done():
                            task.cancel()
                deadline += loop.time() - pause_started
                continue

            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            event_task = asyncio.create_task(event.wait())
            state_task = asyncio.create_task(state_changed.wait())
            try:
                done, pending = await asyncio.wait(
                    {event_task, state_task},
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                if event_task in done and event_task.result():
                    return
                if not done:
                    raise asyncio.TimeoutError
            finally:
                for task in (event_task, state_task):
                    if not task.done():
                        task.cancel()

    def _control_url(self, endpoint: str) -> str:
        endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        if self.router_api_path == "/":
            base = self.upstream_origin
        else:
            base = f"{self.upstream_origin}{self.router_api_path}".rstrip("/")
        return f"{base}{endpoint}"

    async def _control_post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "X-Session-Id": self.bound_session_id,
            "X-Instance-Id": self.bound_instance_id,
        }
        if self._current_version is not None:
            headers["X-Dressage-Expected-Version"] = str(self._current_version)
        url = self._control_url(endpoint)
        client = self._client
        if client is not None:
            response = await client.post(url, json=payload, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=httpx.Timeout(None), trust_env=False) as temp_client:
                response = await temp_client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json() if response.content else {}
        return data if isinstance(data, dict) else {"data": data}

    def _build_app(self) -> FastAPI:
        @asynccontextmanager
        async def _lifespan(_: FastAPI):
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(None),
                limits=httpx.Limits(max_connections=100),
            )
            try:
                yield
            finally:
                if self._client is not None:
                    await self._client.aclose()
                    self._client = None

        app = FastAPI(lifespan=_lifespan)

        @app.get("/__proxy_health")
        async def _health() -> dict[str, object]:
            return {"ok": True, _HEALTH_TOKEN_FIELD: self._health_token}

        @app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
        )
        async def _proxy(request: Request, path: str) -> Response:
            return await self._handle_proxy(request, path)

        return app

    def _is_chat_completion(self, method: str, path: str) -> bool:
        return method.upper() == "POST" and f"/{path}".rstrip("/").endswith("/chat/completions")

    async def _handle_proxy(self, request: Request, path: str) -> Response:
        is_chat = self._is_chat_completion(request.method, path)
        upstream_url = self._join_upstream(path, request.url.query)

        body_bytes = await request.body()
        body_json: dict[str, Any] | None = None
        is_streaming = False
        original_stream = False
        original_chat_request: dict[str, Any] | None = None
        parsed_body: Any = None

        if body_bytes:
            try:
                parsed_body = json.loads(body_bytes)
            except json.JSONDecodeError:
                pass

        if isinstance(parsed_body, dict):
            if is_chat:
                original_chat_request = dict(parsed_body)
                body_json = dict(parsed_body)
                original_stream = bool(original_chat_request.get("stream", False))
                is_streaming = original_stream
            else:
                body_json = parsed_body
        elif is_chat and parsed_body is not None:
            LOGGER.warning(
                "[PROXY REQUEST] Chat completion payload is %s, forwarding without proxy mutation",
                type(parsed_body).__name__,
            )

        LOGGER.info(
            "[PROXY REQUEST] %s /%s -> %s (is_chat=%s, stream=%s)",
            request.method,
            path,
            upstream_url,
            is_chat,
            original_stream,
        )
        LOGGER.info("[PROXY REQUEST] Body content: %s", self._preview_bytes(body_bytes, limit=1000))
        LOGGER.info("[PROXY REQUEST] Body size: %d bytes", len(body_bytes) if body_bytes else 0)
        LOGGER.info("[PROXY REQUEST] Path: /%s, Upstream: %s", path, upstream_url)

        snapshot = await self._enter_chat_request() if is_chat else None
        turn_id = snapshot.turn_id if snapshot else None
        if snapshot is not None and snapshot.max_steps_exceeded:
            await self._record_max_steps_error(snapshot)
            return self._max_steps_exceeded_response(snapshot)

        if is_chat and body_json is not None:
            tools = body_json.get("tools")
            LOGGER.info(
                "[PROXY REQUEST] Top-level request keys: %s",
                sorted(body_json.keys()),
            )
            LOGGER.info(
                "[PROXY REQUEST] model=%s, msg_count=%d, original_stream=%s, has_stream_options=%s, tool_count=%d",
                body_json.get("model"),
                len(body_json.get("messages", [])),
                original_stream,
                "stream_options" in body_json,
                len(tools) if isinstance(tools, list) else 0,
            )

            if (
                body_json.get("temperature") is None
                and self.default_temperature is not None
            ):
                body_json["temperature"] = self.default_temperature

            if body_json.get("stream", False):
                stream_options = body_json.get("stream_options")
                if not isinstance(stream_options, dict):
                    stream_options = {}
                else:
                    stream_options = dict(stream_options)
                stream_options["include_usage"] = True
                body_json["stream_options"] = stream_options
            elif "stream_options" in body_json:
                removed_stream_options = body_json.pop("stream_options")
                LOGGER.info(
                    "[PROXY REQUEST] Removed stream_options because upstream stream=false: %s",
                    json.dumps(removed_stream_options, ensure_ascii=False),
                )

            LOGGER.info(
                "[PROXY REQUEST] Final upstream stream=%s, has_stream_options=%s",
                body_json.get("stream"),
                "stream_options" in body_json,
            )

            if body_json != original_chat_request:
                body_bytes = json.dumps(body_json, ensure_ascii=False).encode("utf-8")
                LOGGER.info("[PROXY REQUEST] Final body size: %d bytes", len(body_bytes))
                LOGGER.info(
                    "[PROXY REQUEST] Final body preview: %s",
                    self._preview_bytes(body_bytes, limit=500),
                )

        headers = self._build_upstream_headers(request.headers, is_chat=is_chat, turn_id=turn_id)
        LOGGER.info("[PROXY REQUEST] Forwarding with headers: %s", list(headers.keys()))
        if is_chat:
            LOGGER.info(
                "[PROXY REQUEST] Sticky header %s=%s",
                self.sticky_header_name,
                headers.get(self.sticky_header_name),
            )

        request_profile = (
            _chat_request_profile(body_json)
            if is_chat and isinstance(body_json, dict)
            else {}
        )
        if is_chat and is_streaming:
            LOGGER.info("[PROXY REQUEST] Using streaming proxy")
            return await self._stream_proxy(
                upstream_url,
                body_bytes,
                headers,
                snapshot,
                request_profile=request_profile,
            )
        LOGGER.info("[PROXY REQUEST] Using plain proxy")
        return await self._plain_proxy(
            request.method,
            upstream_url,
            body_bytes,
            headers,
            snapshot,
            request_profile=request_profile,
        )

    def _join_upstream(self, path: str, query: str | None = None) -> str:
        normalized = path.lstrip("/")
        upstream_url = f"{self.upstream_origin}/{normalized}"
        if query:
            return f"{upstream_url}?{query}"
        return upstream_url

    def _build_upstream_headers(self, original_headers: Any, *, is_chat: bool, turn_id: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        reserved_headers = {"host", "content-length", "transfer-encoding"}
        if is_chat:
            reserved_headers.update(
                {
                    self.sticky_header_name.lower(),
                    "x-session-id",
                    "x-instance-id",
                    "x-turn-id",
                    "x-dressage-partial-rollout",
                    "x-dressage-expected-version",
                }
            )
        for key, value in original_headers.items():
            if key.lower() in reserved_headers:
                continue
            headers[key] = value
        if is_chat:
            self._set_header(headers, self.sticky_header_name, self.bound_session_id)
            self._set_header(headers, "X-Session-Id", self.bound_session_id)
            self._set_header(headers, "X-Instance-Id", self.bound_instance_id)
            if turn_id:
                self._set_header(headers, "X-Turn-Id", turn_id)
            self._set_header(headers, "X-Dressage-Partial-Rollout", "1")
            if self._current_version is not None:
                self._set_header(headers, "X-Dressage-Expected-Version", str(self._current_version))
        LOGGER.info("[PROXY REQUEST] upstream request headers: %s", headers)
        return headers

    @staticmethod
    def _set_header(headers: dict[str, str], name: str, value: str) -> None:
        for existing in list(headers):
            if existing.lower() == name.lower():
                del headers[existing]
        headers[name] = value

    async def _capture_snapshot(self) -> _TurnSnapshot:
        async with self._scope_lock:
            scope = self._turn_scope
            if scope is None:
                return _TurnSnapshot(
                    session_id=self.bound_session_id,
                    turn_id=None,
                    backend_session_id=None,
                    step=0,
                    scope=None,
                )
            step = scope.step_counter
            scope.step_counter += 1
            return _TurnSnapshot(
                session_id=self.bound_session_id,
                turn_id=scope.turn_id,
                backend_session_id=scope.backend_session_id,
                step=step,
                scope=scope,
            )

    async def _enter_chat_request(self) -> _TurnSnapshot:
        """Wait for rollout resume and atomically mark a chat request active."""

        while True:
            async with self._scope_lock:
                if not self._paused:
                    scope = self._turn_scope
                    if scope is None:
                        return _TurnSnapshot(
                            session_id=self.bound_session_id,
                            turn_id=None,
                            backend_session_id=None,
                            step=0,
                            scope=None,
                        )
                    if self.max_steps is not None and scope.step_counter >= self.max_steps:
                        return _TurnSnapshot(
                            session_id=self.bound_session_id,
                            turn_id=scope.turn_id,
                            backend_session_id=scope.backend_session_id,
                            step=scope.step_counter,
                            scope=scope,
                            max_steps_exceeded=True,
                        )
                    step = scope.step_counter
                    scope.step_counter += 1
                    scope.inflight_requests += 1
                    scope.drained.clear()
                    return _TurnSnapshot(
                        session_id=self.bound_session_id,
                        turn_id=scope.turn_id,
                        backend_session_id=scope.backend_session_id,
                        step=step,
                        scope=scope,
                    )
                resume_event = self._resume_event
            await resume_event.wait()

    def _max_steps_exceeded_response(self, snapshot: _TurnSnapshot) -> Response:
        payload = {
            "error": {
                "message": "Turn exceeded max_steps.",
                "type": "rate_limit_error",
                "code": "max_steps_exceeded",
                "details": {
                    "max_steps": self.max_steps,
                    "attempted_step": snapshot.step,
                },
            }
        }
        return Response(
            content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            status_code=429,
            media_type="application/json",
        )

    async def _record_max_steps_error(self, snapshot: _TurnSnapshot) -> None:
        scope = snapshot.scope
        if scope is None:
            return
        payload = self._max_steps_error_payload(snapshot)
        async with self._scope_lock:
            scope.max_steps_error = payload
            scope.max_steps_exceeded.set()

    def _max_steps_error_payload(self, snapshot: _TurnSnapshot) -> dict[str, Any]:
        return {
            "error": "max_steps_exceeded",
            "message": "Turn exceeded max_steps.",
            "details": {
                "session_id": snapshot.session_id,
                "turn_id": snapshot.turn_id,
                "max_steps": self.max_steps,
                "attempted_step": snapshot.step,
                "backend_message": "429 Turn exceeded max_steps.",
                "raw_error_code": "rate_limit_error",
            },
        }

    async def _mark_request_started(self, scope: _TurnScope) -> None:
        async with self._scope_lock:
            scope.inflight_requests += 1
            scope.drained.clear()

    async def _mark_request_finished(self, scope: _TurnScope) -> None:
        async with self._scope_lock:
            if scope.inflight_requests > 0:
                scope.inflight_requests -= 1
            if scope.inflight_requests == 0:
                scope.drained.set()

    async def _record_llm_request(
        self,
        scope: _TurnScope,
        *,
        step: int | None = None,
        seconds: float,
        request_bytes: int,
        response_bytes: int = 0,
        status_code: int | None = None,
        streaming: bool,
        request_profile: dict[str, Any] | None = None,
        response_profile: dict[str, Any] | None = None,
    ) -> None:
        async with self._scope_lock:
            scope.llm_request_count += 1
            scope.llm_request_bytes += max(0, int(request_bytes))
            scope.llm_response_bytes += max(0, int(response_bytes))
            if streaming:
                scope.llm_stream_http_s += max(0.0, float(seconds))
            else:
                scope.llm_plain_http_s += max(0.0, float(seconds))
            if step is not None:
                record: dict[str, Any] = {
                    "step": int(step),
                    "http_s": max(0.0, float(seconds)),
                    "stream": bool(streaming),
                    "status_code": None if status_code is None else int(status_code),
                    "request_bytes": max(0, int(request_bytes)),
                    "response_bytes": max(0, int(response_bytes)),
                }
                for source in (request_profile or {}, response_profile or {}):
                    for key, value in source.items():
                        if isinstance(value, (int, float, str, bool)) or value is None:
                            record[key] = value
                scope.step_records.append(record)

    async def _record_llm_error(self, scope: _TurnScope) -> None:
        async with self._scope_lock:
            scope.llm_error_count += 1

    def _profile_llm_request(
        self,
        snapshot: _TurnSnapshot | None,
        *,
        request_bytes: int,
        streaming: bool,
        request_profile: dict[str, Any] | None = None,
    ) -> _LLMRequestContext:
        return _LLMRequestContext(
            proxy=self,
            snapshot=snapshot,
            request_bytes=request_bytes,
            streaming=streaming,
            request_profile=request_profile,
        )

    async def _plain_proxy(
        self,
        method: str,
        url: str,
        body: bytes,
        headers: dict[str, str],
        snapshot: _TurnSnapshot | None,
        *,
        request_profile: dict[str, Any] | None = None,
    ) -> Response:
        assert self._client is not None
        async with self._profile_llm_request(
            snapshot,
            request_bytes=len(body),
            streaming=False,
            request_profile=request_profile,
        ) as profile:
            response = await self._send_plain_request(method, url, body, headers)
            profile.status_code = response.status_code
            profile.response_bytes = len(response.content)
            profile.response_profile = _chat_response_profile(response.content)
            if response.status_code >= 400:
                profile.is_error = True
                self._log_upstream_error(
                    url=url,
                    status_code=response.status_code,
                    response_headers=dict(response.headers),
                    response_body=response.content,
                    retried=False,
                )
                if snapshot is not None and snapshot.scope is not None:
                    await self._record_context_overflow_error(
                        snapshot.scope,
                        status_code=response.status_code,
                        response_body=response.content,
                    )
                    if await self._record_rollout_invalidated_error(
                        snapshot.scope,
                        status_code=response.status_code,
                        response_body=response.content,
                    ):
                        synthetic = self._synthetic_chat_completion_response()
                        profile.status_code = synthetic.status_code
                        profile.response_bytes = len(synthetic.body or b"")
                        profile.response_profile = _chat_response_profile(synthetic.body or b"")
                        return synthetic
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("transfer-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

    async def _stream_proxy(
        self,
        url: str,
        body: bytes,
        headers: dict[str, str],
        snapshot: _TurnSnapshot | None,
        *,
        request_profile: dict[str, Any] | None = None,
    ) -> Response:
        assert self._client is not None
        profile = self._profile_llm_request(
            snapshot,
            request_bytes=len(body),
            streaming=True,
            request_profile=request_profile,
        )
        await profile.__aenter__()
        try:
            upstream_response = await self._send_stream_request(url, body, headers)
        except Exception as exc:
            await profile.__aexit__(type(exc), exc, exc.__traceback__)
            raise

        if upstream_response.status_code >= 400:
            profile.is_error = True
            profile.status_code = upstream_response.status_code
            error_body = await upstream_response.aread()
            profile.response_bytes = len(error_body)
            profile.response_profile = _chat_response_profile(error_body)
            response_headers = dict(upstream_response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("transfer-encoding", None)
            response_headers.pop("content-length", None)
            self._log_upstream_error(
                url=url,
                status_code=upstream_response.status_code,
                response_headers=dict(upstream_response.headers),
                response_body=error_body,
                retried=False,
            )
            if snapshot is not None and snapshot.scope is not None:
                await self._record_context_overflow_error(
                    snapshot.scope,
                    status_code=upstream_response.status_code,
                    response_body=error_body,
                )
                if await self._record_rollout_invalidated_error(
                    snapshot.scope,
                    status_code=upstream_response.status_code,
                    response_body=error_body,
                ):
                    await upstream_response.aclose()
                    synthetic = self._synthetic_chat_completion_stream_response()
                    profile.status_code = synthetic.status_code
                    profile.response_bytes = 0
                    profile.response_profile = {}
                    await profile.close()
                    return synthetic
            await upstream_response.aclose()
            await profile.close()
            return Response(
                content=error_body,
                status_code=upstream_response.status_code,
                headers=response_headers,
            )

        response_bytes = 0
        stream_buffer = ""
        stream_response_profile: dict[str, Any] = {}

        async def _forward():
            nonlocal response_bytes, stream_buffer, stream_response_profile
            try:
                async for chunk in upstream_response.aiter_bytes():
                    response_bytes += len(chunk)
                    stream_buffer, parsed_profile = _consume_sse_profile(
                        stream_buffer,
                        chunk,
                    )
                    stream_response_profile.update(parsed_profile)
                    yield chunk
            finally:
                await upstream_response.aclose()
                profile.status_code = upstream_response.status_code
                profile.response_bytes = response_bytes
                profile.response_profile = stream_response_profile
                await profile.close()

        response_headers = dict(upstream_response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("transfer-encoding", None)
        response_headers.pop("content-length", None)
        return StreamingResponse(
            _forward(),
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=upstream_response.headers.get("content-type", "text/event-stream"),
        )

    async def _send_plain_request(
        self,
        method: str,
        url: str,
        body: bytes,
        headers: dict[str, str],
    ) -> httpx.Response:
        assert self._client is not None
        request_headers = dict(headers)
        request_headers["content-length"] = str(len(body))
        LOGGER.info("[PROXY REQUEST] Setting content-length: %d", len(body))
        return await self._client.request(method=method, url=url, content=body, headers=request_headers)

    async def _send_stream_request(
        self,
        url: str,
        body: bytes,
        headers: dict[str, str],
    ) -> httpx.Response:
        assert self._client is not None
        request_headers = dict(headers)
        request_headers["content-length"] = str(len(body))
        LOGGER.info("[PROXY REQUEST] Setting content-length: %d", len(body))
        request = self._client.build_request("POST", url, content=body, headers=request_headers)
        return await self._client.send(request, stream=True)

    def _log_upstream_error(
        self,
        *,
        url: str,
        status_code: int,
        response_headers: dict[str, str],
        response_body: bytes,
        retried: bool,
    ) -> None:
        LOGGER.warning(
            "[PROXY REQUEST] Upstream returned %d for %s (content_type=%s, retried=%s, body=%s)",
            status_code,
            url,
            response_headers.get("content-type"),
            retried,
            self._preview_bytes(response_body, limit=2000),
        )

    async def _record_context_overflow_error(
        self,
        scope: _TurnScope,
        *,
        status_code: int,
        response_body: bytes,
    ) -> None:
        payload = self._context_overflow_payload_from_response(
            status_code=status_code,
            response_body=response_body,
        )
        if payload is None:
            return
        async with self._scope_lock:
            scope.context_overflow_error = payload

    async def _record_rollout_invalidated_error(
        self,
        scope: _TurnScope,
        *,
        status_code: int,
        response_body: bytes,
    ) -> bool:
        payload = self._rollout_invalidated_payload_from_response(
            status_code=status_code,
            response_body=response_body,
        )
        if payload is None:
            return False
        async with self._scope_lock:
            scope.rollout_invalidated_error = payload
        return True

    @staticmethod
    def _context_overflow_payload_from_response(
        *,
        status_code: int,
        response_body: bytes,
    ) -> dict[str, Any] | None:
        if status_code != 413 or not response_body:
            return None
        try:
            payload = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or payload.get("error") != "context_overflow":
            return None
        return payload

    @staticmethod
    def _rollout_invalidated_payload_from_response(
        *,
        status_code: int,
        response_body: bytes,
    ) -> dict[str, Any] | None:
        if status_code != 502 or not response_body:
            return None
        try:
            payload = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        detail = payload.get("detail")
        if isinstance(detail, dict):
            candidate = detail
        else:
            candidate = payload
        if candidate.get("error") not in DRESSAGE_ROLLOUT_INVALIDATED_ERRORS:
            return None
        return {str(key): value for key, value in candidate.items()}

    @staticmethod
    def _synthetic_chat_completion_response() -> Response:
        payload = {
            "id": "chatcmpl-rollout-invalidated",
            "object": "chat.completion",
            "created": 0,
            "model": "proxy-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        return Response(
            content=json.dumps(payload).encode("utf-8"),
            status_code=200,
            media_type="application/json",
        )

    @staticmethod
    def _synthetic_chat_completion_stream_response() -> StreamingResponse:
        payload = {
            "id": "chatcmpl-rollout-invalidated",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "proxy-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
        }

        async def _events():
            yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            _events(),
            status_code=200,
            media_type="text/event-stream",
        )

    @staticmethod
    def _preview_bytes(body: bytes, *, limit: int) -> str:
        if not body:
            return ""
        text = body.decode("utf-8", errors="replace")
        if len(text) <= limit:
            return text
        return text[:limit] + "...(truncated)"


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = min(
        len(sorted_values) - 1,
        max(0, int(round((len(sorted_values) - 1) * q))),
    )
    return sorted_values[index]


def _sum_step_fields(records: list[dict[str, Any]]) -> dict[str, float]:
    sums: dict[str, float] = {
        "proxy.llm_prompt_tokens": 0.0,
        "proxy.llm_completion_tokens": 0.0,
        "proxy.llm_total_tokens": 0.0,
        "proxy.llm_tool_call_count": 0.0,
        "proxy.llm_tool_call_steps": 0.0,
        "proxy.llm_length_finish_count": 0.0,
    }
    for record in records:
        sums["proxy.llm_prompt_tokens"] += _float_field(record, "prompt_tokens")
        sums["proxy.llm_completion_tokens"] += _float_field(record, "completion_tokens")
        sums["proxy.llm_total_tokens"] += _float_field(record, "total_tokens")
        tool_calls = _float_field(record, "tool_call_count")
        sums["proxy.llm_tool_call_count"] += tool_calls
        if tool_calls > 0:
            sums["proxy.llm_tool_call_steps"] += 1.0
        if str(record.get("finish_reason") or "").lower() == "length":
            sums["proxy.llm_length_finish_count"] += 1.0
        for key, value in record.items():
            if not key.startswith(("chat_", "sglang_")):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            sums[f"proxy.llm_{key}"] = sums.get(f"proxy.llm_{key}", 0.0) + float(value)
    return sums


def _float_field(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _chat_request_profile(body_json: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(body_json, dict):
        return {}
    messages = body_json.get("messages")
    tools = body_json.get("tools")
    profile: dict[str, Any] = {
        "model": str(body_json.get("model") or ""),
        "request_message_count": float(len(messages) if isinstance(messages, list) else 0),
        "request_tool_schema_count": float(len(tools) if isinstance(tools, list) else 0),
        "request_stream": bool(body_json.get("stream", False)),
    }
    sampling = body_json.get("sampling_params")
    for source in (body_json, sampling if isinstance(sampling, dict) else {}):
        for source_key, target_key in (
            ("max_tokens", "request_max_tokens"),
            ("max_new_tokens", "request_max_tokens"),
            ("temperature", "request_temperature"),
            ("top_p", "request_top_p"),
        ):
            value = source.get(source_key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                profile[target_key] = float(value)
    return profile


def _chat_response_profile(content: bytes) -> dict[str, Any]:
    if not content:
        return {}
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    profile: dict[str, Any] = {}
    usage = payload.get("usage")
    if isinstance(usage, dict):
        profile.update(_chat_usage_profile(usage))
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            finish_reason = first_choice.get("finish_reason")
            if finish_reason is not None:
                profile["finish_reason"] = str(finish_reason)
            message = first_choice.get("message")
            if isinstance(message, dict):
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    profile["tool_call_count"] = float(len(tool_calls))
                content_value = message.get("content")
                if isinstance(content_value, str):
                    profile["content_chars"] = float(len(content_value))
    return profile


def _chat_usage_profile(usage: dict[str, Any]) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    if isinstance(usage, dict):
        for source_key, target_key in (
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = usage.get(source_key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                profile[target_key] = float(value)
        nested_profile = usage.get("profile")
        if isinstance(nested_profile, dict):
            for key, value in nested_profile.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    profile[str(key).replace(".", "_")] = float(value)
    return profile


def _consume_sse_profile(buffer: str, chunk: bytes) -> tuple[str, dict[str, Any]]:
    try:
        buffer += chunk.decode("utf-8")
    except UnicodeDecodeError:
        buffer += chunk.decode("utf-8", errors="ignore")
    response_profile: dict[str, Any] = {}
    while "\n\n" in buffer:
        event, buffer = buffer.split("\n\n", 1)
        for line in event.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            usage = payload.get("usage")
            if isinstance(usage, dict):
                response_profile.update(_chat_usage_profile(usage))
            choices = payload.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    finish_reason = first_choice.get("finish_reason")
                    if finish_reason is not None:
                        response_profile["finish_reason"] = str(finish_reason)
    return buffer, response_profile
