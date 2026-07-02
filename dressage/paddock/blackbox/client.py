"""HTTP client for the Dressage blackbox server protocol."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from dressage.paddock.blackbox.common.command import build_execute_cmd_payload
from dressage.paddock.blackbox.common.http_retry import post_json_with_retry
from dressage.paddock.blackbox.common.utils import _env_float, _env_int
from dressage.sandbox.types import SandboxEndpoint

logger = logging.getLogger(__name__)


class BlackboxServerClient:
    """Client for the blackbox server HTTP API.

    This class is intentionally provider-agnostic.  It only needs a
    ``SandboxEndpoint`` and does not know whether the endpoint came from E2B or
    a local Ray/bubblewrap lease.
    """

    def __init__(self, *, client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = client is None
        limits = httpx.Limits(
            max_connections=_env_int(
                "DRESSAGE_BLACKBOX_HTTP_MAX_CONNECTIONS", 1024, min_value=1
            ),
            max_keepalive_connections=_env_int(
                "DRESSAGE_BLACKBOX_HTTP_MAX_KEEPALIVE_CONNECTIONS", 256, min_value=0
            ),
        )
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(None), trust_env=False, limits=limits
        )

    async def health(self, endpoint: SandboxEndpoint) -> dict[str, Any]:
        response = await self._client.get(
            f"{endpoint.url.rstrip('/')}/health",
            headers=endpoint.headers,
        )
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {"ok": True, "text": response.text}

    async def register_agent(
        self,
        endpoint: SandboxEndpoint,
        *,
        trajectory_id: str,
        instance_id: str,
        session_id: str,
        router_url: str,
        blackbox_type: str,
        backend_options: Any,
        server_config: dict[str, Any],
        router_api_path: str = "/v1",
    ) -> dict[str, Any]:
        payload = {
            "blackbox_type": blackbox_type,
            "router": router_url,
            "router_api_path": router_api_path,
            "bound_instance_id": instance_id,
            "bound_session_id": session_id,
            "backend_options": backend_options,
            "server_config": server_config,
        }
        response = await self._post_agent_with_retry(
            endpoint,
            "/v1/rollout/register",
            json=payload,
            operation="register_agent",
            trajectory_id=trajectory_id,
        )
        return response.json()

    async def call_agent(
        self,
        endpoint: SandboxEndpoint,
        *,
        trajectory_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self._post_agent_with_retry(
            endpoint,
            f"/v1/sessions/{session_id}/messages",
            json={"messages": messages, "metadata": metadata or {}},
            operation="call_agent",
            trajectory_id=trajectory_id,
        )
        return response.json()

    async def execute_cmd(
        self,
        endpoint: SandboxEndpoint,
        *,
        session_id: str,
        cmd: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        payload = build_execute_cmd_payload(cmd=cmd, timeout=timeout)
        response = await self._client.post(
            f"{endpoint.url.rstrip('/')}/v1/sessions/{session_id}/execute_cmd",
            json=payload,
            headers=endpoint.headers,
        )
        response.raise_for_status()
        return response.json()

    async def abort_session(
        self,
        endpoint: SandboxEndpoint,
        *,
        session_id: str,
        timeout: float | None = 10.0,
    ) -> dict[str, Any]:
        """Abort an in-flight blackbox session so it reaches the ABORTED terminal state.

        Needed when a rollout drops/cancels a trajectory mid-turn: the engine-side
        ``abort_all`` only stops the LLM, leaving the *stateful* env session ACTIVE.
        ``has_open_sessions()`` counts ACTIVE as occupied, so the next rollout step's
        rebind is rejected with 409 ("Cannot rebind while active or desynced sessions
        still exist"). The server forces the session to ABORTED within a bounded time.
        """
        response = await self._client.post(
            f"{endpoint.url.rstrip('/')}/v1/sessions/{session_id}/abort",
            headers=endpoint.headers,
            timeout=timeout,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return {
                    "request_id": "",
                    "session_id": session_id,
                    "action": "abort",
                    "state": "aborted",
                    "mode": "missing",
                }
            raise
        return response.json()

    async def pause(
        self,
        endpoint: SandboxEndpoint,
        *,
        reason: str = "weight_update",
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"reason": reason}
        if timeout_seconds is not None:
            payload["timeout_seconds"] = timeout_seconds
        response = await self._client.post(
            f"{endpoint.url.rstrip('/')}/v1/rollout/pause",
            json=payload,
            headers=endpoint.headers,
        )
        response.raise_for_status()
        return response.json()

    async def resume(
        self,
        endpoint: SandboxEndpoint,
        *,
        version: str | None = None,
        reason: str = "weight_update",
    ) -> dict[str, Any]:
        response = await self._client.post(
            f"{endpoint.url.rstrip('/')}/v1/rollout/resume",
            json={"reason": reason, "version": version},
            headers=endpoint.headers,
        )
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post_agent_with_retry(
        self,
        endpoint: SandboxEndpoint,
        path: str,
        *,
        json: dict[str, Any],
        operation: str,
        trajectory_id: str,
    ) -> httpx.Response:
        max_attempts = _env_int(
            "DRESSAGE_BLACKBOX_AGENT_REQUEST_MAX_ATTEMPTS",
            _env_int(
                "DRESSAGE_SANDBOX_AGENT_REQUEST_MAX_ATTEMPTS",
                6,
                min_value=1,
            ),
            min_value=1,
        )
        initial_delay = _env_float(
            "DRESSAGE_BLACKBOX_AGENT_REQUEST_INITIAL_DELAY_SEC",
            _env_float(
                "DRESSAGE_SANDBOX_AGENT_REQUEST_INITIAL_DELAY_SEC",
                1.0,
                min_value=0.0,
            ),
            min_value=0.0,
        )
        max_delay = _env_float(
            "DRESSAGE_BLACKBOX_AGENT_REQUEST_MAX_DELAY_SEC",
            _env_float(
                "DRESSAGE_SANDBOX_AGENT_REQUEST_MAX_DELAY_SEC",
                10.0,
                min_value=0.0,
            ),
            min_value=0.0,
        )
        jitter_fraction = _env_float(
            "DRESSAGE_BLACKBOX_AGENT_REQUEST_JITTER_FRACTION",
            _env_float(
                "DRESSAGE_SANDBOX_AGENT_REQUEST_JITTER_FRACTION",
                0.2,
                min_value=0.0,
            ),
            min_value=0.0,
        )
        return await post_json_with_retry(
            self._client,
            f"{endpoint.url.rstrip('/')}{path}",
            json=json,
            operation=operation,
            trajectory_id=trajectory_id,
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            jitter_fraction=jitter_fraction,
            log_prefix="blackbox server",
            logger=logger,
            headers=endpoint.headers,
        )
