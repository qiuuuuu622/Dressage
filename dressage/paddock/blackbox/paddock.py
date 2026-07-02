"""Paddock implementation for blackbox agent rollouts."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import time
from typing import Any

import httpx

from dressage.paddock.blackbox.client import BlackboxServerClient
from dressage.paddock.blackbox.common.defaults import (
    DEFAULT_BLACKBOX_TYPE,
    merge_backend_options,
    normalize_blackbox_type,
    server_config_for,
)
from dressage.paddock.blackbox.common.state import SandboxState
from dressage.paddock.blackbox.common.utils import (
    _env_float,
    _env_int,
    _exception_summary,
    _jittered_delay,
    _require_public_proxy_url,
    _validate_public_proxy_url,
)
from dressage.paddock.interface import BlackboxPaddock
from dressage.sandbox import SandboxEndpoint, SandboxLease, SandboxServiceSpec, SandboxSpec
from dressage.sandbox.factory import create_sandbox_provider_from_env
from dressage.sandbox.provider import SandboxProvider

logger = logging.getLogger(__name__)
_REGISTER_SEMAPHORES: dict[int, asyncio.Semaphore] = {}
_REGISTER_RECYCLE_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
)


class BlackboxAgentPaddock(BlackboxPaddock):
    """Run blackbox agents through a provider-created sandbox service."""

    def __init__(
        self,
        *,
        provider: SandboxProvider | None = None,
        blackbox_client: BlackboxServerClient | None = None,
        proxy_public_url: str | None = None,
        blackbox_port: int | None = None,
        wait_health: bool | None = None,
    ) -> None:
        self._provider = provider or create_sandbox_provider_from_env()
        self._client = blackbox_client or BlackboxServerClient()
        self._proxy_public_url = _require_public_proxy_url(proxy_public_url)
        self._blackbox_port = int(
            blackbox_port
            or os.environ.get("DRESSAGE_BLACKBOX_PORT")
            or _provider_blackbox_port_env(getattr(self._provider, "name", ""))
            or "31000"
        )
        if wait_health is None:
            wait_health = os.environ.get("DRESSAGE_BLACKBOX_SKIP_HEALTHCHECK") not in {
                "1",
                "true",
                "TRUE",
                "yes",
            }
        self._wait_health = wait_health
        self._leases: dict[str, SandboxLease] = {}
        self._specs: dict[str, SandboxSpec] = {}
        self._states: dict[str, SandboxState] = {}
        self._cleanup_tasks: set[asyncio.Task[dict[str, Any]]] = set()

    async def init(
        self,
        traj_id: str,
        env_type: str | None = None,
        env_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> SandboxState:
        env_args = {**(env_args or {}), **kwargs}
        spec = SandboxSpec(
            trajectory_id=traj_id,
            env_type=env_type,
            env_args=env_args,
            services=(
                SandboxServiceSpec(
                    name="blackbox",
                    port=int(env_args.get("blackbox_port") or self._blackbox_port),
                    health_path="/health",
                ),
            ),
            timeout_sec=env_args.get("sandbox_timeout_sec"),
            metadata={"paddock_mode": "blackbox"},
        )
        return await self._create_state_from_spec(spec)

    async def _create_state_from_spec(self, spec: SandboxSpec) -> SandboxState:
        lease = await self._provider.create(spec)
        endpoint = lease.endpoints.get("blackbox")
        if endpoint is None:
            endpoint = await self._provider.get_public_url(
                lease,
                port=spec.services[0].port,
                service_name="blackbox",
            )
            lease.endpoints["blackbox"] = endpoint
        endpoint = endpoint.normalized()
        if self._wait_health:
            await self._client.health(endpoint)
        self._leases[spec.trajectory_id] = lease
        self._specs[spec.trajectory_id] = spec
        state = SandboxState(
            trajectory_id=spec.trajectory_id,
            sandbox_url=endpoint.url,
            sandbox_id=lease.sandbox_id,
            raw_register_response={
                "provider": lease.provider,
                "sandbox_id": lease.sandbox_id,
                "metadata": lease.metadata,
                "endpoints": {
                    name: endpoint.url for name, endpoint in lease.endpoints.items()
                },
            },
        )
        self._states[spec.trajectory_id] = state
        return state

    async def register_agent(
        self,
        state: SandboxState | str,
        *,
        instance_id: str,
        session_id: str,
        router_url: str | None = None,
        blackbox_type: str = DEFAULT_BLACKBOX_TYPE,
        backend_options: Any = None,
        router_api_path: str = "/v1",
    ) -> dict[str, Any]:
        state = self._resolve_state(state)
        router = _validate_public_proxy_url(router_url or self._proxy_public_url)
        blackbox_type = normalize_blackbox_type(blackbox_type)
        merged_backend_options = merge_backend_options(blackbox_type, backend_options)
        server_config = _server_config_for_provider(self._provider.name, blackbox_type)
        recycle_attempts = _env_int(
            "DRESSAGE_BLACKBOX_REGISTER_RECYCLE_ATTEMPTS",
            1,
            min_value=0,
        )
        async with _register_limit():
            for attempt in range(recycle_attempts + 1):
                try:
                    result = await self._register_agent_with_transport_retries(
                        state,
                        instance_id=instance_id,
                        session_id=session_id,
                        router_url=router,
                        blackbox_type=blackbox_type,
                        backend_options=merged_backend_options,
                        server_config=server_config,
                        router_api_path=router_api_path,
                        recycle_attempt=attempt,
                    )
                    _set_lease_profile_value(
                        state,
                        "profile.register.recycle_attempts",
                        float(attempt),
                    )
                    return result
                except httpx.HTTPStatusError as exc:
                    if not _should_recycle_register_error(exc) or attempt >= recycle_attempts:
                        raise
                    context = self._register_log_context(state)
                    logger.warning(
                        "register_agent failed with retryable status=%s for trajectory_id=%s; "
                        "recycling sandbox slot (%d/%d) endpoint=%s sandbox_id=%s "
                        "slot_id=%s generation=%s",
                        exc.response.status_code,
                        state.trajectory_id,
                        attempt + 1,
                        recycle_attempts,
                        context["endpoint"],
                        context["sandbox_id"],
                        context["slot_id"],
                        context["generation"],
                    )
                    state = await self._recycle_state_for_register(state)
                except _REGISTER_RECYCLE_ERRORS as exc:
                    if attempt >= recycle_attempts:
                        raise
                    context = self._register_log_context(state)
                    logger.warning(
                        "register_agent transport failure for trajectory_id=%s; "
                        "recycling sandbox slot (%d/%d) exc_type=%s endpoint=%s "
                        "sandbox_id=%s slot_id=%s generation=%s detail=%s",
                        state.trajectory_id,
                        attempt + 1,
                        recycle_attempts,
                        type(exc).__name__,
                        context["endpoint"],
                        context["sandbox_id"],
                        context["slot_id"],
                        context["generation"],
                        _exception_summary(exc),
                    )
                    state = await self._recycle_state_for_register(state)

        raise RuntimeError("unreachable register_agent recycle loop exit")

    async def _register_agent_with_transport_retries(
        self,
        state: SandboxState,
        *,
        instance_id: str,
        session_id: str,
        router_url: str,
        blackbox_type: str,
        backend_options: Any,
        server_config: dict[str, Any],
        router_api_path: str,
        recycle_attempt: int,
    ) -> dict[str, Any]:
        retries = _env_int(
            "DRESSAGE_BLACKBOX_REGISTER_TRANSPORT_RETRIES",
            2,
            min_value=0,
        )
        delay = _env_float(
            "DRESSAGE_BLACKBOX_REGISTER_RETRY_INITIAL_DELAY_SEC",
            0.1,
            min_value=0.0,
        )
        max_delay = _env_float(
            "DRESSAGE_BLACKBOX_REGISTER_RETRY_MAX_DELAY_SEC",
            0.5,
            min_value=0.0,
        )
        for transport_attempt in range(retries + 1):
            attempt_start = time.perf_counter()
            try:
                return await self._register_agent_once(
                    state,
                    instance_id=instance_id,
                    session_id=session_id,
                    router_url=router_url,
                    blackbox_type=blackbox_type,
                    backend_options=backend_options,
                    server_config=server_config,
                    router_api_path=router_api_path,
                )
            except _REGISTER_RECYCLE_ERRORS as exc:
                if transport_attempt >= retries:
                    raise
                context = self._register_log_context(state)
                logger.warning(
                    "register_agent transport failure for trajectory_id=%s; "
                    "retrying same sandbox slot (%d/%d) exc_type=%s endpoint=%s "
                    "sandbox_id=%s slot_id=%s generation=%s recycle_attempt=%d "
                    "elapsed=%.3fs detail=%s",
                    state.trajectory_id,
                    transport_attempt + 1,
                    retries,
                    type(exc).__name__,
                    context["endpoint"],
                    context["sandbox_id"],
                    context["slot_id"],
                    context["generation"],
                    recycle_attempt,
                    time.perf_counter() - attempt_start,
                    _exception_summary(exc),
                )
                sleep_for = _jittered_delay(min(delay, max_delay), 0.2) if max_delay > 0 else 0.0
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                if max_delay > 0:
                    delay = min(delay * 2 if delay > 0 else max_delay, max_delay)
        raise RuntimeError("unreachable register_agent transport retry loop exit")

    async def _register_agent_once(
        self,
        state: SandboxState,
        *,
        instance_id: str,
        session_id: str,
        router_url: str,
        blackbox_type: str,
        backend_options: Any,
        server_config: dict[str, Any],
        router_api_path: str,
    ) -> dict[str, Any]:
        lease = self._leases.get(state.trajectory_id)
        endpoint = self._endpoint_for_state(state, lease)
        return await self._client.register_agent(
            endpoint,
            trajectory_id=state.trajectory_id,
            instance_id=instance_id,
            session_id=session_id,
            router_url=router_url,
            blackbox_type=blackbox_type,
            backend_options=backend_options,
            server_config=server_config,
            router_api_path=router_api_path,
        )

    def _register_log_context(self, state: SandboxState) -> dict[str, Any]:
        lease = self._leases.get(state.trajectory_id)
        endpoint = self._endpoint_for_state(state, lease)
        metadata = lease.metadata if lease is not None else {}
        return {
            "endpoint": endpoint.url,
            "sandbox_id": state.sandbox_id,
            "slot_id": metadata.get("slot_id"),
            "generation": metadata.get("generation"),
        }

    async def _recycle_state_for_register(self, state: SandboxState) -> SandboxState:
        spec = self._specs.get(state.trajectory_id)
        if spec is None:
            raise KeyError(
                f"sandbox spec not found for trajectory_id={state.trajectory_id}"
            )
        lease = self._leases.get(state.trajectory_id)
        if lease is None:
            await self._provider.terminate(state.trajectory_id)
        else:
            await self._provider.terminate(lease)
        self._leases.pop(state.trajectory_id, None)
        self._states.pop(state.trajectory_id, None)
        new_state = await self._create_state_from_spec(spec)
        state.sandbox_url = new_state.sandbox_url
        state.sandbox_id = new_state.sandbox_id
        state.raw_register_response = new_state.raw_register_response
        self._states[state.trajectory_id] = state
        return state

    async def call_agent(
        self,
        state: SandboxState | str,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._resolve_state(state)
        return await self._client.call_agent(
            self._endpoint_for_state(state, self._leases.get(state.trajectory_id)),
            trajectory_id=state.trajectory_id,
            session_id=session_id,
            messages=messages,
            metadata=metadata,
        )

    async def execute_cmd(
        self,
        state: SandboxState | str,
        *,
        session_id: str,
        cmd: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        state = self._resolve_state(state)
        return await self._client.execute_cmd(
            self._endpoint_for_state(state, self._leases.get(state.trajectory_id)),
            session_id=session_id,
            cmd=cmd,
            timeout=timeout,
        )

    async def pause(
        self,
        traj_id: str | None = None,
        *,
        reason: str = "weight_update",
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        states = self._select_states(traj_id)
        results = {
            tid: await self._client.pause(
                self._endpoint_for_state(state, self._leases.get(tid)),
                reason=reason,
                timeout_seconds=timeout_seconds,
            )
            for tid, state in states.items()
        }
        return {
            "status": "paused",
            "reason": reason,
            "quiesced": all(bool(result.get("quiesced", True)) for result in results.values()),
            "results": results,
        }

    async def resume(
        self,
        traj_id: str | None = None,
        *,
        version: str | None = None,
        reason: str = "weight_update",
    ) -> dict[str, Any]:
        states = self._select_states(traj_id)
        results = {
            tid: await self._client.resume(
                self._endpoint_for_state(state, self._leases.get(tid)),
                version=version,
                reason=reason,
            )
            for tid, state in states.items()
        }
        return {"status": "resumed", "reason": reason, "version": version, "results": results}

    async def terminate(
        self,
        traj_id: str,
        env_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        terminate_env_args = {**(env_args or {}), **kwargs}
        skip_abort = _skip_abort_on_terminate(terminate_env_args)
        terminate_start = time.perf_counter()
        state = self._states.pop(traj_id, None)
        lease = self._leases.pop(traj_id, None)
        self._specs.pop(traj_id, None)
        if lease is None and state is None:
            return {"terminated": False, "trajectory_id": traj_id, "missing": True}

        if not _background_terminate_enabled():
            return await self._terminate_cleanup(
                traj_id,
                state=state,
                lease=lease,
                terminate_start=terminate_start,
                skip_abort=skip_abort,
            )

        task = asyncio.create_task(
            self._terminate_cleanup(
                traj_id,
                state=state,
                lease=lease,
                terminate_start=terminate_start,
                skip_abort=skip_abort,
            )
        )
        self._track_cleanup_task(task)
        lease_id = None if lease is None else lease.sandbox_id
        return {
            "terminated": True,
            "trajectory_id": traj_id,
            "lease_id": lease_id,
            "release_queued": True,
            "cleanup_background": True,
        }

    async def _terminate_cleanup(
        self,
        traj_id: str,
        *,
        state: SandboxState | None,
        lease: SandboxLease | None,
        terminate_start: float,
        skip_abort: bool = False,
    ) -> dict[str, Any]:
        warn_after = _terminate_warn_sec()
        # Abort the env session before the provider can mark the slot reusable. A
        # trajectory dropped mid-turn (drop-tail / engine abort_all kills its LLM
        # request) otherwise lingers as ACTIVE and can make the next rebind fail with
        # 409. This cleanup may run in the background; provider release/reset still
        # controls when the slot becomes ready again.
        if state is not None and not skip_abort:
            abort_start = time.perf_counter()
            try:
                endpoint = self._endpoint_for_state(state, lease)
                abort_response = await self._client.abort_session(
                    endpoint,
                    session_id=traj_id,
                    timeout=_terminate_abort_timeout_sec(),
                )
                logger.debug(
                    "blackbox abort_session completed session_id=%s mode=%s state=%s",
                    traj_id,
                    abort_response.get("mode"),
                    abort_response.get("state"),
                )
            except Exception as exc:  # noqa: BLE001 - never block release on abort failure
                logger.info(
                    "abort_session best-effort failed for session_id=%s exc_type=%s: %s",
                    traj_id,
                    type(exc).__name__,
                    exc,
                )
            finally:
                abort_seconds = time.perf_counter() - abort_start
                if abort_seconds >= warn_after:
                    logger.warning(
                        "slow blackbox abort_session session_id=%s took %.3fs",
                        traj_id,
                        abort_seconds,
                    )
        elif state is not None:
            logger.debug(
                "skip blackbox abort_session for cleanly finalized session_id=%s",
                traj_id,
            )
        provider_start = time.perf_counter()
        try:
            if lease is None:
                assert state is not None
                result = await self._provider.terminate(state.trajectory_id)
            else:
                result = await self._provider.terminate(lease)
            if isinstance(result, dict):
                profile = {
                    key: value
                    for key, value in result.items()
                    if key.startswith("profile.")
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                }
                if profile:
                    logger.debug(
                        "blackbox terminate profile session_id=%s profile=%s",
                        traj_id,
                        profile,
                    )
            return result
        finally:
            provider_seconds = time.perf_counter() - provider_start
            total_seconds = time.perf_counter() - terminate_start
            if provider_seconds >= warn_after or total_seconds >= warn_after:
                logger.warning(
                    "slow blackbox terminate session_id=%s provider=%.3fs total=%.3fs",
                    traj_id,
                    provider_seconds,
                    total_seconds,
                )

    def _track_cleanup_task(self, task: asyncio.Task[dict[str, Any]]) -> None:
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._discard_cleanup_task)

    def _discard_cleanup_task(self, task: asyncio.Task[dict[str, Any]]) -> None:
        self._cleanup_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 - background cleanup must be observable
            logger.warning("blackbox background terminate cleanup failed: %s", exc)

    async def drain_cleanup_tasks(self) -> None:
        while self._cleanup_tasks:
            tasks = tuple(self._cleanup_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        await self.drain_cleanup_tasks()
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()

    def _resolve_state(self, state: SandboxState | str) -> SandboxState:
        if isinstance(state, SandboxState):
            return state
        if state not in self._states:
            raise KeyError(f"sandbox state not found for trajectory_id={state}")
        return self._states[state]

    def _select_states(self, traj_id: str | None) -> dict[str, SandboxState]:
        if traj_id is None:
            return dict(self._states)
        return {traj_id: self._resolve_state(traj_id)}

    def _endpoint_for_state(
        self,
        state: SandboxState,
        lease: SandboxLease | None,
    ) -> SandboxEndpoint:
        if lease is not None and "blackbox" in lease.endpoints:
            return lease.endpoints["blackbox"].normalized()
        return SandboxEndpoint(url=state.sandbox_url, headers={})


def _provider_blackbox_port_env(provider_name: str) -> str | None:
    if provider_name == "e2b":
        return os.environ.get("DRESSAGE_E2B_BLACKBOX_PORT")
    if provider_name == "local_bwrap":
        return os.environ.get("DRESSAGE_LOCAL_BWRAP_BLACKBOX_PORT")
    return None


def _should_recycle_register_error(exc: httpx.HTTPStatusError) -> bool:
    return exc.response.status_code in {502, 503, 504}


def _background_terminate_enabled() -> bool:
    return os.environ.get("DRESSAGE_BLACKBOX_BACKGROUND_TERMINATE", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _skip_abort_on_terminate(env_args: dict[str, Any]) -> bool:
    value = env_args.get("blackbox_skip_abort")
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "on"}


@asynccontextmanager
async def _register_limit():
    concurrency = _env_int("DRESSAGE_BLACKBOX_REGISTER_CONCURRENCY", 0, min_value=0)
    if concurrency <= 0:
        yield
        return
    semaphore = _REGISTER_SEMAPHORES.get(concurrency)
    if semaphore is None:
        semaphore = asyncio.Semaphore(concurrency)
        _REGISTER_SEMAPHORES[concurrency] = semaphore
    async with semaphore:
        yield


def _set_lease_profile_value(state: SandboxState, key: str, value: float) -> None:
    raw = state.raw_register_response
    metadata = raw.get("metadata") if isinstance(raw, dict) else None
    if isinstance(metadata, dict):
        metadata[key] = value


def _server_config_for_provider(provider_name: str, blackbox_type: str) -> dict[str, Any]:
    config = server_config_for(blackbox_type)
    if provider_name == "local_bwrap":
        # Local blackbox servers are launched by the node supervisor, so runtime
        # root comes from BBS_RUNTIME_ROOT in the server process environment.
        config.pop("runtime_root", None)
    return config


def _terminate_warn_sec() -> float:
    raw = os.environ.get("DRESSAGE_BLACKBOX_TERMINATE_WARN_SEC", "1")
    try:
        value = float(raw)
    except ValueError:
        return 1.0
    return max(0.0, value)


def _terminate_abort_timeout_sec() -> float:
    raw = os.environ.get("DRESSAGE_BLACKBOX_TERMINATE_ABORT_TIMEOUT_SEC", "6")
    try:
        value = float(raw)
    except ValueError:
        return 6.0
    return max(0.1, value)
