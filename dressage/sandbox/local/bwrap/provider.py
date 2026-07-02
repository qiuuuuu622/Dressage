"""Local Ray-managed bubblewrap sandbox provider."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from collections.abc import Awaitable
import os
import time
from typing import Any

import httpx

from dressage.config import local_bwrap_manager_name, local_bwrap_namespace
from dressage.profiling import enabled as profiling_enabled
from dressage.sandbox.local.bwrap.supervisor import (
    POOL_BLACKBOX,
    POOL_COMMAND_ONLY,
    PoolMode,
    normalize_pool_mode,
)
from dressage.sandbox.types import CommandResult, SandboxEndpoint, SandboxLease, SandboxSpec

_ACQUIRE_SEMAPHORES: dict[int, asyncio.Semaphore] = {}


class LocalBwrapSandboxProvider:
    """Local provider backed by a single Ray-managed bwrap slot pool.

    One manager is either ``blackbox`` or ``command_only``.  The provider does
    not create a second pool; it validates that the connected pool matches the
    requested paddock mode.
    """

    name = "local_bwrap"

    def __init__(
        self,
        *,
        manager: Any | None = None,
        ray_address: str | None = None,
        namespace: str | None = None,
        manager_name: str | None = None,
    ) -> None:
        backend = os.environ.get("DRESSAGE_LOCAL_BWRAP_BACKEND")
        if backend and backend.strip().lower() != "ray_pool":
            raise ValueError(
                "DRESSAGE_LOCAL_BWRAP_BACKEND no longer selects production "
                "backends; local_bwrap always uses ray_pool"
            )
        self._namespace = namespace or local_bwrap_namespace()
        self._manager_name = manager_name or local_bwrap_manager_name()
        self._manager = manager or self._connect_manager(ray_address=ray_address)
        self._leases: dict[str, SandboxLease] = {}

    async def create(self, spec: SandboxSpec) -> SandboxLease:
        total_start = time.perf_counter() if profiling_enabled() else 0.0
        paddock_mode = _paddock_mode_from_spec(spec)
        expected_pool_mode = (
            POOL_BLACKBOX if paddock_mode == "blackbox" else POOL_COMMAND_ONLY
        )
        status_before = await _manager_status(self._manager) if profiling_enabled() else None
        pool_mode_start = time.perf_counter() if profiling_enabled() else 0.0
        pool_mode = await _manager_pool_mode(self._manager)
        pool_mode_seconds = (
            time.perf_counter() - pool_mode_start if profiling_enabled() else None
        )
        if pool_mode != expected_pool_mode:
            raise RuntimeError(
                "local_bwrap pool mode does not match paddock mode: "
                f"paddock_mode={paddock_mode!r} requires pool_mode={expected_pool_mode!r}, "
                f"connected pool_mode={pool_mode!r}; stop the current pool and start "
                "the correct DRESSAGE_LOCAL_BWRAP_POOL_MODE"
            )
        acquire_start = time.perf_counter() if profiling_enabled() else 0.0
        health_recycles = 0
        recycle_attempts = (
            _env_int("DRESSAGE_LOCAL_BWRAP_ACQUIRE_HEALTH_RECYCLE_ATTEMPTS", 1, min_value=0)
            if paddock_mode == "blackbox"
            else 0
        )
        payload: dict[str, Any]
        async with _acquire_limit():
            for attempt in range(recycle_attempts + 1):
                payload = await _remote_call(
                    self._manager,
                    "acquire",
                    trajectory_id=spec.trajectory_id,
                    env_type=spec.env_type,
                    env_args=spec.env_args,
                )
                if (
                    paddock_mode != "blackbox"
                    or not _spec_requests_service(spec, "blackbox")
                    or not _env_bool(
                    "DRESSAGE_LOCAL_BWRAP_ACQUIRE_HEALTH_PRECHECK",
                    True,
                    )
                ):
                    break
                sandbox_url = payload.get("sandbox_url")
                if sandbox_url and await _blackbox_health_ok(str(sandbox_url).rstrip("/")):
                    break
                health_recycles += 1
                await _remote_call(
                    self._manager,
                    "release",
                    trajectory_id=spec.trajectory_id,
                    lease_id=payload.get("lease_id"),
                    reason="acquire_health_precheck_failed",
                )
                if attempt >= recycle_attempts:
                    raise RuntimeError(
                        "local_bwrap acquired blackbox slot failed health precheck "
                        f"after {attempt + 1} attempt(s)"
                    )
        acquire_seconds = (
            time.perf_counter() - acquire_start if profiling_enabled() else None
        )
        status_after = await _manager_status(self._manager) if profiling_enabled() else None
        profile_metadata = {}
        if profiling_enabled():
            profile_metadata = {
                "profile.provider.pool_mode": pool_mode_seconds,
                "profile.provider.acquire": acquire_seconds,
                "profile.provider.create": time.perf_counter() - total_start,
                "profile.provider.acquire_health_recycles": float(health_recycles),
                **_status_profile("provider.slot.before", status_before),
                **_status_profile("provider.slot.after", status_after),
            }
        lease = SandboxLease(
            trajectory_id=spec.trajectory_id,
            provider=self.name,
            sandbox_id=payload.get("lease_id"),
            capabilities=(
                {"command", "file", "public_url"}
                if paddock_mode == "blackbox"
                else {"command", "file"}
            ),
            metadata={
                "pool_mode": pool_mode,
                "paddock_mode": paddock_mode,
                "node_id": payload.get("node_id"),
                "node_ip": payload.get("node_ip"),
                "slot_id": payload.get("slot_id"),
                "port": payload.get("port"),
                "generation": payload.get("generation"),
                **profile_metadata,
            },
            raw=payload,
        )
        if paddock_mode == "blackbox":
            sandbox_url = payload.get("sandbox_url")
            if not sandbox_url:
                raise RuntimeError("blackbox local_bwrap lease did not return sandbox_url")
            sandbox_endpoint = SandboxEndpoint(url=str(sandbox_url).rstrip("/"), headers={})
            lease.endpoints["blackbox"] = sandbox_endpoint
            for service in spec.services:
                if service.name == "blackbox":
                    lease.endpoints[service.name] = sandbox_endpoint
                elif payload.get("node_ip") and service.port:
                    lease.endpoints[service.name] = SandboxEndpoint(
                        url=f"http://{payload['node_ip']}:{service.port}",
                        headers={},
                    )
        self._leases[spec.trajectory_id] = lease
        return lease

    async def terminate(self, lease: SandboxLease | str) -> dict[str, Any]:
        trajectory_id = lease if isinstance(lease, str) else lease.trajectory_id
        sandbox_id = None if isinstance(lease, str) else lease.sandbox_id
        known = self._leases.pop(trajectory_id, None)
        lease_id = sandbox_id or (known.sandbox_id if known is not None else None)
        return await _remote_call(
            self._manager,
            "release",
            trajectory_id=trajectory_id,
            lease_id=lease_id,
            reason="paddock_terminate",
        )

    async def get_public_url(
        self,
        lease: SandboxLease,
        *,
        port: int,
        service_name: str | None = None,
    ) -> SandboxEndpoint:
        if "public_url" not in lease.capabilities:
            raise ValueError(
                f"local_bwrap lease {lease.trajectory_id!r} does not expose public URLs"
            )
        if service_name and service_name in lease.endpoints:
            return lease.endpoints[service_name].normalized()
        if "blackbox" in lease.endpoints and int(lease.metadata.get("port") or port) == port:
            return lease.endpoints["blackbox"].normalized()
        node_ip = lease.metadata.get("node_ip")
        if not node_ip:
            raise ValueError("local_bwrap lease does not contain node_ip")
        return SandboxEndpoint(url=f"http://{node_ip}:{port}")

    async def run_command(
        self,
        lease: SandboxLease,
        command: str | list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        stdin: str | bytes | None = None,
    ) -> CommandResult:
        result = await _remote_call(
            self._manager,
            "run_command",
            trajectory_id=lease.trajectory_id,
            lease_id=lease.sandbox_id,
            command=command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            stdin=stdin,
        )
        return CommandResult(
            cmd=result.get("cmd", command),
            stdout=str(result.get("stdout") or ""),
            stderr=str(result.get("stderr") or ""),
            returncode=result.get("returncode"),
            timed_out=bool(result.get("timed_out")),
            metadata={k: v for k, v in result.items() if k not in {"cmd", "stdout", "stderr", "returncode", "timed_out"}},
        )

    async def read_file(
        self,
        lease: SandboxLease,
        path: str,
        *,
        encoding: str | None = "utf-8",
        max_bytes: int | None = None,
    ) -> str | bytes:
        return await _remote_call(
            self._manager,
            "read_file",
            trajectory_id=lease.trajectory_id,
            lease_id=lease.sandbox_id,
            path=path,
            encoding=encoding,
            max_bytes=max_bytes,
        )

    async def write_file(
        self,
        lease: SandboxLease,
        path: str,
        content: str | bytes,
        *,
        encoding: str | None = "utf-8",
        append: bool = False,
    ) -> dict[str, Any]:
        return await _remote_call(
            self._manager,
            "write_file",
            trajectory_id=lease.trajectory_id,
            lease_id=lease.sandbox_id,
            path=path,
            content=content,
            encoding=encoding,
            append=append,
        )

    def _connect_manager(self, *, ray_address: str | None = None) -> Any:
        try:
            import ray
        except ImportError as exc:
            raise ImportError("ray is required for DRESSAGE_SANDBOX_PROVIDER=local_bwrap") from exc
        if not ray.is_initialized():
            ray.init(
                address=ray_address
                or os.environ.get("DRESSAGE_LOCAL_BWRAP_RAY_ADDRESS")
                or os.environ.get("DRESSAGE_RAY_ADDRESS", "auto"),
                namespace=self._namespace,
                ignore_reinit_error=True,
            )
        return ray.get_actor(self._manager_name, namespace=self._namespace)


@asynccontextmanager
async def _acquire_limit():
    concurrency = _env_int("DRESSAGE_LOCAL_BWRAP_ACQUIRE_CONCURRENCY", 0, min_value=0)
    if concurrency <= 0:
        yield
        return
    semaphore = _ACQUIRE_SEMAPHORES.get(concurrency)
    if semaphore is None:
        semaphore = asyncio.Semaphore(concurrency)
        _ACQUIRE_SEMAPHORES[concurrency] = semaphore
    async with semaphore:
        yield


async def _manager_status(manager: Any) -> dict[str, Any] | None:
    try:
        return await _remote_call(manager, "status", force_refresh=True)
    except Exception:
        return None


async def _blackbox_health_ok(base_url: str) -> bool:
    timeout = _env_float(
        "DRESSAGE_LOCAL_BWRAP_ACQUIRE_HEALTH_TIMEOUT_SEC",
        2.0,
        min_value=0.1,
    )
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.get(f"{base_url.rstrip('/')}/health")
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return False
    if isinstance(payload, dict):
        status = str(payload.get("status") or payload.get("state") or "").upper()
        if status in {"ERROR", "SHUTTING_DOWN"}:
            return False
    return True


def _status_profile(prefix: str, status: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(status, dict):
        return {}
    keys = (
        "total_capacity",
        "total_ready",
        "total_leased",
        "total_resetting",
        "total_restarting",
        "total_failed",
        "total_lost",
    )
    result: dict[str, float] = {}
    for key in keys:
        value = status.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[f"profile.{prefix}.{key}"] = float(value)
    leases = status.get("leases")
    if isinstance(leases, dict):
        for key in ("active", "releasing", "tracked", "expired", "lost"):
            value = leases.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[f"profile.{prefix}.leases_{key}"] = float(value)
    return result


def _spec_requests_service(spec: SandboxSpec, service_name: str) -> bool:
    return any(service.name == service_name for service in spec.services)


def _env_int(name: str, default: int, *, min_value: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            value = default
    if min_value is not None:
        value = max(min_value, value)
    return value


def _env_float(name: str, default: float, *, min_value: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = float(raw)
        except ValueError:
            value = default
    if min_value is not None:
        value = max(min_value, value)
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _paddock_mode_from_spec(spec: SandboxSpec) -> str:
    mode = str(spec.metadata.get("paddock_mode") or "blackbox").strip().lower()
    if mode not in {"blackbox", "whitebox"}:
        raise ValueError(
            f"unsupported local_bwrap paddock_mode={mode!r}; expected blackbox|whitebox"
        )
    return mode


async def _manager_pool_mode(manager: Any) -> PoolMode:
    pool_mode_method = getattr(manager, "pool_mode", None)
    if pool_mode_method is not None:
        if not callable(pool_mode_method):
            return normalize_pool_mode(str(pool_mode_method))
        remote = getattr(pool_mode_method, "remote", None)
        if remote is not None:
            return normalize_pool_mode(await _ray_get(remote()))
        value = pool_mode_method()
        if asyncio.iscoroutine(value) or isinstance(value, Awaitable):
            value = await value
        return normalize_pool_mode(str(value))

    status = await _remote_call(manager, "status")
    return normalize_pool_mode(str(status.get("pool_mode")))


async def _remote_call(target: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(target, method_name)
    remote = getattr(method, "remote", None)
    if remote is not None:
        return await _ray_get(remote(*args, **kwargs))
    result = method(*args, **kwargs)
    if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
        return await result
    return result


async def _ray_get(obj_ref: Any) -> Any:
    import ray

    if hasattr(obj_ref, "__await__"):
        try:
            return await obj_ref
        except TypeError:
            pass
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, ray.get, obj_ref)
