"""Ray-pinnable node-local supervisor for local bwrap slots."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import logging
import os
import posixpath
from pathlib import Path
import socket
import time
from typing import Any, Literal
import uuid

import httpx

from dressage.paddock.blackbox.common.defaults import (
    DEFAULT_BLACKBOX_TYPE,
    normalize_blackbox_type,
)
from dressage.sandbox.local.bwrap.slot import (
    SLOT_DEAD,
    SLOT_EMPTY,
    SLOT_FAILED,
    SLOT_LEASED,
    SLOT_LOST,
    SLOT_READY,
    SLOT_RELEASING,
    SLOT_RESETTING,
    SLOT_RESTARTING,
    SLOT_STARTING,
    SlotConfig,
    SlotRuntime,
)
from dressage.sandbox.local.bwrap.runner import LocalSandboxRunner

try:  # Ray is optional for local unit tests.
    import ray
except ImportError:  # pragma: no cover - exercised only in envs without Ray
    ray = None

logger = logging.getLogger(__name__)

HealthChecker = Callable[[str], Awaitable[bool] | bool]
PoolMode = Literal["blackbox", "command_only"]

POOL_BLACKBOX: PoolMode = "blackbox"
POOL_COMMAND_ONLY: PoolMode = "command_only"
_POOL_MODES = {POOL_BLACKBOX, POOL_COMMAND_ONLY}


def normalize_pool_mode(value: str | None) -> PoolMode:
    text = (value or POOL_BLACKBOX).strip().lower().replace("-", "_")
    if text in {"command", "commands", "commandonly", POOL_COMMAND_ONLY}:
        return POOL_COMMAND_ONLY
    if text == POOL_BLACKBOX:
        return POOL_BLACKBOX
    expected = "|".join(sorted(_POOL_MODES))
    raise ValueError(f"unsupported local_bwrap pool mode {value!r}; expected {expected}")


class LocalBwrapNodeSupervisorCore:
    """Manage local bwrap slots on one node."""

    def __init__(
        self,
        *,
        node_id: str,
        node_ip: str | None = None,
        capacity: int,
        base_port: int = 31000,
        bind_host: str | None = None,
        advertise_host: str | None = None,
        base_dir: str | Path | None = None,
        blackbox_type: str = DEFAULT_BLACKBOX_TYPE,
        runner: Any | None = None,
        health_checker: HealthChecker | None = None,
        startup_timeout_sec: float | None = None,
        acquire_timeout_sec: float | None = None,
        acquire_poll_interval_sec: float | None = None,
        health_interval_sec: float | None = None,
        health_timeout_sec: float | None = None,
        reset_strategy: str | None = None,
        start_health_loop: bool = True,
        pool_mode: str | None = None,
    ) -> None:
        if capacity < 0:
            raise ValueError(f"capacity must be non-negative, got {capacity}")
        self.node_id = node_id
        self.pool_mode = normalize_pool_mode(
            pool_mode or os.environ.get("DRESSAGE_LOCAL_BWRAP_POOL_MODE", POOL_BLACKBOX)
        )
        self.node_ip = node_ip or advertise_host or _local_ip()
        self.capacity = capacity
        self.base_port = base_port
        self.bind_host = bind_host or os.environ.get(
            "DRESSAGE_BLACKBOX_BIND_HOST", "0.0.0.0"
        )
        self.advertise_host = advertise_host or os.environ.get(
            "DRESSAGE_BLACKBOX_ADVERTISE_IP", self.node_ip
        )
        self.base_dir = Path(
            base_dir
            or os.environ.get(
                "DRESSAGE_LOCAL_BWRAP_SLOT_BASE_DIR",
                os.environ.get(
                    "DRESSAGE_BLACKBOX_SLOT_BASE_DIR",
                    f"/tmp/dressage-local-bwrap/node-{_safe_name(node_id)}",
                ),
            )
        )
        self.blackbox_type = normalize_blackbox_type(blackbox_type)
        self.runner = runner or LocalSandboxRunner()
        self.health_checker = health_checker
        self.startup_timeout_sec = _env_float(
            "DRESSAGE_BLACKBOX_STARTUP_TIMEOUT_SEC",
            60.0 if startup_timeout_sec is None else startup_timeout_sec,
            min_value=0.0,
        )
        self.acquire_timeout_sec = _env_float(
            "DRESSAGE_BLACKBOX_NODE_ACQUIRE_TIMEOUT_SEC",
            30.0 if acquire_timeout_sec is None else acquire_timeout_sec,
            min_value=0.0,
        )
        self.acquire_poll_interval_sec = _env_float(
            "DRESSAGE_BLACKBOX_NODE_ACQUIRE_POLL_SEC",
            0.1 if acquire_poll_interval_sec is None else acquire_poll_interval_sec,
            min_value=0.0,
        )
        self.health_interval_sec = _env_float(
            "DRESSAGE_BLACKBOX_HEALTH_INTERVAL_SEC",
            10.0 if health_interval_sec is None else health_interval_sec,
            min_value=0.1,
        )
        self.health_timeout_sec = _env_float(
            "DRESSAGE_BLACKBOX_HEALTH_TIMEOUT_SEC",
            2.0 if health_timeout_sec is None else health_timeout_sec,
            min_value=0.0,
        )
        self.reset_strategy = (
            reset_strategy
            or os.environ.get("DRESSAGE_BLACKBOX_RESET_STRATEGY")
            or "hard"
        ).lower()
        self.preserve_session_artifacts = _env_bool(
            "DRESSAGE_BLACKBOX_PRESERVE_SESSION_ARTIFACTS", False
        )
        self.session_archive_dirs = tuple(
            item.strip()
            for item in os.environ.get(
                "DRESSAGE_BLACKBOX_SESSION_ARCHIVE_DIRS", "home,work,runtime,tmp"
            ).split(",")
            if item.strip()
        )
        self.session_archive_max_per_slot = _env_int(
            "DRESSAGE_BLACKBOX_SESSION_ARCHIVE_MAX_PER_SLOT",
            20,
            min_value=0,
        )
        self.session_archive_ttl_sec = _env_float(
            "DRESSAGE_BLACKBOX_SESSION_ARCHIVE_TTL_SEC",
            86400.0,
            min_value=0.0,
        )
        default_reset_concurrency = max(1, min(capacity or 1, 64))
        self.reset_concurrency = _env_int(
            "DRESSAGE_LOCAL_BWRAP_RESET_CONCURRENCY",
            default_reset_concurrency,
            min_value=1,
        )
        self._reset_semaphore = asyncio.Semaphore(self.reset_concurrency)
        self._slots: list[SlotRuntime] = [
            SlotRuntime(
                SlotConfig(
                    slot_id=slot_id,
                    port=base_port + slot_id,
                    bind_host=self.bind_host,
                    advertise_host=self.advertise_host,
                    base_dir=self.base_dir / "slots",
                    blackbox_type=self.blackbox_type,
                    memory_high_bytes=_env_int(
                        "DRESSAGE_BLACKBOX_MEMORY_HIGH_BYTES",
                        1536 * 1024 * 1024,
                        min_value=1,
                    ),
                    memory_max_bytes=_env_int(
                        "DRESSAGE_BLACKBOX_MEMORY_MAX_BYTES",
                        2 * 1024 * 1024 * 1024,
                        min_value=1,
                    ),
                    pids_max=_env_int(
                        "DRESSAGE_BLACKBOX_PIDS_MAX", 128, min_value=1
                    ),
                    nofile=_env_int(
                        "DRESSAGE_BLACKBOX_NOFILE", 512, min_value=1
                    ),
                )
            )
            for slot_id in range(capacity)
        ]
        self._lock = asyncio.Lock()
        self._slot_locks: dict[int, asyncio.Lock] = {
            slot.config.slot_id: asyncio.Lock() for slot in self._slots
        }
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._run_id = f"{_safe_name(self.node_id)}-{uuid.uuid4().hex}"
        self._closed = False
        self._health_task: asyncio.Task[Any] | None = None
        self._start_health_loop_on_pool_start = (
            start_health_loop and self.pool_mode == POOL_BLACKBOX
        )
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.health_timeout_sec), trust_env=False
        )

    async def start_pool(self) -> dict[str, Any]:
        if self._closed:
            return await self.health()
        if self.capacity == 0:
            return await self.health()
        startable = [
            slot
            for slot in self._slots
            if slot.status in {SLOT_EMPTY, SLOT_FAILED, SLOT_DEAD}
            and slot.lease_id is None
        ]
        if self.pool_mode == POOL_COMMAND_ONLY:
            await asyncio.gather(
                *(
                    self._prepare_command_only_slot(slot, "pool_start")
                    for slot in startable
                )
            )
        else:
            await asyncio.gather(*(self._start_slot(slot) for slot in startable))
        if self._start_health_loop_on_pool_start and self._health_task is None:
            self._health_task = asyncio.create_task(self._health_loop())
        return await self.health()

    async def acquire(
        self,
        trajectory_id: str,
        env_type: str | None = None,
        env_args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del env_type, env_args
        deadline = time.monotonic() + self.acquire_timeout_sec
        while True:
            if self._closed:
                raise RuntimeError(f"local_bwrap supervisor {self.node_id} is shut down")
            async with self._lock:
                if self._closed:
                    raise RuntimeError(
                        f"local_bwrap supervisor {self.node_id} is shut down"
                    )
                existing = self._slot_for_trajectory(trajectory_id)
                if existing is not None:
                    return self._lease_payload(existing)

                slot = next((item for item in self._slots if item.is_available), None)
                if slot is not None:
                    lease_id = self._new_lease_id(slot, trajectory_id)
                    slot.status = SLOT_LEASED
                    slot.lease_id = lease_id
                    slot.trajectory_id = trajectory_id
                    slot.acquired_ts = time.time()
                    return self._lease_payload(slot)

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"no ready local_bwrap {self.pool_mode} slot on node "
                    f"{self.node_id} for trajectory_id={trajectory_id}"
                )
            await asyncio.sleep(self.acquire_poll_interval_sec)

    async def release(
        self,
        lease_id: str | None = None,
        trajectory_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            slot = self._slot_for_lease(lease_id, trajectory_id)
            if slot is None:
                return {
                    "released": True,
                    "already_released": True,
                    "lease_id": lease_id,
                    "trajectory_id": trajectory_id,
                    "node_id": self.node_id,
                    "node_ip": self.node_ip,
                }
            payload = self._lease_payload(slot)
            if self._closed:
                slot.status = SLOT_DEAD
                slot.lease_id = None
                slot.trajectory_id = None
                slot.acquired_ts = None
                return {
                    "released": True,
                    "release_queued": False,
                    "slot_reusable": False,
                    "supervisor_closed": True,
                    "already_released": False,
                    "lease_id": payload["lease_id"],
                    "trajectory_id": payload["trajectory_id"],
                    "slot_id": payload["slot_id"],
                    "generation": payload["generation"],
                    "node_id": self.node_id,
                    "node_ip": self.node_ip,
                }
            slot.status = SLOT_RELEASING
            slot.lease_id = None
            slot.trajectory_id = None
            slot.acquired_ts = None

        self._schedule_task(
            self._reset_slot_limited(
                slot,
                reason or "release",
                expected_generation=payload["generation"],
                session_id=payload["trajectory_id"],
                lease_id=payload["lease_id"],
            )
        )
        return {
            "released": True,
            "release_queued": True,
            "slot_reusable": False,
            "already_released": False,
            "lease_id": payload["lease_id"],
            "trajectory_id": payload["trajectory_id"],
            "slot_id": payload["slot_id"],
            "generation": payload["generation"],
            "node_id": self.node_id,
            "node_ip": self.node_ip,
        }

    async def force_release(
        self,
        slot_id: int,
        lease_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        slot = self._slots[slot_id]
        return await self.release(
            lease_id=lease_id or slot.lease_id,
            trajectory_id=slot.trajectory_id,
            reason=reason or "force_release",
        )

    async def restart_slot(
        self,
        slot_id: int,
        reason: str | None = None,
    ) -> dict[str, Any]:
        slot = self._slots[slot_id]
        if self._closed:
            return slot.to_dict()
        async with self._lock:
            if self._closed:
                return slot.to_dict()
            slot.status = SLOT_RESTARTING
            slot.lease_id = None
            slot.trajectory_id = None
            slot.acquired_ts = None
        await self._reset_slot(slot, reason or "manual_restart")
        return slot.to_dict()

    async def health(self) -> dict[str, Any]:
        self._reap_background_tasks()
        counts = self._counts()
        return {
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "hostname": socket.gethostname(),
            "run_id": self._run_id,
            "pool_mode": self.pool_mode,
            "closed": self._closed,
            "capacity": self.capacity,
            "ready": counts.get(SLOT_READY, 0),
            "leased": counts.get(SLOT_LEASED, 0),
            "releasing": counts.get(SLOT_RELEASING, 0),
            "starting": counts.get(SLOT_STARTING, 0),
            "resetting": counts.get(SLOT_RESETTING, 0),
            "restarting": counts.get(SLOT_RESTARTING, 0),
            "failed": counts.get(SLOT_FAILED, 0),
            "dead": counts.get(SLOT_DEAD, 0),
            "lost": counts.get(SLOT_LOST, 0),
            "empty": counts.get(SLOT_EMPTY, 0),
            "background_tasks": len(self._background_tasks),
            "reset_concurrency": self.reset_concurrency,
            "slots": [slot.to_dict() for slot in self._slots],
            "last_error": self._last_error(),
        }

    async def drain(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "draining": True}

    async def undrain(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "draining": False}

    async def logs(self, slot_id: int, tail: int = 200) -> str:
        slot = self._slots[slot_id]
        paths = sorted(slot.config.log_dir.glob("server-*.*"))
        if not paths:
            return ""
        lines: list[str] = []
        for path in paths[-2:]:
            try:
                path_lines = path.read_text(errors="replace").splitlines()
            except OSError as exc:
                lines.append(f"failed to read {path}: {exc}")
                continue
            lines.extend([f"==> {path.name} <==", *path_lines[-tail:]])
        return "\n".join(lines[-tail:])

    async def run_command(
        self,
        *,
        command: str | list[str],
        lease_id: str | None = None,
        trajectory_id: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        stdin: str | bytes | None = None,
    ) -> dict[str, Any]:
        slot = self._active_slot_for_lease(lease_id, trajectory_id)
        async with self._slot_locks[slot.config.slot_id]:
            self._ensure_slot_still_active(slot, lease_id, trajectory_id)
            cmd = self.runner.build_tool_command(slot, command, cwd=cwd)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=self.runner.build_tool_env(slot, extra_env=env),
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            stdin_bytes: bytes | None
            if isinstance(stdin, str):
                stdin_bytes = stdin.encode()
            else:
                stdin_bytes = stdin
            timed_out = False
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(stdin_bytes),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                timed_out = True
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                stdout, stderr = await proc.communicate()
            return {
                "cmd": command,
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
                "returncode": proc.returncode,
                "timed_out": timed_out,
                "lease_id": slot.lease_id,
                "trajectory_id": slot.trajectory_id,
                "slot_id": slot.config.slot_id,
                "node_id": self.node_id,
                "node_ip": self.node_ip,
            }

    async def read_file(
        self,
        *,
        path: str,
        lease_id: str | None = None,
        trajectory_id: str | None = None,
        encoding: str | None = "utf-8",
        max_bytes: int | None = None,
    ) -> str | bytes:
        slot = self._active_slot_for_lease(lease_id, trajectory_id)
        host_path = self._sandbox_path_to_host(slot, path)
        data = host_path.read_bytes()
        if max_bytes is not None:
            data = data[:max_bytes]
        if encoding is None:
            return data
        return data.decode(encoding)

    async def write_file(
        self,
        *,
        path: str,
        content: str | bytes,
        lease_id: str | None = None,
        trajectory_id: str | None = None,
        encoding: str | None = "utf-8",
        append: bool = False,
    ) -> dict[str, Any]:
        slot = self._active_slot_for_lease(lease_id, trajectory_id)
        host_path = self._sandbox_path_to_host(slot, path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "ab" if append else "wb"
        data = content if isinstance(content, bytes) else content.encode(encoding or "utf-8")
        with host_path.open(mode) as handle:
            handle.write(data)
        return {
            "path": path,
            "host_path": str(host_path),
            "bytes": len(data),
            "append": append,
            "lease_id": slot.lease_id,
            "trajectory_id": slot.trajectory_id,
        }

    async def shutdown(self) -> dict[str, Any]:
        self._closed = True
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        results = await asyncio.gather(
            *(self._shutdown_slot(slot) for slot in self._slots),
            return_exceptions=True,
        )
        await self._client.aclose()
        stopped = sum(1 for result in results if not isinstance(result, Exception))
        errors = [
            _exception_summary(result)
            for result in results
            if isinstance(result, Exception)
        ]
        return {
            "node_id": self.node_id,
            "stopped": True,
            "closed": True,
            "slots_stopped": stopped,
            "errors": errors,
        }

    async def _shutdown_slot(self, slot: SlotRuntime) -> None:
        async with self._slot_locks[slot.config.slot_id]:
            await self.runner.stop(slot)
            slot.process = None
            slot.process_pid = None
            slot.lease_id = None
            slot.trajectory_id = None
            slot.acquired_ts = None
            slot.status = SLOT_EMPTY

    async def _prepare_command_only_slot(self, slot: SlotRuntime, reason: str) -> None:
        async with self._slot_locks[slot.config.slot_id]:
            if self._closed:
                slot.status = SLOT_EMPTY
                return
            slot.status = SLOT_RESETTING
            slot.generation += 1
            slot.rotate_cleanup_token(supervisor_run_id=self._run_id)
            slot.process = None
            slot.process_pid = None
            slot.lease_id = None
            slot.trajectory_id = None
            slot.acquired_ts = None
            slot.last_error = None
            slot.config.reset_runtime_dirs(
                preserve_artifacts=False,
                generation=slot.generation,
                reason=reason,
                metadata={"node_id": self.node_id, "node_ip": self.node_ip},
            )
            slot.status = SLOT_READY

    async def _start_slot(self, slot: SlotRuntime) -> None:
        if self.pool_mode == POOL_COMMAND_ONLY:
            await self._prepare_command_only_slot(slot, "start_slot")
            return
        if self._closed:
            slot.status = SLOT_EMPTY
            return
        slot.status = SLOT_STARTING
        slot.generation += 1
        slot.rotate_cleanup_token(supervisor_run_id=self._run_id)
        slot.process = None
        slot.process_pid = None
        slot.last_error = None
        slot.config.clear_runtime_dirs()
        try:
            proc = await self.runner.start(slot)
            slot.process = proc
            slot.process_pid = getattr(proc, "pid", None)
            if self._closed:
                await self.runner.stop(slot)
                slot.process = None
                slot.process_pid = None
                slot.status = SLOT_EMPTY
                return
            await self._wait_until_healthy(slot)
            if self._closed:
                await self.runner.stop(slot)
                slot.process = None
                slot.process_pid = None
                slot.status = SLOT_EMPTY
                return
            slot.status = SLOT_READY
            slot.last_health_ts = time.time()
        except Exception as exc:
            slot.status = SLOT_FAILED
            slot.last_error = _exception_summary(exc)
            logger.exception(
                "failed to start local_bwrap blackbox slot node_id=%s slot_id=%s",
                self.node_id,
                slot.config.slot_id,
            )

    async def _reset_slot_limited(
        self,
        slot: SlotRuntime,
        reason: str,
        *,
        expected_generation: int | None = None,
        session_id: str | None = None,
        lease_id: str | None = None,
    ) -> None:
        async with self._reset_semaphore:
            await self._reset_slot(
                slot,
                reason,
                expected_generation=expected_generation,
                session_id=session_id,
                lease_id=lease_id,
            )

    async def _reset_slot(
        self,
        slot: SlotRuntime,
        reason: str,
        *,
        expected_generation: int | None = None,
        session_id: str | None = None,
        lease_id: str | None = None,
    ) -> None:
        async with self._slot_locks[slot.config.slot_id]:
            if self._closed:
                await self.runner.stop(slot)
                slot.process = None
                slot.process_pid = None
                slot.lease_id = None
                slot.trajectory_id = None
                slot.acquired_ts = None
                slot.status = SLOT_EMPTY
                return
            if expected_generation is not None and slot.generation != expected_generation:
                logger.debug(
                    "skip stale blackbox slot reset node_id=%s slot_id=%s "
                    "expected_generation=%s current_generation=%s",
                    self.node_id,
                    slot.config.slot_id,
                    expected_generation,
                    slot.generation,
                )
                return
            if self.pool_mode == POOL_COMMAND_ONLY:
                slot.status = SLOT_RESETTING
                await self.runner.stop(slot)
                slot.process = None
                slot.process_pid = None
                archive_path = slot.config.reset_runtime_dirs(
                    preserve_artifacts=self.preserve_session_artifacts,
                    session_id=session_id,
                    lease_id=lease_id,
                    generation=expected_generation,
                    reason=reason,
                    archive_dirs=self.session_archive_dirs,
                    archive_max_per_slot=self.session_archive_max_per_slot,
                    archive_ttl_sec=self.session_archive_ttl_sec,
                    metadata={"node_id": self.node_id, "node_ip": self.node_ip},
                )
                if archive_path is not None:
                    logger.info(
                        "archived local_bwrap command-only slot artifacts "
                        "node_id=%s slot_id=%s session_id=%s archive_path=%s",
                        self.node_id,
                        slot.config.slot_id,
                        session_id,
                        archive_path,
                    )
                slot.generation += 1
                slot.rotate_cleanup_token(supervisor_run_id=self._run_id)
                slot.lease_id = None
                slot.trajectory_id = None
                slot.acquired_ts = None
                slot.last_error = None
                slot.status = SLOT_EMPTY if self._closed else SLOT_READY
                return
            if self.reset_strategy == "soft":
                slot.config.clear_runtime_dirs()
                slot.status = SLOT_EMPTY if self._closed else SLOT_READY
                slot.last_error = None
                return
            slot.status = SLOT_RESTARTING
            await self.runner.stop(slot)
            slot.process = None
            slot.process_pid = None
            if self._closed:
                slot.status = SLOT_EMPTY
                slot.lease_id = None
                slot.trajectory_id = None
                slot.acquired_ts = None
                return
            if expected_generation is not None and slot.generation != expected_generation:
                logger.debug(
                    "skip stale blackbox slot restart node_id=%s slot_id=%s "
                    "expected_generation=%s current_generation=%s",
                    self.node_id,
                    slot.config.slot_id,
                    expected_generation,
                    slot.generation,
                )
                return
            archive_path = slot.config.reset_runtime_dirs(
                preserve_artifacts=self.preserve_session_artifacts,
                session_id=session_id,
                lease_id=lease_id,
                generation=expected_generation,
                reason=reason,
                archive_dirs=self.session_archive_dirs,
                archive_max_per_slot=self.session_archive_max_per_slot,
                archive_ttl_sec=self.session_archive_ttl_sec,
                metadata={"node_id": self.node_id, "node_ip": self.node_ip},
            )
            if archive_path is not None:
                logger.info(
                    "archived blackbox slot artifacts node_id=%s slot_id=%s "
                    "session_id=%s archive_path=%s",
                    self.node_id,
                    slot.config.slot_id,
                    session_id,
                    archive_path,
                )
            if self._closed:
                slot.status = SLOT_EMPTY
                return
            await self._start_slot(slot)

    async def _wait_until_healthy(self, slot: SlotRuntime) -> None:
        if _env_bool("DRESSAGE_BLACKBOX_SKIP_HEALTHCHECK", False):
            return
        deadline = time.monotonic() + self.startup_timeout_sec
        while True:
            if await self._check_slot_health(slot):
                return
            proc = slot.process
            returncode = None if proc is None else getattr(proc, "returncode", None)
            if returncode is not None:
                raise RuntimeError(
                    f"slot {slot.config.slot_id} exited before health check passed "
                    f"with returncode={returncode}; recent logs:\n"
                    f"{self._slot_log_tail(slot)}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"slot {slot.config.slot_id} health check timed out; "
                    f"process_pid={slot.process_pid}; recent logs:\n"
                    f"{self._slot_log_tail(slot)}"
                )
            await asyncio.sleep(0.2)

    async def _check_slot_health(self, slot: SlotRuntime) -> bool:
        proc = slot.process
        if proc is not None and getattr(proc, "returncode", None) is not None:
            return False
        if self.health_checker is not None:
            result = self.health_checker(slot.sandbox_url)
            if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                return bool(await result)
            return bool(result)
        try:
            response = await self._client.get(f"{slot.sandbox_url}/health")
        except httpx.HTTPError:
            return False
        return response.status_code < 500

    def _slot_log_tail(self, slot: SlotRuntime, *, tail: int = 80) -> str:
        paths = sorted(slot.config.log_dir.glob("server-*.*"))
        if not paths:
            return "<no slot logs>"
        lines: list[str] = []
        for path in paths[-4:]:
            try:
                path_lines = path.read_text(errors="replace").splitlines()
            except OSError as exc:
                lines.append(f"==> {path.name} <==")
                lines.append(f"failed to read {path}: {exc}")
                continue
            lines.extend([f"==> {path.name} <==", *path_lines[-tail:]])
        return "\n".join(lines[-tail:])

    async def _health_loop(self) -> None:
        while not self._closed:
            try:
                if self.pool_mode == POOL_BLACKBOX:
                    await self._check_live_slots_once()
            except Exception:
                logger.exception("local_bwrap supervisor health loop failed")
            await asyncio.sleep(self.health_interval_sec)

    async def _check_live_slots_once(self) -> None:
        if self._closed:
            return
        for slot in self._slots:
            if slot.status not in {SLOT_READY, SLOT_LEASED}:
                continue
            ok = await self._check_slot_health(slot)
            if ok:
                slot.last_health_ts = time.time()
                continue
            if slot.status == SLOT_LEASED:
                slot.status = SLOT_LOST
                slot.last_error = "health check failed while leased"
            else:
                slot.status = SLOT_DEAD
                self._schedule_task(
                    self._reset_slot(
                        slot,
                        "health_check_failed",
                        expected_generation=slot.generation,
                    )
                )

    def _slot_for_trajectory(self, trajectory_id: str) -> SlotRuntime | None:
        return next(
            (
                slot
                for slot in self._slots
                if slot.trajectory_id == trajectory_id
                and slot.status in {SLOT_LEASED, SLOT_LOST}
            ),
            None,
        )

    def _slot_for_lease(
        self,
        lease_id: str | None,
        trajectory_id: str | None,
    ) -> SlotRuntime | None:
        for slot in self._slots:
            if slot.status not in {SLOT_LEASED, SLOT_LOST, SLOT_RELEASING, SLOT_RESETTING}:
                continue
            if lease_id is not None and slot.lease_id == lease_id:
                return slot
            if lease_id is None and trajectory_id is not None:
                if slot.trajectory_id == trajectory_id:
                    return slot
        return None

    def _active_slot_for_lease(
        self,
        lease_id: str | None,
        trajectory_id: str | None,
    ) -> SlotRuntime:
        slot = self._slot_for_lease(lease_id, trajectory_id)
        if slot is None or slot.status not in {SLOT_LEASED, SLOT_LOST}:
            raise KeyError(
                "active local_bwrap lease not found "
                f"lease_id={lease_id!r} trajectory_id={trajectory_id!r}"
            )
        return slot

    def _ensure_slot_still_active(
        self,
        slot: SlotRuntime,
        lease_id: str | None,
        trajectory_id: str | None,
    ) -> None:
        if slot.status not in {SLOT_LEASED, SLOT_LOST}:
            raise RuntimeError(
                f"slot {slot.config.slot_id} is no longer active; status={slot.status}"
            )
        if lease_id is not None and slot.lease_id != lease_id:
            raise RuntimeError(
                f"slot {slot.config.slot_id} lease changed from {lease_id!r} "
                f"to {slot.lease_id!r}"
            )
        if trajectory_id is not None and slot.trajectory_id != trajectory_id:
            raise RuntimeError(
                f"slot {slot.config.slot_id} trajectory changed from {trajectory_id!r} "
                f"to {slot.trajectory_id!r}"
            )

    def _sandbox_path_to_host(self, slot: SlotRuntime, path: str) -> Path:
        raw = str(path or "").strip()
        if not raw:
            raise ValueError("path must be non-empty")
        if not raw.startswith("/"):
            raw = "/workspace/" + raw
        normalized = posixpath.normpath(raw)
        if normalized == ".":
            normalized = "/workspace"
        mounts: tuple[tuple[str, Path], ...] = (
            ("/workspace", slot.config.work_dir),
            ("/home/blackbox", slot.config.home_dir),
            ("/tmp", slot.config.tmp_dir),
            (
                "/workspace_sandbox/blackbox_server_runtime",
                slot.config.runtime_dir,
            ),
        )
        for prefix, host_root in mounts:
            if normalized == prefix or normalized.startswith(prefix + "/"):
                rel = normalized[len(prefix) :].lstrip("/")
                candidate = (host_root / rel).resolve(strict=False)
                root = host_root.resolve(strict=False)
                try:
                    inside = candidate == root or candidate.is_relative_to(root)
                except ValueError:
                    inside = False
                if not inside:
                    raise ValueError(f"path escapes sandbox mount: {path!r}")
                return candidate
        raise ValueError(
            f"unsupported sandbox path {path!r}; expected /workspace, "
            "/home/blackbox, /tmp, or /workspace_sandbox/blackbox_server_runtime"
        )

    def _lease_payload(self, slot: SlotRuntime) -> dict[str, Any]:
        return {
            "lease_id": slot.lease_id,
            "trajectory_id": slot.trajectory_id,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "slot_id": slot.config.slot_id,
            "port": slot.config.port,
            "sandbox_url": slot.sandbox_url if self.pool_mode == POOL_BLACKBOX else None,
            "generation": slot.generation,
            "ready": slot.status in {SLOT_LEASED, SLOT_LOST},
            "status": slot.status,
            "pool_mode": self.pool_mode,
        }

    def _new_lease_id(self, slot: SlotRuntime, trajectory_id: str) -> str:
        return (
            f"lease-{_safe_name(self.node_id)[:12]}-{slot.config.slot_id:04d}-"
            f"gen{slot.generation}-{_safe_name(trajectory_id)[:32]}-"
            f"{uuid.uuid4().hex[:8]}"
        )

    def _counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for slot in self._slots:
            counts[slot.status] = counts.get(slot.status, 0) + 1
        return counts

    def _last_error(self) -> str | None:
        for slot in reversed(self._slots):
            if slot.last_error:
                return slot.last_error
        return None

    def _schedule_task(self, coro: Awaitable[Any]) -> None:
        if self._closed:
            close = getattr(coro, "close", None)
            if close is not None:
                close()
            return
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)

        def _done(done: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(done)
            try:
                done.result()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("local_bwrap supervisor background task failed")

        task.add_done_callback(_done)

    def _reap_background_tasks(self) -> None:
        for task in list(self._background_tasks):
            if task.done():
                self._background_tasks.discard(task)


def _local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _safe_name(value: Any) -> str:
    text = str(value)
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    return "".join(ch if ch in allowed else "_" for ch in text).strip("._") or "node"


def _env_int(name: str, default: int, *, min_value: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("invalid %s=%r; falling back to %d", name, value, default)
        return default
    if parsed < min_value:
        logger.warning("invalid %s=%r; falling back to %d", name, value, default)
        return default
    return parsed


def _env_float(name: str, default: float, *, min_value: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        logger.warning("invalid %s=%r; falling back to %.3f", name, value, default)
        return default
    if parsed < min_value:
        logger.warning("invalid %s=%r; falling back to %.3f", name, value, default)
        return default
    return parsed


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _exception_summary(exc: BaseException) -> str:
    return " ".join(str(exc).splitlines()) or type(exc).__name__


if ray is not None:

    @ray.remote(num_cpus=0, num_gpus=0, max_concurrency=2048)
    class LocalBwrapNodeSupervisor(LocalBwrapNodeSupervisorCore):
        pass

else:

    class _MissingRayActor:
        def options(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise ImportError("ray is required to create LocalBwrapNodeSupervisor actors")

    LocalBwrapNodeSupervisor = _MissingRayActor()
