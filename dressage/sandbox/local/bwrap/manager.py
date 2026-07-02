"""Global Ray actor and testable core for local bwrap slot allocation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
import logging
import os
import time
from typing import Any

from dressage.sandbox.local.bwrap.supervisor import (
    POOL_BLACKBOX,
    LocalBwrapNodeSupervisor,
    PoolMode,
    normalize_pool_mode,
)

try:  # Ray is optional for local unit tests.
    import ray
except ImportError:  # pragma: no cover - exercised only in envs without Ray
    ray = None

logger = logging.getLogger(__name__)

LEASE_ACTIVE = "ACTIVE"
LEASE_RELEASING = "RELEASING"
LEASE_RELEASED = "RELEASED"
LEASE_EXPIRED = "EXPIRED"
LEASE_LOST = "LOST"


@dataclass(slots=True)
class NodeRecord:
    node_id: str
    node_ip: str
    capacity: int
    supervisor: Any
    hostname: str | None = None
    alive: bool = True
    used: int = 0
    free: int = 0
    ready: int = 0
    leased: int = 0
    resetting: int = 0
    restarting: int = 0
    failed: int = 0
    lost: int = 0
    pending_acquires: int = 0
    last_heartbeat_ts: float = 0.0
    draining: bool = False
    last_error: str | None = None

    def update_from_health(self, payload: dict[str, Any]) -> None:
        self.node_ip = str(payload.get("node_ip") or self.node_ip)
        self.hostname = payload.get("hostname") or self.hostname
        self.capacity = int(payload.get("capacity") or self.capacity)
        self.ready = int(payload.get("ready") or 0)
        self.leased = int(payload.get("leased") or 0)
        self.resetting = int(payload.get("resetting") or 0)
        self.restarting = int(payload.get("restarting") or 0)
        self.failed = int(payload.get("failed") or 0)
        self.lost = int(payload.get("lost") or 0)
        self.used = self.leased + self.resetting + self.restarting
        self.free = self.ready
        self.alive = True
        self.last_heartbeat_ts = time.time()
        self.last_error = payload.get("last_error")

    def mark_lost(self, exc: BaseException) -> None:
        self.alive = False
        self.free = 0
        self.ready = 0
        self.last_error = _exception_summary(exc)
        self.last_heartbeat_ts = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "hostname": self.hostname,
            "capacity": self.capacity,
            "alive": self.alive,
            "ready": self.ready,
            "leased": self.leased,
            "used": self.used,
            "free": self.free,
            "resetting": self.resetting,
            "restarting": self.restarting,
            "failed": self.failed,
            "lost": self.lost,
            "pending_acquires": self.pending_acquires,
            "draining": self.draining,
            "last_heartbeat_ts": self.last_heartbeat_ts,
            "last_error": self.last_error,
        }


@dataclass(slots=True)
class LeaseRecord:
    lease_id: str
    trajectory_id: str
    node_id: str
    node_ip: str
    slot_id: int
    port: int
    sandbox_url: str | None
    acquired_ts: float
    deadline_ts: float | None
    generation: int
    status: str = LEASE_ACTIVE

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        lease_ttl_sec: float | None,
    ) -> "LeaseRecord":
        now = time.time()
        return cls(
            lease_id=str(payload["lease_id"]),
            trajectory_id=str(payload["trajectory_id"]),
            node_id=str(payload["node_id"]),
            node_ip=str(payload["node_ip"]),
            slot_id=int(payload["slot_id"]),
            port=int(payload["port"]),
            sandbox_url=(
                str(payload["sandbox_url"]).rstrip("/")
                if payload.get("sandbox_url") is not None
                else None
            ),
            acquired_ts=now,
            deadline_ts=None if lease_ttl_sec is None else now + lease_ttl_sec,
            generation=int(payload.get("generation") or 0),
            status=LEASE_ACTIVE,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "trajectory_id": self.trajectory_id,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "slot_id": self.slot_id,
            "port": self.port,
            "sandbox_url": self.sandbox_url,
            "acquired_ts": self.acquired_ts,
            "deadline_ts": self.deadline_ts,
            "generation": self.generation,
            "status": self.status,
            "ready": self.status == LEASE_ACTIVE,
        }


class LocalBwrapClusterManagerCore:
    """Testable slot allocator used by the detached Ray manager actor."""

    def __init__(
        self,
        *,
        total_servers: int | None = None,
        base_port: int = 31000,
        acquire_timeout_sec: float | None = None,
        acquire_poll_interval_sec: float | None = None,
        lease_ttl_sec: float | None = None,
        status_refresh_interval_sec: float | None = None,
        namespace: str | None = None,
        proxy_url: str | None = None,
        pool_mode: str | None = None,
    ) -> None:
        self.pool_mode: PoolMode = normalize_pool_mode(
            pool_mode or os.environ.get("DRESSAGE_LOCAL_BWRAP_POOL_MODE", POOL_BLACKBOX)
        )
        self.total_servers = total_servers
        self.base_port = base_port
        self.acquire_timeout_sec = _env_float(
            "DRESSAGE_BLACKBOX_ACQUIRE_TIMEOUT_SEC",
            1800.0 if acquire_timeout_sec is None else acquire_timeout_sec,
            min_value=0.0,
        )
        self.acquire_poll_interval_sec = _env_float(
            "DRESSAGE_BLACKBOX_ACQUIRE_POLL_SEC",
            0.25 if acquire_poll_interval_sec is None else acquire_poll_interval_sec,
            min_value=0.0,
        )
        self.lease_ttl_sec = lease_ttl_sec
        if self.lease_ttl_sec is None:
            env_ttl = os.environ.get("DRESSAGE_BLACKBOX_LEASE_TTL_SEC")
            self.lease_ttl_sec = float(env_ttl) if env_ttl else 2100.0
        if self.lease_ttl_sec <= 0:
            self.lease_ttl_sec = None
        self.status_refresh_interval_sec = _env_float(
            "DRESSAGE_BLACKBOX_STATUS_REFRESH_SEC",
            (
                2.0
                if status_refresh_interval_sec is None
                else status_refresh_interval_sec
            ),
            min_value=0.0,
        )
        self.namespace = namespace or os.environ.get(
            "DRESSAGE_LOCAL_BWRAP_RAY_NAMESPACE",
            "dressage",
        )
        self.proxy_url = proxy_url
        self.nodes: dict[str, NodeRecord] = {}
        self.leases: dict[str, LeaseRecord] = {}
        self._leases_by_id: dict[str, str] = {}
        self._last_refresh_ts = 0.0
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._lock = asyncio.Lock()
        self._closed = False

    async def add_supervisor(
        self,
        *,
        node_id: str,
        node_ip: str,
        capacity: int,
        supervisor: Any,
        hostname: str | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("local_bwrap cluster manager is shut down")
        self.nodes[node_id] = NodeRecord(
            node_id=node_id,
            node_ip=node_ip,
            capacity=capacity,
            supervisor=supervisor,
            hostname=hostname,
        )
        payload = await self._refresh_node(self.nodes[node_id])
        if payload is not None and payload.get("pool_mode") is not None:
            node_mode = payload.get("pool_mode")
            if normalize_pool_mode(str(node_mode)) != self.pool_mode:
                self.nodes.pop(node_id, None)
                raise RuntimeError(
                    "local_bwrap supervisor pool mode mismatch: "
                    f"manager={self.pool_mode!r} supervisor={node_mode!r}"
                )
        return self.nodes[node_id].to_dict()

    async def acquire(
        self,
        trajectory_id: str,
        env_type: str | None = None,
        env_args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.acquire_timeout_sec
        env_args = env_args or {}
        while True:
            if self._closed:
                raise RuntimeError("local_bwrap cluster manager is shut down")
            node: NodeRecord | None = None
            async with self._lock:
                if self._closed:
                    raise RuntimeError("local_bwrap cluster manager is shut down")
                await self._reconcile_locked()
                existing = self.leases.get(trajectory_id)
                if existing is not None and existing.status == LEASE_ACTIVE:
                    return existing.to_dict()
                if existing is not None and existing.status == LEASE_RELEASING:
                    # The same trajectory may retry while its previous slot is
                    # still being reset. Do not lease another slot for it until
                    # the background release has completed and removed the lease.
                    node = None
                else:
                    if existing is not None:
                        self._drop_lease_locked(existing)
                    await self._refresh_nodes_if_needed_locked(force=False)
                    node = self._select_node_locked()
                    if node is not None:
                        node.pending_acquires += 1

            if node is not None:
                try:
                    payload = await _remote_call(
                        node.supervisor,
                        "acquire",
                        trajectory_id=trajectory_id,
                        env_type=env_type,
                        env_args=env_args,
                    )
                except asyncio.CancelledError:
                    async with self._lock:
                        current = self.nodes.get(node.node_id)
                        if current is not None:
                            current.pending_acquires = max(
                                0, current.pending_acquires - 1
                            )
                    raise
                except Exception as exc:
                    async with self._lock:
                        current = self.nodes.get(node.node_id)
                        if current is not None:
                            current.pending_acquires = max(
                                0, current.pending_acquires - 1
                            )
                            current.mark_lost(exc)
                    logger.warning(
                        "local_bwrap supervisor acquire failed on node_id=%s: %s",
                        node.node_id,
                        _exception_summary(exc),
                    )
                else:
                    lease = LeaseRecord.from_payload(
                        payload,
                        lease_ttl_sec=self.lease_ttl_sec,
                    )
                    cleanup_acquired_slot = False
                    return_existing: dict[str, Any] | None = None
                    manager_closed = False
                    async with self._lock:
                        current = self.nodes.get(node.node_id)
                        if current is not None:
                            current.pending_acquires = max(
                                0, current.pending_acquires - 1
                            )
                        if self._closed:
                            cleanup_acquired_slot = True
                            manager_closed = True
                        elif current is None or not current.alive:
                            cleanup_acquired_slot = True
                        else:
                            existing = self.leases.get(trajectory_id)
                            if existing is not None and existing.status == LEASE_ACTIVE:
                                cleanup_acquired_slot = True
                                return_existing = existing.to_dict()
                            elif existing is not None and existing.status == LEASE_RELEASING:
                                cleanup_acquired_slot = True
                            else:
                                if existing is not None:
                                    self._drop_lease_locked(existing)
                                self.leases[trajectory_id] = lease
                                self._leases_by_id[lease.lease_id] = trajectory_id
                                current.used += 1
                                current.leased += 1
                                current.free = max(0, current.free - 1)
                                current.ready = max(0, current.ready - 1)
                                return lease.to_dict()

                    if cleanup_acquired_slot:
                        await self._release_uncommitted_acquire(
                            node=node,
                            payload=payload,
                            reason="manager_acquire_not_committed",
                        )
                    if return_existing is not None:
                        return return_existing
                    if manager_closed:
                        raise RuntimeError("local_bwrap cluster manager is shut down")

            if time.monotonic() >= deadline:
                raise TimeoutError(f"no local_bwrap {self.pool_mode} slot available")
            await asyncio.sleep(self.acquire_poll_interval_sec)

    async def release(
        self,
        trajectory_id: str | None = None,
        lease_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            lease = self._lookup_lease_locked(trajectory_id, lease_id)
            if lease is None:
                return {
                    "released": True,
                    "already_released": True,
                    "trajectory_id": trajectory_id,
                    "lease_id": lease_id,
                }
            if self._closed:
                self._drop_lease_locked(lease)
                return {
                    "released": True,
                    "manager_closed": True,
                    "already_released": False,
                    "trajectory_id": lease.trajectory_id,
                    "lease_id": lease.lease_id,
                    "slot_id": lease.slot_id,
                    "node_id": lease.node_id,
                    "node_ip": lease.node_ip,
                    "slot_reusable": False,
                }
            if lease.status == LEASE_RELEASING:
                return {
                    "released": True,
                    "release_queued": True,
                    "already_releasing": True,
                    "trajectory_id": lease.trajectory_id,
                    "lease_id": lease.lease_id,
                    "slot_id": lease.slot_id,
                    "node_id": lease.node_id,
                    "node_ip": lease.node_ip,
                    "slot_reusable": False,
                }

            node = self.nodes.get(lease.node_id)
            lease.status = LEASE_RELEASING

        if node is None:
            async with self._lock:
                if self.leases.get(lease.trajectory_id) is lease:
                    lease.status = LEASE_LOST
                    self._drop_lease_locked(lease)
            return {
                "released": True,
                "node_lost": True,
                "trajectory_id": lease.trajectory_id,
                "lease_id": lease.lease_id,
                "slot_reusable": False,
            }

        task = asyncio.create_task(
            self._release_lease_in_background(
                lease=lease,
                node_id=node.node_id,
                reason=reason or "manager_release",
            )
        )
        self._track_background_task(task)
        return {
            "released": True,
            "release_queued": True,
            "already_released": False,
            "trajectory_id": lease.trajectory_id,
            "lease_id": lease.lease_id,
            "slot_id": lease.slot_id,
            "node_id": node.node_id,
            "node_ip": node.node_ip,
            "slot_reusable": False,
        }

    async def run_command(
        self,
        *,
        trajectory_id: str | None = None,
        lease_id: str | None = None,
        command: str | list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        stdin: str | bytes | None = None,
    ) -> dict[str, Any]:
        lease, node = await self._active_lease_and_node(trajectory_id, lease_id)
        return await _remote_call(
            node.supervisor,
            "run_command",
            lease_id=lease.lease_id,
            trajectory_id=lease.trajectory_id,
            command=command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            stdin=stdin,
        )

    async def read_file(
        self,
        *,
        trajectory_id: str | None = None,
        lease_id: str | None = None,
        path: str,
        encoding: str | None = "utf-8",
        max_bytes: int | None = None,
    ) -> str | bytes:
        lease, node = await self._active_lease_and_node(trajectory_id, lease_id)
        return await _remote_call(
            node.supervisor,
            "read_file",
            lease_id=lease.lease_id,
            trajectory_id=lease.trajectory_id,
            path=path,
            encoding=encoding,
            max_bytes=max_bytes,
        )

    async def write_file(
        self,
        *,
        trajectory_id: str | None = None,
        lease_id: str | None = None,
        path: str,
        content: str | bytes,
        encoding: str | None = "utf-8",
        append: bool = False,
    ) -> dict[str, Any]:
        lease, node = await self._active_lease_and_node(trajectory_id, lease_id)
        return await _remote_call(
            node.supervisor,
            "write_file",
            lease_id=lease.lease_id,
            trajectory_id=lease.trajectory_id,
            path=path,
            content=content,
            encoding=encoding,
            append=append,
        )

    async def wait_ready(self, timeout_s: int | float = 600) -> dict[str, Any]:
        target = self.total_servers
        deadline = time.monotonic() + timeout_s
        while True:
            status = await self.status(force_refresh=True)
            effective_target = target or status["total_capacity"]
            if status["total_capacity"] > 0:
                required_ready = min(effective_target, status["total_capacity"])
            else:
                required_ready = effective_target
            if status["total_ready"] >= required_ready:
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "local_bwrap pool did not reach ready target "
                    f"{effective_target} before timeout; status={status}"
                )
            await asyncio.sleep(1.0)

    async def status(self, *, force_refresh: bool = False) -> dict[str, Any]:
        self._reap_background_tasks()
        async with self._lock:
            if self._closed:
                return self._status_locked()
            await self._reconcile_locked()
            await self._refresh_nodes_if_needed_locked(force=force_refresh)
            return self._status_locked()

    async def reconcile(self) -> dict[str, Any]:
        async with self._lock:
            if self._closed:
                return self._status_locked()
            await self._reconcile_locked()
            return self._status_locked()

    async def drain_node(self, node_id: str) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("local_bwrap cluster manager is shut down")
        async with self._lock:
            node = self.nodes[node_id]
            node.draining = True
        await _remote_call(node.supervisor, "drain")
        return node.to_dict()

    async def undrain_node(self, node_id: str) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("local_bwrap cluster manager is shut down")
        async with self._lock:
            node = self.nodes[node_id]
            node.draining = False
        await _remote_call(node.supervisor, "undrain")
        return node.to_dict()

    async def restart_node_pool(self, node_id: str) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("local_bwrap cluster manager is shut down")
        node = self.nodes[node_id]
        await _remote_call(node.supervisor, "start_pool")
        await self._refresh_node(node)
        return node.to_dict()

    async def shutdown(self) -> dict[str, Any]:
        self._closed = True
        tasks = tuple(self._background_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()
        results: dict[str, Any] = {}
        for node_id, node in self.nodes.items():
            try:
                results[node_id] = await _remote_call(node.supervisor, "shutdown")
            except Exception as exc:
                results[node_id] = {"stopped": False, "error": _exception_summary(exc)}
            node.alive = False
            node.free = 0
            node.ready = 0
            node.used = 0
            node.leased = 0
            node.resetting = 0
            node.restarting = 0
        for lease in list(self.leases.values()):
            lease.status = LEASE_LOST
            self._drop_lease_locked(lease)
        return {"stopped": True, "closed": True, "nodes": results}

    def supervisor_handles(self) -> dict[str, Any]:
        return {node_id: node.supervisor for node_id, node in self.nodes.items()}

    def planned_capacities(
        self,
        candidates: list[tuple[dict[str, Any], int]],
    ) -> list[tuple[dict[str, Any], int]]:
        total_available = sum(capacity for _, capacity in candidates)
        if not candidates or total_available <= 0:
            return []
        if self.total_servers is None or self.total_servers >= total_available:
            return [(node, capacity) for node, capacity in candidates if capacity > 0]

        target = max(0, self.total_servers)
        planned: list[tuple[dict[str, Any], int]] = []
        fractional: list[tuple[float, int]] = []
        assigned = 0
        for index, (node, capacity) in enumerate(candidates):
            raw = target * capacity / total_available
            count = min(capacity, int(raw))
            planned.append((node, count))
            assigned += count
            fractional.append((raw - count, index))

        for _, index in sorted(fractional, reverse=True):
            if assigned >= target:
                break
            node, count = planned[index]
            capacity = candidates[index][1]
            if count < capacity:
                planned[index] = (node, count + 1)
                assigned += 1
        return [(node, capacity) for node, capacity in planned if capacity > 0]

    def _track_background_task(self, task: asyncio.Task[Any]) -> None:
        if self._closed:
            task.cancel()
            return
        self._background_tasks.add(task)

        def _done(done: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(done)
            try:
                done.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning(
                    "local_bwrap release background task failed: %s",
                    _exception_summary(exc),
                )

        task.add_done_callback(_done)

    def _reap_background_tasks(self) -> None:
        for task in tuple(self._background_tasks):
            if not task.done():
                continue
            self._background_tasks.discard(task)
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning(
                    "local_bwrap release background task failed: %s",
                    _exception_summary(exc),
                )

    async def _release_uncommitted_acquire(
        self,
        *,
        node: NodeRecord,
        payload: dict[str, Any],
        reason: str,
    ) -> None:
        try:
            await _remote_call(
                node.supervisor,
                "release",
                lease_id=payload.get("lease_id"),
                trajectory_id=payload.get("trajectory_id"),
                reason=reason,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort leak prevention
            logger.warning(
                "local_bwrap cleanup for uncommitted acquire failed on node_id=%s: %s",
                node.node_id,
                _exception_summary(exc),
            )

    async def _release_lease_in_background(
        self,
        *,
        lease: LeaseRecord,
        node_id: str,
        reason: str,
    ) -> None:
        if self._closed:
            return
        async with self._lock:
            node = self.nodes.get(node_id)
        if node is None:
            async with self._lock:
                if self.leases.get(lease.trajectory_id) is lease:
                    lease.status = LEASE_LOST
                    self._drop_lease_locked(lease)
            return

        release_error: BaseException | None = None
        try:
            await _remote_call(
                node.supervisor,
                "release",
                lease_id=lease.lease_id,
                trajectory_id=lease.trajectory_id,
                reason=reason,
            )
        except Exception as exc:
            release_error = exc
            logger.warning(
                "local_bwrap supervisor release failed on node_id=%s: %s",
                node.node_id,
                _exception_summary(exc),
            )

        if release_error is not None:
            async with self._lock:
                current = self.nodes.get(node_id)
                if current is not None:
                    current.mark_lost(release_error)
                if self.leases.get(lease.trajectory_id) is lease:
                    lease.status = LEASE_LOST
            return

        deadline = time.monotonic() + _env_float(
            "DRESSAGE_BLACKBOX_BACKGROUND_RELEASE_TIMEOUT_SEC",
            600.0,
            min_value=1.0,
        )
        last_payload: dict[str, Any] | None = None
        last_error: BaseException | None = None
        while True:
            if self._closed:
                return
            try:
                last_payload = await _remote_call(node.supervisor, "health")
                last_error = None
            except Exception as exc:
                last_error = exc
                last_payload = None

            if last_payload is not None:
                async with self._lock:
                    current = self.nodes.get(node_id)
                    if current is not None:
                        current.update_from_health(last_payload)
                if _slot_release_complete(last_payload, lease):
                    async with self._lock:
                        if self.leases.get(lease.trajectory_id) is lease:
                            self._drop_lease_locked(lease)
                    return

            if time.monotonic() >= deadline:
                async with self._lock:
                    current = self.nodes.get(node_id)
                    if current is not None:
                        if last_payload is not None:
                            current.update_from_health(last_payload)
                        elif last_error is not None:
                            current.mark_lost(last_error)
                    if self.leases.get(lease.trajectory_id) is lease:
                        lease.status = LEASE_LOST
                logger.warning(
                    "local_bwrap background release timed out for trajectory_id=%s "
                    "lease_id=%s node_id=%s slot_id=%s",
                    lease.trajectory_id,
                    lease.lease_id,
                    node_id,
                    lease.slot_id,
                )
                return

            await asyncio.sleep(0.25)

    async def _refresh_nodes_if_needed_locked(self, *, force: bool) -> None:
        if self._closed:
            return
        if not force:
            elapsed = time.time() - self._last_refresh_ts
            if elapsed < self.status_refresh_interval_sec:
                return
        for node in self.nodes.values():
            await self._refresh_node(node)
        self._last_refresh_ts = time.time()

    async def _refresh_node(self, node: NodeRecord) -> dict[str, Any] | None:
        try:
            payload = await _remote_call(node.supervisor, "health")
        except Exception as exc:
            node.mark_lost(exc)
            for lease in self.leases.values():
                if lease.node_id == node.node_id and lease.status in {LEASE_ACTIVE, LEASE_RELEASING}:
                    lease.status = LEASE_LOST
            return None
        node.update_from_health(payload)
        return payload

    async def _reconcile_locked(self) -> None:
        if self._closed:
            return
        now = time.time()
        expired = [
            lease
            for lease in self.leases.values()
            if lease.status == LEASE_ACTIVE
            and lease.deadline_ts is not None
            and lease.deadline_ts <= now
        ]
        for lease in expired:
            lease.status = LEASE_EXPIRED
            node = self.nodes.get(lease.node_id)
            self._drop_lease_locked(lease)
            if node is not None:
                try:
                    await _remote_call(
                        node.supervisor,
                        "force_release",
                        slot_id=lease.slot_id,
                        lease_id=lease.lease_id,
                        reason="lease_ttl_expired",
                    )
                except Exception as exc:
                    node.mark_lost(exc)

    async def _active_lease_and_node(
        self,
        trajectory_id: str | None,
        lease_id: str | None,
    ) -> tuple[LeaseRecord, NodeRecord]:
        async with self._lock:
            lease = self._lookup_lease_locked(trajectory_id, lease_id)
            if lease is None or lease.status != LEASE_ACTIVE:
                raise KeyError(
                    "active local_bwrap lease not found "
                    f"trajectory_id={trajectory_id!r} lease_id={lease_id!r}"
                )
            node = self.nodes.get(lease.node_id)
            if node is None or not node.alive:
                raise RuntimeError(
                    f"node for local_bwrap lease is unavailable: node_id={lease.node_id}"
                )
            return lease, node

    def _lookup_lease_locked(
        self,
        trajectory_id: str | None,
        lease_id: str | None,
    ) -> LeaseRecord | None:
        if trajectory_id is not None and trajectory_id in self.leases:
            return self.leases[trajectory_id]
        if lease_id is not None:
            trajectory = self._leases_by_id.get(lease_id)
            if trajectory is not None:
                return self.leases.get(trajectory)
        return None

    def _drop_lease_locked(self, lease: LeaseRecord) -> None:
        self.leases.pop(lease.trajectory_id, None)
        self._leases_by_id.pop(lease.lease_id, None)
        lease.status = LEASE_RELEASED

    def _select_node_locked(self) -> NodeRecord | None:
        candidates = [
            node
            for node in self.nodes.values()
            if node.alive
            and not node.draining
            and node.free - node.pending_acquires > 0
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda node: (
                (node.used + node.pending_acquires) / max(node.capacity, 1),
                -(node.free - node.pending_acquires),
                node.node_id,
            ),
        )

    def _status_locked(self) -> dict[str, Any]:
        nodes = [node.to_dict() for node in self.nodes.values()]
        active = [
            lease for lease in self.leases.values() if lease.status == LEASE_ACTIVE
        ]
        releasing = [
            lease for lease in self.leases.values() if lease.status == LEASE_RELEASING
        ]
        return {
            "closed": self._closed,
            "pool_mode": self.pool_mode,
            "total_capacity": sum(node["capacity"] for node in nodes),
            "total_ready": sum(node["ready"] for node in nodes),
            "total_leased": sum(node["leased"] for node in nodes),
            "total_resetting": sum(node.get("resetting", 0) for node in nodes),
            "total_restarting": sum(node["restarting"] for node in nodes),
            "total_failed": sum(node["failed"] for node in nodes),
            "total_lost": sum(node["lost"] for node in nodes),
            "nodes": nodes,
            "leases": {
                "active": len(active),
                "releasing": len(releasing),
                "tracked": len(self.leases),
                "expired": sum(
                    1 for lease in self.leases.values() if lease.status == LEASE_EXPIRED
                ),
                "lost": sum(
                    1 for lease in self.leases.values() if lease.status == LEASE_LOST
                ),
            },
        }


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
    if ray is None:
        raise ImportError("ray is required to resolve Ray object references")
    if hasattr(obj_ref, "__await__"):
        try:
            return await obj_ref
        except TypeError:
            pass
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, ray.get, obj_ref)


def _slot_release_complete(payload: dict[str, Any], lease: LeaseRecord) -> bool:
    slots = payload.get("slots")
    if not isinstance(slots, list):
        # Older or fake supervisors do not report per-slot state. In that case,
        # absence of leased/resetting/restarting capacity is the best available
        # completion signal.
        return (
            int(payload.get("leased") or 0) == 0
            and int(payload.get("resetting") or 0) == 0
            and int(payload.get("restarting") or 0) == 0
        )
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        if int(slot.get("slot_id", -1)) != lease.slot_id:
            continue
        status = str(slot.get("status") or "").upper()
        lease_id = slot.get("lease_id")
        trajectory_id = slot.get("trajectory_id")
        if status == "READY" and not lease_id and not trajectory_id:
            return True
        return False
    return False

def _node_id(node: dict[str, Any]) -> str:
    return str(node.get("NodeID") or node.get("node_id") or node.get("NodeId"))


def _node_ip(node: dict[str, Any]) -> str:
    return str(
        node.get("NodeManagerAddress")
        or node.get("node_ip")
        or node.get("NodeManagerHostname")
        or "127.0.0.1"
    )


def _node_hostname(node: dict[str, Any]) -> str | None:
    value = node.get("NodeName") or node.get("NodeManagerHostname")
    return str(value) if value else None


def _exception_summary(exc: BaseException) -> str:
    return " ".join(str(exc).splitlines()) or type(exc).__name__


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


if ray is not None:

    @ray.remote(num_cpus=0.2, num_gpus=0, max_concurrency=4096)
    class LocalBwrapClusterManager:
        def __init__(
            self,
            *,
            total_servers: int | None = None,
            base_port: int = 31000,
            proxy_url: str | None = None,
            namespace: str | None = None,
            acquire_timeout_sec: float | None = None,
            acquire_poll_interval_sec: float | None = None,
            lease_ttl_sec: float | None = None,
            pool_mode: str | None = None,
        ) -> None:
            self.namespace = namespace or os.environ.get(
                "DRESSAGE_LOCAL_BWRAP_RAY_NAMESPACE",
                "dressage",
            )
            self.core = LocalBwrapClusterManagerCore(
                total_servers=total_servers,
                base_port=base_port,
                proxy_url=proxy_url,
                namespace=self.namespace,
                acquire_timeout_sec=acquire_timeout_sec,
                acquire_poll_interval_sec=acquire_poll_interval_sec,
                lease_ttl_sec=lease_ttl_sec,
                pool_mode=pool_mode,
            )

        async def pool_mode(self) -> str:
            return self.core.pool_mode

        async def init_pool(self) -> dict[str, Any]:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            if self.core._closed:
                raise RuntimeError(
                    "local_bwrap cluster manager is shut down; destroy the detached "
                    "actor and create a new one"
                )

            candidates: list[tuple[dict[str, Any], int]] = []
            for node in ray.nodes():
                if not node.get("Alive"):
                    continue
                resources = node.get("Resources", {})
                slots = int(resources.get("local_bwrap_slots", 0))
                if slots > 0:
                    candidates.append((node, slots))

            for node, capacity in self.core.planned_capacities(candidates):
                node_id = _node_id(node)
                node_ip = _node_ip(node)
                supervisor_name = (
                    "dressage_local_bwrap_supervisor_"
                    f"{_safe_name(self.core.pool_mode)[:16]}_"
                    f"{_safe_name(node_id)[:20]}"
                )
                supervisor = LocalBwrapNodeSupervisor.options(
                    name=supervisor_name,
                    namespace=self.namespace,
                    lifetime="detached",
                    get_if_exists=True,
                    num_cpus=0,
                    num_gpus=0,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=False,
                    ),
                ).remote(
                    node_id=node_id,
                    node_ip=node_ip,
                    capacity=capacity,
                    base_port=self.core.base_port,
                    pool_mode=self.core.pool_mode,
                )
                await _ray_get(supervisor.start_pool.remote())
                await self.core.add_supervisor(
                    node_id=node_id,
                    node_ip=node_ip,
                    capacity=capacity,
                    supervisor=supervisor,
                    hostname=_node_hostname(node),
                )
            return await self.core.status(force_refresh=True)

        async def init_cluster(self) -> dict[str, Any]:
            return await self.init_pool()

        async def wait_ready(self, timeout_s: int | float = 600) -> dict[str, Any]:
            return await self.core.wait_ready(timeout_s=timeout_s)

        async def acquire(
            self,
            trajectory_id: str,
            env_type: str | None = None,
            env_args: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            return await self.core.acquire(
                trajectory_id=trajectory_id,
                env_type=env_type,
                env_args=env_args or {},
            )

        async def release(
            self,
            trajectory_id: str | None = None,
            lease_id: str | None = None,
            reason: str | None = None,
        ) -> dict[str, Any]:
            return await self.core.release(
                trajectory_id=trajectory_id,
                lease_id=lease_id,
                reason=reason,
            )

        async def run_command(
            self,
            *,
            trajectory_id: str | None = None,
            lease_id: str | None = None,
            command: str | list[str],
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout: float | None = None,
            stdin: str | bytes | None = None,
        ) -> dict[str, Any]:
            return await self.core.run_command(
                trajectory_id=trajectory_id,
                lease_id=lease_id,
                command=command,
                cwd=cwd,
                env=env,
                timeout=timeout,
                stdin=stdin,
            )

        async def read_file(
            self,
            *,
            trajectory_id: str | None = None,
            lease_id: str | None = None,
            path: str,
            encoding: str | None = "utf-8",
            max_bytes: int | None = None,
        ) -> str | bytes:
            return await self.core.read_file(
                trajectory_id=trajectory_id,
                lease_id=lease_id,
                path=path,
                encoding=encoding,
                max_bytes=max_bytes,
            )

        async def write_file(
            self,
            *,
            trajectory_id: str | None = None,
            lease_id: str | None = None,
            path: str,
            content: str | bytes,
            encoding: str | None = "utf-8",
            append: bool = False,
        ) -> dict[str, Any]:
            return await self.core.write_file(
                trajectory_id=trajectory_id,
                lease_id=lease_id,
                path=path,
                content=content,
                encoding=encoding,
                append=append,
            )

        async def status(self) -> dict[str, Any]:
            return await self.core.status(force_refresh=True)

        async def reconcile(self) -> dict[str, Any]:
            return await self.core.reconcile()

        async def drain_node(self, node_id: str) -> dict[str, Any]:
            return await self.core.drain_node(node_id)

        async def undrain_node(self, node_id: str) -> dict[str, Any]:
            return await self.core.undrain_node(node_id)

        async def restart_node_pool(self, node_id: str) -> dict[str, Any]:
            return await self.core.restart_node_pool(node_id)

        async def shutdown(self, destroy_supervisors: bool | None = None) -> dict[str, Any]:
            if destroy_supervisors is None:
                destroy_supervisors = _env_bool(
                    "DRESSAGE_LOCAL_BWRAP_DESTROY_ACTORS_ON_STOP", True
                )
            supervisors = self.core.supervisor_handles()
            result = await self.core.shutdown()
            destroyed: dict[str, bool] = {}
            if destroy_supervisors:
                for node_id, supervisor in supervisors.items():
                    try:
                        ray.kill(supervisor, no_restart=True)
                    except Exception as exc:
                        destroyed[node_id] = False
                        result.setdefault("destroy_errors", {})[node_id] = _exception_summary(
                            exc
                        )
                    else:
                        destroyed[node_id] = True
            result["supervisor_actors_destroyed"] = destroyed
            return result

else:

    class _MissingRayActor:
        def options(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise ImportError("ray is required to create LocalBwrapClusterManager actors")

    LocalBwrapClusterManager = _MissingRayActor()


def _safe_name(value: Any) -> str:
    text = str(value)
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    return "".join(ch if ch in allowed else "_" for ch in text).strip("._") or "node"
