from __future__ import annotations

import asyncio

import pytest

from dressage.sandbox.local.bwrap.manager import LocalBwrapClusterManagerCore


class FakeSupervisor:
    def __init__(
        self,
        *,
        node_id: str,
        node_ip: str,
        capacity: int,
        ready: int,
        leased: int = 0,
    ) -> None:
        self.node_id = node_id
        self.node_ip = node_ip
        self.capacity = capacity
        self.ready = ready
        self.leased = leased
        self.acquire_calls: list[str] = []
        self.release_calls: list[tuple[str | None, str | None]] = []
        self.force_release_calls: list[tuple[int, str | None]] = []
        self.shutdown_calls = 0

    async def health(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "hostname": self.node_id,
            "capacity": self.capacity,
            "ready": self.ready,
            "leased": self.leased,
            "restarting": 0,
            "failed": 0,
            "lost": 0,
            "last_error": None,
        }

    async def acquire(self, trajectory_id: str, env_type=None, env_args=None) -> dict:
        del env_type, env_args
        self.acquire_calls.append(trajectory_id)
        if self.ready <= 0:
            raise TimeoutError("no slot")
        slot_id = self.leased
        self.ready -= 1
        self.leased += 1
        return {
            "lease_id": f"lease-{self.node_id}-{slot_id}-{trajectory_id}",
            "trajectory_id": trajectory_id,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "slot_id": slot_id,
            "port": 31000 + slot_id,
            "sandbox_url": f"http://{self.node_ip}:{31000 + slot_id}",
            "generation": 1,
            "ready": True,
        }

    async def release(
        self,
        lease_id: str | None = None,
        trajectory_id: str | None = None,
        reason: str | None = None,
    ) -> dict:
        del reason
        self.release_calls.append((lease_id, trajectory_id))
        self.leased = max(0, self.leased - 1)
        self.ready += 1
        return {
            "released": True,
            "lease_id": lease_id,
            "trajectory_id": trajectory_id,
            "node_id": self.node_id,
        }

    async def force_release(
        self,
        slot_id: int,
        lease_id: str | None = None,
        reason: str | None = None,
    ) -> dict:
        del reason
        self.force_release_calls.append((slot_id, lease_id))
        self.leased = max(0, self.leased - 1)
        self.ready += 1
        return {"released": True, "lease_id": lease_id, "slot_id": slot_id}

    async def drain(self) -> dict:
        return {"draining": True}

    async def undrain(self) -> dict:
        return {"draining": False}

    async def shutdown(self) -> dict:
        self.shutdown_calls += 1
        self.ready = 0
        self.leased = 0
        return {"node_id": self.node_id, "stopped": True}


def test_cluster_manager_default_acquire_timeout_is_longer(monkeypatch):
    monkeypatch.delenv("DRESSAGE_BLACKBOX_ACQUIRE_TIMEOUT_SEC", raising=False)

    manager = LocalBwrapClusterManagerCore()

    assert manager.acquire_timeout_sec == 1800.0


def test_cluster_manager_uses_least_loaded_with_free_tie_breaker():
    asyncio.run(_run_cluster_manager_uses_least_loaded_with_free_tie_breaker())


async def _run_cluster_manager_uses_least_loaded_with_free_tie_breaker():
    manager = LocalBwrapClusterManagerCore(
        acquire_timeout_sec=0.01,
        status_refresh_interval_sec=0,
    )
    small = FakeSupervisor(
        node_id="small", node_ip="10.0.0.1", capacity=4, ready=2, leased=2
    )
    large = FakeSupervisor(
        node_id="large", node_ip="10.0.0.2", capacity=8, ready=4, leased=4
    )
    await manager.add_supervisor(
        node_id="small", node_ip=small.node_ip, capacity=small.capacity, supervisor=small
    )
    await manager.add_supervisor(
        node_id="large", node_ip=large.node_ip, capacity=large.capacity, supervisor=large
    )

    lease = await manager.acquire("traj-1")

    assert lease["node_id"] == "large"
    assert large.acquire_calls == ["traj-1"]
    assert small.acquire_calls == []


def test_cluster_manager_acquire_and_release_are_idempotent():
    asyncio.run(_run_cluster_manager_acquire_and_release_are_idempotent())


async def _run_cluster_manager_acquire_and_release_are_idempotent():
    manager = LocalBwrapClusterManagerCore(
        acquire_timeout_sec=0.01,
        status_refresh_interval_sec=0,
    )
    supervisor = FakeSupervisor(
        node_id="node-a", node_ip="10.0.0.10", capacity=2, ready=2
    )
    await manager.add_supervisor(
        node_id="node-a",
        node_ip=supervisor.node_ip,
        capacity=supervisor.capacity,
        supervisor=supervisor,
    )

    first = await manager.acquire("traj-1")
    second = await manager.acquire("traj-1")
    released = await manager.release("traj-1", first["lease_id"])
    released_again = await manager.release("traj-1", first["lease_id"])

    assert second == first
    assert supervisor.acquire_calls == ["traj-1"]
    assert released["released"] is True
    assert released["release_queued"] is True
    assert released_again["already_releasing"] is True

    for _ in range(20):
        await asyncio.sleep(0.01)
        status = await manager.status(force_refresh=True)
        if status["leases"]["tracked"] == 0:
            break
    else:
        raise AssertionError("release did not complete")

    released_after_completion = await manager.release("traj-1", first["lease_id"])
    assert released_after_completion["already_released"] is True


def test_cluster_manager_release_is_not_blocked_by_inflight_supervisor_acquire():
    asyncio.run(_run_cluster_manager_release_is_not_blocked_by_inflight_supervisor_acquire())


async def _run_cluster_manager_release_is_not_blocked_by_inflight_supervisor_acquire():
    manager = LocalBwrapClusterManagerCore(
        acquire_timeout_sec=0.2,
        acquire_poll_interval_sec=0.001,
        status_refresh_interval_sec=999,
    )
    supervisor = BlockingAcquireSupervisor(
        node_id="node-a",
        node_ip="10.0.0.10",
        capacity=2,
        ready=2,
        blocked_trajectory="traj-slow",
    )
    await manager.add_supervisor(
        node_id="node-a",
        node_ip=supervisor.node_ip,
        capacity=supervisor.capacity,
        supervisor=supervisor,
    )

    held = await manager.acquire("traj-held")
    slow_acquire = asyncio.create_task(manager.acquire("traj-slow"))
    await asyncio.wait_for(supervisor.acquire_started.wait(), timeout=0.05)

    released = await asyncio.wait_for(
        manager.release("traj-held", held["lease_id"]), timeout=0.05
    )

    supervisor.acquire_can_finish.set()
    slow = await asyncio.wait_for(slow_acquire, timeout=0.1)
    assert released["release_queued"] is True
    assert slow["trajectory_id"] == "traj-slow"


def test_cluster_manager_cancelled_acquire_releases_pending_reservation():
    asyncio.run(_run_cluster_manager_cancelled_acquire_releases_pending_reservation())


async def _run_cluster_manager_cancelled_acquire_releases_pending_reservation():
    manager = LocalBwrapClusterManagerCore(
        acquire_timeout_sec=0.2,
        acquire_poll_interval_sec=0.001,
        status_refresh_interval_sec=999,
    )
    supervisor = BlockingAcquireSupervisor(
        node_id="node-a",
        node_ip="10.0.0.10",
        capacity=1,
        ready=1,
        blocked_trajectory="traj-slow",
    )
    await manager.add_supervisor(
        node_id="node-a",
        node_ip=supervisor.node_ip,
        capacity=supervisor.capacity,
        supervisor=supervisor,
    )

    acquire_task = asyncio.create_task(manager.acquire("traj-slow"))
    await asyncio.wait_for(supervisor.acquire_started.wait(), timeout=0.05)
    acquire_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await acquire_task

    status = await manager.status(force_refresh=False)
    assert status["nodes"][0]["pending_acquires"] == 0
    assert status["total_ready"] == 1


def test_cluster_manager_does_not_allocate_draining_node():
    asyncio.run(_run_cluster_manager_does_not_allocate_draining_node())


async def _run_cluster_manager_does_not_allocate_draining_node():
    manager = LocalBwrapClusterManagerCore(
        acquire_timeout_sec=0.01,
        status_refresh_interval_sec=0,
    )
    drained = FakeSupervisor(
        node_id="drained", node_ip="10.0.0.1", capacity=8, ready=8
    )
    fallback = FakeSupervisor(
        node_id="fallback", node_ip="10.0.0.2", capacity=1, ready=1
    )
    await manager.add_supervisor(
        node_id="drained",
        node_ip=drained.node_ip,
        capacity=drained.capacity,
        supervisor=drained,
    )
    await manager.add_supervisor(
        node_id="fallback",
        node_ip=fallback.node_ip,
        capacity=fallback.capacity,
        supervisor=fallback,
    )

    await manager.drain_node("drained")
    lease = await manager.acquire("traj-1")

    assert lease["node_id"] == "fallback"
    assert drained.acquire_calls == []


def test_cluster_manager_expires_leases():
    asyncio.run(_run_cluster_manager_expires_leases())


async def _run_cluster_manager_expires_leases():
    manager = LocalBwrapClusterManagerCore(
        acquire_timeout_sec=0.01,
        lease_ttl_sec=0.01,
        status_refresh_interval_sec=0,
    )
    supervisor = FakeSupervisor(
        node_id="node-a", node_ip="10.0.0.10", capacity=1, ready=1
    )
    await manager.add_supervisor(
        node_id="node-a",
        node_ip=supervisor.node_ip,
        capacity=supervisor.capacity,
        supervisor=supervisor,
    )

    lease = await manager.acquire("traj-ttl")
    await asyncio.sleep(0.02)
    status = await manager.reconcile()

    assert status["leases"]["active"] == 0
    assert supervisor.force_release_calls == [(lease["slot_id"], lease["lease_id"])]


class BlockingAcquireSupervisor(FakeSupervisor):
    def __init__(self, *, blocked_trajectory: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.blocked_trajectory = blocked_trajectory
        self.acquire_started = asyncio.Event()
        self.acquire_can_finish = asyncio.Event()

    async def acquire(self, trajectory_id: str, env_type=None, env_args=None) -> dict:
        del env_type, env_args
        self.acquire_calls.append(trajectory_id)
        if trajectory_id == self.blocked_trajectory:
            self.acquire_started.set()
            await self.acquire_can_finish.wait()
        if self.ready <= 0:
            raise TimeoutError("no slot")
        slot_id = self.leased
        self.ready -= 1
        self.leased += 1
        return {
            "lease_id": f"lease-{self.node_id}-{slot_id}-{trajectory_id}",
            "trajectory_id": trajectory_id,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "slot_id": slot_id,
            "port": 31000 + slot_id,
            "sandbox_url": f"http://{self.node_ip}:{31000 + slot_id}",
            "generation": 1,
            "ready": True,
        }


class BlockingHealthSupervisor(FakeSupervisor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.health_started = asyncio.Event()
        self.health_can_finish = asyncio.Event()
        self.block_health = True

    async def health(self) -> dict:
        if self.block_health:
            self.health_started.set()
            await self.health_can_finish.wait()
        return await super().health()


class SlowReleaseSupervisor(FakeSupervisor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.release_started = asyncio.Event()
        self.release_can_finish = asyncio.Event()

    async def release(
        self,
        lease_id: str | None = None,
        trajectory_id: str | None = None,
        reason: str | None = None,
    ) -> dict:
        del reason
        self.release_calls.append((lease_id, trajectory_id))
        self.release_started.set()
        await self.release_can_finish.wait()
        self.leased = max(0, self.leased - 1)
        self.ready += 1
        return {
            "released": True,
            "lease_id": lease_id,
            "trajectory_id": trajectory_id,
            "node_id": self.node_id,
        }


def test_cluster_manager_release_does_not_reuse_slot_until_background_reset_finishes():
    asyncio.run(
        _run_cluster_manager_release_does_not_reuse_slot_until_background_reset_finishes()
    )


async def _run_cluster_manager_release_does_not_reuse_slot_until_background_reset_finishes():
    manager = LocalBwrapClusterManagerCore(
        acquire_timeout_sec=0.02,
        acquire_poll_interval_sec=0.001,
        status_refresh_interval_sec=999,
    )
    supervisor = SlowReleaseSupervisor(
        node_id="node-a", node_ip="10.0.0.10", capacity=1, ready=1
    )
    await manager.add_supervisor(
        node_id="node-a",
        node_ip=supervisor.node_ip,
        capacity=supervisor.capacity,
        supervisor=supervisor,
    )

    lease = await manager.acquire("traj-1")
    released = await asyncio.wait_for(
        manager.release("traj-1", lease["lease_id"]), timeout=0.05
    )
    await asyncio.wait_for(supervisor.release_started.wait(), timeout=0.05)

    assert released["release_queued"] is True
    assert released["slot_reusable"] is False
    status_while_releasing = await manager.reconcile()
    assert status_while_releasing["leases"]["releasing"] == 1
    assert status_while_releasing["total_ready"] == 0

    try:
        await manager.acquire("traj-2")
    except TimeoutError:
        pass
    else:
        raise AssertionError("slot was reused before release cleanup finished")

    supervisor.release_can_finish.set()
    for _ in range(20):
        await asyncio.sleep(0.01)
        status = await manager.status(force_refresh=True)
        if status["leases"]["tracked"] == 0 and status["total_ready"] == 1:
            break
    else:
        raise AssertionError("slot did not become ready after background release")

    second = await manager.acquire("traj-2")
    assert second["trajectory_id"] == "traj-2"
    assert supervisor.release_calls == [(lease["lease_id"], "traj-1")]


def test_cluster_manager_release_is_not_blocked_by_slow_health_refresh():
    asyncio.run(_run_cluster_manager_release_is_not_blocked_by_slow_health_refresh())


async def _run_cluster_manager_release_is_not_blocked_by_slow_health_refresh():
    manager = LocalBwrapClusterManagerCore(
        acquire_timeout_sec=0.2,
        acquire_poll_interval_sec=0.001,
        status_refresh_interval_sec=0,
    )
    supervisor = BlockingHealthSupervisor(
        node_id="node-a", node_ip="10.0.0.10", capacity=2, ready=2
    )
    supervisor.block_health = False
    await manager.add_supervisor(
        node_id="node-a",
        node_ip=supervisor.node_ip,
        capacity=supervisor.capacity,
        supervisor=supervisor,
    )

    lease = await manager.acquire("traj-held")
    supervisor.block_health = True
    status_task = asyncio.create_task(manager.status(force_refresh=True))
    await asyncio.wait_for(supervisor.health_started.wait(), timeout=0.05)

    released = await asyncio.wait_for(
        manager.release("traj-held", lease["lease_id"]), timeout=0.05
    )

    supervisor.health_can_finish.set()
    await asyncio.wait_for(status_task, timeout=0.2)

    assert released["released"] is True
    assert released["release_queued"] is True
    assert released["slot_reusable"] is False


def test_cluster_manager_release_is_not_blocked_by_acquire_health_refresh():
    asyncio.run(_run_cluster_manager_release_is_not_blocked_by_acquire_health_refresh())


async def _run_cluster_manager_release_is_not_blocked_by_acquire_health_refresh():
    manager = LocalBwrapClusterManagerCore(
        acquire_timeout_sec=0.2,
        acquire_poll_interval_sec=0.001,
        status_refresh_interval_sec=0,
    )
    supervisor = BlockingHealthSupervisor(
        node_id="node-a", node_ip="10.0.0.10", capacity=2, ready=2
    )
    supervisor.block_health = False
    await manager.add_supervisor(
        node_id="node-a",
        node_ip=supervisor.node_ip,
        capacity=supervisor.capacity,
        supervisor=supervisor,
    )

    held = await manager.acquire("traj-held")
    supervisor.ready = 1
    supervisor.block_health = True
    acquire_task = asyncio.create_task(manager.acquire("traj-refresh"))
    await asyncio.wait_for(supervisor.health_started.wait(), timeout=0.05)

    released = await asyncio.wait_for(
        manager.release("traj-held", held["lease_id"]), timeout=0.05
    )

    supervisor.health_can_finish.set()
    acquired = await asyncio.wait_for(acquire_task, timeout=0.2)

    assert released["release_queued"] is True
    assert acquired["trajectory_id"] == "traj-refresh"


def test_cluster_manager_shutdown_stops_supervisors_and_blocks_acquire():
    asyncio.run(_run_cluster_manager_shutdown_stops_supervisors_and_blocks_acquire())


async def _run_cluster_manager_shutdown_stops_supervisors_and_blocks_acquire():
    manager = LocalBwrapClusterManagerCore(
        acquire_timeout_sec=0.01,
        status_refresh_interval_sec=0,
    )
    supervisor = FakeSupervisor(
        node_id="node-a", node_ip="10.0.0.10", capacity=1, ready=1
    )
    await manager.add_supervisor(
        node_id="node-a",
        node_ip=supervisor.node_ip,
        capacity=supervisor.capacity,
        supervisor=supervisor,
    )
    await manager.acquire("traj-1")

    shutdown = await manager.shutdown()
    status = await manager.status(force_refresh=True)

    assert shutdown["stopped"] is True
    assert shutdown["closed"] is True
    assert supervisor.shutdown_calls == 1
    assert status["closed"] is True
    assert status["total_ready"] == 0
    assert status["leases"]["tracked"] == 0
    with pytest.raises(RuntimeError, match="shut down"):
        await manager.acquire("traj-after-shutdown")


def test_cluster_manager_shutdown_is_idempotent():
    asyncio.run(_run_cluster_manager_shutdown_is_idempotent())


async def _run_cluster_manager_shutdown_is_idempotent():
    manager = LocalBwrapClusterManagerCore(status_refresh_interval_sec=0)
    supervisor = FakeSupervisor(
        node_id="node-a", node_ip="10.0.0.10", capacity=1, ready=1
    )
    await manager.add_supervisor(
        node_id="node-a",
        node_ip=supervisor.node_ip,
        capacity=supervisor.capacity,
        supervisor=supervisor,
    )

    first = await manager.shutdown()
    second = await manager.shutdown()

    assert first["stopped"] is True
    assert second["stopped"] is True
    assert supervisor.shutdown_calls == 2
