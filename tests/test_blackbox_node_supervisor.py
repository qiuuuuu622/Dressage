from __future__ import annotations

import asyncio
import json

import pytest

from dressage.sandbox.local.bwrap.supervisor import LocalBwrapNodeSupervisorCore


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode or 0


class FakeRunner:
    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0

    async def start(self, slot):
        self.starts += 1
        return FakeProcess(1000 + slot.config.slot_id + self.starts * 10)

    async def stop(self, slot, timeout_sec=None):
        del timeout_sec
        self.stops += 1
        if slot.process is not None:
            slot.process.terminate()
            await slot.process.wait()


class DeadRunner:
    async def start(self, slot):
        slot.config.ensure_dirs()
        (slot.config.log_dir / f"server-{slot.generation}.err").write_text(
            "child failed loudly\n"
        )
        proc = FakeProcess(2000 + slot.config.slot_id)
        proc.returncode = 127
        return proc

    async def stop(self, slot, timeout_sec=None):
        del slot, timeout_sec


class BlockingStopRunner(FakeRunner):
    def __init__(self) -> None:
        super().__init__()
        self.stop_started = asyncio.Event()
        self.stop_can_finish = asyncio.Event()

    async def stop(self, slot, timeout_sec=None):
        del timeout_sec
        self.stops += 1
        self.stop_started.set()
        await self.stop_can_finish.wait()
        if slot.process is not None:
            slot.process.terminate()
            await slot.process.wait()


class CountingBlockingStopRunner(FakeRunner):
    def __init__(self) -> None:
        super().__init__()
        self.active_stops = 0
        self.max_active_stops = 0
        self.stop_started_count = 0
        self.stop_started = asyncio.Event()
        self.stop_can_finish = asyncio.Event()

    async def stop(self, slot, timeout_sec=None):
        del timeout_sec
        self.stops += 1
        self.stop_started_count += 1
        self.active_stops += 1
        self.max_active_stops = max(self.max_active_stops, self.active_stops)
        self.stop_started.set()
        try:
            await self.stop_can_finish.wait()
            if slot.process is not None:
                slot.process.terminate()
                await slot.process.wait()
        finally:
            self.active_stops -= 1


def test_node_supervisor_starts_leases_and_releases_slots(tmp_path):
    asyncio.run(_run_node_supervisor_starts_leases_and_releases_slots(tmp_path))


async def _run_node_supervisor_starts_leases_and_releases_slots(tmp_path):
    runner = FakeRunner()
    supervisor = LocalBwrapNodeSupervisorCore(
        node_id="node-a",
        node_ip="10.0.0.12",
        capacity=2,
        base_port=31000,
        base_dir=tmp_path,
        runner=runner,
        health_checker=lambda url: url.startswith("http://10.0.0.12:"),
        reset_strategy="soft",
        start_health_loop=False,
    )
    async def abort_noop(*args, **kwargs):
        del args, kwargs

    supervisor._abort_blackbox_session = abort_noop

    ready = await supervisor.start_pool()
    first = await supervisor.acquire("traj-1")
    second = await supervisor.acquire("traj-1")
    released = await supervisor.release(
        lease_id=first["lease_id"],
        trajectory_id="traj-1",
        reason="test",
    )
    for _ in range(20):
        status = await supervisor.health()
        if status["ready"] == 2 and status["leased"] == 0:
            break
        await asyncio.sleep(0.01)
    status = await supervisor.health()
    await supervisor.shutdown()

    assert ready["ready"] == 2
    assert runner.starts == 2
    assert first == second
    assert first["sandbox_url"] == "http://10.0.0.12:31000"
    assert released["released"] is True
    assert status["ready"] == 2
    assert status["leased"] == 0


def test_node_supervisor_command_only_does_not_start_or_restart_server(tmp_path):
    asyncio.run(_run_node_supervisor_command_only_does_not_start_or_restart_server(tmp_path))


async def _run_node_supervisor_command_only_does_not_start_or_restart_server(tmp_path):
    runner = FakeRunner()
    supervisor = LocalBwrapNodeSupervisorCore(
        node_id="node-a",
        node_ip="10.0.0.12",
        capacity=1,
        base_port=31000,
        base_dir=tmp_path,
        runner=runner,
        health_checker=lambda url: False,
        reset_strategy="hard",
        start_health_loop=True,
        pool_mode="command_only",
    )

    ready = await supervisor.start_pool()
    lease = await supervisor.acquire("traj-1")
    released = await supervisor.release(
        lease_id=lease["lease_id"],
        trajectory_id="traj-1",
        reason="test",
    )
    for _ in range(20):
        await asyncio.sleep(0)
    status = await supervisor.health()

    assert ready["pool_mode"] == "command_only"
    assert runner.starts == 0
    assert runner.stops == 1
    assert lease["sandbox_url"] is None
    assert released["release_queued"] is True
    assert status["ready"] == 1
    assert status["leased"] == 0

    await supervisor.shutdown()


def test_node_supervisor_hard_reset_restarts_process(tmp_path):
    asyncio.run(_run_node_supervisor_hard_reset_restarts_process(tmp_path))


async def _run_node_supervisor_hard_reset_restarts_process(tmp_path):
    runner = FakeRunner()
    supervisor = LocalBwrapNodeSupervisorCore(
        node_id="node-a",
        node_ip="10.0.0.12",
        capacity=1,
        base_port=31000,
        base_dir=tmp_path,
        runner=runner,
        health_checker=lambda url: True,
        reset_strategy="hard",
        start_health_loop=False,
    )

    await supervisor.start_pool()
    lease = await supervisor.acquire("traj-1")
    await supervisor.release(
        lease_id=lease["lease_id"],
        trajectory_id="traj-1",
        reason="test",
    )
    for _ in range(10):
        await asyncio.sleep(0)
    status = await supervisor.health()
    await supervisor.shutdown()

    assert runner.starts == 2
    assert runner.stops >= 1
    assert status["ready"] == 1
    assert status["leased"] == 0


def test_node_supervisor_limits_concurrent_resets(tmp_path, monkeypatch):
    monkeypatch.setenv("DRESSAGE_LOCAL_BWRAP_RESET_CONCURRENCY", "1")
    asyncio.run(_run_node_supervisor_limits_concurrent_resets(tmp_path))


async def _run_node_supervisor_limits_concurrent_resets(tmp_path):
    runner = CountingBlockingStopRunner()
    supervisor = LocalBwrapNodeSupervisorCore(
        node_id="node-a",
        node_ip="10.0.0.12",
        capacity=2,
        base_port=31000,
        base_dir=tmp_path,
        runner=runner,
        health_checker=lambda url: True,
        reset_strategy="hard",
        start_health_loop=False,
    )

    await supervisor.start_pool()
    first = await supervisor.acquire("traj-1")
    second = await supervisor.acquire("traj-2")
    await supervisor.release(
        lease_id=first["lease_id"],
        trajectory_id="traj-1",
        reason="test",
    )
    await supervisor.release(
        lease_id=second["lease_id"],
        trajectory_id="traj-2",
        reason="test",
    )
    await asyncio.wait_for(runner.stop_started.wait(), timeout=0.1)
    await asyncio.sleep(0.02)

    assert runner.stop_started_count == 1
    assert runner.max_active_stops == 1

    runner.stop_can_finish.set()
    for _ in range(20):
        status = await supervisor.health()
        if status["ready"] == 2 and runner.stops == 2:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("limited resets did not complete")
    assert status["reset_concurrency"] == 1
    await supervisor.shutdown()


def test_node_supervisor_reports_reset_phase_profile(tmp_path):
    asyncio.run(_run_node_supervisor_reports_reset_phase_profile(tmp_path))


async def _run_node_supervisor_reports_reset_phase_profile(tmp_path):
    runner = FakeRunner()
    supervisor = LocalBwrapNodeSupervisorCore(
        node_id="node-a",
        node_ip="10.0.0.12",
        capacity=1,
        base_port=31000,
        base_dir=tmp_path,
        runner=runner,
        health_checker=lambda url: True,
        reset_strategy="hard",
        start_health_loop=False,
    )

    await supervisor.start_pool()
    lease = await supervisor.acquire("traj-profile")
    await supervisor.release(
        lease_id=lease["lease_id"],
        trajectory_id="traj-profile",
        reason="test",
    )
    for _ in range(20):
        status = await supervisor.health()
        if status["ready"] == 1 and status["leased"] == 0:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("profiled reset did not complete")

    profile = status["profile"]
    assert profile["reset.completed.count"] >= 1
    assert profile["reset.queue_wait.max"] >= 0
    assert profile["reset.runner_stop.max"] >= 0
    assert profile["reset.reset_dirs.max"] >= 0
    assert profile["reset.runner_start.max"] >= 0
    assert profile["reset.healthcheck.max"] >= 0
    await supervisor.shutdown()


def test_node_supervisor_reports_logs_when_process_exits(tmp_path):
    asyncio.run(_run_node_supervisor_reports_logs_when_process_exits(tmp_path))


async def _run_node_supervisor_reports_logs_when_process_exits(tmp_path):
    supervisor = LocalBwrapNodeSupervisorCore(
        node_id="node-a",
        node_ip="10.0.0.12",
        capacity=1,
        base_port=31000,
        base_dir=tmp_path,
        runner=DeadRunner(),
        health_checker=lambda url: False,
        startup_timeout_sec=0.1,
        start_health_loop=False,
    )

    status = await supervisor.start_pool()
    await supervisor.shutdown()

    assert status["failed"] == 1
    assert "exited before health check passed" in status["last_error"]
    assert "child failed loudly" in status["last_error"]


def test_node_supervisor_archives_session_artifacts_before_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv("DRESSAGE_BLACKBOX_PRESERVE_SESSION_ARTIFACTS", "1")
    monkeypatch.setenv("DRESSAGE_BLACKBOX_SESSION_ARCHIVE_MAX_PER_SLOT", "20")
    monkeypatch.setenv("DRESSAGE_BLACKBOX_SESSION_ARCHIVE_TTL_SEC", "0")
    asyncio.run(_run_node_supervisor_archives_session_artifacts_before_reuse(tmp_path))


async def _run_node_supervisor_archives_session_artifacts_before_reuse(tmp_path):
    runner = FakeRunner()
    supervisor = LocalBwrapNodeSupervisorCore(
        node_id="node-a",
        node_ip="10.0.0.12",
        capacity=1,
        base_port=31000,
        base_dir=tmp_path,
        runner=runner,
        health_checker=lambda url: True,
        reset_strategy="hard",
        start_health_loop=False,
    )

    await supervisor.start_pool()
    lease = await supervisor.acquire("bbs-session-archive")
    slot_dir = tmp_path / "slots" / "0000"
    (slot_dir / "home" / "home.txt").write_text("home")
    (slot_dir / "work" / "work.txt").write_text("work")
    (slot_dir / "runtime" / "runtime.txt").write_text("runtime")
    (slot_dir / "tmp" / "tmp.txt").write_text("tmp")

    released = await supervisor.release(
        lease_id=lease["lease_id"],
        trajectory_id="bbs-session-archive",
        reason="test-archive",
    )
    for _ in range(20):
        await asyncio.sleep(0)
    status = await supervisor.health()
    await supervisor.shutdown()

    archive = slot_dir / "archives" / "bbs-session-archive"
    assert released["release_queued"] is True
    assert released["slot_reusable"] is False
    assert status["ready"] == 1
    assert runner.starts == 2
    assert (archive / "home" / "home.txt").read_text() == "home"
    assert (archive / "work" / "work.txt").read_text() == "work"
    assert (archive / "runtime" / "runtime.txt").read_text() == "runtime"
    assert (archive / "tmp" / "tmp.txt").read_text() == "tmp"
    metadata = json.loads((archive / "metadata.json").read_text())
    assert metadata["session_id"] == "bbs-session-archive"
    assert metadata["bound_session_id"] == "bbs-session-archive"
    assert metadata["lease_id"] == lease["lease_id"]
    assert metadata["generation"] == lease["generation"]
    assert list((slot_dir / "home").iterdir()) == []
    assert list((slot_dir / "work").iterdir()) == []


def test_node_supervisor_shutdown_during_release_does_not_restart_slot(tmp_path):
    asyncio.run(_run_node_supervisor_shutdown_during_release_does_not_restart_slot(tmp_path))


async def _run_node_supervisor_shutdown_during_release_does_not_restart_slot(tmp_path):
    runner = BlockingStopRunner()
    supervisor = LocalBwrapNodeSupervisorCore(
        node_id="node-a",
        node_ip="10.0.0.12",
        capacity=1,
        base_port=31000,
        base_dir=tmp_path,
        runner=runner,
        health_checker=lambda url: True,
        reset_strategy="hard",
        start_health_loop=False,
    )

    await supervisor.start_pool()
    lease = await supervisor.acquire("traj-1")
    await supervisor.release(
        lease_id=lease["lease_id"],
        trajectory_id="traj-1",
        reason="test",
    )
    await asyncio.wait_for(runner.stop_started.wait(), timeout=0.1)
    shutdown_task = asyncio.create_task(supervisor.shutdown())
    await asyncio.sleep(0)
    runner.stop_can_finish.set()
    shutdown = await asyncio.wait_for(shutdown_task, timeout=0.2)
    status = await supervisor.health()

    assert shutdown["stopped"] is True
    assert runner.starts == 1
    assert status["closed"] is True
    assert status["ready"] == 0
    assert status["empty"] == 1


def test_node_supervisor_shutdown_blocks_new_work(tmp_path):
    asyncio.run(_run_node_supervisor_shutdown_blocks_new_work(tmp_path))


async def _run_node_supervisor_shutdown_blocks_new_work(tmp_path):
    runner = FakeRunner()
    supervisor = LocalBwrapNodeSupervisorCore(
        node_id="node-a",
        node_ip="10.0.0.12",
        capacity=1,
        base_port=31000,
        base_dir=tmp_path,
        runner=runner,
        health_checker=lambda url: True,
        reset_strategy="hard",
        start_health_loop=False,
    )

    await supervisor.shutdown()
    await supervisor.start_pool()

    assert runner.starts == 0
    with pytest.raises(RuntimeError, match="shut down"):
        await supervisor.acquire("traj-after-shutdown")
