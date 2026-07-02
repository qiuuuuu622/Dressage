# Slow Terminate Reset Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add lifecycle profiling for slow blackbox terminate/reset and move manager health refresh out of the global lock so release/acquire/register are less likely to queue behind slow supervisor health RPCs.

**Architecture:** Reuse the existing `dressage.profiling` sample profile path for rollout-visible metrics and add low-volume aggregate logging for manager/supervisor actor internals. Keep the slot state machine unchanged: release can queue cleanup quickly, but slots become reusable only after supervisor reset/restart and healthcheck mark them READY.

**Tech Stack:** Python 3.12, asyncio, Ray actors, httpx, pytest, existing Dressage local_bwrap manager/supervisor/provider modules.

## Global Constraints

- Do not enable or rework `fast_finalize`.
- Do not mark a slot READY before reset/restart and healthcheck finish.
- Do not change agent loop, SGLang generation, or proxy parse behavior.
- Do not replace hard reset with soft reset.
- Do not change the external blackbox session API.
- Preserve `LEASED -> RELEASING -> RESETTING/RESTARTING -> READY`.

---

## File Structure

- Modify `dressage/profiling.py`: add small, reusable duration helpers for lock-wait style timing and percentile summaries if needed by actor-local aggregate logs.
- Modify `dressage/paddock/blackbox/paddock.py`: record abort/provider terminate durations in logs and, when possible, copy provider lifecycle metadata back into lease/state metadata.
- Modify `dressage/sandbox/local/bwrap/provider.py`: attach manager release response metrics to terminate results; keep provider API unchanged.
- Modify `dressage/sandbox/local/bwrap/manager.py`: add manager-side release/health-refresh timing, move node health refresh out of `_lock`, and add tests for slow health refresh not blocking release.
- Modify `dressage/sandbox/local/bwrap/supervisor.py`: add reset phase timing to `SlotRuntime` metadata or low-volume logs without changing reset semantics.
- Modify `tests/test_ray_blackbox_scheduler.py`: manager concurrency tests.
- Modify `tests/test_blackbox_node_supervisor.py`: supervisor reset phase profiling tests.
- Optionally modify `examples/scripts/default/dressage_env_defaults.sh`: add env defaults only if new profiling knobs are required.

## Task 1: Manager Release Is Not Blocked By Slow Health Refresh

**Files:**
- Modify: `tests/test_ray_blackbox_scheduler.py`
- Modify: `dressage/sandbox/local/bwrap/manager.py`

**Interfaces:**
- Consumes: `LocalBwrapClusterManagerCore.release(trajectory_id, lease_id, reason=None)`.
- Produces: manager release remains fast even if a concurrent status/acquire path is waiting on `supervisor.health()`.

- [ ] **Step 1: Add failing test supervisor**

Append this test helper to `tests/test_ray_blackbox_scheduler.py` near `BlockingAcquireSupervisor`:

```python
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
```

- [ ] **Step 2: Add failing concurrency test**

Append this test to `tests/test_ray_blackbox_scheduler.py`:

```python
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
```

- [ ] **Step 3: Verify the test fails on the old implementation**

Run:

```bash
pytest -q tests/test_ray_blackbox_scheduler.py::test_cluster_manager_release_is_not_blocked_by_slow_health_refresh
```

Expected before implementation: timeout while waiting for `manager.release()` because `status(force_refresh=True)` holds the manager lock while awaiting `supervisor.health()`.

- [ ] **Step 4: Implement lock-free health refresh**

In `dressage/sandbox/local/bwrap/manager.py`, add a lock-free refresh helper:

```python
async def _refresh_nodes_outside_lock(self, *, force: bool) -> None:
    async with self._lock:
        if self._closed:
            return
        elapsed = time.time() - self._last_refresh_ts
        if not force and elapsed < self.status_refresh_interval_sec:
            return
        nodes = list(self.nodes.values())

    results = await asyncio.gather(
        *(self._refresh_node_payload(node) for node in nodes),
        return_exceptions=True,
    )

    async with self._lock:
        if self._closed:
            return
        for node, result in zip(nodes, results, strict=False):
            current = self.nodes.get(node.node_id)
            if current is None:
                continue
            if isinstance(result, BaseException):
                current.mark_lost(result)
                for lease in self.leases.values():
                    if lease.node_id == current.node_id and lease.status in {
                        LEASE_ACTIVE,
                        LEASE_RELEASING,
                    }:
                        lease.status = LEASE_LOST
                continue
            current.update_from_health(result)
        self._last_refresh_ts = time.time()


async def _refresh_node_payload(self, node: NodeRecord) -> dict[str, Any]:
    return await _remote_call(node.supervisor, "health")
```

Keep `_refresh_node()` for call sites that already hold the lock during startup or shutdown, but move hot `status()` to call `_refresh_nodes_outside_lock()` before taking the final status lock.

- [ ] **Step 5: Update `status()` to avoid lock-held health RPC**

Change `status()` to:

```python
async def status(self, *, force_refresh: bool = False) -> dict[str, Any]:
    self._reap_background_tasks()
    if force_refresh:
        await self._refresh_nodes_outside_lock(force=True)
    async with self._lock:
        if self._closed:
            return self._status_locked()
        await self._reconcile_locked()
        if not force_refresh:
            await self._refresh_nodes_if_needed_locked(force=False)
        return self._status_locked()
```

If this still leaves `acquire()` using lock-held refresh, defer acquire restructuring to Task 3 so Task 1 remains focused on proving status/health does not block release.

- [ ] **Step 6: Run the focused test**

Run:

```bash
pytest -q tests/test_ray_blackbox_scheduler.py::test_cluster_manager_release_is_not_blocked_by_slow_health_refresh
```

Expected: pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add dressage/sandbox/local/bwrap/manager.py tests/test_ray_blackbox_scheduler.py
git commit -m "perf: keep manager release off slow health refresh"
```

## Task 2: Add Supervisor Reset Phase Profiling

**Files:**
- Modify: `dressage/sandbox/local/bwrap/supervisor.py`
- Modify: `tests/test_blackbox_node_supervisor.py`

**Interfaces:**
- Consumes: `LocalBwrapNodeSupervisorCore.health()`.
- Produces: health payload includes aggregate reset timing fields under a new `profile` dictionary.

- [ ] **Step 1: Add test for reset phase profile**

Append this test to `tests/test_blackbox_node_supervisor.py`:

```python
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
    assert profile["reset.completed"] >= 1
    assert profile["reset.queue_wait.max"] >= 0
    assert profile["reset.runner_stop.max"] >= 0
    assert profile["reset.reset_dirs.max"] >= 0
    assert profile["reset.runner_start.max"] >= 0
    assert profile["reset.healthcheck.max"] >= 0
    await supervisor.shutdown()
```

- [ ] **Step 2: Add a small in-memory profile accumulator**

In `LocalBwrapNodeSupervisorCore.__init__`, add:

```python
self._profile_values: dict[str, list[float]] = {}
```

Add methods:

```python
def _profile_observe(self, key: str, value: float) -> None:
    values = self._profile_values.setdefault(key, [])
    values.append(float(value))
    if len(values) > 2048:
        del values[: len(values) - 2048]

def _profile_summary(self) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, values in self._profile_values.items():
        if not values:
            continue
        ordered = sorted(values)
        result[f"{key}.count"] = float(len(ordered))
        result[f"{key}.max"] = float(ordered[-1])
        result[f"{key}.p95"] = float(ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))])
        result[f"{key}.mean"] = float(sum(ordered) / len(ordered))
    return result
```

- [ ] **Step 3: Expose profile in health**

In `health()`, add:

```python
"profile": self._profile_summary(),
```

- [ ] **Step 4: Time reset phases**

In `_reset_slot_limited()`, record queue wait:

```python
queued_at = time.perf_counter()
async with self._reset_semaphore:
    self._profile_observe("reset.queue_wait", time.perf_counter() - queued_at)
    await self._reset_slot(
        slot,
        reason,
        expected_generation=expected_generation,
        session_id=session_id,
        lease_id=lease_id,
    )
```

In hard reset branch of `_reset_slot()`, wrap `runner.stop`, `reset_runtime_dirs`, `_start_slot`:

```python
stop_start = time.perf_counter()
await self.runner.stop(slot)
self._profile_observe("reset.runner_stop", time.perf_counter() - stop_start)

dirs_start = time.perf_counter()
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
self._profile_observe("reset.reset_dirs", time.perf_counter() - dirs_start)

start_start = time.perf_counter()
await self._start_slot(slot)
self._profile_observe("reset.runner_start", time.perf_counter() - start_start)
self._profile_observe("reset.completed", 1.0)
```

In `_wait_until_healthy()`, measure healthcheck duration around the loop and observe `reset.healthcheck`.

- [ ] **Step 5: Run supervisor tests**

Run:

```bash
pytest -q tests/test_blackbox_node_supervisor.py::test_node_supervisor_reports_reset_phase_profile tests/test_blackbox_node_supervisor.py::test_node_supervisor_limits_concurrent_resets
```

Expected: both pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add dressage/sandbox/local/bwrap/supervisor.py tests/test_blackbox_node_supervisor.py
git commit -m "perf: profile local bwrap reset phases"
```

## Task 3: Move Acquire Refresh Out Of Manager Lock

**Files:**
- Modify: `dressage/sandbox/local/bwrap/manager.py`
- Modify: `tests/test_ray_blackbox_scheduler.py`

**Interfaces:**
- Consumes: Task 1 `_refresh_nodes_outside_lock(force: bool)`.
- Produces: acquire uses cached readiness first and performs slow health refresh outside the manager lock.

- [ ] **Step 1: Add test for acquire refresh not blocking release**

Append this test to `tests/test_ray_blackbox_scheduler.py`:

```python
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
```

- [ ] **Step 2: Refactor acquire refresh**

In `acquire()`, avoid calling `_refresh_nodes_if_needed_locked()` while holding `_lock`. Use this shape:

```python
needs_refresh = False
async with self._lock:
    await self._reconcile_locked()
    existing = self.leases.get(trajectory_id)
    # Keep the current existing-active, existing-releasing, and stale-lease
    # handling here; only the health refresh moves outside the lock.
    node = self._select_node_locked()
    if node is None:
        needs_refresh = True

if needs_refresh:
    await self._refresh_nodes_outside_lock(force=True)
    await asyncio.sleep(0)
    continue
```

Keep `pending_acquires` and cleanup of uncommitted acquire unchanged.

- [ ] **Step 3: Run manager tests**

Run:

```bash
pytest -q tests/test_ray_blackbox_scheduler.py
```

Expected: all manager tests pass.

- [ ] **Step 4: Commit**

Run:

```bash
git add dressage/sandbox/local/bwrap/manager.py tests/test_ray_blackbox_scheduler.py
git commit -m "perf: refresh local bwrap nodes outside acquire lock"
```

## Task 4: Surface Lifecycle Metrics In Rollout Logs

**Files:**
- Modify: `dressage/sandbox/local/bwrap/provider.py`
- Modify: `dressage/paddock/blackbox/paddock.py`
- Modify: `tests/test_sandbox_provider_layer.py`

**Interfaces:**
- Consumes: manager release response and supervisor health `profile`.
- Produces: terminate result includes numeric `profile.provider.terminate`, `profile.manager.release`, and optional reset profile values without changing callers that ignore the response.

- [ ] **Step 1: Add provider terminate profile test**

Append this test to `tests/test_sandbox_provider_layer.py`:

```python
def test_local_bwrap_provider_terminate_includes_profile(monkeypatch):
    asyncio.run(_run_local_bwrap_provider_terminate_includes_profile(monkeypatch))


async def _run_local_bwrap_provider_terminate_includes_profile(monkeypatch):
    monkeypatch.setenv("DRESSAGE_PROFILE", "1")
    manager = FakeLocalManager()
    provider = LocalBwrapSandboxProvider(manager=manager)
    lease = await provider.create(
        SandboxSpec(
            trajectory_id="traj-profile",
            services=(SandboxServiceSpec(name="blackbox", port=31000),),
            metadata={"paddock_mode": "blackbox"},
        )
    )

    result = await provider.terminate(lease)

    assert result["released"] is True
    assert result["profile.provider.terminate"] >= 0
```

- [ ] **Step 2: Add provider timing**

In `LocalBwrapSandboxProvider.terminate()`, wrap manager release:

```python
start = time.perf_counter() if profiling_enabled() else 0.0
result = await _remote_call(
    self._manager,
    "release",
    trajectory_id=trajectory_id,
    lease_id=lease_id,
    reason="paddock_terminate",
)
if profiling_enabled() and isinstance(result, dict):
    result["profile.provider.terminate"] = time.perf_counter() - start
return result
```

- [ ] **Step 3: Log terminate result profile in paddock**

In `_terminate_cleanup()`, capture the provider result before returning:

```python
result = await self._provider.terminate(lease)
if isinstance(result, dict):
    profile = {k: v for k, v in result.items() if k.startswith("profile.") and isinstance(v, (int, float))}
    if profile:
        logger.debug("blackbox terminate profile session_id=%s profile=%s", traj_id, profile)
return result
```

- [ ] **Step 4: Run focused provider tests**

Run:

```bash
pytest -q tests/test_sandbox_provider_layer.py::test_local_bwrap_provider_terminate_includes_profile
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add dressage/sandbox/local/bwrap/provider.py dressage/paddock/blackbox/paddock.py tests/test_sandbox_provider_layer.py
git commit -m "perf: surface local bwrap terminate profile"
```

## Task 5: Full Test And H200 Validation

**Files:**
- Read: `logs/run_blackbox_qwen3.5_35b_a3b_sync_local_h200x4_20260702_090613.log`
- Read: new H200 run log generated by validation script.

**Interfaces:**
- Consumes: Tasks 1-4 commits.
- Produces: before/after metrics for slow terminate and rollout_time.

- [ ] **Step 1: Run local unit test set**

Run:

```bash
pytest -q tests/test_ray_blackbox_scheduler.py tests/test_blackbox_node_supervisor.py tests/test_sandbox_provider_layer.py tests/test_new_paddock_layers.py
```

Expected: pass.

- [ ] **Step 2: Start H200 end-to-end run**

Run:

```bash
bash examples/scripts/run_blackbox_qwen3.5_35b_a3b_sync_local_h200x4.sh
```

Expected: script starts four SGLang engines and completes at least `perf 0`.

- [ ] **Step 3: Extract perf0 slow terminate stats**

Run this against the new run log:

```bash
NEW_LOG="$(ls -t logs/run_blackbox_qwen3.5_35b_a3b_sync_local_h200x4_*.log | head -n 1)"
python3 - <<'PY'
import re, statistics, sys
from pathlib import Path
p = Path(sys.argv[1])
lines = p.read_text(errors="replace").splitlines()
perf_idx = next((i for i, line in enumerate(lines) if "perf 0:" in line), len(lines))
pat = re.compile(r"slow blackbox terminate .* provider=([0-9.]+)s total=([0-9.]+)s")
values = [float(m.group(1)) for line in lines[:perf_idx] if (m := pat.search(line))]
if not values:
    print("count=0")
    raise SystemExit
values.sort()
def q(pct):
    return values[min(len(values)-1, max(0, int(len(values)*pct/100)-1))]
print({
    "count": len(values),
    "mean": round(statistics.fmean(values), 3),
    "p50": q(50),
    "p90": q(90),
    "p95": q(95),
    "max": values[-1],
})
PY "${NEW_LOG}"
```

Expected: provider p95 below baseline `12.645s`, no increase in register 500/502, 404, ReadTimeout, or ReadError.

- [ ] **Step 4: Record validation result**

Append a short validation section to `reports/h200_local_bwrap_concurrency_20260702.md` with:

```text
commit range:
run log:
rollout_time:
effective_tokens_per_gpu_per_sec:
slow terminate count:
provider p50/p90/p95/max:
register 500/502:
404 / ReadTimeout / ReadError:
reward_mean:
truncated_ratio:
response_len mean:
```

- [ ] **Step 5: Commit validation note**

Run:

```bash
git add reports/h200_local_bwrap_concurrency_20260702.md
git commit -m "docs: record slow terminate optimization validation"
```

## Self-Review

- Spec coverage: lifecycle profiling is covered by Tasks 2 and 4; manager lock-free health refresh is covered by Tasks 1 and 3; H200 validation is covered by Task 5.
- Placeholder scan: no placeholder markers remain.
- Type consistency: manager methods keep existing public signatures; supervisor health adds an optional `profile` dictionary; provider terminate still returns a dictionary.
