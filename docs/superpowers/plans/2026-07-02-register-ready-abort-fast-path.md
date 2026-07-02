# Register Readiness And Abort Fast Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce sandbox lifecycle instability by making register transport failures diagnosable/retryable and making clean-session abort return quickly.

**Architecture:** Keep existing external HTTP APIs intact. Add same-slot register retry and structured diagnostics in the paddock layer, then add server-side abort classification so safe inactive sessions avoid expensive adapter abort. Unsafe sessions keep the existing real abort path.

**Tech Stack:** Python 3.12, asyncio, httpx, FastAPI blackbox server, local_bwrap Ray pool, pytest.

## Global Constraints

- Do not change public blackbox HTTP routes or response model compatibility.
- Do not return a slot to READY while active command, in-flight turn, active adapter request, or DESYNCED state can still mutate session state.
- Do not reduce rollout concurrency as the primary fix.
- Preserve existing provider recycle behavior when same-slot register retries are exhausted.

---

### Task 1: Register Transport Diagnostics And Same-Slot Retry

**Files:**
- Modify: `dressage/paddock/blackbox/paddock.py`
- Test: `tests/test_new_paddock_layers.py`

**Interfaces:**
- Consumes: `BlackboxAgentPaddock._register_agent_once(state, ...) -> dict[str, Any]`
- Produces: same public `register_agent()` behavior plus private same-slot transport retry before recycling.

- [ ] Step 1: Add tests for same-slot retry before recycle.

Create a fake client whose first `register_agent()` call raises `httpx.ReadError` and second call succeeds. Assert provider terminate is not called and the same state is used.

Run: `pytest -q tests/test_new_paddock_layers.py -k register`
Expected before implementation: failure because same-slot retry does not exist.

- [ ] Step 2: Implement same-slot transport retry.

Add env knobs: `DRESSAGE_BLACKBOX_REGISTER_TRANSPORT_RETRIES=2`, `DRESSAGE_BLACKBOX_REGISTER_RETRY_INITIAL_DELAY_SEC=0.1`, `DRESSAGE_BLACKBOX_REGISTER_RETRY_MAX_DELAY_SEC=0.5`. Retry only `_REGISTER_RECYCLE_ERRORS` before entering the existing recycle path.

- [ ] Step 3: Add structured register failure logging.

Log `exc_type`, endpoint URL, trajectory id, sandbox id, slot id, generation, recycle attempt, transport attempt, and elapsed seconds. Use lease metadata already stored by local_bwrap.

- [ ] Step 4: Run register tests.

Run: `pytest -q tests/test_new_paddock_layers.py tests/test_sandbox_provider_layer.py`
Expected: pass.

- [ ] Step 5: Commit.

Run: `git add dressage/paddock/blackbox/paddock.py tests/test_new_paddock_layers.py && git commit -m "fix: retry blackbox register transport races"`

---

### Task 2: Abort Fast Path Classification

**Files:**
- Modify: `blackbox_server/core/server.py`
- Test: `tests/blackbox_server/test_server.py`

**Interfaces:**
- Consumes: `BlackboxServer.abort_session(session_id: str) -> AbortResponse`
- Produces: private safe-abort classification that returns `mode="fast_finalize"` only for inactive clean sessions.

- [ ] Step 1: Add tests for clean inactive fast abort.

Create a session with no active command, no active adapter request, and no in-flight lock. Call `/v1/sessions/{id}/abort`. Assert `mode == "fast_finalize"`, state is aborted, and adapter abort is not called.

Run: `pytest -q tests/blackbox_server/test_server.py -k abort`
Expected before implementation: failure because mode is not `fast_finalize` or adapter abort is called.

- [ ] Step 2: Add tests for unsafe sessions.

Cover DESYNCED, active command, and adapter `has_active_request=True`. Assert the real abort path is used and adapter abort is called when needed.

- [ ] Step 3: Implement classification.

Use `_active_cmd_processes` for active command detection, adapter `has_active_request(session)` for active adapter detection, session state for DESYNCED, and non-blocking session lock acquisition for in-flight detection. If adapter active-request probing raises, classify as unsafe.

- [ ] Step 4: Implement fast path.

For safe inactive sessions, set `session.state = SessionState.ABORTED`, save the session, return `AbortResponse(..., mode="fast_finalize")`, and skip `_adapter.abort_session(session)`.

- [ ] Step 5: Run server tests.

Run: `pytest -q tests/blackbox_server/test_server.py tests/blackbox_server/test_opencode_adapter.py tests/blackbox_server/test_openclaw_adapter.py`
Expected: pass.

- [ ] Step 6: Commit.

Run: `git add blackbox_server/core/server.py tests/blackbox_server/test_server.py && git commit -m "fix: fast abort clean blackbox sessions"`

---

### Task 3: Integration Test And H200 Validation

**Files:**
- Runtime logs under `logs/`

**Interfaces:**
- Consumes: committed code from Tasks 1-2
- Produces: validation summary with counts for register failures, context overflow, finalize 404, slow terminate provider/total, and `perf0`.

- [ ] Step 1: Run focused unit suite.

Run: `pytest -q tests/test_ray_blackbox_scheduler.py tests/test_blackbox_node_supervisor.py tests/test_sandbox_provider_layer.py tests/test_new_paddock_layers.py tests/blackbox_server/test_server.py`
Expected: pass.

- [ ] Step 2: Start one H200 `perf0` validation when GPUs are free.

Run the existing four-card script through the background wrapper.
Expected: reaches `perf 0` without DeepGEMM/CUDA graph crash.

- [ ] Step 3: Compare against 20260702_090613 and 20260702_120538.

Report `perf/rollout_time`, `perf/effective_tokens_per_gpu_per_sec`, register transport failure count before perf0, finalize 404 count before perf0, context overflow count before perf0, and slow abort/terminate count/p50/p95/max.

## Self-Review

- Spec coverage: register diagnostics, same-slot retry, abort fast path, unsafe session preservation, and validation are covered by Tasks 1-3.
- Placeholder scan: no placeholder tasks are present; every task has files, commands, and expected results.
- Type consistency: all public entry points keep current names and signatures; new helpers are private to their modules.
