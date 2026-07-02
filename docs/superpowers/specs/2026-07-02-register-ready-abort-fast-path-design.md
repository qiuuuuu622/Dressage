# Register Readiness And Abort Fast Path Design

## Problem

The 2026-07-02 H200 run showed two remaining sandbox lifecycle costs after the provider/manager lock fixes:

- `register_agent transport failure` still appears during initial slot startup and once after `perf0`, which means a slot can be handed to rollout before its blackbox server is truly usable for `/v1/rollout/register`.
- `slow blackbox terminate` no longer spends time in provider release, but still spends 1-6 seconds in `/v1/sessions/{session_id}/abort`; this delays slot reuse and can create cleanup waves that affect the next rollout round.

The current provider fix worked: provider release is now millisecond-scale. The remaining work is to make slot readiness stronger and make abort cheap for sessions that are already cleanly inactive.

## Goals

- Preserve Dressage rollout semantics: a slot must not be reused while an active command, in-flight turn, active adapter request, or desynced session can still mutate state.
- Keep the public blackbox API unchanged. Existing callers still use `/health`, `/v1/rollout/register`, and `/v1/sessions/{session_id}/abort`.
- Make register failures diagnosable by logging exception type, endpoint, sandbox id, slot id, generation, attempt, and elapsed time.
- Absorb short readiness races with same-slot register retries before recycling the slot.
- Add a server-side abort fast path for sessions that are already cleanly inactive, returning a mode that can be profiled.

## Non-Goals

- Do not reduce rollout parallelism as the primary fix.
- Do not mark slots READY without a successful health/readiness check after reset.
- Do not skip real abort for in-flight, active-command, active-adapter, or desynced sessions.
- Do not change SGLang generation configuration in this change.

## Design

### Register Readiness

`local_bwrap` already performs an acquire-time blackbox health precheck, but that check only proves the HTTP server is responsive. The first implementation keeps the provider boundary stable and strengthens the paddock register behavior:

- `BlackboxServerClient.register_agent()` continues to call `/v1/rollout/register`.
- `BlackboxAgentPaddock.register_agent()` retries transport failures against the same endpoint for a short jittered backoff before recycling the slot.
- If same-slot retries fail, the existing recycle path remains in force.
- Warnings include structured context from the lease metadata: endpoint URL, sandbox id, slot id, generation, exception type, recycle attempt, local retry attempt, and elapsed seconds.

This is intentionally less invasive than adding a new provider `/ready` API first. It directly addresses the observed failure mode and gives enough evidence to decide whether a later supervisor-level `/ready` endpoint is necessary.

### Abort Fast Path

`BlackboxServer.abort_session()` currently terminates active commands and calls adapter abort for most non-ABORTED sessions. That is correct for unsafe sessions, but too expensive for sessions that have already completed their last message and have no active work.

The server will classify a session before doing expensive abort work:

- `has_inflight_turn`: session lock is currently held by message handling, or the session has a turn that has not reached a terminal response state.
- `has_active_cmd`: the server has an active command process for the session.
- `has_active_adapter_request`: the adapter reports active backend work for the session.
- `is_desynced`: session state is `DESYNCED`.

If none of those are true, `/abort` returns quickly with `mode="fast_finalize"` after marking the session `ABORTED`. It does not call adapter abort. If any unsafe condition is true, the existing real abort path runs.

The fast path deliberately reuses the existing ABORTED terminal state in this phase. A new CLOSED/COMPLETED state would be cleaner but has a larger migration surface; the fast-path mode makes behavior visible without changing external state semantics.

## Error Handling

- Register same-slot retries apply only to transport exceptions: connect, pool, read, timeout, and remote protocol errors. HTTP 502/503/504 still use the existing recycle behavior.
- If same-slot retries and recycle attempts are exhausted, the original exception type is propagated.
- Abort fast path is allowed only when the server can prove no active mutation is in progress. If adapter active-request probing fails, the server treats the session as unsafe and uses the real abort path.
- Active command sessions must terminate the command before the session is marked terminal.

## Testing

- Unit tests for register logging and same-slot retry before recycle.
- Unit tests for register retry exhaustion preserving the existing recycle behavior.
- Blackbox server tests for abort fast path on clean inactive sessions.
- Blackbox server tests proving active command, desynced session, and active adapter request still use real abort.
- Existing sandbox provider and blackbox paddock tests must continue to pass.

## Expected Runtime Effect

- Startup/recycle register transport failures should drop or become clearly attributable to a specific exception and slot generation.
- Clean-session aborts should move from seconds to milliseconds, reducing cleanup waves between rollout rounds.
- Single `perf0` may improve only slightly because its dominant cost is agent/SGLang generation. Multi-round stability should improve because failed register and slow abort no longer spill into the next round.
