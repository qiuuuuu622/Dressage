# Slow Terminate / Reset Profiling Design

日期：2026-07-02

## Goal

降低 H200 blackbox rollout 中 `slow blackbox terminate` 的尾延迟，并确认它是否阻塞 `acquire/register`，但不改变 Dressage 的 agentic RL 隔离语义。

## Current Evidence

最近一次 H200 四卡端到端运行日志：

```text
logs/run_blackbox_qwen3.5_35b_a3b_sync_local_h200x4_20260702_090613.log
```

按 `perf 0` 之前统计：

```text
slow terminate count: 141
provider mean: 5.084s
provider p50: 3.541s
provider p90: 12.204s
provider p95: 12.645s
provider max: 13.662s
total mean: 5.099s
total p95: 12.650s
total max: 13.663s
```

这说明 rollout 关键路径里的慢点主要在 `provider.terminate()` 等待 manager / supervisor release 链路，而不是 paddock 本地 cleanup 统计本身。

`perf 0` 之后统计：

```text
slow terminate count: 49
provider mean: 1.066s
provider p50: 0.032s
provider p95: 2.913s
total mean: 4.033s
total p95: 6.013s
```

这部分更像 rollout 结束后的 background cleanup drain，不能和 perf0 前的关键路径混为一谈。

## Non-Goals

本轮不做这些事情：

- 不启用或重做 `fast_finalize`。
- 不把 slot 在 reset 完成前提前标记 READY。
- 不改 agent loop、SGLang generation、proxy parse。
- 不把 hard reset 替换成 soft reset。
- 不改变 blackbox session API 的外部语义。

## Design Constraints

Dressage 的 sandbox 隔离语义必须保持：

```text
LEASED -> RELEASING -> RESETTING/RESTARTING -> READY
```

只有 reset/restart 和 healthcheck 完成后，slot 才能重新进入 READY。任何优化都不能让一个 trajectory 的文件、进程、session 状态污染下一个 trajectory。

manager 的全局锁仍然必要，用来保护 lease 状态一致性。但锁里不能等待慢 RPC 或慢 health refresh，否则 release/acquire/register 会在同一个入口排队。

## Suspected Bottlenecks

### 1. Manager release 可能被锁内 refresh 或 reconcile 间接阻塞

当前 `manager.acquire()` 已经把 supervisor acquire RPC 移到锁外，但 manager 仍有一些锁内路径可能等待 supervisor：

```text
_refresh_nodes_if_needed_locked()
  -> _refresh_node()
     -> supervisor.health()
```

如果大量 acquire/status 正好触发 health refresh，`release()` 想拿 manager 锁只做 `LEASE_RELEASING` 标记，也可能被拖住。

### 2. Supervisor reset/restart 波峰

hard reset 包括：

```text
runner.stop()
reset_runtime_dirs()
runner.start()
wait_until_healthy()
```

这些不是纯 CPU 任务，而是进程、文件系统、HTTP healthcheck、Ray actor 调度混合等待。当前 `DRESSAGE_LOCAL_BWRAP_RESET_CONCURRENCY=64` 可能仍会在 64 核 H200 上制造波峰。

### 3. Manager background release completion polling 粒度粗

manager release 已经快速返回给 provider，但后台要通过 supervisor health 判断 slot 是否 READY，然后 drop lease。当前 poll 默认是 1s。它不一定解释 12s p95，但会影响同一个 trajectory 重试或统计上的 releasing 滞留。

## Proposed Architecture

分两步做：先观测，再做低风险结构优化。

### Phase 1: Add lifecycle profiling

新增轻量生命周期计时，不依赖全局 `DRESSAGE_PROFILE` 才能看到关键聚合日志。采样或聚合输出，避免每个 session 打大量日志。

需要拆出的指标：

```text
paddock.abort_session_s
paddock.provider_terminate_s

manager.release_lock_wait_s
manager.release_mark_s
manager.release_rpc_queue_submit_s
manager.background_supervisor_release_rpc_s
manager.background_wait_ready_s
manager.health_refresh_s
manager.health_refresh_lock_commit_s

supervisor.release_lock_wait_s
supervisor.release_mark_s
supervisor.reset_queue_wait_s
supervisor.runner_stop_s
supervisor.reset_dirs_s
supervisor.runner_start_s
supervisor.healthcheck_s

provider.create_acquire_s
provider.terminate_s
register_agent_s
```

输出口径要能区分：

- perf0 前。
- perf0 后。
- p50 / p90 / p95 / max。
- 按 node_id / slot_id 聚合。
- acquire/register 发生时 manager `leases.releasing`、`total_ready`、`total_resetting`、`total_restarting` 的状态。

### Phase 2: Move manager health refresh out of the global lock

把 manager health refresh 改成两段式：

```text
lock: snapshot nodes and last_refresh_ts
unlock: call supervisor.health() concurrently with bounded fanout
lock: commit NodeRecord health payloads if still valid
```

`release()` 应继续只做短锁状态切换：

```text
lookup lease
mark LEASE_RELEASING
enqueue background release task
return
```

`acquire()` 应优先使用 cached node readiness。只有没有 candidate 或 cache 明显过期时，才触发锁外 refresh。这样 release 不会被 acquire 的 health refresh 拖住。

## Data Flow

优化后的 release 关键路径：

```text
paddock.terminate()
  -> best-effort abort_session()
  -> provider.terminate()
  -> manager.release()
       lock: mark lease RELEASING
       unlock: create background release task
       return to provider
  -> paddock cleanup returns

manager background task
  -> supervisor.release()
       lock: mark slot RELEASING, clear lease ids
       enqueue reset task
       return
  -> poll supervisor.health() until slot READY
  -> lock: drop lease

supervisor reset task
  -> wait reset semaphore
  -> stop slot server/bwrap
  -> reset runtime dirs
  -> start slot server/bwrap
  -> healthcheck
  -> mark READY
```

## Error Handling

如果 supervisor release RPC 失败：

- manager 标记 node lost。
- 对应 lease 标记 LOST。
- 不把 slot 误标 READY。

如果 background wait_ready 超时：

- manager 记录 timeout 日志。
- lease 标记 LOST。
- node 状态按最近一次 health payload 或异常更新。

如果 health refresh 失败：

- 只标记对应 node lost。
- 不阻塞其他 node。
- 不让 release 因 refresh 失败而长时间等待。

## Testing Plan

单元测试至少覆盖：

1. 慢 supervisor health refresh 不阻塞 manager release。
2. health refresh 锁外并发后，node 状态能正确提交。
3. health refresh 失败只影响对应 node，不破坏已有 active/releasing lease 状态。
4. release 仍然在 reset 完成前保持 slot 不可复用。
5. profiling 聚合不会改变默认行为。

H200 端到端验证：

```text
scripts/run_blackbox_qwen3.5_35b_a3b_sync_local_h200x4.sh
```

比较指标：

```text
rollout_time
effective_tokens_per_gpu_per_sec
slow blackbox terminate count
provider terminate p50/p90/p95/max
manager release lock wait p95/max
supervisor reset queue wait p95/max
register 500/502 count
404 / ReadTimeout / ReadError count
reward_mean
truncated_ratio
response_len mean
```

## Success Criteria

第一版成功标准：

- register 500/502 不增加。
- 404 / ReadTimeout / ReadError 不增加。
- perf0 前 provider terminate p95 低于上一轮 `12.645s`。
- `rollout_time` 不回退；理想目标降低 10-30s。
- 如果 rollout_time 没明显下降，profiling 必须能解释当前剩余瓶颈在哪里。

## Rollout Safety

本设计不改变以下边界：

- session abort 仍然是 best-effort cleanup。
- provider release/reset 仍然决定 slot 何时可复用。
- slot 未 READY 前不会被新 trajectory acquire。
- hard reset 仍然是默认策略。

因此这属于调度层和观测层优化，不改变 Dressage agentic RL 的设计初衷。

## Spec Self-Review

- Placeholder scan: no placeholder markers remain.
- Consistency check: all proposed changes preserve LEASED -> RELEASING -> RESETTING/RESTARTING -> READY.
- Scope check: scoped to lifecycle profiling and manager health refresh locking; reset strategy tuning is validation work, not core behavior change.
- Ambiguity check: success metrics and non-goals are explicit.
