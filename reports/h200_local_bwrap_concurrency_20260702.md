# H200 Local Bwrap 并发调度优化总结

日期：2026-07-02  
代码基线：`5d4b0fc`  
优化提交：`47ee14f perf: reduce local bwrap acquire and reset contention`

## 背景

这轮优化针对 Qwen3.5-35B-A3B 在 H200 上做 agentic RL blackbox rollout 时的尾延迟问题。现象是 `slow blackbox terminate` 很多，且日志中 `provider` 时间几乎等于 `total` 时间，说明慢点主要发生在 sandbox provider 到 local_bwrap manager/supervisor 的释放路径，而不是 agent 本身的生成或工具调用。

旧基线中，第一轮 rollout 到 `perf 0` 为止：

- `rollout_time = 722.09s`
- `slow blackbox terminate = 249`
- `provider p95 = 10.74s`
- `provider max = 12.64s`
- `blackbox rollout failed = 2`
- `404 Not Found = 1`

## 相关组件关系

### Paddock

`dressage/paddock/blackbox/paddock.py` 负责 rollout 视角的 sandbox 生命周期：

1. 创建 sandbox state。
2. 调用 provider 分配 sandbox lease。
3. 注册 blackbox agent。
4. 调用 agent。
5. finalize / abort / terminate。

`slow blackbox terminate` 的日志在 paddock 层记录，但它统计的慢点主要是 `_provider.terminate()`。

### Provider

`dressage/sandbox/local/bwrap/provider.py` 是 rollout 进程里的资源入口。它不直接管理 256 个 slot，只负责把 `create()` / `terminate()` 转成 RPC 发给 Ray detached manager。

通俗理解：provider 是前台窗口，manager 是总调度中心。

### Manager

`dressage/sandbox/local/bwrap/manager.py` 管全局 lease 状态：

- 哪个 trajectory 占了哪个 slot。
- 哪个 node 有多少 ready/free/leased/resetting slot。
- release 是否已排队。
- 防止两个 trajectory 同时拿到同一个 slot。

manager 的锁是必要的，但锁只应该保护内存状态，不能持锁等待远程 RPC。

### Supervisor

`dressage/sandbox/local/bwrap/supervisor.py` 是节点本地 slot 管理器。它负责：

- 启动 blackbox server。
- 分配 node-local slot。
- release 后 reset/restart slot。
- healthcheck。

### Runner / Sandbox

`dressage/sandbox/local/bwrap/runner.py` 负责实际 bwrap 进程的 start/stop。hard reset 会触发：

- stop server/bwrap 进程。
- 清 runtime/work/tmp 等目录。
- 扫 `/proc` 清残留进程。
- restart server。
- healthcheck。

这些操作不是纯 CPU 计算，而是进程、IO、HTTP、Ray actor 调度混合等待，所以 64 核 CPU 不一定能被打满。

## 根因分析

### 1. `manager.acquire()` 持全局锁等待 supervisor RPC

旧逻辑大致是：

```python
async with self._lock:
    ...
    payload = await _remote_call(node.supervisor, "acquire", ...)
```

这会导致一个慢 acquire 持有 manager 全局锁。此时其他 release 只是想标记 `LEASE_RELEASING`，也必须排队等锁。

表现就是许多 `terminate()` 同一秒成批完成，且 `provider=8s/10s/12s`，说明它们不是各自执行慢，而是在共享入口排队。

### 2. hard reset 无界并发

旧 supervisor release 后直接 `_schedule_task(_reset_slot(...))`，256 个 slot 如果同时释放，就可能同时进行 bwrap stop/start、目录清理、healthcheck 和 `/proc` 扫描。

这会制造 reset 风暴，进一步放大 actor 队列和系统资源等待。

### 3. slot 生命周期必须严格保持

不能为了快而提前复用 slot。设计边界是：

```text
LEASED -> RELEASING -> RESETTING/RESTARTING -> READY
```

只有 reset 完成后才能 READY。否则会破坏 agentic RL 的隔离语义，导致一个 trajectory 的 session/文件/进程污染下一个 trajectory。

## 已完成的修复

### 1. manager acquire 两段式提交

新逻辑：

1. 短暂拿 manager 锁。
2. 检查已有 lease。
3. 选择候选 node。
4. 记录 `pending_acquires`。
5. 释放 manager 锁。
6. 锁外等待 supervisor acquire RPC。
7. RPC 返回后再短暂拿锁提交 lease。
8. 如果提交失败或状态变化，释放刚拿到但未提交的 slot。

这样 release 不会被慢 supervisor acquire 堵住。

### 2. `pending_acquires`

新增 `NodeRecord.pending_acquires`，用于避免多个 acquire 并发选择同一个 node 后过度预留。

node 选择逻辑从只看 `free` 改成看：

```text
free - pending_acquires
```

负载排序从：

```text
used / capacity
```

改成：

```text
(used + pending_acquires) / capacity
```

### 3. acquire 取消安全

如果 acquire task 在等待 supervisor RPC 时被取消，会回收 `pending_acquires`，避免 manager 认为有 slot 被预留但永远不提交。

### 4. 未提交 lease 的回收

如果 supervisor acquire 成功，但 manager 提交时发现：

- manager 已关闭；
- node 已 lost；
- trajectory 已经有 active lease；
- trajectory 正在 releasing；

则调用 `_release_uncommitted_acquire()` 释放刚拿到的 slot，防止资源泄漏。

### 5. supervisor reset 并发限制

新增环境变量：

```bash
DRESSAGE_LOCAL_BWRAP_RESET_CONCURRENCY=64
```

默认值写入 `examples/scripts/default/dressage_env_defaults.sh`。代码层默认是 `min(capacity, 64)`。

release 仍然快速返回，但后台 reset 会经过 semaphore：

```text
release() -> SLOT_RELEASING -> enqueue reset task -> reset semaphore -> _reset_slot() -> READY
```

这不改变 slot 状态机，只限制同时进行 hard reset 的数量。

## 单元测试

已补充测试：

- 慢 supervisor acquire 不阻塞 manager release。
- acquire 被取消后不泄漏 `pending_acquires`。
- supervisor reset 并发限制生效。
- 原有 local_bwrap provider/manager/supervisor 生命周期测试保持通过。

测试命令：

```bash
pytest -q tests/test_config.py tests/test_ray_blackbox_scheduler.py tests/test_blackbox_node_supervisor.py tests/test_sandbox_provider_layer.py
```

结果：

```text
41 passed in 2.13s
```

## 端到端结果

对比口径：3 张 H200，`DRESSAGE_PROFILE=0`，`TOOL_CALL_PARSE_BACKEND=sglang_api`，`REASONING_PARSE_BACKEND=sglang_api`，都统计到第一条 `perf 0` 为止。

| 指标 | 旧基线 | 新 patch | 变化 |
|---|---:|---:|---:|
| `rollout_time` | `722.09s` | `701.77s` | `-20.32s` / `-2.81%` |
| effective TPS | `962.81` | `1095.83` | `+13.82%` |
| slow terminate 数 | `249` | `115` | `-53.8%` |
| provider mean | `5.67s` | `3.17s` | `-44.1%` |
| provider p50 | `5.51s` | `2.68s` | `-51.4%` |
| provider p90 | `9.58s` | `5.23s` | `-45.4%` |
| provider p95 | `10.74s` | `5.99s` | `-44.2%` |
| provider max | `12.64s` | `6.57s` | `-48.0%` |
| rollout failed | `2` | `0` | 修复 |
| 404 | `1` | `0` | 修复 |
| ReadTimeout / ReadError | `0 / 0` | `0 / 0` | 无新增 |

其他 rollout 指标：

| 指标 | 旧基线 | 新 patch |
|---|---:|---:|
| reward | `0.6172` | `0.6426` |
| response_len mean | `2038.8` | `2156.1` |
| truncated_ratio | `0.0098` | `0.0131` |
| segments | `1023` | `1070` |

## 结论

这次优化没有改变 Dressage 的 agentic RL 设计语义。它修的是 local_bwrap 资源调度层的并发缺陷：

- manager 不再持全局锁等待远程 acquire。
- release 能更快进入 manager 并排队 cleanup。
- reset 风暴被限制。
- slot reset 完成前仍然不会 READY。

结果上，terminate 尾延迟明显下降：slow terminate 数和 provider p95/max 基本砍半。端到端 rollout time 提升约 2.8%。收益没有更大，是因为当前总体大头仍在 agent loop、LLM 生成和工具交互，而不是 terminate。

## 后续建议

1. 继续拆 `call_agent` 内部耗时：LLM generate、tool call、sandbox command、proxy parse。
2. 对 `DRESSAGE_LOCAL_BWRAP_RESET_CONCURRENCY` 做参数扫：32 / 64 / 96，找 H200 当前 64 核环境下的最优点。
3. 如果 404/finalize 再出现，再回到 session 状态机，补 clean inactive session fast path，但不要提前复用未 reset slot。
4. reset 本身仍可继续优化：减少 hard reset 中 `/proc` 扫描、目录清理和 healthcheck 的尾延迟。
5. 长期可以考虑多个 supervisor actor 分片管理 slot，降低单 actor 的队列压力。
