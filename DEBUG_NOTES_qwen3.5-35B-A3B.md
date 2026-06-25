# Qwen3.5-35B-A3B RL 调试笔记与修改记录

记录在单机 8×H100 上跑 `examples/scripts/run_blackbox_qwen3.5_35b_a3b_sync_local.sh`
(slime + Megatron 训练,sglang 推理,opencode/bwrap 沙箱 agentic rollout)过程中
定位的问题、做的修改、以及尚未解决的根因。

---

## 0. TL;DR — 改了哪些

| # | 改动 | 文件 / 位置 | 解决的问题 |
|---|---|---|---|
| 1 | `BASE_FOLDER` 默认 `/root/model_dist` | run 脚本 | proxy 加载 tokenizer 失败 |
| 2 | `--sglang-disable-custom-all-reduce` | run 脚本 SGLANG_ARGS | **sglang cuda graph capture 失败**(见 §2) |
| 3 | `--router-policy round_robin` | run 脚本 SGLANG_ARGS | 单引擎塌缩(见 §3) |
| 4 | `make_runtime_id()` 固定为 `"bbs-session"` | `blackbox_server/runtime/paths.py` | 前缀缓存命中 0(见 §4) |
| 5 | bwrap `--setenv OMP/OPENBLAS/MKL/NUMEXPR=1` | `dressage/sandbox/local/bwrap/runner.py` | 沙箱 code-exec 抢满 CPU(见 §5) |
| 6 | 沙箱池 `dressage_apply_local_bwrap_defaults 16` | run 脚本 | 并发被 8 槽卡死(见 §5) |
| 7 | `--rollout-num-gpus-per-engine 2`(4 引擎 TP2) | run 脚本 | TP4 通信浪费 |
| 8 | `--rollout-batch-size 24`、`--sglang-max-prefill-tokens 16384` | run 脚本 | 喂大并发 / 长 prompt prefill |
| — | `--log-probs-chunk-size` 调小、mem-fraction 取舍 | run 脚本 | 训练侧 OOM(见 §6) |

> ⚠️ 这些是吞吐/正确性的工程修复;**两个根因未解决**:NVSwitch 硬件故障(§7)、
> rollout 训练后乱码 = 权重转换 bug(§8)。

---

## 1. 启动 / 路径
模型在 `BASE_FOLDER=/root/model_dist`,三份:`Qwen3.5-35B-A3B`(HF,给 sglang)、
`_torch_dist`(Megatron,`--ref-load`)、`_slime`(训练 checkpoint,`--load/--save`)。
HF 与 torch_dist 必须同一份权重两种格式;`--ref-load` 只能吃 Megatron 格式。

```bash
cd /root/Dressage
BASE_FOLDER=/root/model_dist nohup bash examples/scripts/run_blackbox_qwen3.5_35b_a3b_sync_local.sh \
  > log/run_main.log 2>&1 &
```

## 2. sglang CUDA graph capture 失败(custom_all_reduce)
**现象**:多次 CUDA error 999 之后,sglang 起不来:
`Capture cuda graph failed: custom_all_reduce.cuh:37: CUDA error: invalid argument`。
**原因**:sglang 自定义 all-reduce 在 cuda graph capture 时建 P2P/IPC,在被污染的 CUDA
状态下报 invalid argument。**修复**:`--sglang-disable-custom-all-reduce`(改用 NCCL allreduce,
绕过该 capture 路径)。注意:`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 与 colocate 的
`torch_memory_saver` 冲突,**不能全局设**。

## 3. 单引擎塌缩(router 默认 cache_aware)
**现象**:4 个引擎(TP2)只有 1 个在跑、其余 6 卡闲。
**原因**:未设 `--router-policy` 时 sglang-router 默认 **`cache_aware`**;它按前缀相似度把请求
发往同一 worker。每个请求都带**完全相同的 ~7.4k opencode system prompt**,在 cache_aware 看来
前缀全一样 → **全部塞给同一个引擎**。**修复**:`--router-policy round_robin`(均摊到 4 引擎,
约 +40% 总吞吐且不浪费 GPU)。

## 4. 前缀缓存命中 0(工作目录唯一)
**现象**:`prefix_cache_hit_rate=0`,~7.4k 样板每样本重复 prefill。
**原因**:opencode 把**每 session 唯一的工作目录** `bbs-<时间戳>-<uuid>` 写进 system prompt 的
`<env>` 块 → prompt 文本每条都变 → radix 前缀缓存对不上。注意:缓存认的是 **prompt 文本**
(目录名在里面),不是磁盘内容。**修复**:`make_runtime_id()` 返回固定 `"bbs-session"`,
`ensure_runtime_dir` 改为先清残留再建(各 slot 的 runtime_root 在 host 上独立 → 并发安全;
binding 串行)。

## 5. 并发瓶颈 = 沙箱池(不是 sglang)
**现象**:提交 192 序列,但任意时刻只有 ~8 个在生成(actGen≈8),sglang 严重吃不饱。
**原因**:每条轨迹要占一个 opencode 沙箱 server,**沙箱池只有 8 个槽**
(`DRESSAGE_LOCAL_BWRAP_TOTAL_SERVERS=8`)→ 并发被卡在 8。**修复**:
`dressage_apply_local_bwrap_defaults 16`(池→16)。沙箱大部分时间在等模型(I/O),idle 几乎不耗 CPU,
且 colocate 下 rollout 与 train 分相位(train 时沙箱睡),所以可 oversubscribe。
**但**模型生成的代码(numpy/scipy/BLAS)会抢满核 → bwrap 加 `--setenv OMP_NUM_THREADS=1`
(及 OPENBLAS/MKL/NUMEXPR)**只钉沙箱内、不影响训练侧 CPU 优化器**。

## 6. 训练侧 OOM
`fused_vocab_parallel_cross_entropy` 显存爆 → 调小 `--log-probs-chunk-size`
(buffer = chunk×(vocab/TP)×4B,且 clone 翻倍);colocate 下 `--sglang-mem-fraction-static`
别太高(挤训练显存)。

## 7. ⚠️ 未解决根因 A:NVSwitch 硬件故障
节点 NVSwitch3 反复 `dmesg: SXid 22013 Non-fatal Minion Link DLREQ interrupt`(多条 link:
61/46/42/38/...),多条 NVLink `Replay Errors` 95~192(健康应为 0)。每隔约 15–60 分钟打断在飞的
NCCL → `Cuda failure 999` → 训练崩 / 或 sglang `ncclCommInitRank` 起不来。容器内 GPU 无法 reset。
**判据**:崩溃栈在 NVLink P2P(CP attention / 权重广播 / NCCL init);带 SXid=硬件。
**唯一真解:换节点 / 报修。** 排查:`dmesg -T | grep -i sxid`、`nvidia-smi nvlink -e`。

## 8. ⚠️ 未解决根因 B:训练一步后 rollout 乱码(权重转换 bug)
**现象**:rollout #0(sglang 直接加载 HF)正常;第一次 `update_weights`(Megatron→sglang)后,
rollout 全是多语言乱码 token、`finish_reason=length`,reward→0。
**定位**:colocate + 默认 `--megatron-to-hf-mode raw` → 转换走手写函数
`convert_qwen3_5_to_hf`(`slime/backends/megatron_utils/megatron_to_hf/qwen3_5.py`),
**不经 mbridge / megatron.bridge**。可疑:gated-attention 的 QKV split/reshape(q_proj 含
[query;output_gate])、MoE fused grouped-GEMM expert 布局 / EP 聚合,导出的 HF 权重与 sglang
期望不一致 → 第一次同步后即乱码。(另:`slime_plugins/mbridge/qwen3_5.py` 的 bridge 模式
`_weight_to_hf_format` 缺 gated/MoE 逆变换,是另一条路径上的真 bug,但 raw 模式不走它。)
**状态:未修,需对 `convert_qwen3_5_to_hf` 的特殊层导出做正确性核对。**

---

## 9. 任务 / 数据说明
`examples/data/dressage_dapo_prompts.jsonl`(3000 条)全为 `task_type: math` 纯 CoT 题。
但跑在 `blackbox_type=opencode` agent 里:每条轨迹由沙箱内的 opencode agent 驱动,system prompt
是 opencode 的(~9.7k 字符)+ 10 个工具(~21.7k 字符 JSON)→ prompt ~96% 是框架样板,题目仅 ~4%。
模型**自主**决定:大多数(~94%)直接 CoT,少数算量大的题(~6%)写 Python 在沙箱里跑(真执行)。
截断(~28%)是单次回复撞 `--rollout-max-response-len 8192`(非 context 撑爆,context 才用到 ~16k/32768)
→ 需要可放大到 16384。

## 10. 怎么拉 sglang 指标(log warning 压日志时)
```bash
IP=10.244.31.217   # 引擎 bind 内网 IP
curl -s http://$IP:15000/server_info | python3 -c \
 'import sys,json;d=json.load(sys.stdin)["internal_states"];i=d[0] if isinstance(d,list) else d;print(i["last_gen_throughput"],i["memory_usage"])'
curl -s http://$IP:8000/workers          # 各引擎 load(活跃请求分布)
curl -s http://$IP:5030/metrics | grep smg_worker   # router 每 worker 指标 / policy
curl -s http://$IP:8800/health           # proxy active_sglang_generations
```
`internal_states` 可能是 list(取 `[0]`);`token_capacity` 是**每引擎**的。
