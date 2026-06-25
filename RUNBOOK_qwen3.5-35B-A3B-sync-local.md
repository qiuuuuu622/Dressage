# Qwen3.5-35B-A3B Dressage Sync-Local 运行与调优笔记

本文记录在单机 8×H100 上,用 `examples/scripts/run_blackbox_qwen3.5_35b_a3b_sync_local.sh`
跑 Qwen3.5-35B-A3B(MoE,3B 激活)RL 训练(slime + Megatron 训练,sglang 推理,
bwrap 沙箱 agentic rollout)的踩坑、修复与推理调优结论。

---

## 1. 快速启动

模型权重需放在 `BASE_FOLDER` 下(默认 `/root/model_dist`),包含三份:

| 路径 | 格式 | 用途 |
|---|---|---|
| `Qwen3.5-35B-A3B/` | HuggingFace(config.json + safetensors) | sglang 推理 + tokenizer(`--hf-checkpoint`) |
| `Qwen3.5-35B-A3B_torch_dist/` | Megatron 分布式 checkpoint(`release/`) | 训练 ref / 冷启 policy(`--ref-load`) |
| `Qwen3.5-35B-A3B_slime/` | Megatron 训练 checkpoint | policy 加载/保存(`--load`/`--save`) |

启动:

```bash
cd /root/Dressage
BASE_FOLDER=/root/model_dist nohup bash examples/scripts/run_blackbox_qwen3.5_35b_a3b_sync_local.sh \
  > /root/Dressage/log/run_main.log 2>&1 &
```

> ⚠️ HF 与 torch_dist 必须是**同一份权重的两种格式**;`--ref-load` 只能吃 Megatron
> 格式,**不能**指向 HF 文件夹。

---

## 2. 并行与序列长度

- 训练并行:`TP=2, PP=1, CP=4, EP=8`(world=8 → DP=1)。
- **最大训练序列 = `MAX_TOKENS_PER_GPU(8192) × CP`**;CP=4 → 32768。proxy 的
  `--context-window` 即由此算出(32768)。CP=2 则砍半到 16384。
- 只有**训练**用 CP(ring attention,full/softmax 注意力);**sglang 推理不用 CP**。

---

## 3. 已修复的坑

| 问题 | 现象 | 修复 |
|---|---|---|
| `BASE_FOLDER` 默认 `/root` | proxy 加载 tokenizer 失败(HFValidationError) | 默认改为 `/root/model_dist` |
| `PYTHONBUFFERED=16` 拼写错误 | 日志块缓冲,进度看着"卡住" | 应为 `PYTHONUNBUFFERED=1`;实时看 proxy 日志 |
| sglang `custom_all_reduce` 起不来 | `custom_all_reduce.cuh: CUDA error: invalid argument` | 加 `--sglang-disable-custom-all-reduce`(改用 NCCL allreduce) |
| `expandable_segments` 全局生效 | sglang `TorchMemorySaver disabled ... not supported` | **不要**全局设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(与 colocate offload 冲突) |
| 训练侧 OOM | `fused_vocab_parallel_cross_entropy` 显存爆 | 调小 `--log-probs-chunk-size`(buffer = chunk×(vocab/TP)×4B,且 clone 翻倍);colocate 下别把 `mem-fraction-static` 调太高 |

---

## 4. sglang 推理调优

诊断方法(`--sglang-log-level warning` 压了日志、`/metrics` 未开时):

```bash
IP=10.244.31.217   # 引擎 bind 的内网 IP,不是 127.0.0.1
curl -s http://$IP:15000/server_info | \
  python3 -c 'import sys,json; d=json.load(sys.stdin)["internal_states"];
              i=d[0] if isinstance(d,list) else d;
              print(i["last_gen_throughput"], i["memory_usage"])'
```
> `/get_server_info` 已弃用,用 `/server_info`。`internal_states` 可能是 list(取 `[0]`)。
> KV `token_capacity` 是**每引擎**的(不是全局)。

**引擎布局**:`--rollout-num-gpus-per-engine` 决定引擎数 = 8 / 该值。
- `4` → 2 引擎 TP4(3B 激活下 TP 通信浪费)
- `2` → **4 引擎 TP2**(TP 通信减半 + 负载更分散,推荐;实测不 OOM)

**关键结论(实测)**:
- 瓶颈最初是 **batch 饿死**(并发 = `rollout-batch-size × n-samples`)。KV 用不满
  (164 万 token/引擎),提 `mem-fraction` 无用,**该提的是 batch**。
- `rollout-batch-size 8→16→24` 后,4 引擎真实聚合吞吐到 **~1100–1500 tok/s**。
- 之后瓶颈转移为:**① 批次排空震荡**(单轮 churn + sync rollout 一次性提交不回填,
  decode batch 单调缩水,吞吐在 ~190↔375/引擎 之间摆)+ **② 引擎间不均**
  (`consistent_hashing` → 改 `round_robin`)。
- **GPU util 只有 ~75% 不是算力瓶颈**:decode 是显存带宽瓶颈,且小/排空批次下每步
  调度开销 + TP/MoE 通信停顿造成 kernel 间空隙。根治靠:维持更大持续批次
  (async / partial-rollout 回填)、修 NVLink、或上投机解码。

**当前推荐配置**(SGLANG_ARGS / ROLLOUT_ARGS):
```
--rollout-num-gpus-per-engine 2      # 4 引擎 TP2
--rollout-batch-size 24              # 192 并发(96/引擎,KV 安全)
--sglang-max-running-requests 128    # 每引擎上限,未顶到
--sglang-max-prefill-tokens 16384    # 长 prompt(~7.6k)prefill 更快
--router-policy round_robin          # 削引擎不均
--sglang-mem-fraction-static 0.6     # colocate 下给训练留余量
```

> `--moe-enable-deepep` / `--moe-token-dispatcher-type flex` 是**训练侧** MoE EP 通信
> 加速,不影响 sglang;且会更压 NVLink,NVLink 不健康时建议关。

---

## 5. 已知硬件问题 ⚠️

部署节点的 **NVSwitch3 故障**:`dmesg` 反复 `SXid 22013 Non-fatal Minion Link DLREQ
interrupt`,多条 NVLink `Replay Errors` 高达 95~192(健康应为 0)。每隔约 15–60 分钟
打断在飞的 NCCL → `Cuda failure 999 'unknown error'` → 训练崩。容器内 GPU 无法 reset。

**判据**:崩溃栈在 NVLink P2P(CP attention `flash_attn_p2p_communicate` / 权重广播);
带 SXid = 硬件,不带 = 软件。**这不是脚本能修的,需换节点 / 报修。**

排查命令:
```bash
dmesg -T | grep -iE 'sxid|xid'
nvidia-smi nvlink -e        # 看各链路 Replay/CRC Errors
```

---

## 6. 任务说明

当前数据集(`examples/data/dressage_dapo_prompts.jsonl`,3000 条)全为 `task_type: math`
的纯 CoT 题(提示词要求逐步推理 + `\boxed{}` 答案),**不触发工具/沙箱 → 单轮**
(`segments_per_trajectory=1`)。若要 agentic 多轮(写代码 + 沙箱执行),需换成需要工具
的任务,或改提示词要求"写代码并执行"。

观察到的训练健康指标:`raw_reward`、`truncated_ratio`(早期高截断 = 回复撞 8192 上限,
需放大 `--rollout-max-response-len` 或确认推理质量)。
