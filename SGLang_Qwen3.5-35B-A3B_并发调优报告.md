# SGLang 混合注意力模型推理调优报告

> 目标模型:**Qwen3.5-35B-A3B**(MoE + Hybrid Linear/Full Attention)
> 部署:8×H100 80GB,`--colocate` RL 训练 + rollout
> 参考脚本:`examples/scripts/run_blackbox_qwen3.5_35b_a3b_sync_local.sh`
> SGLang 源码版本:`v0.5.12.post1`(`dl/sglang/`)

---

## 0. 结论先行

调优目标是**推理效率最高**(吞吐 tokens/s、TPOT),不是最大并发数。两者不是一回事,且你当前配置在两个维度上都不优:

- **并发维度**:实际只跑到 **~12/引擎(总~48)**,而非设定的 256;
- **效率维度**:cuda graph 按默认 256 捕获(33 个点,含 127MB/点的 logits buffer),浪费显存;EAGLE 的 mamba 中间态在 256 并发下吃掉 23.6GB。

**效率最优配置(推荐,保留 EAGLE、右-size 到饱和拐点):**

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.80
   --sglang-disable-custom-all-reduce
   --sglang-reasoning-parser qwen3
   --sglang-tool-call-parser qwen3_coder
   --sglang-log-level warning
   --sglang-chunked-prefill-size 4096
   --sglang-max-prefill-tokens 16384
   --sglang-max-running-requests "${B_SAT:-96}"        # = 实测饱和拐点,不是 256
   --sglang-router-port "${SGLANG_ROUTER_PORT}"
   --router-policy consistent_hashing

   --sglang-speculative-algorithm EAGLE               # 保留:decode 吞吐倍增器
   --sglang-speculative-num-steps 2
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 3

   --sglang-mamba-scheduler-strategy extra_buffer
   --sglang-mamba-full-memory-ratio 0.5              # 2.0→0.5(方向反了:KV 应得 2× mamba)
   --sglang-mamba-ssm-dtype bfloat16                 # 递归状态 fp32→bf16,per_req 砍半(需校验精度)
   --sglang-cuda-graph-max-bs "${B_SAT:-96}"          # 对齐拐点:省捕获显存,拐点后零损失
   --sglang-enable-metrics
)
```

`B_SAT` 须实测(§13),经验起点 96,跑通后上调到吞吐拐点(多落在 96–160)。

**最大并发替代档**(上下文极长、KV 是硬瓶颈时):删 EAGLE、`max_running_requests=128`、`cuda_graph_max_bs=160`,见 §12。

下文给出完整推导。

---

## 1. 模型架构事实(来自 HF config.json)

`Qwen/Qwen3.5-35B-A3B` 的 `text_config`:

| 字段 | 值 | 含义 |
|---|---|---|
| `num_hidden_layers` | 40 | |
| `layer_types` | 30× `linear_attention` + 10× `full_attention` | 每 4 层 1 层 full(`full_attention_interval=4`) |
| `linear_num_value_heads` | 32 | GDN value 头数 |
| `linear_value_head_dim` | 128 | value 头维 |
| `linear_num_key_heads` | 16 | GDN key 头数(n_groups) |
| `linear_key_head_dim` | 128 | SSM state_size |
| `linear_conv_kernel_dim` | 4 | ShortConv 卷积核 |
| `num_attention_heads` | 16 | full attention 头数 |
| `num_key_value_heads` | 2 | GQA KV 组数 |
| `head_dim` | 256 | full attention 头维 |
| `mamba_ssm_dtype` | `float32` | SSM 时序状态精度(4B) |
| `num_experts` / `num_experts_per_tok` | 256 / 8 | **激活参数 ~3B(A3B)** |
| `hidden_size` | 2048 | |

**线性层占 75%(30/40)**——sglang 必须为每请求的固定递归状态开 mamba cache。slime 侧实现见 `slime/slime_plugins/models/qwen3_5.py`:每 4 层 1 层 full GQA,其余替换为 `Qwen3_5GatedDeltaNet`(门控 delta 规则线性注意力,类 Mamba SSM)。

---

## 2. SGLang 四资源耦合模型

sglang 把这类模型识别为 `mambaish_config`。四个资源通过 **`max_mamba_cache_size`(请求粒度状态槽)** 耦合:

```
剩余显存 rest_memory
   ├── (EAGLE 投机先扣) per_req × max_running_requests × speculative_num_draft_tokens
   └── 按 mamba_full_memory_ratio 切分:
          mamba_state_memory ──/ per_req ──► max_mamba_cache_size (槽数)
          kv_cache_memory    ──/ per_token ──► max_total_num_tokens (token 池)

并发上限 = min(
    max_running_requests,
    max_mamba_cache_size // R,        # R = mamba 槽/请求倍数
    max_total_num_tokens / avg_ctx   # KV token 池
)
其中 R:
  extra_buffer + overlap  → 5     (3 基础 + 2 ping-pong)
  extra_buffer + no-overlap → 4
  no_buffer (radix on)    → 3

吞吐 = 并发 × decode速率  (decode速率受 EAGLE 加速比 s_eagle 与 cuda graph 影响)
```

关键代码(v0.5.12.post1):
- 内存切分方程 / EAGLE 预留:`python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py:78-137`
- 并发上限反算:`同文件:818-836`
- 倍数常数:`同文件:51-53`(`RATIO=3`、`OVERLAP=2`、`NO_OVERLAP=1`)
- mamba pool 分配 + ping-pong:`python/sglang/srt/mem_cache/memory_pool.py:490-617`
- extra_buffer / no_buffer 校验:`python/sglang/srt/server_args.py:2510-2571`
- CUDA graph 捕获集生成:`python/sglang/srt/server_args.py:1604-1634`
- cuda graph 中 mamba_track 开关:`python/sglang/srt/model_executor/cuda_graph_runner.py:698-701`

---

## 3. 推导:每请求 mamba 状态字节数

来自 `configs/mamba_utils.py:116-125`(`mamba_cache_per_req`)与 `configs/qwen3_next.py:283-302`(`Mamba2StateShape.create`),TP=2:

```
intermediate_size = head_v_dim × num_v_heads = 128 × 32 = 4096
conv_dim          = intermediate_size + 2 × n_groups × state_size
                  = 4096 + 2 × 16 × 128 = 8192
conv_state_shape  = (conv_dim / TP, conv_kernel - 1) = (8192/2, 3) = (4096, 3)
conv_numel        = 4096 × 3 = 12 288          (dtype bf16 → 2B)

temporal_state_shape = (num_v_heads / TP, head_dim, state_size)
                     = (32/2, 128, 128) = (16, 128, 128)
ssm_numel            = 16 × 128 × 128 = 262 144   (dtype float32 → 4B / bf16 → 2B)

每层每请求 = conv_numel×2 + ssm_numel×(ssm_dtype_bytes)
  fp32: 24 576 + 1 048 576 = 1 073 152 B
  bf16: 24 576 +   524 288 =   548 864 B

× 30 个线性层:
  fp32: 32 194 560 B ≈ 30.7 MB / 请求
  bf16: 16 465 920 B ≈ 15.7 MB / 请求
```

> 每 TP rank 的值(temporal 与 conv 都已 `divide(..., tp_world_size)`)。

## 4. 推导:每 token KV 字节数

full attention 层(10 层),GQA(`num_kv_heads=2`),TP=2,bf16:

```
每 token 每层 = (num_kv_heads / TP) × head_dim × 2(K+V) × dtype_bytes
            = (2/2) × 256 × 2 × 2 = 1024 B
× 10 层 = 10 240 B ≈ 10 KiB / token   (每 rank)
```

---

## 5. 诊断:当前配置为何并发被限死

当前参数:EAGLE on(`num_draft_tokens=3`)、`mamba_ssm_dtype=float32`、`ratio=2.0`、`extra_buffer`+overlap(÷5)、`max_running_requests=256`、`mem_fraction_static=0.8`。

设 `rest ≈ 28 GB`(权重 35B bf16 / TP2 ≈ 35GB/rank 后的余量,以启动日志 `available KV cache memory` 为准):

**① EAGLE 投机先扣的 mamba 中间态**(`model_runner_kv_cache_mixin.py:84-98`):

```
= per_req × max_running_requests × speculative_num_draft_tokens
= 30.7 MB × 256 × 3 = 23.6 GB     ← 几乎吃光 rest
```

**② 剩余按 ratio=2.0 切分(mamba 拿 2/3):**

```
剩余 = 28 − 23.6 = 4.4 GB
mamba pool = 4.4 × 2/3 = 2.9 GB → 槽 = 2900/30.7 ≈ 95 → 并发 = 95 // 5 = 19
KV pool    = 4.4 × 1/3 = 1.5 GB → token = 1.5e9/10240 ≈ 146k → 并发(12k 均长)≈ 12
```

**实际并发 ≈ min(256, 19, 12) ≈ 10–12 / 引擎**(×4 引擎 ≈ 40–48 总)。

三个根因:
- **A. EAGLE 在 hybrid 模型上的乘性成本**:`per_req × M × draft_tokens`,dense 模型只多一份 draft KV,hybrid 乘以每请求全量递归状态 ×3。
- **B. `ratio=2.0` 方向反了**:KV 是按 token 线性增长的大头,2/3 给 mamba 饿死 KV。
- **C. `mamba_ssm_dtype=float32`** 使 per_req 翻倍(30.7 vs 15.7MB)。

---

## 6. 关键翻转:并发 ≠ 效率,存在饱和拐点 B_sat

§5 优化的"并发上限"不等于"推理效率最高"。decode 是带宽/算力 bound,吞吐随并发上升直到**饱和拐点 B_sat**,之后:

- 吞吐**基本不增**(已被带宽/算力限死),
- KV token 池压力**线性上升** → 触发 retraction/recompute → 反而变慢。

```
吞吐 ───────────╱────── 饱和(B_sat)
              ╱
            ╱
          ╱
        ───────────────────── 并发
        0     B_sat        N_max
```

**效率最优 ≈ 刚过 B_sat,而不是 N_max。** 盲目冲并发(如 §5 反推的 127)在 B_sat 之后纯属浪费显存。

## 7. A3B MoE 的特殊性:B_sat 偏高

`num_experts_per_tok=8 / 256`,**激活参数仅 ~3B**,远小于 35B 总量。每 token decode 实际加载的权重小 → 单 token 带宽成本低 → 需要更大 batch 才能打满 GPU:

- dense 35B:B_sat 可能 ~32–64 就饱和;
- **A3B MoE:B_sat 明显更高(~96–192,需实测)**——高并发在此模型上 *确实* 能换吞吐,直到拐点。

这决定了:**对 A3B,冲并发到 B_sat 是有意义的**(不是纯浪费),最优策略是"对齐 B_sat"而非"砍到极小"或"冲到 N_max"。

---

## 8. CUDA graph 右-sizing:三重收益

`cuda_graph_max_bs` 决定捕获集(`server_args.py:1604-1634`):

- 非 spec: `[1,2,4,8,12] + range(16,257,8) + ...` → max_bs=256 时约 **33 个**捕获点;
- spec(EAGLE)on: `[1..8]+[10..32 step2]+[40..64 step4]+...` → 更密,约 **45 个**且每个含 draft buffer。

每个捕获点占静态 buffer:logits(`max_bs × vocab 248320 × 2B`,256 时单点 127MB)、hidden、MoE buffer,以及 **mamba track buffer**。后者取决于 `cuda_graph_runner.py:698-701`:

```
enable_mamba_track = enable_mamba_extra_buffer() and spec_algorithm.is_none()
```

| EAGLE | cuda graph 里 mamba_track | mamba 状态成本落在哪 |
|---|---|---|
| off | **ON**(∝ cuda_graph_max_bs) | cuda graph 捕获 buffer |
| on | **OFF** | EAGLE 中间态 pool(∝ max_running_requests) |

把 `cuda_graph_max_bs` 从 256 降到 B_sat(如 96):

1. 捕获点 33→~16,buffer 尺寸 256→96(37%),**省下数 GB** 给 KV/mamba/EAGLE;
2. 拐点之后本就无吞吐收益,**零效率损失**;
3. EAGLE off 时同步缩小 mamba track buffer。

**"cuda graph 太大"的直接解法:对齐 B_sat。**

---

## 9. EAGLE 重新评估:效率场景下应保留

§5 诊断里 EAGLE 是并发杀手,但它是**真实的 decode 吞吐倍增器**(verify 步分摊权重加载),删它丢 ~1.5–2× decode 速度。效率视角判据是吞吐/延迟:

**吞吐 break-even 公式**(两者都在拐点前,under-saturated):

```
EAGLE 更优  ⟺  s_eagle × N_eagle  >  N_noeagle
其中 s_eagle = 有效加速比(num_steps=2,topk=1,实测 ~1.5–2.0)
```

A3B 的 B_sat 偏高 → 若不 EAGLE 也到不了 B_sat(under-saturated),则 EAGLE 的 s_eagle 直接乘进吞吐,**净正**。前提:EAGLE 中间态可负担——用 bf16 ssm + `max_running_requests=B_sat`(而非 256):

```
M=96, bf16: 中间态 = 15.7MB × 96 × 3 = 4.5 GB   (256 时 23.6GB,砍 80%)
```

→ **效率最优保留 EAGLE**,通过 bf16 + 右-sized M 压到可负担。

## 10. ratio 最优公式

令 mamba 并发 = KV 并发(extra_buffer+overlap 即 ÷5):

```
mamba_reqs = rest × r/(1+r) / per_req / 5
kv_reqs    = rest × 1/(1+r) / kv_per_token / avg_len
令其相等 →  r_opt = 5 × per_req / (kv_per_token × avg_len)
```

代入 `kv_per_token=10240 B`:

| avg 上下文 | r_opt(bf16 ssm, per_req=15.7MB) | r_opt(fp32 ssm, per_req=30.7MB) |
|---|---|---|
| 8k  | 0.96 | 1.88 |
| 12k | 0.64 | 1.25 |
| 16k | 0.48 | 0.94 |

agent rollout 上下文普遍 12–16k(脚本 `CONTEXT_WINDOW 6144` + `ROLLOUT_MAX_RESPONSE_LEN 10240`),故 **bf16 下 r_opt ≈ 0.5–0.64**。`2.0` 仅在 ctx≈3.7k 时才最优——与负载完全不符。

> 实际取 `r` 时让 mamba 与 KV 并发都在 B_sat 之上(留 ~15% 余量),不必严格等于上表。

---

## 11. 效率最优配置与验算

见 §0 配置块。`B_SAT=96` 验算(frac 0.82,rest≈30GB):

```
EAGLE 中间态 = 15.7MB × 96 × 3 = 4.5 GB → 剩 25.5 GB
mamba pool  = 25.5 × 0.5/1.5 = 8.5 GB → 槽 541 → 并发 541//5 = 108 (>96 ✓)
KV pool     = 25.5 × 1/1.5 = 17 GB → 1.66M tok → /12k = 138 (>96 ✓)
并发 = 96(由 B_sat 卡,故意)
decode 吞吐 ≈ 96 × s_eagle   vs  无EAGLE ≈ 127 × 1.0
s_eagle > 1.32 即 EAGLE 净胜 → 实测 1.5–2.0 → 明显更优
```

mamba(108)、KV(138)均 > B_sat(96),说明缓存未触底、由拐点卡定——这正是效率最优的标志(资源有余量、不浪费)。

## 12. 最大并发替代档

上下文极长、KV 是硬瓶颈、要同时挂最多请求时:

```bash
   # 删整段 EAGLE(speculative-*)
   --sglang-max-running-requests 128          # 可达上限而非 256
   --sglang-cuda-graph-max-bs 160            # 对齐 128 留余量
   --sglang-mamba-full-memory-ratio 0.5
   --sglang-mamba-ssm-dtype bfloat16
   --sglang-mem-fraction-static 0.82
```
验算:无 EAGLE 预留,mamba=10GB→127 并发,KV=20GB→162,实际 ~127/引擎。

| 目标 | 配置要点 | 适用 |
|---|---|---|
| **最高吞吐/最低延迟**(主) | 保留 EAGLE、N=B_sat、**cuda_graph=B_sat**、bf16 ssm | A3B 未饱和、decode 占比高 |
| **最大并发**(备) | 删 EAGLE、N=127、cuda_graph=160 | 上下文极长、KV 是硬瓶颈 |

关键差异:**EAGLE 留还是删**、**cuda_graph_max_bs 对齐 B_sat 还是留大**。

---

## 13. 找到真正最优:profiling B_sat

静态公式给不出 B_sat,必须实测:

1. 固定 §0 配置,`B_SAT` 依次设 48/96/128/160,跑同一批 rollout prompt。
2. 读 `--sglang-enable-metrics` 的 `gen_throughput`(tokens/s/engine)与 `decode_ts`(TPOT)。
3. 画 吞吐~并发 曲线,找拐点 B_sat(吞吐不再显著上升处)。
4. 回填 `max_running_requests = B_sat`、`cuda_graph_max_bs = B_sat`(取下一个 cuda graph 捕获点,如 96/128/160)。
5. 用 §10 公式微调 `mamba_full_memory_ratio`,使 mamba 与 KV 并发都在 B_sat 之上(留 ~15% 余量)。
6. 若 EAGLE 实测 `s_eagle<1.3` 或 acceptance 低,退回 §12 删 EAGLE 方案。

> 经验起点:A3B MoE on H100/TP2,B_sat 多落在 96–160;先用 96 跑通,再上调至吞吐拐点。

---

## 14. 验证清单

1. 启动日志读 `available KV cache memory` 与 `max_mamba_cache_size`,反推实际 `rest` 与槽数,与本文公式对账。
2. bf16 ssm:对比 rollout 通过率 / reward 与 fp32 基线(Qwen3.5-35B-A3B 在 bf16 下的数值稳定性需实测);不可接受则保留 fp32,此时 r_opt 升到 ~0.9–1.25,仍远低于 2.0。
3. colocate 训练:确认 rollout 阶段 `mem_fraction_static=0.82` 不与训练激活竞争导致 OOM(关注 weight-update 后 rollout 恢复)。
4. 若保留 EAGLE:`max_running_requests` 必须满足自洽方程 `M = (rest − per_req×M×draft_tokens×1e-9) × r/(1+r) / per_req / 5`,迭代求解,不可直接设 256。
5. slime 是否转发 `--sglang-mamba-ssm-dtype` / `--sglang-cuda-graph-max-bs` 取决于部署版白名单;不支持时前者改用环境变量 `SGLANG_MAMBA_SSM_DTYPE=bfloat16`(`configs/mamba_utils.py:95`),后者影响较小保留默认。

---

## 15. 一句话总结

**最高效率 = 把 `max_running_requests` 与 `cuda_graph_max_bs` 都对齐到实测饱和拐点 B_sat,保留 EAGLE(bf16 ssm 使其可负担),ratio 用 `r_opt = 5×per_req/(kv_per_token×avg_len)` 微调。** cuda graph 大不是要忍,而是因为它没对齐拐点——右-size 它同时省显存、保效率。

---

## 16. 关键代码索引(SGLang v0.5.12.post1)

| 主题 | 位置 |
|---|---|
| 内存切分方程 / EAGLE 预留 / ratio 切分 | `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py:78-137` |
| 并发上限反算(max_mamba_cache_size // ratio) | 同文件 `:818-836` |
| mamba 倍数常数(3/+1/+2) | 同文件 `:51-53`,`:186-198` |
| mamba pool 分配 + ping-pong | `python/sglang/srt/mem_cache/memory_pool.py:490-617` |
| extra_buffer / no_buffer 校验 | `python/sglang/srt/server_args.py:2510-2571` |
| `enable_mamba_extra_buffer` | `server_args.py:7220` |
| `cuda_graph_max_bs` 默认档位 | `server_args.py:1425-1488` |
| CUDA graph 捕获集生成 | `server_args.py:1604-1634` |
| cuda graph 中 mamba_track 开关 | `python/sglang/srt/model_executor/cuda_graph_runner.py:698-701` |
| `mamba_cache_per_req` 公式 | `python/sglang/srt/configs/mamba_utils.py:116-125` |
| Qwen3.5 cache params | `python/sglang/srt/configs/qwen3_next.py:283-302` |
| `SGLANG_ENABLE_SPEC_V2` 默认 True | `python/sglang/srt/environ.py:470` |
| `SGLANG_MAMBA_SSM_DTYPE` 环境变量 | `python/sglang/srt/configs/mamba_utils.py:95` |
