# Qwen3.5-35B-A3B MTP 补丁集

本目录的补丁让 MTP 在 RL(colocate)训练里**推理和训练两端都跑起来**。

## 1. sglang colocate + MTP/EAGLE 投机解码补丁
**文件**: `cudagraph_recapture_fix.patch` + `apply_cudagraph_patch.sh`
**目标**: `sglang/srt/managers/scheduler_update_weights_mixin.py`

修两件事:
- **rollout 乱码**: 在线权重更新(update_weights_from_tensor)后给 CUDA graph 打
  `_cuda_graphs_need_recapture` 标记,onload(resume_memory_occupation)时重捕获。
- **MTP/EAGLE colocate onload 崩溃**: spec-v2 的 `EAGLEWorkerV2` 没有 `.model_runner`,
  draft 的 ModelRunner 暴露为 `.draft_runner`。补丁加兜底:
  `getattr(draft_worker,"model_runner",None) or getattr(draft_worker,"draft_runner",None)`,
  否则 onload 时 `AttributeError: ... has no attribute 'model_runner'`。

应用(幂等,脚本启动时自动调用):
    bash apply_cudagraph_patch.sh
判定标记是 `draft_runner`;若检测到**旧版**补丁(只有 `_cuda_graphs_need_recapture`、
无 `draft_runner` 兜底)会报警告,需先 `git checkout` 还原该文件再重跑。

> 注意:此补丁改的是容器内 sglang 库文件,容器重建会丢,所以固化在此目录。

## 2. Megatron MTP 权重转换补丁(checkpoint 预处理)
**文件**: `preprocess_mtp_experts.py`

slime(#1702)的 Qwen3.5 MTP bridge 期望 MTP MoE experts 是 **individual**
(`mtp.layers.0.mlp.experts.{i}.gate_proj/up_proj/down_proj.weight`),
但原始 HF checkpoint 存的是 **fused**(`gate_up_proj [E,2F,H]` / `down_proj [E,H,F]`)。
不预处理直接转会报 `KeyError: mtp.layers.0.mlp.experts.0.gate_proj.weight`。

本脚本把 fused 拆成 per-expert,产出一个新 HF 目录(软链原 shard + 一个新 shard + 重写 index)。
**无需改 slime 任何代码**,用 stock bridge 即可。

用法:
    # 1) 预处理(拆 fused experts)
    python preprocess_mtp_experts.py /root/model_dist/Qwen3.5-35B-A3B \
                                     /root/model_dist/Qwen3.5-35B-A3B_mtp_individual
    # 2) 用拆好的目录做 megatron(torch_dist)转换,得到带 MTP 的 ckpt
    #    slime 转换的 --hf-checkpoint 指向 *_mtp_individual
    #    -> 产出 Qwen3.5-35B-A3B_torch_dist_mtp(含 540 个 MTP key,bit-exact)

## 3. slime MTP 模型补丁(submodule,无法直接提交)
**文件**: `slime_mtp_model_args.patch` + `apply_slime_mtp_patch.sh`
**目标**: `<slime>/scripts/models/qwen3.5-35B-A3B.sh`

slime 是 submodule(THUDM/slime),其改动无法提进 Dressage 库,故以补丁自愈:
给 MODEL_ARGS 加 `--mtp-num-layers 1`(开启 Megatron MTP 层)。
run 脚本启动期自动 `bash apply_slime_mtp_patch.sh`(幂等,已应用则 skip)。

## 训练脚本里的 MTP 开关(参考)
- **训练 loss**: `--enable-mtp-training` + `--mtp-loss-scaling-factor 0.1`;
  模型 sh 里 `--mtp-num-layers 1`;`--ref-load .../Qwen3.5-35B-A3B_torch_dist_mtp`。
- **推理加速**: `--sglang-speculative-algorithm EAGLE`
  `--sglang-speculative-num-steps 3 --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4`;
  sglang 的 `--hf-checkpoint` 仍指向**原始 fused** 目录。
- EAGLE mem:colocate 下 `--sglang-mem-fraction-static 0.75`(训练时已 offload,安全)。
- 实测 spec_accept_length ≈ 2.2–2.5。
