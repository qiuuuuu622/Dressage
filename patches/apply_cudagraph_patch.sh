#!/bin/bash
# 幂等应用 SGLang CUDA graph 重捕获补丁
#   修复1: 在线权重更新(rollout->train->rollout)后 CUDA graph 未重捕获导致 rollout 乱码
#   修复2: colocate + MTP/EAGLE 投机解码 onload 时崩溃
#          (EAGLEWorkerV2 没有 .model_runner,draft ModelRunner 暴露为 .draft_runner)
# 用法: bash apply_cudagraph_patch.sh   (脚本启动时自动调用)
set -e
PATCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PATCH_FILE="${PATCH_DIR}/cudagraph_recapture_fix.patch"
SGLANG_DIR="$(python3 -c "import os,sglang; print(os.path.dirname(sglang.__file__))" 2>/dev/null || true)"
if [[ -z "${SGLANG_DIR}" ]]; then echo "[cudagraph-patch] sglang not importable, skip"; exit 0; fi
TARGET="${SGLANG_DIR}/srt/managers/scheduler_update_weights_mixin.py"
if [[ ! -f "${TARGET}" ]]; then echo "[cudagraph-patch] target not found: ${TARGET}, skip"; exit 0; fi
# 用 draft_runner 这个唯一标记判断是否已是「带 EAGLEWorkerV2 兜底」的新版补丁
if grep -q "draft_runner" "${TARGET}"; then
  echo "[cudagraph-patch] already applied (draft_runner fallback present), skip"
elif grep -q "_cuda_graphs_need_recapture" "${TARGET}"; then
  echo "[cudagraph-patch] WARNING: 检测到旧版补丁(无 draft_runner 兜底),MTP/EAGLE colocate 会崩;请先 git checkout 还原该文件再重跑本脚本" >&2
elif patch "${TARGET}" --dry-run < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch "${TARGET}" < "${PATCH_FILE}"
  echo "[cudagraph-patch] applied to ${TARGET}"
else
  echo "[cudagraph-patch] WARNING: patch did not apply cleanly (sglang version mismatch?), check manually" >&2
fi
