#!/bin/bash
# 幂等应用 slime MTP 模型补丁:给 qwen3.5-35B-A3B 的 MODEL_ARGS 加 --mtp-num-layers 1
# slime 是 submodule(THUDM/slime),改动无法直接提进 Dressage 库,故以补丁形式自愈。
# 用法: bash apply_slime_mtp_patch.sh   (run 脚本启动时自动调用)
set -e
PATCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PATCH_FILE="${PATCH_DIR}/slime_mtp_model_args.patch"
SLIME_ROOT="${SLIME_ROOT:-$(cd "${PATCH_DIR}/../slime" 2>/dev/null && pwd)}"
TARGET="${SLIME_ROOT}/scripts/models/qwen3.5-35B-A3B.sh"
if [[ ! -f "${TARGET}" ]]; then echo "[slime-mtp-patch] target not found: ${TARGET}, skip"; exit 0; fi
if grep -q "mtp-num-layers" "${TARGET}"; then
  echo "[slime-mtp-patch] already applied, skip"
elif patch "${TARGET}" --dry-run < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch "${TARGET}" < "${PATCH_FILE}"
  echo "[slime-mtp-patch] applied to ${TARGET}"
else
  echo "[slime-mtp-patch] WARNING: patch did not apply cleanly (slime version mismatch?), check manually" >&2
fi
