#!/bin/bash
# Idempotently apply the SGLang DeepGEMM JIT activation guard.
#
# Fixes Qwen3.5/Qwen3 A3B DeepGEMM startup crashes such as:
#   silu_and_mul_masked_post_quant.cuh:319 RuntimeCheck(num_threads >= num_experts)
#
# Upstream:
#   sgl-project/sglang#26025 - fallback DeepGEMM activation for unsupported shapes
#   sgl-project/sglang#27377 - add the missing D // 8 < E guard for Qwen 3.5 35B
set -e

PATCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PATCH_FILE="${PATCH_DIR}/deepgemm_jit_activation_guard.patch"

TARGET=""
if [[ -n "${SGLANG_ROOT:-}" ]]; then
  TARGET="${SGLANG_ROOT}/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py"
else
  SGLANG_PKG_DIR="$(python3 -c "import os,sglang; print(os.path.dirname(sglang.__file__))" 2>/dev/null || true)"
  if [[ -n "${SGLANG_PKG_DIR}" ]]; then
    TARGET="${SGLANG_PKG_DIR}/srt/layers/moe/moe_runner/deep_gemm.py"
  else
    for CANDIDATE_ROOT in \
      "${PATCH_DIR}/../../sglang" \
      "/sgl-workspace/sglang" \
      "/root/sglang" \
      "/root/Dressage/../sglang"; do
      CANDIDATE_TARGET="${CANDIDATE_ROOT}/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py"
      if [[ -f "${CANDIDATE_TARGET}" ]]; then
        TARGET="${CANDIDATE_TARGET}"
        break
      fi
    done
  fi
fi

if [[ -z "${TARGET}" || ! -f "${TARGET}" ]]; then
  echo "[deepgemm-patch] target not found: ${TARGET}, skip"
  echo "[deepgemm-patch] set SGLANG_ROOT=/path/to/sglang if SGLang is a source checkout"
  exit 0
fi

if grep -Eq "D // 8 < E|D // 8 >= E" "${TARGET}"; then
  echo "[deepgemm-patch] already applied (D // 8 guard present), skip"
elif patch "${TARGET}" --dry-run < "${PATCH_FILE}" >/dev/null 2>&1; then
  patch "${TARGET}" < "${PATCH_FILE}"
  echo "[deepgemm-patch] applied to ${TARGET}"
elif grep -q "if N % 4 != 0 or G % 4 != 0:" "${TARGET}"; then
  # v0.5.13 already has #26025 but misses #27377. Add only the missing guard.
  perl -0pi -e 's/if N % 4 != 0 or G % 4 != 0:/if N % 4 != 0 or G % 4 != 0 or D \\/\\/ 8 < E:/' "${TARGET}"
  echo "[deepgemm-patch] applied missing D // 8 < E guard to ${TARGET}"
elif grep -q "envs.SGLANG_OPT_USE_JIT_EP_ACTIVATION.get() and N % 4 == 0 and G % 4 == 0" "${TARGET}"; then
  # Some checkouts already fold the v0.5.13 guard into use_jit_ep_activation.
  # Add the missing Qwen3.5 guard while preserving that style.
  perl -0pi -e 's@envs\.SGLANG_OPT_USE_JIT_EP_ACTIVATION\.get\(\) and N % 4 == 0 and G % 4 == 0@envs.SGLANG_OPT_USE_JIT_EP_ACTIVATION.get() and N % 4 == 0 and G % 4 == 0 and D // 8 >= E@' "${TARGET}"
  echo "[deepgemm-patch] applied missing D // 8 >= E guard to ${TARGET}"
else
  echo "[deepgemm-patch] WARNING: patch did not apply cleanly (sglang version mismatch?), check manually" >&2
fi
