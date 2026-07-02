#!/bin/bash
# Idempotently apply the SGLang GDN/Mamba TARGET_VERIFY state fix.
#
# Upstream intent: sgl-project/sglang#29449
#   In EAGLE/NEXTN TARGET_VERIFY with extra_buffer, a finished/retracted request
#   can linger in the verify batch after its Mamba/GDN state slot was freed.
#   Such rows must not commit stale recurrent state back into a possibly reused
#   live slot.
set -e

SGLANG_DIR="$(python3 -c "import os,sglang; print(os.path.dirname(sglang.__file__))" 2>/dev/null || true)"
if [[ -z "${SGLANG_DIR}" ]]; then
  echo "[gdn-mtp-verify-state-patch] sglang not importable, skip"
  exit 0
fi

SCHEDULE_TARGET="${SGLANG_DIR}/srt/managers/schedule_batch.py"
HYBRID_TARGET="${SGLANG_DIR}/srt/layers/attention/hybrid_linear_attn_backend.py"

if [[ ! -f "${SCHEDULE_TARGET}" || ! -f "${HYBRID_TARGET}" ]]; then
  echo "[gdn-mtp-verify-state-patch] target not found, skip"
  echo "[gdn-mtp-verify-state-patch] schedule=${SCHEDULE_TARGET} hybrid=${HYBRID_TARGET}"
  exit 0
fi

python3 - <<'PY' "${SCHEDULE_TARGET}" "${HYBRID_TARGET}"
from pathlib import Path
import sys

schedule = Path(sys.argv[1])
hybrid = Path(sys.argv[2])

changed = False

s = schedule.read_text()
if "# Dressage patch: mark freed Mamba/GDN verify rows as -1" in s:
    print(f"[gdn-mtp-verify-state-patch] schedule already applied: {schedule}")
else:
    old = """                idx = (\n                    torch.tensor(\n                        [req.mamba_next_track_idx for req in self.reqs],\n                        dtype=torch.int64,\n                        pin_memory=True,\n                    )\n                    .unsqueeze(1)\n                    .to(device=all_buffers.device, non_blocking=True)\n                )\n                self.mamba_track_indices = (\n                    torch.gather(all_buffers, 1, idx).squeeze(1).to(torch.int64)\n                )\n"""
    new = """                # Dressage patch: mark freed Mamba/GDN verify rows as -1.\n                # A finished/retracted request can linger in TARGET_VERIFY after\n                # its state slot was released, leaving mamba_next_track_idx=None.\n                # Gather with a safe in-range index, then mark those rows so the\n                # verify commit path skips stale recurrent-state writes.\n                next_track_idx = [req.mamba_next_track_idx for req in self.reqs]\n                none_rows = [i for i, v in enumerate(next_track_idx) if v is None]\n                for i in none_rows:\n                    next_track_idx[i] = 0\n                idx = (\n                    torch.tensor(\n                        next_track_idx,\n                        dtype=torch.int64,\n                        pin_memory=True,\n                    )\n                    .unsqueeze(1)\n                    .to(device=all_buffers.device, non_blocking=True)\n                )\n                self.mamba_track_indices = (\n                    torch.gather(all_buffers, 1, idx).squeeze(1).to(torch.int64)\n                )\n                if none_rows:\n                    self.mamba_track_indices[\n                        torch.tensor(none_rows, dtype=torch.int64, device=all_buffers.device)\n                    ] = -1\n"""
    if old not in s:
        raise SystemExit(f"[gdn-mtp-verify-state-patch] schedule patch did not match expected code: {schedule}")
    schedule.write_text(s.replace(old, new))
    changed = True
    print(f"[gdn-mtp-verify-state-patch] patched schedule: {schedule}")

h = hybrid.read_text()
if "# Dressage patch: skip freed Mamba/GDN verify rows during commit" in h:
    print(f"[gdn-mtp-verify-state-patch] hybrid already applied: {hybrid}")
else:
    old = """        request_number = last_correct_step_indices.shape[0]\n\n        state_indices_tensor = (\n"""
    new = """        request_number = last_correct_step_indices.shape[0]\n\n        # Dressage patch: skip freed Mamba/GDN verify rows during commit.\n        # Rows marked with mamba_track_indices < 0 correspond to requests that\n        # finished/retracted and released their recurrent-state slot while still\n        # lingering in TARGET_VERIFY. Clamp their accepted step to -1 so the\n        # fused scatter kernel's step<0 guard drops both SSM and conv commits.\n        if mamba_track_indices is not None:\n            last_correct_step_indices = torch.where(\n                mamba_track_indices[:request_number] < 0,\n                torch.full_like(last_correct_step_indices, -1),\n                last_correct_step_indices,\n            )\n\n        state_indices_tensor = (\n"""
    if old not in h:
        raise SystemExit(f"[gdn-mtp-verify-state-patch] hybrid patch did not match expected code: {hybrid}")
    hybrid.write_text(h.replace(old, new, 1))
    changed = True
    print(f"[gdn-mtp-verify-state-patch] patched hybrid: {hybrid}")

if not changed:
    print("[gdn-mtp-verify-state-patch] already applied, skip")
PY
