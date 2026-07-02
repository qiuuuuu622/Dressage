from __future__ import annotations

import shutil
from pathlib import Path

from blackbox_server.core.models import utcnow  # noqa: F401  (kept for compatibility)


def make_runtime_id() -> str:
    # Stable runtime id on purpose: the agent (e.g. opencode) embeds its working
    # directory into the system prompt. A per-session unique id (timestamp+uuid)
    # makes every rollout's prompt differ, defeating sglang's radix/prefix cache
    # so the large system-prompt + tool-definition prefix (~7k tokens) gets
    # re-prefilled on every sample. Each pool server has its own (per-slot)
    # runtime_root mounted to the same in-sandbox path, and bindings are handled
    # sequentially, so a constant id is collision-free and makes the reported
    # working directory identical across sessions -> prefix cache hits.
    return "bbs-session"


def ensure_runtime_dir(runtime_root: str, runtime_id: str) -> str:
    runtime_dir = Path(runtime_root) / runtime_id
    # Reuse the stable path: clear any stale contents left by a previous binding
    # (e.g. if cleanup was skipped) instead of failing on an existing directory.
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir, ignore_errors=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return str(runtime_dir)


def remove_runtime_dir(runtime_dir: str | None) -> None:
    if not runtime_dir:
        return
    shutil.rmtree(runtime_dir, ignore_errors=True)
