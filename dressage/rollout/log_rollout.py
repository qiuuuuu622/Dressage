"""Dressage rollout-side logging hook.

Registered via ``--custom-rollout-log-function-path``.  Adds
``raw_reward_trajectory_mean`` to the extra metrics dict, then returns
``False`` so slime's default ``_log_rollout_data`` continues with its
standard metric collection (response_len stats, reward stats, etc.).

The trajectory-mean metric gives a clean "fraction of trajectories that
got reward 1.0" (for binary rewards) without multi-segment trajectories
contributing N times more weight than single-segment ones.
"""

from __future__ import annotations

import logging
from typing import Any

from dressage.training.log_helpers import compute_trajectory_mean_raw_reward

logger = logging.getLogger(__name__)


def _sample_has_trainable_loss(sample: Any) -> bool:
    if getattr(sample, "remove_sample", False):
        return False

    try:
        response_length = int(getattr(sample, "response_length", 0) or 0)
    except (TypeError, ValueError):
        response_length = 0
    if response_length <= 0:
        return False

    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is None:
        return True
    return any(int(value) != 0 for value in loss_mask)


def log_rollout_data(
    rollout_id: int,
    args: Any,
    samples: list,
    extra_metrics: dict,
    rollout_time: float,
) -> bool:
    """Append trajectory-mean raw_reward to *extra_metrics*, then let slime log.

    Reads ``parent_traj_id``, ``segment_index``, and ``reward`` from each
    sample's ``metadata`` dict.  Samples without ``parent_traj_id`` are
    skipped (single-segment mode).

    Returns ``False`` so slime's default logging continues — the extra
    metric rides through ``log_dict = {**(rollout_extra_metrics or {})}``
    in slime's ``_log_rollout_data``.
    """
    parent_traj_ids: list[str] = []
    segment_indices: list[int] = []
    raw_rewards: list[float] = []

    has_multi_segment = False
    for sample in samples:
        meta = getattr(sample, "metadata", None) or {}
        ptid = meta.get("parent_traj_id")
        if ptid is None:
            continue
        has_multi_segment = True
        if not _sample_has_trainable_loss(sample):
            continue
        parent_traj_ids.append(str(ptid))
        segment_indices.append(int(meta.get("segment_index", 0)))
        r = getattr(sample, "reward", None)
        raw_rewards.append(float(r) if r is not None else 0.0)

    if not has_multi_segment:
        return False

    traj_mean = compute_trajectory_mean_raw_reward(
        parent_traj_ids, raw_rewards, segment_indices
    )
    if traj_mean is not None:
        extra_metrics["rollout/raw_reward_trajectory_mean"] = traj_mean

    return False
