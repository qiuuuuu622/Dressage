"""Synchronous rollout entrypoint for Dressage on colocate setups.

Used when actor and sglang share the same GPUs (e.g. qwen3.5-35B-A3B on
8xH100 with `--colocate`). Mirrors the dressage retry / empty-batch /
failure-summary semantics of `fully_async_rollout`, but runs to completion
per `rollout_id` so the framework can offload the sglang engine before
training kicks in.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from typing import Any

from dressage.profiling import add_duration, enabled as profiling_enabled, summarize_profiles
from dressage.rollout.fully_async_rollout import (
    _allow_empty_train_batch,
    _flatten_multi_segment_result,
    _group_failure_summary,
    _group_has_trainable_tokens,
    _increment_retry,
    _is_aborted_group,
    _retry_count,
)

logger = logging.getLogger(__name__)

try:
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.rollout.sglang_rollout import GenerateState, abort, generate_and_rm_group
    from slime.utils.async_utils import run
except ImportError:
    GenerateState = None  # type: ignore[assignment]
    generate_and_rm_group = None  # type: ignore[assignment]
    abort = None  # type: ignore[assignment]
    RolloutFnTrainOutput = None  # type: ignore[assignment]

    def run(coro):  # type: ignore[no-redef]
        return asyncio.run(coro)

from dressage.rollout.multi_segment import compute_multi_segment_metrics


def _max_retries() -> int:
    return int(os.environ.get("DRESSAGE_ROLLOUT_MAX_RETRIES", "2"))

_LAST_DROPPED_TAIL: dict[str, int] = {"count": 0}
_LONG_PROMPT_QUEUE: list[list[Any]] = []
_ROUND_TYPE: dict[str, str] = {"type": "short"}


def _tail_batch_enabled() -> bool:
    """Tail batching (RollPacker) 开关。默认开启;设 DRESSAGE_TAIL_BATCHING=0 关闭。"""
    val = os.environ.get("DRESSAGE_TAIL_BATCHING", "").lower()
    if val in {"0", "false", "no", "off"}:
        return False
    return True


def _long_round_threshold() -> float:
    """队列长度达到 target × 此值时触发长轮次。默认 1.0。"""
    try:
        return max(0.5, float(os.environ.get("DRESSAGE_TAIL_BATCH_LONG_THRESHOLD", "1.0")))
    except (TypeError, ValueError):
        return 1.0


def _clean_for_requeue(group: list[Any]) -> list[Any]:
    """清洗被 abort 的 group,使其可在长轮次中重新提交生成。

    清除 session_id、部分生成的 response/tokens/reward 等,
    保证长轮次中由当前权重全新生成(on-policy)。
    """
    for sample in group:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            sample.metadata = metadata
        old_sid = metadata.get("session_id") or getattr(sample, "session_id", None)
        if old_sid is not None:
            metadata["last_deferred_session_id"] = old_sid
        metadata.pop("session_id", None)
        metadata.pop("parent_traj_id", None)
        metadata.pop("segment_index", None)
        if hasattr(sample, "session_id"):
            sample.session_id = None
        sample.response = ""
        sample.response_length = 0
        sample.tokens = getattr(sample, "tokens", None) or []
        sample.loss_mask = []
        sample.rollout_log_probs = []
        sample.reward = None
        sample.remove_sample = False
    return group


def _oversample_factor() -> float:
    """过采样系数:实际 submit = ceil(rollout_batch_size * 此系数)。

    默认 1.25(多发 25%),收满 target 个成功 group 即返回,abort 最慢的尾巴。
    设为 1.0 即退回旧的"全 submit、等所有完成"行为。
    """
    try:
        v = float(os.environ.get("DRESSAGE_SYNC_OVERSAMPLE", "1.25"))
    except (TypeError, ValueError):
        v = 1.25
    return v if v >= 1.0 else 1.0


async def _submit_group(
    args: Any,
    group: list[Any],
    state: Any,
    pendings: set[asyncio.Task],
    task_to_group: dict[asyncio.Task, list[Any]],
) -> None:
    if generate_and_rm_group is None:
        raise RuntimeError("slime.rollout.sglang_rollout.generate_and_rm_group is unavailable")
    task = asyncio.create_task(
        generate_and_rm_group(
            args,
            group,
            sampling_params=state.sampling_params.copy(),
            evaluation=False,
        )
    )
    pendings.add(task)
    task_to_group[task] = group


async def _run_sync_rollout(
    args: Any,
    rollout_id: int,
    data_buffer: Any,
) -> list[list[Any]]:
    if GenerateState is None or generate_and_rm_group is None:
        raise RuntimeError(
            "Dressage sync rollout requires slime.rollout.sglang_rollout to be importable"
        )

    target = int(getattr(args, "rollout_batch_size", 1))
    total_start = time.perf_counter()
    rollout_profile: dict[str, Any] = {}
    max_retries = _max_retries()
    start = time.perf_counter()
    state = GenerateState(args)
    add_duration(rollout_profile, "sync.generate_state", time.perf_counter() - start)
    data: list[list[Any]] = []
    pendings: set[asyncio.Task] = set()
    task_to_group: dict[asyncio.Task, list[Any]] = {}

    oversample = _oversample_factor()
    n_submit = max(target, math.ceil(target * oversample))

    # 过采样:一次性多 submit n_submit 个 group
    start = time.perf_counter()
    groups = data_buffer.get_samples(n_submit)
    add_duration(rollout_profile, "sync.data_buffer_get_samples", time.perf_counter() - start)
    if len(groups) < target:
        raise RuntimeError(
            f"data_buffer.get_samples({n_submit}) returned {len(groups)} groups (< target "
            f"{target}); cannot fill a rollout batch."
        )
    for group in groups:
        await _submit_group(args, group, state, pendings, task_to_group)

    # 够数即返回:收满 target 个成功 group 就停;失败的在有过采样余量时直接丢,
    # 余量耗尽却还没收满才再补一波。
    while pendings and len(data) < target:
        wait_start = time.perf_counter()
        done, pendings = await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)
        add_duration(rollout_profile, "sync.wait_first_completed", time.perf_counter() - wait_start)
        for task in done:
            group_for_task = task_to_group.pop(task)
            if len(data) >= target:
                # 已收满,本批里多完成的算富余 -> 丢
                continue
            error: BaseException | None = None
            result_group: list[Any] | None = None
            try:
                result_group = task.result()
            except BaseException as exc:  # noqa: BLE001 - mirror fully_async behavior
                error = exc

            if result_group is not None:
                result_group = _flatten_multi_segment_result(result_group)

            failed = error is not None or _is_aborted_group(result_group or group_for_task)
            if not failed:
                data.append(result_group)
                continue

            summary = _group_failure_summary(
                result_group if result_group is not None else group_for_task, error
            )
            if _retry_count(group_for_task) < max_retries:
                _increment_retry(group_for_task)
                logger.warning(
                    "resubmitting rollout group for retry (attempt %d/%d): %s",
                    _retry_count(group_for_task),
                    max_retries,
                    summary,
                )
                await _submit_group(args, group_for_task, state, pendings, task_to_group)
            else:
                logger.error(
                    "rollout group exhausted retries, dropping (oversample covers): %s",
                    summary,
                )

        # 过采样余量被失败耗尽却还没收满 -> 再补一波
        if len(data) < target and not pendings:
            need = target - len(data)
            extra = max(need, math.ceil(need * oversample))
            start = time.perf_counter()
            more = data_buffer.get_samples(extra)
            add_duration(rollout_profile, "sync.data_buffer_get_samples", time.perf_counter() - start)
            if not more:
                break
            for group in more:
                await _submit_group(args, group, state, pendings, task_to_group)

    # abort 慢长尾:收满 target 后,先让 sglang 引擎中止所有在飞请求(abort_all),
    # 否则 colocate 训练前 offload 的 is_fully_idle() 断言会失败;再 cancel python task。
    dropped_tail = len(pendings)
    if pendings and abort is not None and not getattr(state, "aborted", False):
        try:
            abort_start = time.perf_counter()
            await abort(args, rollout_id)  # POST /abort_request {abort_all} 给所有引擎
            add_duration(rollout_profile, "sync.abort_tail", time.perf_counter() - abort_start)
        except Exception as exc:  # noqa: BLE001 - never block offload on abort errors
            logger.warning(
                "engine abort_request failed (will still cancel tasks): %s", exc
            )
    queued_tail = 0
    for task in pendings:
        task.cancel()
        original_group = task_to_group.pop(task, None)
        if original_group is not None and _tail_batch_enabled():
            _clean_for_requeue(original_group)
            _LONG_PROMPT_QUEUE.append(original_group)
            queued_tail += 1
    if pendings:
        gather_start = time.perf_counter()
        await asyncio.gather(*pendings, return_exceptions=True)
        add_duration(rollout_profile, "sync.cancel_tail_gather", time.perf_counter() - gather_start)
    pendings = set()
    if _tail_batch_enabled():
        _LAST_DROPPED_TAIL["count"] = 0
        if queued_tail:
            logger.info(
                "sync rollout (short): collected %d/%d good groups, queued %d slow-tail "
                "groups for long round (oversample=%.2f, queue_size=%d)",
                len(data),
                target,
                queued_tail,
                oversample,
                len(_LONG_PROMPT_QUEUE),
            )
    else:
        if dropped_tail:
            logger.info(
                "sync rollout: collected %d/%d good groups, aborted %d slow-tail groups "
                "(oversample=%.2f)",
                len(data),
                target,
                dropped_tail,
                oversample,
            )
        _LAST_DROPPED_TAIL["count"] = dropped_tail

    reset_start = time.perf_counter()
    state.reset()
    add_duration(rollout_profile, "sync.state_reset", time.perf_counter() - reset_start)
    add_duration(rollout_profile, "sync.rollout_total", time.perf_counter() - total_start)

    data = sorted(data, key=lambda group: getattr(group[0], "index", 0))
    if not _allow_empty_train_batch() and not any(
        _group_has_trainable_tokens(group) for group in data
    ):
        summaries = [_group_failure_summary(group) for group in data[: min(3, len(data))]]
        raise RuntimeError(
            "Dressage sync rollout produced no trainable samples; "
            "refusing to train on failed placeholder samples. "
            f"First failures: {' | '.join(summaries)}. "
            "Set DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH=1 to keep the previous behavior."
        )

    if profiling_enabled():
        for group in data:
            for sample in group:
                metadata = getattr(sample, "metadata", None)
                if not isinstance(metadata, dict):
                    continue
                profile = metadata.get("dressage_profile")
                if not isinstance(profile, dict):
                    profile = {}
                    metadata["dressage_profile"] = profile
                profile.update(rollout_profile)

    return data


async def _run_long_round(
    args: Any,
    rollout_id: int,
    data_buffer: Any,
    target: int,
) -> list[list[Any]]:
    """长轮次: 从 _LONG_PROMPT_QUEUE 取 target 个 group, 不过采样、不 abort。

    与短轮次的区别:
    - 数据来源: _LONG_PROMPT_QUEUE 而非 data_buffer.get_samples()
    - 过采样: η=1.0 (不过采样)
    - 尾部处理: 无 abort, 等全部完成
    - 失败补位: 从 data_buffer 取新 prompt 填充
    """
    if GenerateState is None or generate_and_rm_group is None:
        raise RuntimeError(
            "Dressage sync rollout requires slime.rollout.sglang_rollout to be importable"
        )

    total_start = time.perf_counter()
    rollout_profile: dict[str, Any] = {}
    start = time.perf_counter()
    state = GenerateState(args)
    add_duration(rollout_profile, "sync_long.generate_state", time.perf_counter() - start)
    data: list[list[Any]] = []
    pendings: set[asyncio.Task] = set()
    task_to_group: dict[asyncio.Task, list[Any]] = {}
    max_retries = _max_retries()

    # 从长尾队列取 target 个 group
    groups = _LONG_PROMPT_QUEUE[:target]
    del _LONG_PROMPT_QUEUE[:target]
    logger.info(
        "sync rollout (long): starting long round with %d groups from queue "
        "(target=%d, queue_remaining=%d)",
        len(groups),
        target,
        len(_LONG_PROMPT_QUEUE),
    )
    for group in groups:
        await _submit_group(args, group, state, pendings, task_to_group)

    # 等全部完成, 不 abort
    while pendings and len(data) < target:
        wait_start = time.perf_counter()
        done, pendings = await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)
        add_duration(rollout_profile, "sync_long.wait_first_completed", time.perf_counter() - wait_start)
        for task in done:
            group_for_task = task_to_group.pop(task)
            if len(data) >= target:
                continue
            error: BaseException | None = None
            result_group: list[Any] | None = None
            try:
                result_group = task.result()
            except BaseException as exc:  # noqa: BLE001
                error = exc

            if result_group is not None:
                result_group = _flatten_multi_segment_result(result_group)

            failed = error is not None or _is_aborted_group(result_group or group_for_task)
            if not failed:
                data.append(result_group)
                continue

            summary = _group_failure_summary(
                result_group if result_group is not None else group_for_task, error
            )
            if _retry_count(group_for_task) < max_retries:
                _increment_retry(group_for_task)
                logger.warning(
                    "long round: resubmitting group for retry (attempt %d/%d): %s",
                    _retry_count(group_for_task),
                    max_retries,
                    summary,
                )
                await _submit_group(args, group_for_task, state, pendings, task_to_group)
            else:
                logger.error(
                    "long round: group exhausted retries, dropping: %s", summary
                )

        # 长轮次中失败耗尽且还没收满 -> 从 data_buffer 补充
        if len(data) < target and not pendings:
            need = target - len(data)
            start = time.perf_counter()
            more = data_buffer.get_samples(need)
            add_duration(rollout_profile, "sync_long.data_buffer_get_samples", time.perf_counter() - start)
            if not more:
                break
            for group in more:
                await _submit_group(args, group, state, pendings, task_to_group)

    # 长轮次不应有尾部 abort, 但安全起见检查
    if pendings:
        logger.warning("long round: unexpected pending tasks, cancelling")
        for task in pendings:
            task.cancel()
            task_to_group.pop(task, None)
        gather_start = time.perf_counter()
        await asyncio.gather(*pendings, return_exceptions=True)
        add_duration(rollout_profile, "sync_long.cancel_tail_gather", time.perf_counter() - gather_start)
    pendings = set()

    logger.info(
        "sync rollout (long): collected %d/%d good groups (queue_remaining=%d)",
        len(data),
        target,
        len(_LONG_PROMPT_QUEUE),
    )

    reset_start = time.perf_counter()
    state.reset()
    add_duration(rollout_profile, "sync_long.state_reset", time.perf_counter() - reset_start)
    add_duration(rollout_profile, "sync_long.rollout_total", time.perf_counter() - total_start)

    data = sorted(data, key=lambda group: getattr(group[0], "index", 0))
    if not _allow_empty_train_batch() and not any(
        _group_has_trainable_tokens(group) for group in data
    ):
        summaries = [_group_failure_summary(group) for group in data[: min(3, len(data))]]
        raise RuntimeError(
            "Dressage sync rollout (long round) produced no trainable samples; "
            "refusing to train on failed placeholder samples. "
            f"First failures: {' | '.join(summaries)}. "
            "Set DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH=1 to keep the previous behavior."
        )

    if profiling_enabled():
        for group in data:
            for sample in group:
                metadata = getattr(sample, "metadata", None)
                if not isinstance(metadata, dict):
                    continue
                profile = metadata.get("dressage_profile")
                if not isinstance(profile, dict):
                    profile = {}
                    metadata["dressage_profile"] = profile
                profile.update(rollout_profile)

    return data


def generate_rollout_sync(
    args: Any,
    rollout_id: int,
    data_buffer: Any,
    evaluation: bool = False,
):
    if evaluation:
        raise ValueError("Dressage sync rollout does not support evaluation mode")

    target = int(getattr(args, "rollout_batch_size", 1))

    if _tail_batch_enabled() and len(_LONG_PROMPT_QUEUE) >= int(target * _long_round_threshold()):
        _ROUND_TYPE["type"] = "long"
        data = run(_run_long_round(args, rollout_id, data_buffer, target))
    else:
        _ROUND_TYPE["type"] = "short"
        data = run(_run_sync_rollout(args, rollout_id, data_buffer))

    metrics: dict[str, Any] = compute_multi_segment_metrics(
        [sample for group in data for sample in group]
    )
    metrics["rollout/dropped_tail_groups"] = float(_LAST_DROPPED_TAIL.get("count", 0))
    metrics["rollout/round_type"] = 1.0 if _ROUND_TYPE["type"] == "long" else 0.0
    metrics["rollout/long_prompt_queue_size"] = float(len(_LONG_PROMPT_QUEUE))
    if profiling_enabled():
        metrics.update(
            summarize_profiles(
                [sample for group in data for sample in group],
                metric_prefix="profile",
            )
        )
    if RolloutFnTrainOutput is None:
        return data
    return RolloutFnTrainOutput(samples=data, metrics=metrics)
