"""Lightweight rollout profiling helpers.

Enabled by setting ``DRESSAGE_PROFILE=1``.  Profile data is kept in normal
metadata dictionaries so it survives Ray task boundaries and sample expansion.
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
import os
import statistics
import time
from typing import Any, Iterable


def enabled() -> bool:
    value = os.environ.get("DRESSAGE_PROFILE", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _profile_dict(container: dict[str, Any]) -> dict[str, Any]:
    profile = container.get("dressage_profile")
    if not isinstance(profile, dict):
        profile = {}
        container["dressage_profile"] = profile
    return profile


def add_duration(container: dict[str, Any], key: str, seconds: float) -> None:
    if not enabled():
        return
    profile = _profile_dict(container)
    profile_add(profile, key, seconds)


def set_value(container: dict[str, Any], key: str, value: Any) -> None:
    if not enabled():
        return
    profile_set(_profile_dict(container), key, value)


def profile_add(profile: dict[str, Any] | None, key: str, seconds: float) -> None:
    """Accumulate a duration directly into a profile dictionary."""

    if not enabled() or profile is None:
        return
    profile[key] = float(profile.get(key, 0.0)) + float(seconds)


def profile_set(profile: dict[str, Any] | None, key: str, value: Any) -> None:
    """Set a profile value directly on a profile dictionary."""

    if not enabled() or profile is None:
        return
    profile[key] = value


def profile_public(profile: dict[str, Any] | None) -> dict[str, float]:
    """Return numeric profile values safe to expose in response usage."""

    if not profile:
        return {}
    return {
        str(key): float(value)
        for key, value in profile.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


@contextmanager
def profile_span(profile: dict[str, Any] | None, key: str):
    """Measure a synchronous block into an existing profile dictionary."""

    if not enabled() or profile is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        profile_add(profile, key, time.perf_counter() - start)


@asynccontextmanager
async def async_profile_span(profile: dict[str, Any] | None, key: str):
    """Measure an async block into an existing profile dictionary."""

    if not enabled() or profile is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        profile_add(profile, key, time.perf_counter() - start)


@contextmanager
def time_block(container: dict[str, Any], key: str):
    if not enabled():
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        add_duration(container, key, time.perf_counter() - start)


@asynccontextmanager
async def async_time_block(container: dict[str, Any], key: str):
    if not enabled():
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        add_duration(container, key, time.perf_counter() - start)


def summarize_values(values: Iterable[float], *, prefix: str) -> dict[str, float]:
    data = sorted(float(v) for v in values if v is not None)
    if not data:
        return {}
    count = len(data)

    def percentile(q: float) -> float:
        if count == 1:
            return data[0]
        index = min(count - 1, max(0, int(round((count - 1) * q))))
        return data[index]

    return {
        f"{prefix}/count": float(count),
        f"{prefix}/mean": float(statistics.fmean(data)),
        f"{prefix}/p50": percentile(0.50),
        f"{prefix}/p95": percentile(0.95),
        f"{prefix}/max": data[-1],
    }


def summarize_profiles(
    samples: Iterable[Any],
    *,
    metadata_key: str = "dressage_profile",
    metric_prefix: str = "profile",
) -> dict[str, float]:
    by_key: dict[str, list[float]] = {}
    for sample in samples:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        profile = metadata.get(metadata_key)
        if not isinstance(profile, dict):
            continue
        for key, value in profile.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                by_key.setdefault(key, []).append(float(value))

    metrics: dict[str, float] = {}
    for key, values in sorted(by_key.items()):
        safe_key = key.replace(".", "_").replace(" ", "_")
        metrics.update(summarize_values(values, prefix=f"{metric_prefix}/{safe_key}"))
    return metrics
