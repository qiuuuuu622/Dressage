"""Small profiling helpers for blackbox server response metadata."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from functools import wraps
import time
from typing import Any, Awaitable, Callable, TypeVar

R = TypeVar("R")


class ProfileRecorder:
    """Collect lightweight stage timings without scattering timer code."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._values[key] = value

    def add_duration(self, key: str, seconds: float) -> None:
        self._values[key] = float(self._values.get(key, 0.0)) + float(seconds)

    @contextmanager
    def measure(self, key: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add_duration(key, time.perf_counter() - start)

    @asynccontextmanager
    async def measure_async(self, key: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add_duration(key, time.perf_counter() - start)

    def snapshot(self) -> dict[str, Any]:
        return dict(self._values)


def profile_response(total_key: str) -> Callable[[Callable[..., Awaitable[R]]], Callable[..., Awaitable[R]]]:
    """Inject a ProfileRecorder and attach its snapshot to ``response.profile``."""

    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> R:
            profile = ProfileRecorder()
            start = time.perf_counter()
            response = await func(*args, profile=profile, **kwargs)
            profile.add_duration(total_key, time.perf_counter() - start)
            response_profile = getattr(response, "profile", None)
            if isinstance(response_profile, dict):
                response_profile.update(profile.snapshot())
            return response

        return wrapper

    return decorator
