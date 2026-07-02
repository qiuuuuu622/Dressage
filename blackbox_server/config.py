from __future__ import annotations

import os

from pydantic import BaseModel, Field


def _get_env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


class BlackboxServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    runtime_root: str = "/tmp/blackbox_server"
    max_sessions: int = Field(default=1, ge=1)
    max_turns: int = 200
    backend_timeout: float = 960.0
    execute_cmd_timeout: float = Field(default=600.0, gt=0)
    router_timeout: int = 600000
    shutdown_timeout: float = 30.0
    health_check_interval: float = 2.0
    health_check_timeout: float = 60.0
    runtime_health_check_interval: float = 10.0
    runtime_health_check_retries: int = Field(default=10, ge=1)
    runtime_health_check_retry_delay: float = Field(default=2.0, ge=0.0)

    @classmethod
    def from_env(cls) -> "BlackboxServerConfig":
        return cls(
            host=_get_env_str("BBS_HOST", "0.0.0.0"),
            port=_get_env_int("BBS_PORT", 23456),
            runtime_root=_get_env_str("BBS_RUNTIME_ROOT", "/tmp/blackbox_server"),
            max_sessions=_get_env_int("BBS_MAX_SESSIONS", 1),
            max_turns=_get_env_int("BBS_MAX_TURNS", 200),
            backend_timeout=_get_env_float("BBS_BACKEND_TIMEOUT", 960.0),
            execute_cmd_timeout=_get_env_float("BBS_EXECUTE_CMD_TIMEOUT", 600.0),
            router_timeout=_get_env_int("BBS_ROUTER_TIMEOUT", 600000),
            shutdown_timeout=_get_env_float("BBS_SHUTDOWN_TIMEOUT", 30.0),
            runtime_health_check_interval=_get_env_float(
                "BBS_RUNTIME_HEALTH_CHECK_INTERVAL", 10.0
            ),
            runtime_health_check_retries=_get_env_int(
                "BBS_RUNTIME_HEALTH_CHECK_RETRIES", 10
            ),
            runtime_health_check_retry_delay=_get_env_float(
                "BBS_RUNTIME_HEALTH_CHECK_RETRY_DELAY", 2.0
            ),
        )


class ServerConfigOverride(BaseModel):
    runtime_root: str | None = None
    max_sessions: int | None = Field(default=None, ge=1)
    max_turns: int | None = None
    backend_timeout: float | None = None
    execute_cmd_timeout: float | None = Field(default=None, gt=0)
    router_timeout: int | None = None
    runtime_health_check_interval: float | None = None
    runtime_health_check_retries: int | None = Field(default=None, ge=1)
    runtime_health_check_retry_delay: float | None = Field(default=None, ge=0.0)

    def apply(self, config: BlackboxServerConfig) -> BlackboxServerConfig:
        updates = self.model_dump(exclude_none=True)
        return config.model_copy(update=updates)

    def explicit_values(self) -> dict[str, int | float | str]:
        return self.model_dump(exclude_none=True)
