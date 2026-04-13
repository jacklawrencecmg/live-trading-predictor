"""
Application settings with startup validation.

All settings are read from environment variables (or a .env file).
Pydantic validators run at import time — a misconfigured environment raises
immediately rather than failing silently at the first database call.

Validation rules
----------------
- DATABASE_URL  must start with ``postgresql`` (rejects SQLite in production)
- SECRET_KEY    must not be the default dev value when ENV=production
- Risk bounds are enforced: max_daily_loss_pct in (0, 1],
  max_position_size_pct in (0, 1], starting_capital > 0
- Kill switch state is warned (not blocked) at startup when active
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

_DEV_SECRET = "dev-secret-key"


class Settings(BaseSettings):
    # ── Environment ───────────────────────────────────────────────────────────
    env: Literal["development", "staging", "production"] = "development"

    # ── Infrastructure ────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/options_research"
    redis_url: str = "redis://localhost:6379/0"
    secret_key: str = _DEV_SECRET

    # ── Risk defaults ─────────────────────────────────────────────────────────
    max_daily_loss_pct: float = 0.02       # 2% of capital
    max_position_size_pct: float = 0.05    # 5% per position
    cooldown_minutes: int = 15
    starting_capital: float = 100_000.0
    kill_switch: bool = False

    # ── Market data ───────────────────────────────────────────────────────────
    default_symbol: str = "SPY"
    candle_interval: str = "5m"
    candle_lookback_days: int = 5

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    json_logs: bool = True

    class Config:
        env_file = ".env"

    # ── Field-level validators ────────────────────────────────────────────────

    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, v: str) -> str:
        if not v.startswith("postgresql"):
            raise ValueError(
                f"DATABASE_URL must use a PostgreSQL DSN (got: {v!r}). "
                "SQLite is not supported in this application."
            )
        return v

    @field_validator("max_daily_loss_pct")
    @classmethod
    def _validate_max_daily_loss(cls, v: float) -> float:
        if not (0 < v <= 1.0):
            raise ValueError(
                f"max_daily_loss_pct must be in (0, 1]; got {v}. "
                "A value > 1 would allow total-capital loss in a single day."
            )
        return v

    @field_validator("max_position_size_pct")
    @classmethod
    def _validate_max_position_size(cls, v: float) -> float:
        if not (0 < v <= 1.0):
            raise ValueError(
                f"max_position_size_pct must be in (0, 1]; got {v}."
            )
        return v

    @field_validator("starting_capital")
    @classmethod
    def _validate_capital(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(
                f"starting_capital must be positive; got {v}."
            )
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}; got {v!r}.")
        return upper

    # ── Cross-field validators ────────────────────────────────────────────────

    @model_validator(mode="after")
    def _production_safety_checks(self) -> "Settings":
        if self.env == "production":
            if self.secret_key == _DEV_SECRET:
                raise ValueError(
                    "SECRET_KEY must not be the default dev value in production. "
                    "Set a strong random SECRET_KEY environment variable."
                )
            if "localhost" in self.database_url or "127.0.0.1" in self.database_url:
                raise ValueError(
                    "DATABASE_URL points to localhost in a production environment. "
                    "Set DATABASE_URL to your production database host."
                )

        if self.kill_switch:
            # Warn but do not block — the kill switch may be intentionally set.
            logger.warning(
                "Kill switch is ACTIVE at startup (KILL_SWITCH=true). "
                "All inference calls will be blocked until it is deactivated."
            )

        return self


settings = Settings()
