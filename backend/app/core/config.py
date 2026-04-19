from __future__ import annotations

import os
from dataclasses import dataclass


def _parse_allowed_origins() -> tuple[str, ...]:
    raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
    origins = tuple(origin.strip() for origin in raw_origins.split(",") if origin.strip())
    return origins or ("*",)


@dataclass(frozen=True)
class Settings:
    app_name: str = "Mutual Fund Facts API"
    app_version: str = "1.0.0"
    allowed_origins: tuple[str, ...] = _parse_allowed_origins()


settings = Settings()
