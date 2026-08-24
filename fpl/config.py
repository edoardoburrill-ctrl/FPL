"""Configuration and filesystem layout.

Everything the tool writes lives under ``FPL_DATA_DIR`` (default ``./data``) so a
checkout can be wiped and re-synced without touching source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

FPL_API_BASE = "https://fantasy.premierleague.com/api"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# The Odds API's key for the English top flight.
ODDS_SPORT_KEY = "soccer_epl"


def _root() -> Path:
    return Path(os.environ.get("FPL_DATA_DIR", Path.cwd() / "data")).resolve()


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def db(self) -> Path:
        return self.root / "fpl.db"

    @property
    def snapshots(self) -> Path:
        return self.root / "snapshots"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    def ensure(self) -> "Paths":
        for p in (self.root, self.snapshots, self.cache):
            p.mkdir(parents=True, exist_ok=True)
        return self


def paths() -> Paths:
    return Paths(_root()).ensure()


@dataclass(frozen=True)
class Settings:
    entry_id: int | None
    odds_api_key: str | None
    # Blend weight on market-implied team strength for the imminent gameweek.
    # Decays for gameweeks further out (see models.team_strength).
    market_weight_now: float = 0.85
    # Per-gameweek discount on future expected points in the optimiser.
    horizon_discount: float = 0.85
    default_horizon: int = 5


def settings() -> Settings:
    entry = os.environ.get("FPL_ENTRY_ID")
    return Settings(
        entry_id=int(entry) if entry and entry.isdigit() else None,
        odds_api_key=os.environ.get("ODDS_API_KEY") or None,
    )
