"""Configuration and filesystem layout.

Everything the tool writes lives under ``FPL_DATA_DIR`` (default ``./data``) so a
checkout can be wiped and re-synced without touching source.
"""

from __future__ import annotations

import os
import re
import tomllib
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


class InvalidEntryId(ValueError):
    """Raised when something that is not an FPL entry id is offered as one."""


UUID_SHAPE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def parse_entry_id(raw: str | int | None) -> int | None:
    """Validate an FPL entry id, with a useful message when it is the wrong thing.

    The commonest mistake is offering a premierleague.com account id, which is a
    UUID and looks authoritative but is not what any FPL endpoint accepts. Worth
    naming explicitly rather than failing with a 404 later.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if UUID_SHAPE.match(text):
        raise InvalidEntryId(
            f"{text!r} is a premierleague.com account id, not an FPL entry id. "
            "The entry id is the number in the URL of your Points page: "
            "fantasy.premierleague.com/entry/<NUMBER>/event/2 — note that FPL "
            "reissues entry ids every season, so last season's will not work."
        )
    raise InvalidEntryId(
        f"{text!r} is not an FPL entry id. It should be a plain number, taken "
        "from fantasy.premierleague.com/entry/<NUMBER>/event/2"
    )


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


CONFIG_FILENAME = "fpl.toml"


def _config_file() -> dict:
    """Read ``fpl.toml`` from the working directory or the repository root.

    A config file means the entry id lives with the project rather than in a
    shell profile that has to be re-exported on every new machine and every new
    terminal. An FPL entry id is public information — anyone can view
    ``/entry/<id>/`` — so it is fine to commit; the odds API key is not, and
    stays in the environment.
    """
    for directory in (Path.cwd(), *Path(__file__).resolve().parents[:3]):
        candidate = directory / CONFIG_FILENAME
        if candidate.exists():
            try:
                return tomllib.loads(candidate.read_text())
            except tomllib.TOMLDecodeError:
                return {}
    return {}


def settings() -> Settings:
    # Precedence: environment first so a one-off override works without editing
    # a file, then the committed config. A malformed value is surfaced, not
    # silently ignored: quietly treating it as absent is how you end up
    # wondering why your squad never syncs.
    config = _config_file()
    raw_entry = os.environ.get("FPL_ENTRY_ID") or config.get("entry_id")
    return Settings(
        entry_id=parse_entry_id(raw_entry),
        odds_api_key=os.environ.get("ODDS_API_KEY") or None,
    )
