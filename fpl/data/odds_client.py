"""Bookmaker odds, and matching them to FPL fixtures.

The odds feed names teams in its own way ("Nottingham Forest", "Wolverhampton
Wanderers"), FPL names them in another ("Nott'm Forest", "Wolves"). Getting that
mapping wrong silently attaches the wrong match's odds to a fixture, which is far
worse than having no odds at all — so unmatched events are reported rather than
guessed at, and the caller falls back to the fitted ratings model.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import httpx

from fpl.config import ODDS_API_BASE, ODDS_SPORT_KEY, paths
from fpl.models.devig import devig
from fpl.state.resolve import normalise as fold_name

# FPL short names and common feed spellings that normalisation alone misses.
TEAM_ALIASES = {
    "nottinghamforest": "nottmforest",
    "wolverhamptonwanderers": "wolves",
    "brightonandhovealbion": "brighton",
    "brightonhovealbion": "brighton",
    "tottenhamhotspur": "spurs",
    "tottenham": "spurs",
    "manchestercity": "mancity",
    "manchesterunited": "manutd",
    "newcastleunited": "newcastle",
    "westhamunited": "westham",
    "leedsunited": "leeds",
    "leicestercity": "leicester",
    "ipswichtown": "ipswich",
    "afcbournemouth": "bournemouth",
    "sheffieldunited": "sheffieldutd",
    "lutontown": "luton",
    "norwichcity": "norwich",
    "stokecity": "stoke",
    "swanseacity": "swansea",
    "cardiffcity": "cardiff",
    "hullcity": "hull",
    "birminghamcity": "birmingham",
    "coventrycity": "coventry",
}


def normalise_team(name: str) -> str:
    text = re.sub(r"\b(fc|afc)\b", " ", (name or "").lower())
    text = fold_name(text)
    return TEAM_ALIASES.get(text, text)


@dataclass(frozen=True)
class FixtureOdds:
    """De-vigged market probabilities for one match."""

    home_team: str
    away_team: str
    kickoff: datetime | None
    p_home: float
    p_draw: float
    p_away: float
    p_over: float | None = None
    total_line: float = 2.5
    bookmakers: int = 0

    @property
    def key(self) -> tuple[str, str]:
        return normalise_team(self.home_team), normalise_team(self.away_team)


class OddsClient:
    def __init__(
        self,
        api_key: str | None,
        base_url: str = ODDS_API_BASE,
        cache_dir: Path | None = None,
        ttl_seconds: float = 3600.0,
        offline: bool = False,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir or paths().cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self.offline = offline
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _cache_path(self) -> Path:
        return self.cache_dir / "odds__epl.json"

    def fetch_raw(self, *, force: bool = False) -> list[dict]:
        cache = self._cache_path()
        if not force and cache.exists():
            age = time.time() - cache.stat().st_mtime
            if age < self.ttl or self.offline:
                return json.loads(cache.read_text())

        if not self.enabled or self.offline:
            if cache.exists():
                return json.loads(cache.read_text())
            return []

        url = f"{self.base_url}/sports/{ODDS_SPORT_KEY}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "uk,eu",
            "markets": "h2h,totals",
            "oddsFormat": "decimal",
        }
        try:
            response = httpx.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            if cache.exists():
                return json.loads(cache.read_text())
            return []

        cache.write_text(json.dumps(payload))
        return payload

    def fixture_odds(self, *, force: bool = False) -> list[FixtureOdds]:
        return [parse_event(event) for event in self.fetch_raw(force=force) if event]


def _consensus(values: Sequence[float]) -> float | None:
    """Median across bookmakers — robust to one book being stale or wrong."""
    clean = [v for v in values if v and v > 1.0]
    if not clean:
        return None
    clean.sort()
    mid = len(clean) // 2
    return clean[mid] if len(clean) % 2 else 0.5 * (clean[mid - 1] + clean[mid])


def parse_event(event: Mapping) -> FixtureOdds:
    """Turn one Odds API event into de-vigged probabilities.

    Consensus is taken per outcome across bookmakers before de-vigging, so one
    book with a stale price cannot drag the whole fixture.
    """
    home = event.get("home_team", "")
    away = event.get("away_team", "")
    kickoff = None
    if event.get("commence_time"):
        try:
            kickoff = datetime.fromisoformat(
                event["commence_time"].replace("Z", "+00:00")
            ).astimezone(timezone.utc)
        except ValueError:
            kickoff = None

    home_prices: list[float] = []
    draw_prices: list[float] = []
    away_prices: list[float] = []
    over_prices: list[float] = []
    under_prices: list[float] = []
    line = 2.5
    books = 0

    for book in event.get("bookmakers", []) or []:
        books += 1
        for market in book.get("markets", []) or []:
            outcomes = market.get("outcomes", []) or []
            if market.get("key") == "h2h":
                for outcome in outcomes:
                    name, price = outcome.get("name"), outcome.get("price")
                    if name == home:
                        home_prices.append(price)
                    elif name == away:
                        away_prices.append(price)
                    elif name == "Draw":
                        draw_prices.append(price)
            elif market.get("key") == "totals":
                for outcome in outcomes:
                    if outcome.get("point") is None:
                        continue
                    if abs(float(outcome["point"]) - 2.5) > 1e-6:
                        continue
                    line = 2.5
                    if outcome.get("name") == "Over":
                        over_prices.append(outcome.get("price"))
                    elif outcome.get("name") == "Under":
                        under_prices.append(outcome.get("price"))

    h, d, a = _consensus(home_prices), _consensus(draw_prices), _consensus(away_prices)
    if not (h and d and a):
        raise ValueError(f"incomplete 1X2 market for {home} v {away}")
    fair = devig([h, d, a])

    p_over = None
    over, under = _consensus(over_prices), _consensus(under_prices)
    if over and under:
        p_over = float(devig([over, under])[0])

    return FixtureOdds(
        home_team=home,
        away_team=away,
        kickoff=kickoff,
        p_home=float(fair[0]),
        p_draw=float(fair[1]),
        p_away=float(fair[2]),
        p_over=p_over,
        total_line=line,
        bookmakers=books,
    )


def match_to_fixtures(
    odds: Iterable[FixtureOdds],
    fixtures: Sequence[Mapping],
    teams: Sequence[Mapping],
) -> tuple[dict[int, FixtureOdds], list[FixtureOdds]]:
    """Attach odds to FPL fixture ids.

    Returns ``(matched, unmatched)``. Anything that cannot be matched
    unambiguously is returned as unmatched rather than being force-fitted.
    """
    by_name: dict[str, int] = {}
    for team in teams:
        for key in ("name", "short_name"):
            value = team.get(key)
            if value:
                by_name[normalise_team(str(value))] = int(team["id"])

    fixture_index: dict[tuple[int, int], int] = {}
    for fixture in fixtures:
        if fixture.get("finished"):
            continue
        home, away = fixture.get("team_h"), fixture.get("team_a")
        if home and away:
            fixture_index[(int(home), int(away))] = int(fixture["id"])

    matched: dict[int, FixtureOdds] = {}
    unmatched: list[FixtureOdds] = []
    for event in odds:
        home_key, away_key = event.key
        home_id, away_id = by_name.get(home_key), by_name.get(away_key)
        fixture_id = (
            fixture_index.get((home_id, away_id))
            if home_id and away_id
            else None
        )
        if fixture_id is None:
            unmatched.append(event)
        else:
            matched[fixture_id] = event
    return matched, unmatched
