"""Real historical FPL data, for validation the live API cannot provide.

Two things this codebase needs and the official API cannot supply, because it
only shows the present: proof that our models predict real outcomes better
than doing nothing, and proof that FPL's own tie-break rule is implemented
correctly against real matches rather than three worked examples from a blog.

Source: github.com/vaastav/Fantasy-Premier-League, a long-running public mirror
of FPL's own per-gameweek data (MIT licensed, `master` branch). It is not FPL's
API and could in principle drift from ground truth or disappear; nothing in the
live system depends on it, only the backtest and the bonus-rule verification do.

Cached identically to the live client: fetch once, keep, fall back to a stale
copy on failure. Seasons this old will never change, so the cache never
expires — there is no TTL here, unlike the live client's snapshots.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import httpx

from fpl.config import paths

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master"
USER_AGENT = "fpl-advisor/0.1 (+https://github.com/edoardoburrill-ctrl/FPL)"


class HistoricalDataUnavailable(RuntimeError):
    pass


def _read_csv(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


class HistoricalClient:
    def __init__(self, cache_dir: Path | None = None, timeout: float = 60.0) -> None:
        self.cache_dir = cache_dir or (paths().cache / "historical")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def _fetch(self, relative_path: str) -> str:
        cache_file = self.cache_dir / relative_path.replace("/", "__")
        if cache_file.exists():
            return cache_file.read_text()

        url = f"{BASE_URL}/{relative_path}"
        try:
            response = httpx.get(
                url, timeout=self.timeout, headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
            text = response.text
        except Exception as exc:
            raise HistoricalDataUnavailable(
                f"could not fetch {url} ({exc.__class__.__name__}: {exc})"
            ) from exc

        cache_file.write_text(text)
        return text

    def merged_gw(self, season: str) -> list[dict]:
        """Every player's stat line for every gameweek of ``season`` (e.g. '2024-25')."""
        return _read_csv(self._fetch(f"data/{season}/gws/merged_gw.csv"))

    def teams(self, season: str) -> list[dict]:
        return _read_csv(self._fetch(f"data/{season}/teams.csv"))

    def fixtures(self, season: str) -> list[dict]:
        return _read_csv(self._fetch(f"data/{season}/fixtures.csv"))


@dataclass(frozen=True)
class MatchBPS:
    """One real match's BPS and bonus, both keyed by element id.

    Reconstructed from `merged_gw.csv` by grouping player-rows on `fixture`,
    which carries exactly the same `bps` and `bonus` fields FPL awarded — no
    need to parse the fixtures file's embedded per-event stats blob at all.
    """

    fixture: int
    gameweek: int
    bps: dict[int, int]
    bonus: dict[int, int]


def iter_match_bps(rows: list[dict]) -> Iterator[MatchBPS]:
    """Group per-player historical rows into one entry per real match."""
    by_fixture: dict[tuple[str, str], dict[int, dict]] = {}
    for row in rows:
        key = (row.get("fixture", ""), row.get("GW", ""))
        if not key[0]:
            continue
        by_fixture.setdefault(key, {})[int(row["element"])] = row

    for (fixture, gw), players in by_fixture.items():
        # A player who did not feature has bps=0 and is not part of the real
        # ranking FPL computed; only rows with actual minutes contributed to it.
        bps = {e: int(r["bps"]) for e, r in players.items() if int(r.get("minutes", 0) or 0) > 0}
        bonus = {e: int(players[e]["bonus"]) for e in bps}
        if bps:
            yield MatchBPS(fixture=int(fixture), gameweek=int(gw), bps=bps, bonus=bonus)
