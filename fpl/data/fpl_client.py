"""Client for the official (public, unauthenticated) FPL API.

Two design points matter here:

1. **Everything is cached to disk.** Development and testing happen in
   environments with no route to ``fantasy.premierleague.com``; the models are
   built against cached JSON so the whole engine is exercisable offline.
2. **Nothing is interpreted here.** This layer returns raw decoded JSON. Scoring
   rules, in particular, are read from ``game_settings`` downstream rather than
   being baked into any parsing step.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

from fpl.config import FPL_API_BASE, paths

USER_AGENT = "fpl-advisor/0.1 (+https://github.com/edoardoburrill-ctrl/FPL)"


class FPLUnavailable(RuntimeError):
    """Raised when the API cannot be reached and no cached copy exists."""


class FPLNotFound(FPLUnavailable):
    """Raised when the API answers 404 — the resource genuinely is not there.

    Kept distinct from a transport failure because the two need opposite
    responses: a network problem should fall back to cached data and retry
    later, whereas a 404 on an entry means the id is wrong and no amount of
    retrying will help.
    """


class FPLClient:
    def __init__(
        self,
        base_url: str = FPL_API_BASE,
        cache_dir: Path | None = None,
        ttl_seconds: float = 900.0,
        offline: bool = False,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir or paths().cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self.offline = offline
        self.timeout = timeout

    # -- plumbing ---------------------------------------------------------

    def _cache_path(self, endpoint: str) -> Path:
        slug = endpoint.strip("/").replace("/", "__") or "root"
        return self.cache_dir / f"{slug}.json"

    def _read_cache(self, endpoint: str, max_age: float | None) -> Any | None:
        path = self._cache_path(endpoint)
        if not path.exists():
            return None
        if max_age is not None and (time.time() - path.stat().st_mtime) > max_age:
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    def get(self, endpoint: str, *, force: bool = False) -> Any:
        """Fetch ``endpoint``, preferring a fresh cache entry.

        Falls back to a *stale* cache entry if the network fails, so a blocked
        or flaky connection degrades to slightly old data rather than an error.
        """
        if not force:
            cached = self._read_cache(endpoint, self.ttl)
            if cached is not None:
                return cached

        if self.offline:
            stale = self._read_cache(endpoint, None)
            if stale is not None:
                return stale
            raise FPLUnavailable(f"offline and no cached copy of {endpoint!r}")

        url = f"{self.base_url}/{endpoint.strip('/')}/"
        try:
            response = httpx.get(
                url, timeout=self.timeout, headers={"User-Agent": USER_AGENT}
            )
            if response.status_code == 404:
                # Never serve stale data for a 404: the resource is not there,
                # and a cached copy would hide a wrong id behind old results.
                raise FPLNotFound(f"{url} does not exist (404)")
            response.raise_for_status()
            payload = response.json()
        except FPLNotFound:
            raise
        except Exception as exc:  # network, HTTP status, or decode failure
            stale = self._read_cache(endpoint, None)
            if stale is not None:
                return stale
            raise FPLUnavailable(
                f"could not fetch {url} ({exc.__class__.__name__}: {exc}) "
                "and no cached copy exists"
            ) from exc

        self._cache_path(endpoint).write_text(json.dumps(payload))
        return payload

    # -- endpoints --------------------------------------------------------

    def bootstrap(self, *, force: bool = False) -> dict:
        """Players, teams, gameweeks, and the authoritative scoring rules."""
        return self.get("bootstrap-static", force=force)

    def fixtures(self, *, force: bool = False) -> list[dict]:
        return self.get("fixtures", force=force)

    def element_summary(self, element_id: int, *, force: bool = False) -> dict:
        """Per-fixture history for one player."""
        return self.get(f"element-summary/{element_id}", force=force)

    def event_live(self, gameweek: int, *, force: bool = False) -> dict:
        return self.get(f"event/{gameweek}/live", force=force)

    def entry(self, entry_id: int, *, force: bool = False) -> dict:
        return self.get(f"entry/{entry_id}", force=force)

    def entry_history(self, entry_id: int, *, force: bool = False) -> dict:
        return self.get(f"entry/{entry_id}/history", force=force)

    def entry_picks(self, entry_id: int, gameweek: int, *, force: bool = False) -> dict:
        return self.get(f"entry/{entry_id}/event/{gameweek}/picks", force=force)

    def league_standings(self, league_id: int, *, force: bool = False) -> dict:
        return self.get(f"leagues-classic/{league_id}/standings", force=force)
