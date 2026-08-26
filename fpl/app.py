"""Wiring: clients, store, state and the fitted projection context in one place."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from fpl.config import Paths, Settings, paths, settings
from fpl.data.fpl_client import FPLClient, FPLNotFound, FPLUnavailable
from fpl.data.odds_client import FixtureOdds, OddsClient, match_to_fixtures
from fpl.data.store import Store
from fpl.models.projection import ProjectionContext, build_context
from fpl.models.scoring import load_rules
from fpl.state.ledger import SquadState, state_from_api

# Deep sync pulls per-fixture minutes logs, which is one request per player. Only
# players plausibly worth owning justify that, so the pool is filtered first.
DEEP_SYNC_MIN_MINUTES = 45
DEEP_SYNC_MAX_PLAYERS = 320


@dataclass
class App:
    paths: Paths
    settings: Settings
    client: FPLClient
    odds_client: OddsClient
    store: Store
    offline: bool = False

    @classmethod
    def create(cls, offline: bool = False, entry_id: int | None = None) -> "App":
        p = paths()
        s = settings()
        if entry_id:
            s = Settings(
                entry_id=entry_id,
                odds_api_key=s.odds_api_key,
                market_weight_now=s.market_weight_now,
                horizon_discount=s.horizon_discount,
                default_horizon=s.default_horizon,
            )
        return cls(
            paths=p,
            settings=s,
            client=FPLClient(cache_dir=p.cache, offline=offline),
            odds_client=OddsClient(s.odds_api_key, cache_dir=p.cache, offline=offline),
            store=Store(p.db),
            offline=offline,
        )

    # -- state -----------------------------------------------------------

    def load_state(self) -> SquadState:
        return SquadState.load(self.paths.state)

    def save_state(self, state: SquadState) -> None:
        state.save(self.paths.state)

    def entry_id(self) -> int | None:
        return self.settings.entry_id or self.load_state().entry_id

    # -- data ------------------------------------------------------------

    def bootstrap(self, force: bool = False) -> dict:
        return self.client.bootstrap(force=force)

    def fixtures(self, force: bool = False) -> list[dict]:
        return self.client.fixtures(force=force)

    def odds(self, bootstrap: Mapping, fixtures: Sequence[Mapping], force: bool = False):
        """Fetch odds and attach them to fixture ids.

        Returns ``(matched, unmatched)``; unmatched events are surfaced rather
        than silently dropped, because a name-matching failure is the difference
        between a market-anchored projection and a model-only one.
        """
        if not self.odds_client.enabled and not self.odds_client._cache_path().exists():
            return {}, []
        try:
            events = self.odds_client.fixture_odds(force=force)
        except Exception:
            return {}, []
        return match_to_fixtures(events, fixtures, bootstrap["teams"])

    def histories(self, bootstrap: Mapping, deep: bool = False) -> dict[int, list[dict]]:
        """Per-fixture minutes logs for players worth modelling carefully."""
        if not deep:
            return {}
        candidates = [
            el
            for el in bootstrap["elements"]
            if int(el.get("minutes", 0) or 0) >= DEEP_SYNC_MIN_MINUTES
        ]
        candidates.sort(key=lambda el: -int(el.get("minutes", 0) or 0))
        result: dict[int, list[dict]] = {}
        for element in candidates[:DEEP_SYNC_MAX_PLAYERS]:
            try:
                summary = self.client.element_summary(int(element["id"]))
            except FPLUnavailable:
                continue
            result[int(element["id"])] = list(summary.get("history", []))
        return result

    # -- projections -----------------------------------------------------

    def context(self, force: bool = False, use_odds: bool = True) -> ProjectionContext:
        bootstrap = self.bootstrap(force=force)
        fixtures = self.fixtures(force=force)
        matched: dict[int, FixtureOdds] = {}
        if use_odds:
            matched, _ = self.odds(bootstrap, fixtures, force=force)
        return build_context(
            bootstrap, fixtures, odds=matched, rules=load_rules(self.paths.root / "rules.json")
        )

    # -- sync ------------------------------------------------------------

    def sync(self, deep: bool = False, force: bool = True) -> dict:
        """Pull everything, snapshot it, and refresh squad state."""
        report: dict = {"errors": []}
        bootstrap = self.bootstrap(force=force)
        fixtures = self.fixtures(force=force)

        current = None
        for event in bootstrap.get("events", []):
            if event.get("is_next"):
                current = int(event["id"])
                break

        stamp = self.store.record_snapshot(
            bootstrap["elements"], event=current, note="sync"
        )
        report["observed_at"] = stamp
        report["players"] = len(bootstrap["elements"])
        report["next_event"] = current

        matched, unmatched = self.odds(bootstrap, fixtures, force=force)
        report["odds_matched"] = len(matched)
        report["odds_unmatched"] = [f"{o.home_team} v {o.away_team}" for o in unmatched]

        # Persist the raw payloads so a later run — or a backtest — can replay
        # exactly what was visible at this moment.
        snapshot_dir = self.paths.snapshots / stamp.replace(":", "")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "bootstrap-static.json").write_text(json.dumps(bootstrap))
        (snapshot_dir / "fixtures.json").write_text(json.dumps(fixtures))

        entry = self.entry_id()
        if entry:
            try:
                report.update(self._sync_entry(entry, bootstrap, current))
            except FPLNotFound:
                report["errors"].append(
                    f"FPL has no entry {entry}. Check the number in the URL of "
                    "your Points page — fantasy.premierleague.com/entry/"
                    "<NUMBER>/event/2 — remembering that FPL reissues entry ids "
                    "every season. Player data was still synced, and you can "
                    "record your squad by hand with `fpl set-squad`."
                )
            except FPLUnavailable as exc:
                report["errors"].append(f"entry sync failed: {exc}")
        else:
            report["errors"].append(
                "no FPL entry id configured — set FPL_ENTRY_ID or pass --entry"
            )

        if deep:
            report["deep_histories"] = len(self.histories(bootstrap, deep=True))

        return report

    def _sync_entry(self, entry_id: int, bootstrap: Mapping, current: int | None) -> dict:
        entry = self.client.entry(entry_id, force=True)
        history = self.client.entry_history(entry_id, force=True)

        # Picks are public only once a gameweek's deadline has passed.
        last_played = None
        for row in history.get("current", []) or []:
            last_played = int(row["event"])
        if last_played is None:
            return {"entry": entry_id, "picks": "none yet"}

        picks = self.client.entry_picks(entry_id, last_played, force=True)
        prices = {int(e["id"]): int(e["now_cost"]) for e in bootstrap["elements"]}
        existing = self.load_state()
        state = state_from_api(entry, picks, history, existing=existing, prices=prices)
        self.save_state(state)
        return {
            "entry": entry_id,
            "squad_gameweek": state.gameweek,
            "free_transfers": state.free_transfers,
            "bank": state.bank,
        }
