"""Turning the historical archive into the same shapes the live pipeline expects.

The payoff of this module is that `models.projection.build_context` and
`project_gameweek` do not need to know or care whether their input came from
the live API or a historical archive — they consume the same bootstrap-static
and fixtures shapes either way. That is what lets the backtest exercise the
*real* projection pipeline rather than a parallel one built just for testing.

The one rule that makes this a backtest and not a leaderboard: `snapshot_as_of`
aggregates only rows from gameweeks strictly before the target. A model fitted
on a snapshot that saw the target gameweek's own results would not be predicting
anything — it would be describing what already happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from fpl.data.historical import HistoricalClient
from fpl.models.scoring import DEF, FWD, GK, MID

POSITION_CODE = {"GK": GK, "GKP": GK, "DEF": DEF, "MID": MID, "FWD": FWD}

CUMULATIVE_FIELDS = (
    "minutes", "starts", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "saves", "yellow_cards", "red_cards", "own_goals",
    "penalties_missed", "penalties_saved", "bps", "bonus",
)


def _num(row: Mapping, key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class SeasonData:
    season: str
    rows: list[dict]  # every player-gameweek row, real merged_gw.csv shape
    teams: list[dict]  # id, name, short_name
    fixtures: list[dict]  # id, event, team_h, team_a, scores, finished
    team_id_by_name: dict[str, int] = field(default_factory=dict)
    max_gameweek: int = 0

    def actuals(self, gameweek: int) -> dict[int, dict]:
        """Real outcomes for one gameweek: element -> stat line.

        `clean_sheets` and `bonus` are read straight from the archive, the
        same source `total_points` already comes from, for
        `backtest/calibration.py` to check the projection engine's
        probabilities against.
        """
        return {
            int(r["element"]): {
                "total_points": _num(r, "total_points"),
                "xP": _num(r, "xP"),
                "minutes": _num(r, "minutes"),
                "clean_sheets": _num(r, "clean_sheets"),
                "bonus": _num(r, "bonus"),
            }
            for r in self.rows
            if int(r.get("GW", 0) or 0) == gameweek
        }


def load_season(client: HistoricalClient, season: str) -> SeasonData:
    teams = client.teams(season)
    team_id_by_name = {t["name"]: int(t["id"]) for t in teams}
    rows = client.merged_gw(season)
    fixtures = client.fixtures(season)
    max_gw = max((int(r.get("GW", 0) or 0) for r in rows), default=0)
    return SeasonData(season, rows, teams, fixtures, team_id_by_name, max_gw)


def _element_type(rows: list[dict]) -> int:
    for row in rows:
        code = POSITION_CODE.get(row.get("position"))
        if code:
            return code
    return MID


def snapshot_as_of(data: SeasonData, gameweek: int) -> tuple[dict, list[dict]]:
    """A bootstrap-static-shaped snapshot built only from gameweeks < ``gameweek``.

    Aggregates each player's cumulative stats and takes the most recent price
    and team as of that point — exactly what a live sync would have captured
    had it been run right before this gameweek's real deadline.
    """
    seen = [r for r in data.rows if int(r.get("GW", 0) or 0) < gameweek]

    by_element: dict[int, list[dict]] = {}
    for row in seen:
        by_element.setdefault(int(row["element"]), []).append(row)

    elements: list[dict] = []
    for element_id, rows in by_element.items():
        rows_sorted = sorted(rows, key=lambda r: int(r.get("GW", 0) or 0))
        latest = rows_sorted[-1]
        totals = {field: sum(_num(r, field) for r in rows) for field in CUMULATIVE_FIELDS}
        xg = sum(_num(r, "expected_goals") for r in rows)
        xa = sum(_num(r, "expected_assists") for r in rows)

        elements.append(
            {
                "id": element_id,
                "web_name": latest.get("name", str(element_id)).split(" ")[-1],
                "first_name": "",
                "second_name": latest.get("name", str(element_id)),
                "team": data.team_id_by_name.get(latest.get("team", ""), 0),
                "element_type": _element_type(rows),
                "now_cost": int(_num(latest, "value")) or 50,
                "status": "a",
                "chance_of_playing_next_round": None,
                "expected_goals": f"{xg:.2f}",
                "expected_assists": f"{xa:.2f}",
                "selected_by_percent": "0",
                **{k: totals[k] for k in CUMULATIVE_FIELDS},
            }
        )

    fixtures = [
        {
            "id": int(f["id"]),
            "event": int(f["event"]) if f.get("event") else None,
            "team_h": int(f["team_h"]),
            "team_a": int(f["team_a"]),
            "team_h_score": int(f["team_h_score"]) if f.get("team_h_score") else None,
            "team_a_score": int(f["team_a_score"]) if f.get("team_a_score") else None,
            "finished": str(f.get("finished")).lower() == "true"
            and int(f["event"] or 0) < gameweek,
        }
        for f in data.fixtures
        if f.get("event")
    ]

    events = [
        {
            "id": gw,
            "deadline_time": None,
            "finished": gw < gameweek,
            "is_current": gw == gameweek - 1,
            "is_next": gw == gameweek,
        }
        for gw in range(1, data.max_gameweek + 1)
    ]

    bootstrap = {
        "elements": elements,
        "teams": [
            {"id": int(t["id"]), "name": t["name"], "short_name": t.get("short_name", t["name"][:3])}
            for t in data.teams
        ],
        "events": events,
    }
    return bootstrap, fixtures
