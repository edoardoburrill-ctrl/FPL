"""Walk a real season forward with a strict information cutoff.

The loop is deliberately simple: at each gameweek boundary, build a snapshot
from only the gameweeks already played, run it through the same
`build_context`/`project_gameweek` the live tool uses, and compare the result
to what actually happened. No transfers, no squad, no optimiser — this
validates the *projection engine*, which is the thing every other decision in
the system is built on. Replaying the optimiser's transfer decisions against a
real season is a larger, separate piece of work (see `DECISIONS.md`).
"""

from __future__ import annotations

from dataclasses import dataclass

from fpl.backtest.adapter import SeasonData, load_season, snapshot_as_of
from fpl.backtest.metrics import GameweekResult, SeasonReport
from fpl.data.historical import HistoricalClient
from fpl.models.projection import build_context, project_gameweek

MIN_MINUTES_TO_COUNT = 1  # only score predictions against players who actually played


def replay_season(
    season: str,
    start_gameweek: int = 6,
    end_gameweek: int | None = None,
    client: HistoricalClient | None = None,
    progress=None,
) -> SeasonReport:
    """Project every gameweek from ``start_gameweek`` on, no lookahead, and score it.

    ``start_gameweek`` defaults to 6 rather than 1: the team-strength model
    needs some real matches to fit against, and the first handful of gameweeks
    would just be testing the prior, not the model.
    """
    client = client or HistoricalClient()
    data = load_season(client, season)
    end_gameweek = end_gameweek or data.max_gameweek

    results: list[GameweekResult] = []
    for gameweek in range(start_gameweek, end_gameweek + 1):
        bootstrap, fixtures = snapshot_as_of(data, gameweek)
        if not any(f.get("event") == gameweek for f in fixtures):
            continue  # a blank gameweek in the archive itself; nothing to score

        context = build_context(bootstrap, fixtures)
        projections = project_gameweek(context, gameweek)
        actuals = data.actuals(gameweek)

        result = GameweekResult(gameweek=gameweek)
        for element_id, projection in projections.items():
            actual = actuals.get(element_id)
            if not actual or actual["minutes"] < MIN_MINUTES_TO_COUNT:
                continue
            result.elements.append(element_id)
            result.predicted.append(projection.expected_points)
            result.actual.append(actual["total_points"])
            result.xp_baseline.append(actual["xP"])

        if result.n:
            results.append(result)
        if progress:
            progress(gameweek, result.n)

    return SeasonReport(season=season, gameweeks=results)
