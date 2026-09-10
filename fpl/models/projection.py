"""The pipeline: raw API payloads in, expected points per player per gameweek out.

Ordering of concerns:

1. Work out what each team's goal expectation is in each upcoming fixture, from
   the market where odds exist and from fitted ratings where they do not.
2. Fit empirical-Bayes priors across the whole player pool, then shrink each
   player's per-90 rates toward the prior for his position and price tier.
3. Forecast minutes.
4. Assemble expected points, term by term, for every fixture in the gameweek —
   so a double gameweek sums two fixtures and a blank sums none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from fpl.data.odds_client import FixtureOdds
from fpl.models import player_rates as rates_mod
from fpl.models.minutes import (
    MinutesForecast,
    forecast_minutes,
    price_tier,
    price_tier_priors,
    synthetic_history,
)
from fpl.models.points import PlayerRates, PlayerProjection, project_player
from fpl.models.scoring import GK, ScoringRules, load_rules
from fpl.models.team_strength import (
    DEFAULT_RHO,
    MarketSolution,
    TeamRatings,
    blend_goal_rates,
    conceded_distribution,
    fit_ratings,
    score_matrix,
    solve_from_market,
)

# Statistics we fit priors for. Each maps to a field on the bootstrap element.
RATE_STATS = (
    "expected_goals",
    "expected_assists",
    "saves",
    "bps",
    "yellow_cards",
    "red_cards",
)

DEFCON_COMPONENTS = ("clearances_blocks_interceptions", "tackles")
DEFCON_MID_FWD_EXTRA = ("recoveries",)


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def defcon_season_total(element: Mapping) -> float:
    """Season defensive-contribution *count* for a player.

    Prefers the component stats over any pre-aggregated field, because the
    aggregate has at times been published as points rather than actions and
    getting that wrong would inflate every defender's projection.
    """
    total = sum(_num(element.get(name)) for name in DEFCON_COMPONENTS)
    if int(element.get("element_type", 3)) != 2:
        total += sum(_num(element.get(name)) for name in DEFCON_MID_FWD_EXTRA)
    if total == 0:
        total = _num(element.get("defensive_contribution"))
    return total


def matches_played(fixtures: Sequence[Mapping]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for fixture in fixtures:
        if not fixture.get("finished"):
            continue
        for key in ("team_h", "team_a"):
            team = fixture.get(key)
            if team:
                counts[int(team)] = counts.get(int(team), 0) + 1
    return counts


def finished_matches(fixtures: Sequence[Mapping]) -> list[dict]:
    """Fixtures in the shape :func:`fit_ratings` wants."""
    rows: list[dict] = []
    for fixture in fixtures:
        if not fixture.get("finished"):
            continue
        if fixture.get("team_h_score") is None or fixture.get("team_a_score") is None:
            continue
        rows.append(
            {
                "home": int(fixture["team_h"]),
                "away": int(fixture["team_a"]),
                "home_goals": int(fixture["team_h_score"]),
                "away_goals": int(fixture["team_a_score"]),
                "day": float(fixture.get("event") or 0),
            }
        )
    return rows


@dataclass
class ProjectionContext:
    """Everything fitted once per sync, reused for every gameweek projected."""

    elements: list[dict]
    teams: list[dict]
    fixtures: list[dict]
    events: list[dict]
    rules: ScoringRules
    ratings: TeamRatings | None
    priors: dict[str, rates_mod.RatePriors]
    defcon_priors: rates_mod.RatePriors
    team_contexts: dict[int, rates_mod.TeamContext]
    start_priors: dict[tuple[int, int], float]
    matches: dict[int, int]
    odds: dict[int, FixtureOdds] = field(default_factory=dict)
    league_average_goals: float = 1.4

    def element(self, element_id: int) -> dict:
        return self._by_id[int(element_id)]

    def __post_init__(self) -> None:
        self._by_id = {int(e["id"]): e for e in self.elements}

    @property
    def current_event(self) -> int | None:
        for event in self.events:
            if event.get("is_next"):
                return int(event["id"])
        for event in self.events:
            if not event.get("finished"):
                return int(event["id"])
        return None


def build_context(
    bootstrap: Mapping,
    fixtures: Sequence[Mapping],
    odds: Mapping[int, FixtureOdds] | None = None,
    rules: ScoringRules | None = None,
    ratings_half_life: float = 12.0,
) -> ProjectionContext:
    """Fit everything that does not depend on which gameweek we are projecting."""
    elements = [dict(e) for e in bootstrap["elements"]]
    teams = [dict(t) for t in bootstrap["teams"]]
    events = [dict(e) for e in bootstrap.get("events", [])]
    rules = rules or load_rules()

    # Defensive-contribution counts are derived, so materialise them once.
    for element in elements:
        element["_defcon"] = defcon_season_total(element)

    played = matches_played(fixtures)
    history = finished_matches(fixtures)
    ratings: TeamRatings | None = None
    if len(history) >= 10:
        # `day` is the gameweek index, so the half-life is in gameweeks.
        ratings = fit_ratings(history, half_life_days=ratings_half_life)

    priors = {stat: rates_mod.fit_priors(elements, stat) for stat in RATE_STATS}
    defcon_priors = rates_mod.fit_priors(elements, "_defcon")
    contexts = rates_mod.team_contexts(elements, teams, played)

    start_priors = price_tier_priors(elements)

    league_goals = 1.4
    if history:
        league_goals = float(
            np.mean([m["home_goals"] + m["away_goals"] for m in history]) / 2.0
        )

    return ProjectionContext(
        elements=elements,
        teams=teams,
        fixtures=[dict(f) for f in fixtures],
        events=events,
        rules=rules,
        ratings=ratings,
        priors=priors,
        defcon_priors=defcon_priors,
        team_contexts=contexts,
        start_priors=start_priors,
        matches=played,
        odds=dict(odds or {}),
        league_average_goals=league_goals,
    )


def fixture_goal_rates(
    ctx: ProjectionContext, fixture: Mapping, gameweeks_ahead: int = 0
) -> tuple[float, float]:
    """``(home goal expectation, away goal expectation)`` for one fixture."""
    home, away = int(fixture["team_h"]), int(fixture["team_a"])

    model: tuple[float, float]
    if ctx.ratings is not None:
        model = ctx.ratings.expected_goals(home, away)
    else:
        model = (ctx.league_average_goals * 1.1, ctx.league_average_goals * 0.9)

    market: tuple[float, float] | None = None
    event_odds = ctx.odds.get(int(fixture["id"]))
    if event_odds is not None:
        solution: MarketSolution = solve_from_market(
            event_odds.p_home,
            event_odds.p_draw,
            event_odds.p_away,
            p_over=event_odds.p_over,
            line=event_odds.total_line,
        )
        market = (solution.lam_home, solution.lam_away)

    return blend_goal_rates(market, model, gameweeks_ahead)


def player_rates(ctx: ProjectionContext, element: Mapping) -> PlayerRates:
    """Shrunk per-90 rates for one player, expressed as shares where relevant."""
    team_context = ctx.team_contexts.get(
        int(element["team"]),
        rates_mod.TeamContext(ctx.league_average_goals, 0),
    )

    goals_rate = rates_mod.shrunk_rate(element, ctx.priors["expected_goals"])
    assists_rate = rates_mod.shrunk_rate(element, ctx.priors["expected_assists"])

    nineties = _num(element.get("minutes")) / 90.0
    return PlayerRates(
        goals_share=rates_mod.output_share(goals_rate, team_context),
        assists_share=rates_mod.output_share(assists_rate, team_context),
        saves_per_90=(
            rates_mod.shrunk_rate(element, ctx.priors["saves"])
            if int(element["element_type"]) == GK
            else 0.0
        ),
        defcon_per_90=rates_mod.shrunk_rate(element, ctx.defcon_priors),
        bps_per_90=rates_mod.shrunk_rate(element, ctx.priors["bps"]),
        yellow_per_90=rates_mod.shrunk_rate(element, ctx.priors["yellow_cards"]),
        red_per_90=rates_mod.shrunk_rate(element, ctx.priors["red_cards"]),
    )


def minutes_forecast(
    ctx: ProjectionContext,
    element: Mapping,
    history: Sequence[Mapping] | None = None,
) -> MinutesForecast:
    team_matches = ctx.matches.get(int(element["team"]), 0)
    rows = list(history) if history else synthetic_history(element, team_matches)
    position = int(element["element_type"])
    tier = price_tier(_num(element.get("now_cost")), ctx.elements, position)
    prior = ctx.start_priors.get((position, tier), 0.4)
    return forecast_minutes(rows, element, prior_start_rate=prior)


def gameweek_fixtures(ctx: ProjectionContext, gameweek: int) -> list[dict]:
    return [
        f
        for f in ctx.fixtures
        if f.get("event") == gameweek and not f.get("finished")
    ]


def project_gameweek(
    ctx: ProjectionContext,
    gameweek: int,
    histories: Mapping[int, Sequence[Mapping]] | None = None,
    gameweeks_ahead: int | None = None,
) -> dict[int, PlayerProjection]:
    """Expected points for every player in ``gameweek``.

    A player with two fixtures (a double gameweek) has both summed; a player
    whose team blanks gets an empty projection worth zero, which is exactly what
    the optimiser needs to see.
    """
    histories = histories or {}
    current = ctx.current_event or gameweek
    ahead = gameweeks_ahead if gameweeks_ahead is not None else max(0, gameweek - current)

    fixtures = gameweek_fixtures(ctx, gameweek)
    by_team: dict[int, list[tuple[dict, bool]]] = {}
    for fixture in fixtures:
        by_team.setdefault(int(fixture["team_h"]), []).append((fixture, True))
        by_team.setdefault(int(fixture["team_a"]), []).append((fixture, False))

    goal_rates = {
        int(f["id"]): fixture_goal_rates(ctx, f, ahead) for f in fixtures
    }

    projections: dict[int, PlayerProjection] = {}
    for element in ctx.elements:
        element_id = int(element["id"])
        team = int(element["team"])
        position = int(element["element_type"])
        minutes = minutes_forecast(ctx, element, histories.get(element_id))

        combined = PlayerProjection(
            element=element_id,
            fixture=None,
            position=position,
            minutes=minutes,
            breakdown={},
        )

        for fixture, is_home in by_team.get(team, []):
            lam_home, lam_away = goal_rates[int(fixture["id"])]
            team_goals = lam_home if is_home else lam_away
            opponent_goals = lam_away if is_home else lam_home
            matrix = score_matrix(lam_home, lam_away, DEFAULT_RHO)
            conceded = conceded_distribution(matrix, home=is_home)

            single = project_player(
                element_id,
                position,
                minutes,
                player_rates(ctx, element),
                team_goals=team_goals,
                opponent_goals=opponent_goals,
                conceded_pmf=conceded,
                rules=ctx.rules,
                fixture=int(fixture["id"]),
                league_average_conceded=ctx.league_average_goals,
            )
            for key, value in single.breakdown.items():
                combined.breakdown[key] = combined.breakdown.get(key, 0.0) + value
            # Independent fixtures (a double gameweek), so variances add too.
            for key, value in single.variance.items():
                combined.variance[key] = combined.variance.get(key, 0.0) + value

        projections[element_id] = combined

    return projections
