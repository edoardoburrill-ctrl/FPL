"""Turning a projection context and squad state into optimiser inputs."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from fpl.models.points import PlayerProjection
from fpl.models.projection import ProjectionContext, project_gameweek
from fpl.optimize.milp import BENCH_WEIGHT, Player
from fpl.state.ledger import SquadState

# Players this unlikely to play are noise in the program and slow it down.
MIN_START_PROBABILITY = 0.05

# A player counts as a "likely starter" for the purpose of estimating how often
# a starting XI needs an auto-sub only above this P(start) — a fringe squad
# player's own blank risk says nothing about whether *your* eleven needs a sub.
LIKELY_STARTER_THRESHOLD = 0.5
# Below this many likely starters at a position, the estimate is too thin to
# trust (an early-season universe, or a small backtest/test pool); fall back
# to the constant.
MIN_STARTERS_FOR_ESTIMATE = 15


def horizon_gameweeks(ctx: ProjectionContext, horizon: int, start: int | None = None) -> list[int]:
    first = start or ctx.current_event or 1
    last = max((int(e["id"]) for e in ctx.events), default=first)
    return [gw for gw in range(first, first + horizon) if gw <= last]


def project_horizon(
    ctx: ProjectionContext,
    gameweeks: Sequence[int],
    histories: Mapping[int, Sequence[Mapping]] | None = None,
) -> dict[int, dict[int, PlayerProjection]]:
    return {gw: project_gameweek(ctx, gw, histories=histories) for gw in gameweeks}


def build_players(
    ctx: ProjectionContext,
    projections: Mapping[int, Mapping[int, PlayerProjection]],
    state: SquadState | None = None,
    include_unlikely: bool = False,
) -> list[Player]:
    """Assemble the optimiser's player list, with correct selling prices."""
    gameweeks = list(projections)
    state = state or SquadState()
    owned = {h.element: h for h in state.holdings}

    players: list[Player] = []
    for element in ctx.elements:
        element_id = int(element["id"])
        points = {
            gw: projections[gw][element_id].expected_points
            for gw in gameweeks
            if element_id in projections[gw]
        }

        if not include_unlikely and element_id not in owned:
            likely = max(
                (
                    projections[gw][element_id].minutes.p_start
                    for gw in gameweeks
                    if element_id in projections[gw]
                ),
                default=0.0,
            )
            if likely < MIN_START_PROBABILITY:
                continue

        now_cost = int(element["now_cost"])
        holding = owned.get(element_id)
        players.append(
            Player(
                element=element_id,
                position=int(element["element_type"]),
                team=int(element["team"]),
                buy_price=now_cost,
                sell_price=holding.selling_price(now_cost) if holding else now_cost,
                points=points,
                name=str(element.get("web_name", element_id)),
            )
        )
    return players


def bench_weights_for_gameweek(
    projections: Mapping[int, PlayerProjection],
    threshold: float = LIKELY_STARTER_THRESHOLD,
    min_starters: int = MIN_STARTERS_FOR_ESTIMATE,
) -> dict[int, float]:
    """P(a starter at this position blanks), averaged over likely starters.

    This is what actually triggers an FPL auto-sub — a bench player only scores
    if someone in the XI he could replace registers zero minutes — so it is a
    better-grounded stand-in for the constant ``BENCH_WEIGHT`` than a single
    number picked once and used all season: a congested run of fixtures or an
    international break genuinely raises rotation risk, and this tracks it,
    gameweek by gameweek, straight from the minutes model that projects
    everyone else. ``p_out`` is exactly the quantity needed here — it is
    already "P(zero involvement this fixture)" in ``MinutesForecast``.

    Restricted to players who are themselves likely to start: an obvious
    squad-filler's own blank risk is not evidence about whether the XI he'd
    theoretically replace ever needs an auto-sub.
    """
    by_position: dict[int, list[float]] = {}
    for projection in projections.values():
        if projection.minutes.p_start < threshold:
            continue
        by_position.setdefault(projection.position, []).append(projection.minutes.p_out)

    weights: dict[int, float] = {}
    for position, blanks in by_position.items():
        if len(blanks) < min_starters:
            continue
        weights[position] = float(np.mean(blanks))
    return weights


def bench_weight_table(
    projections: Mapping[int, Mapping[int, PlayerProjection]],
) -> dict[int, dict[int, float]]:
    """Per-gameweek, per-position bench weight: ``gameweek -> position -> weight``.

    Falls back to ``BENCH_WEIGHT``'s constants for any gameweek or position
    where there isn't enough signal to estimate from (very early season, or a
    small backtest/test universe).
    """
    return {
        gw: {**BENCH_WEIGHT, **bench_weights_for_gameweek(gw_projections)}
        for gw, gw_projections in projections.items()
    }
