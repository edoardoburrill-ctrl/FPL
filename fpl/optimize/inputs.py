"""Turning a projection context and squad state into optimiser inputs."""

from __future__ import annotations

from typing import Mapping, Sequence

from fpl.models.points import PlayerProjection
from fpl.models.projection import ProjectionContext, project_gameweek
from fpl.optimize.milp import Player
from fpl.state.ledger import SquadState

# Players this unlikely to play are noise in the program and slow it down.
MIN_START_PROBABILITY = 0.05

# P(a starter in this position blanks and the slot goes to a bench player),
# independent of which specific player fills it. The player-specific half of
# the auto-sub chance -- whether *this* player would actually get minutes if
# summoned -- comes from the minutes model's `p_appears` instead. Calibrated
# so a bench player with a coin-flip chance of playing (p_appears = 0.5)
# reproduces the optimiser's old flat constant: this is what Phase 3's plan
# meant by "the auto-sub probability the minutes model could supply" -- a
# nailed starter tactically left out of the XI and a genuine fringe player are
# not equally likely to be summoned, and a single constant per position could
# not tell them apart.
AUTO_SUB_TRIGGER_PROBABILITY = {1: 0.12, 2: 0.26, 3: 0.26, 4: 0.26}


def auto_sub_weight(position: int, p_appears: float) -> float:
    """P(a bench player with this appearance chance is auto-subbed in)."""
    trigger = AUTO_SUB_TRIGGER_PROBABILITY.get(position, 0.24)
    return trigger * min(1.0, max(0.0, p_appears))


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

        position = int(element["element_type"])
        bench_weight = {
            gw: auto_sub_weight(position, projections[gw][element_id].minutes.p_appears)
            for gw in gameweeks
            if element_id in projections[gw]
        }

        now_cost = int(element["now_cost"])
        holding = owned.get(element_id)
        players.append(
            Player(
                element=element_id,
                position=position,
                team=int(element["team"]),
                buy_price=now_cost,
                sell_price=holding.selling_price(now_cost) if holding else now_cost,
                points=points,
                bench_weight=bench_weight,
                name=str(element.get("web_name", element_id)),
            )
        )
    return players
