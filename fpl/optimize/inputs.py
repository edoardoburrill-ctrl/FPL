"""Turning a projection context and squad state into optimiser inputs."""

from __future__ import annotations

from typing import Mapping, Sequence

from fpl.models.points import PlayerProjection
from fpl.models.projection import ProjectionContext, project_gameweek
from fpl.optimize.milp import Player
from fpl.state.ledger import SquadState

# Players this unlikely to play are noise in the program and slow it down.
MIN_START_PROBABILITY = 0.05


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
        appearance = {
            gw: projections[gw][element_id].minutes.p_appears
            for gw in gameweeks
            if element_id in projections[gw]
        }
        std = {
            gw: projections[gw][element_id].point_std
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
                appearance=appearance,
                std=std,
                name=str(element.get("web_name", element_id)),
            )
        )
    return players
