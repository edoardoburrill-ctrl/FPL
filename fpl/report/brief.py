"""The weekly brief: what to actually do before the deadline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Sequence

from fpl.models.points import PlayerProjection
from fpl.models.projection import ProjectionContext
from fpl.optimize.milp import Decision, Plan, Player, optimise, prune_pool
from fpl.state.ledger import TRANSFER_HIT_COST, SquadState

# A transfer worth less than this over the whole horizon is noise, not an edge.
MEANINGFUL_GAIN = 0.75
# Two captaincy options this close are a coin-flip, whatever the ranking says.
COIN_FLIP_MARGIN = 0.35


@dataclass
class TransferMove:
    out_element: int
    in_element: int
    out_name: str
    in_name: str
    out_sell_price: int
    in_buy_price: int


@dataclass
class Brief:
    gameweek: int
    deadline: datetime | None
    plan: Plan
    hold_plan: Plan
    moves: list[TransferMove]
    gain: float
    captain: int | None
    vice: int | None
    captain_margin: float
    captain_alternatives: list[tuple[int, float]]
    starting: list[int]
    bench: list[int]
    hits: int
    free_transfers: int
    bank: float
    chips_available: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def recommends_transfer(self) -> bool:
        return bool(self.moves)

    @property
    def gain_is_meaningful(self) -> bool:
        return self.gain >= MEANINGFUL_GAIN

    @property
    def captain_is_close(self) -> bool:
        return self.captain_margin < COIN_FLIP_MARGIN


def deadline_for(ctx: ProjectionContext, gameweek: int) -> datetime | None:
    for event in ctx.events:
        if int(event["id"]) == gameweek and event.get("deadline_time"):
            try:
                return datetime.fromisoformat(
                    str(event["deadline_time"]).replace("Z", "+00:00")
                ).astimezone(timezone.utc)
            except ValueError:
                return None
    return None


def pair_transfers(
    decision: Decision, players: Mapping[int, Player]
) -> list[TransferMove]:
    """Pair sales with purchases by position, which is how a manager reads them.

    The optimiser only knows it bought three players and sold three; presenting
    that as "sell X, buy Y" needs the positional pairing that the squad-shape
    constraint guarantees exists.
    """
    outgoing = sorted(decision.transfers_out, key=lambda e: players[e].position)
    incoming = sorted(decision.transfers_in, key=lambda e: players[e].position)

    moves: list[TransferMove] = []
    for out_element, in_element in zip(outgoing, incoming):
        out_player, in_player = players[out_element], players[in_element]
        moves.append(
            TransferMove(
                out_element=out_element,
                in_element=in_element,
                out_name=out_player.name,
                in_name=in_player.name,
                out_sell_price=out_player.sell_price,
                in_buy_price=in_player.buy_price,
            )
        )
    return moves


def build_brief(
    ctx: ProjectionContext,
    state: SquadState,
    players: Sequence[Player],
    projections: Mapping[int, Mapping[int, PlayerProjection]],
    gameweeks: Sequence[int],
    *,
    discount: float = 0.85,
    max_hits: int = 2,
    wildcard_gameweek: int | None = None,
    time_limit: float = 90.0,
) -> Brief:
    """Solve the horizon, then solve it again holding, and compare."""
    gameweek = gameweeks[0]
    pool = prune_pool(players, gameweeks, always_include=state.element_ids)
    by_id = {p.element: p for p in pool}

    common = dict(
        current_squad=state.element_ids,
        bank=state.bank,
        free_transfers=state.free_transfers,
        discount=discount,
        max_hits_per_gameweek=max_hits,
        time_limit=time_limit,
    )

    plan = optimise(pool, gameweeks, wildcard_gameweek=wildcard_gameweek, **common)
    # The value of this week's move is the difference between acting and holding,
    # measured over the whole horizon — not the difference in this gameweek
    # alone, which would ignore what the transfer sets up or forecloses.
    hold = optimise(pool, gameweeks, transfer_caps={gameweek: 0}, **common)

    warnings: list[str] = []
    if not plan.solved:
        warnings.append(f"optimiser returned {plan.status}; no plan available")
        return Brief(
            gameweek=gameweek, deadline=deadline_for(ctx, gameweek), plan=plan,
            hold_plan=hold, moves=[], gain=0.0, captain=None, vice=None,
            captain_margin=0.0, captain_alternatives=[], starting=[], bench=[],
            hits=0, free_transfers=state.free_transfers, bank=state.bank / 10,
            chips_available=state.chips_available(gameweek), warnings=warnings,
        )

    if not plan.proven_optimal:
        warnings.append(
            "solver stopped at the time limit; the plan is good but not proven "
            "optimal — raise --time-limit for a longer search"
        )

    decision = plan.first()
    gain = plan.objective - (hold.objective if hold.solved else plan.objective)

    ranked = sorted(
        decision.starting,
        key=lambda e: -projections[gameweek][e].expected_points,
    )
    margin = 0.0
    if len(ranked) > 1:
        margin = (
            projections[gameweek][ranked[0]].expected_points
            - projections[gameweek][ranked[1]].expected_points
        )

    alternatives = [
        (element, projections[gameweek][element].expected_points)
        for element in ranked[:5]
    ]

    # Only flag rotation risk in the eleven. A £4.2m bench player who never
    # starts is doing exactly the job he was bought for, and warning about him
    # every week trains you to ignore the warnings that matter.
    for element in decision.starting:
        forecast = projections[gameweek][element].minutes
        if 0.0 < forecast.p_start < 0.65:
            warnings.append(
                f"{by_id[element].name} is only {forecast.p_start:.0%} to start"
            )

    return Brief(
        gameweek=gameweek,
        deadline=deadline_for(ctx, gameweek),
        plan=plan,
        hold_plan=hold,
        moves=pair_transfers(decision, by_id),
        gain=gain,
        captain=decision.captain,
        vice=decision.vice,
        captain_margin=margin,
        captain_alternatives=alternatives,
        starting=decision.starting,
        bench=[e for e in decision.squad if e not in decision.starting],
        hits=decision.hits,
        free_transfers=decision.free_transfers,
        bank=decision.bank,
        chips_available=state.chips_available(gameweek),
        warnings=warnings,
    )
