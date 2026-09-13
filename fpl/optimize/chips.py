"""When to play each chip.

Chip timing is usually decided by folklore — "bench boost on a double gameweek",
"triple captain the best home fixture". Those heuristics are often right, but they
are not measured, and they cannot tell you that this week's double is worth 3.1
points more than the next one, which is the only form in which the question can
actually be answered.

Bench Boost and Triple Captain are decided *inside* the integer program: the
solver is allowed to play them in any eligible gameweek and picks the one that
maximises the trajectory, which also means it will build toward them — buying a
bench that plays ahead of a Bench Boost, for instance.

Wildcard and Free Hit restructure the problem rather than adding a term to it, so
they are evaluated by re-solving the horizon with the chip active in each
candidate gameweek and comparing. Free Hit's squad revert is modelled explicitly
in the MILP (a temporary squad/eleven that never touches the persistent one), not
approximated with wildcard semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from fpl.optimize.milp import Plan, Player, optimise
from fpl.state.ledger import FIRST_HALF_LAST_GAMEWEEK, SquadState

# Chip timing needs a near-neutral discount, not the transfer planner's 0.85.
#
# The horizon discount exists because a transfer plan for gameweek 8 will be
# revised long before it is executed, so points promised then are worth less than
# points available now. That argument does not transfer to chips. A chip decision
# is a comparison between playing it this week and playing it in gameweek 8, and
# discounting the later branch at 0.85^5 would shrink a 12-point double-gameweek
# Bench Boost to 5 — advising you to burn chips early, which is precisely the
# mistake chip strategy is about avoiding. A small residual discount is kept
# because a distant chip week can still be disrupted by a fixture change.
CHIP_DISCOUNT = 0.97

CHIP_LABELS = {
    "wildcard": "Wildcard",
    "freehit": "Free Hit",
    "3xc": "Triple Captain",
    "bboost": "Bench Boost",
}


def eligible_gameweeks(
    state: SquadState, chip: str, gameweeks: Sequence[int]
) -> list[int]:
    """Gameweeks in the horizon where ``chip`` may legally be played.

    Both sets expire: a first-half chip not played by the GW19 deadline is lost,
    and a second-half chip cannot be played early. So eligibility depends on
    which half each candidate gameweek falls in, not just on what has been used.
    """
    used = state.chips_used.get(chip, [])
    used_halves = {1 if gw <= FIRST_HALF_LAST_GAMEWEEK else 2 for gw in used}
    return [
        gw
        for gw in gameweeks
        if (1 if gw <= FIRST_HALF_LAST_GAMEWEEK else 2) not in used_halves
    ]


@dataclass
class ChipAdvice:
    chip: str
    gameweek: int | None
    gain: float
    status: str
    plan: Plan | None = None

    @property
    def label(self) -> str:
        return CHIP_LABELS.get(self.chip, self.chip)

    @property
    def worth_playing(self) -> bool:
        return self.gameweek is not None and self.gain > 0.0


def chip_calendar(
    pool: Sequence[Player],
    gameweeks: Sequence[int],
    state: SquadState,
    *,
    include_restructuring: bool = False,
    discount: float = CHIP_DISCOUNT,
    **solver_kwargs,
) -> tuple[Plan, list[ChipAdvice]]:
    """Baseline plan plus the best gameweek for each available chip.

    The gain reported for each chip is the improvement over playing no chip at
    all across the same horizon — which is the number that should drive the
    decision, because holding a chip has no value in itself.
    """
    common = dict(
        current_squad=state.element_ids,
        bank=state.bank,
        free_transfers=state.free_transfers,
        discount=discount,
        **solver_kwargs,
    )

    baseline = optimise(pool, gameweeks, **common)
    if not baseline.solved:
        return baseline, []

    advice: list[ChipAdvice] = []

    boost_weeks = eligible_gameweeks(state, "bboost", gameweeks)
    if boost_weeks:
        plan = optimise(pool, gameweeks, bench_boost_gameweeks=boost_weeks, **common)
        advice.append(_summarise("bboost", plan, baseline))

    triple_weeks = eligible_gameweeks(state, "3xc", gameweeks)
    if triple_weeks:
        plan = optimise(
            pool, gameweeks, triple_captain_gameweeks=triple_weeks, **common
        )
        advice.append(_summarise("3xc", plan, baseline))

    if include_restructuring:
        for chip in ("wildcard", "freehit"):
            weeks = eligible_gameweeks(state, chip, gameweeks)
            best: ChipAdvice | None = None
            for gameweek in weeks:
                kwarg = "wildcard_gameweek" if chip == "wildcard" else "freehit_gameweek"
                plan = optimise(pool, gameweeks, **{kwarg: gameweek}, **common)
                if not plan.solved:
                    continue
                gain = plan.objective - baseline.objective
                if best is None or gain > best.gain:
                    best = ChipAdvice(chip, gameweek, gain, plan.status, plan)
            if best is not None:
                advice.append(best)

    advice.sort(key=lambda a: -a.gain)
    return baseline, advice


def _summarise(chip: str, plan: Plan, baseline: Plan) -> ChipAdvice:
    if not plan.solved:
        return ChipAdvice(chip, None, 0.0, plan.status)
    played = next((d for d in plan.decisions if d.chip == chip), None)
    return ChipAdvice(
        chip=chip,
        gameweek=played.gameweek if played else None,
        gain=plan.objective - baseline.objective,
        status=plan.status,
        plan=plan,
    )
