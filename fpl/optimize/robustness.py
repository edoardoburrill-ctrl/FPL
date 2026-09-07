"""How much of a recommended transfer is signal, versus model noise?

The MILP (`milp.py`) maximises expected points against a single point
estimate of every player's projection. That is the right thing to optimise,
but a point estimate is not the truth — `models/points.py` now carries a
variance alongside every projection's mean — and an optimiser that only ever
sees the mean has no way to notice when a recommendation rests on a
razor-thin, noise-sized edge over the alternative.

The fix is not to make the MILP itself risk-aware — that would need a
stochastic program, a much larger undertaking for benefit that is mostly
captured more cheaply here. Instead: perturb every player-gameweek
projection by a draw from its own uncertainty, re-solve, and repeat. A
transfer that survives in 95% of resamples is a real edge; one that survives
in 40% is a coin-flip that should not cost a hit.

This deliberately resamples the *inputs* (player point estimates), not the
solver's own tie-breaking — the MILP is solved to (as close to) optimality
each time, so all the variation in the reported plans comes from the
projection uncertainty, which is the thing actually worth quantifying.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from fpl.models.points import PlayerProjection
from fpl.optimize.milp import Player, optimise
from fpl.optimize.transfers import TransferMove, pair_transfers


@dataclass(frozen=True)
class TransferStability:
    """One transfer move, and how often it survived resampling."""

    move: TransferMove
    frequency: float  # fraction of resamples in which this exact move appeared


@dataclass
class RobustnessReport:
    n_resamples: int
    n_solved: int
    moves: list[TransferStability]

    def frequency_of(self, out_element: int, in_element: int) -> float:
        for stability in self.moves:
            if (
                stability.move.out_element == out_element
                and stability.move.in_element == in_element
            ):
                return stability.frequency
        return 0.0


def resample_players(
    players: Sequence[Player],
    projections: Mapping[int, Mapping[int, PlayerProjection]],
    rng: np.random.Generator,
) -> list[Player]:
    """One bootstrap draw: every player-gameweek point estimate perturbed by a
    draw from its own projection variance.

    ``projections`` is ``{gameweek: {element: PlayerProjection}}`` — the same
    shape `project_gameweek` returns for a horizon. A player-gameweek missing
    from ``projections`` (blanks, or a player outside the pruned pool) keeps
    its point estimate unperturbed rather than being treated as zero-variance
    by omission, since a blank is already zero for a good reason unrelated to
    uncertainty.
    """
    resampled: list[Player] = []
    for player in players:
        new_points = dict(player.points)
        for gameweek, mean in player.points.items():
            projection = projections.get(gameweek, {}).get(player.element)
            if projection is None:
                continue
            sd = projection.points_sd
            if sd <= 0:
                continue
            draw = rng.normal(mean, sd)
            new_points[gameweek] = max(0.0, float(draw))
        resampled.append(replace(player, points=new_points))
    return resampled


def transfer_stability(
    players: Sequence[Player],
    projections: Mapping[int, Mapping[int, PlayerProjection]],
    gameweeks: Sequence[int],
    current_squad: Sequence[int],
    bank: int,
    free_transfers: int,
    *,
    n_resamples: int = 20,
    seed: int | None = None,
    time_limit: float = 15.0,
    rng: np.random.Generator | None = None,
    **optimise_kwargs,
) -> RobustnessReport:
    """Re-solve the horizon across bootstrap resamples of the projections.

    Returns, for every transfer move that appeared in at least one resample's
    first-gameweek plan, the fraction of resamples in which it appeared. A
    short ``time_limit`` per solve is deliberate: two dozen near-optimal
    solves are more informative here than one proven-optimal one, and the
    MILP's own legality re-check (`milp._is_legal`) still guards every one of
    them, so a time-limited resample can never surface an illegal plan.
    """
    if n_resamples < 1:
        raise ValueError("need at least one resample")
    rng = rng if rng is not None else np.random.default_rng(seed)
    by_id = {p.element: p for p in players}

    counts: dict[tuple[int, int], int] = {}
    example: dict[tuple[int, int], TransferMove] = {}
    n_solved = 0

    for _ in range(n_resamples):
        sample = resample_players(players, projections, rng)
        plan = optimise(
            sample,
            gameweeks,
            current_squad,
            bank,
            free_transfers,
            time_limit=time_limit,
            **optimise_kwargs,
        )
        if not plan.solved:
            continue
        n_solved += 1
        decision = plan.first()
        for move in pair_transfers(decision, by_id):
            key = (move.out_element, move.in_element)
            counts[key] = counts.get(key, 0) + 1
            example.setdefault(key, move)

    moves = [
        TransferStability(move=example[key], frequency=count / n_resamples)
        for key, count in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
    return RobustnessReport(n_resamples=n_resamples, n_solved=n_solved, moves=moves)
