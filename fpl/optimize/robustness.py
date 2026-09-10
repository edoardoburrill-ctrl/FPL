"""Bootstrap-robust transfer recommendations.

The MILP maximises expected points against a single point estimate per
player-gameweek. That estimate carries real, known uncertainty — every term
in `models/points.py` is computed from an underlying distribution, and
`PlayerProjection.point_std` already reports the standard deviation implied
by those same distributions (see `models/points.py`'s variance helpers).
The optimiser itself has no way to tell a real edge from noise it is
over-fitting to: it will recommend a hit for a 0.3-point edge as readily as
for a 5-point one, if that is what the point estimate says.

This resamples the projections `n_resamples` times — perturbing each
player-gameweek's expected points by a draw from `Normal(0, point_std)` — and
re-solves the full horizon against each resample. For every transfer the
point-estimate plan recommends, the reported frequency is how often that
exact (sell, buy) pair reappears in the resampled solves: chosen in 95% of
resamples is a real edge; chosen in 40% is a coin-flip that should not cost a
hit. This is exactly the check PLAN.md's "Robustness" section describes.

Deliberately opt-in and slow: `n_resamples` full re-solves, the same cost
class as `fpl chips --restructuring`. Default `n_resamples=30` and a tight
per-solve time limit keep a five-gameweek horizon in the low minutes; raise
`n_resamples` for a more precise frequency at proportionally more cost.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fpl.optimize.milp import Player, optimise
from fpl.optimize.transfers import TransferMove, pair_transfers

DEFAULT_RESAMPLES = 30
# Each bootstrap solve only needs to rank plausible transfers, not match the
# headline plan's proof of optimality — a short per-solve limit keeps the
# whole batch's wall-clock roughly independent of the headline solve's own
# --time-limit.
DEFAULT_PER_SOLVE_TIME_LIMIT = 20.0


@dataclass(frozen=True)
class TransferStability:
    move: TransferMove
    frequency: float  # fraction of successful resamples choosing this exact pair
    resamples: int  # how many resamples the frequency is measured over

    @property
    def is_robust(self) -> bool:
        return self.frequency >= 0.8

    @property
    def is_coin_flip(self) -> bool:
        return self.frequency < 0.5


def resample_points(
    players: Sequence[Player], rng: np.random.Generator
) -> list[Player]:
    """One bootstrap draw: perturb every player-gameweek estimate by its own
    analytic standard deviation, everything else about the player unchanged."""
    resampled = []
    for player in players:
        noisy_points = {
            gw: ep + rng.normal(0.0, player.std.get(gw, 0.0))
            for gw, ep in player.points.items()
        }
        resampled.append(
            Player(
                element=player.element,
                position=player.position,
                team=player.team,
                buy_price=player.buy_price,
                sell_price=player.sell_price,
                points=noisy_points,
                appearance=player.appearance,
                std=player.std,
                name=player.name,
            )
        )
    return resampled


def bootstrap_transfer_stability(
    players: Sequence[Player],
    gameweeks: Sequence[int],
    baseline_moves: Sequence[TransferMove],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    per_solve_time_limit: float = DEFAULT_PER_SOLVE_TIME_LIMIT,
    **optimise_kwargs,
) -> list[TransferStability]:
    """Re-solve the horizon `n_resamples` times and score `baseline_moves`
    against how often each one recurs.

    `optimise_kwargs` should match whatever produced `baseline_moves` (bank,
    free_transfers, current_squad, discount, wildcard_gameweek, ...) so the
    resamples are answering the same question the headline plan answered.
    """
    if not baseline_moves:
        return []

    optimise_kwargs.setdefault("time_limit", per_solve_time_limit)
    rng = np.random.default_rng(seed)
    counts: Counter[tuple[int, int]] = Counter()
    successful = 0

    for _ in range(n_resamples):
        noisy_players = resample_points(players, rng)
        plan = optimise(noisy_players, gameweeks, **optimise_kwargs)
        if not plan.solved:
            continue
        successful += 1
        decision = plan.first()
        noisy_by_id = {p.element: p for p in noisy_players}
        for move in pair_transfers(decision, noisy_by_id):
            counts[(move.out_element, move.in_element)] += 1

    results = []
    for move in baseline_moves:
        pair = (move.out_element, move.in_element)
        frequency = (counts[pair] / successful) if successful else 0.0
        results.append(
            TransferStability(move=move, frequency=frequency, resamples=successful)
        )
    return results
