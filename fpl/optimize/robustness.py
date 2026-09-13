"""Bootstrap-robust transfer recommendations.

The MILP maximises expected points against a single point estimate for every
player-gameweek. That estimate is the best available guess, but it is still a
guess — and an optimiser handed one number per player cannot tell you whether
the transfer it picked is a real edge or a coin-flip that happened to land on
one side of the line.

`PLAN.md` calls for the fix directly: solve across bootstrap resamples of the
projections, and report how often each recommended transfer survives. A
transfer chosen in 95% of resamples is real; one chosen in 40% of them is not
worth a hit.

**What the resampling noise represents.** The fast analytic path
(`models/points.py`) does not yet carry a per-player variance — every term is
folded into a single expected value before it reaches the optimiser. Rather
than block this on threading a full variance decomposition through the
projection engine, each player-gameweek is perturbed by noise whose scale
grows with the square root of the projection itself. This mirrors how the
dominant scoring terms actually behave — goals, assists and bonus are
Poisson-like counts, whose variance scales with their mean — and it is the
same shape of assumption the rest of the codebase already leans on (see
`models/points.py`'s threshold and Poisson machinery). It is deliberately not
fitted to the backtest: `sigma_scale` is chosen so a mid-table player's sigma
lands near the season-long MAE the backtest measured (`docs/DECISIONS.md`),
as a sanity check on the order of magnitude, not a calibration claim.

This is an approximation, and a labelled one — a real per-player variance
computed inside `models/points.py` (folding the count-like terms' own Poisson
variance together, rather than reusing one blanket rule) would be a strict
improvement without changing anything downstream of `Player.points`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from fpl.optimize.milp import Player, optimise

# Chosen so a ~5-point player's sigma (~1.8) lands near the backtest's measured
# season MAE of 1.842 (docs/DECISIONS.md) — a sanity-checked order of
# magnitude, not a fitted constant.
DEFAULT_SIGMA_SCALE = 0.8
DEFAULT_MIN_SIGMA = 0.4


def resample_points(
    players: Sequence[Player],
    gameweeks: Sequence[int],
    rng: np.random.Generator,
    *,
    sigma_scale: float = DEFAULT_SIGMA_SCALE,
    min_sigma: float = DEFAULT_MIN_SIGMA,
) -> list[Player]:
    """One bootstrap draw: every player-gameweek projection perturbed by noise
    whose scale grows with the square root of the projection (see module
    docstring). Prices, positions and teams are untouched — only the numbers
    the optimiser is uncertain about move.
    """
    resampled: list[Player] = []
    for player in players:
        noisy_points: dict[int, float] = {}
        for gw in gameweeks:
            if gw not in player.points:
                continue
            mean = player.points[gw]
            sigma = max(min_sigma, sigma_scale * math.sqrt(max(mean, 0.0)))
            noisy_points[gw] = mean + rng.normal(0.0, sigma)
        resampled.append(replace(player, points=noisy_points))
    return resampled


def _paired_moves(transfers_out: Sequence[int], transfers_in: Sequence[int],
                   positions: Mapping[int, int]) -> list[tuple[int, int]]:
    """Pair sales with purchases by position — the same rule the brief uses to
    present "sell X, buy Y" rather than two unordered lists.
    """
    outgoing = sorted(transfers_out, key=lambda e: positions.get(e, 0))
    incoming = sorted(transfers_in, key=lambda e: positions.get(e, 0))
    return list(zip(outgoing, incoming))


@dataclass
class BootstrapReport:
    """How often each first-gameweek transfer survived resampling."""

    requested: int
    # Resamples that actually produced a plan. A time-limited solve can fail
    # on an unlucky draw; those are excluded from the frequencies below
    # rather than silently counted as agreement.
    solved: int = 0
    move_frequency: dict[tuple[int, int], float] = field(default_factory=dict)
    hold_frequency: float = 0.0

    def frequency(self, out_element: int, in_element: int) -> float:
        return self.move_frequency.get((out_element, in_element), 0.0)


def bootstrap_transfers(
    players: Sequence[Player],
    gameweeks: Sequence[int],
    optimise_kwargs: Mapping[str, Any],
    *,
    n_resamples: int = 20,
    sigma_scale: float = DEFAULT_SIGMA_SCALE,
    min_sigma: float = DEFAULT_MIN_SIGMA,
    seed: int | None = None,
    time_limit: float | None = 20.0,
) -> BootstrapReport:
    """Re-solve the horizon across ``n_resamples`` perturbed projections and
    report how often each first-gameweek transfer recurs.

    ``optimise_kwargs`` should be the same keyword arguments the real solve
    used (current squad, bank, free transfers, discount, wildcard gameweek,
    ...), so the only thing that changes between resamples is the noise on
    the projections themselves.
    """
    rng = np.random.default_rng(seed)
    positions = {p.element: p.position for p in players}

    counts: dict[tuple[int, int], int] = {}
    solved = 0
    held = 0
    for _ in range(n_resamples):
        noisy = resample_points(
            players, gameweeks, rng, sigma_scale=sigma_scale, min_sigma=min_sigma
        )
        plan = optimise(noisy, gameweeks, time_limit=time_limit, **optimise_kwargs)
        if not plan.solved:
            continue
        solved += 1
        decision = plan.first()
        if not decision.transfers_in and not decision.transfers_out:
            held += 1
            continue
        for move in _paired_moves(decision.transfers_out, decision.transfers_in, positions):
            counts[move] = counts.get(move, 0) + 1

    denominator = max(solved, 1)
    return BootstrapReport(
        requested=n_resamples,
        solved=solved,
        move_frequency={move: count / denominator for move, count in counts.items()},
        hold_frequency=held / denominator,
    )
