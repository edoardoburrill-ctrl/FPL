"""Per-90 attacking rates, shrunk toward what a player of this type usually does.

Two ideas do the work here.

**Empirical Bayes.** A striker with 3 goals from 2 starts has not become a 1.5
goals-per-90 player. Modelling counts as Poisson with a Gamma prior fitted to the
actual distribution of rates in the player pool gives a posterior mean that
starts near the prior and moves toward the observation exactly as fast as the
evidence justifies. The prior's strength is *estimated from the data* by method
of moments, not picked by hand.

**Rates as shares, not absolutes.** A player's projection should respond to the
fixture. Instead of using an absolute goals-per-90, we express it as a share of
his team's expected goals, so multiplying by the fixture's team goal expectation
scales him up against weak defences and down against strong ones automatically —
no hand-tuned difficulty multiplier anywhere in the pipeline.

Expected goals and assists are used in preference to actual goals and assists
because they stabilise several times faster; over a season the difference is
small, but in September it is the difference between a signal and a coin-flip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

MIN_NINETIES_FOR_PRIOR = 0.5


@dataclass(frozen=True)
class GammaPrior:
    """Gamma(alpha, beta) prior over a Poisson rate, in events per 90 minutes."""

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / self.beta

    @property
    def strength(self) -> float:
        """Prior strength in 90-minute units — how much evidence it is worth."""
        return self.beta

    def posterior_mean(self, events: float, nineties: float) -> float:
        return (events + self.alpha) / (nineties + self.beta)


def fit_rate_prior(
    observations: Sequence[tuple[float, float]],
    floor_variance: float = 1e-4,
) -> GammaPrior:
    """Fit a Gamma prior by method of moments.

    ``observations`` are ``(events, nineties)`` pairs. Under Poisson sampling the
    observed rate has variance ``v + m / n`` where ``v`` is the genuine spread of
    talent and ``m / n`` is sampling noise; subtracting the sampling term is what
    stops us mistaking small-sample noise for real variation between players and
    consequently under-shrinking.
    """
    rows = [(e, n) for e, n in observations if n >= MIN_NINETIES_FOR_PRIOR]
    if not rows:
        return GammaPrior(alpha=1.0, beta=10.0)

    events = np.array([r[0] for r in rows], dtype=float)
    nineties = np.array([r[1] for r in rows], dtype=float)

    mean = float(events.sum() / nineties.sum())
    if mean <= 0:
        return GammaPrior(alpha=1e-3, beta=10.0)

    rates = events / nineties
    weights = nineties / nineties.sum()
    total_var = float(np.sum(weights * (rates - mean) ** 2))
    sampling_var = float(np.sum(weights * mean / nineties))
    between_var = max(total_var - sampling_var, floor_variance * mean)

    beta = mean / between_var
    alpha = mean * beta
    return GammaPrior(alpha=float(alpha), beta=float(beta))


def _price_tier(cost: float, costs: np.ndarray, tiers: int) -> int:
    if costs.size == 0:
        return 0
    edges = np.quantile(costs, np.linspace(0, 1, tiers + 1))
    return int(np.clip(np.digitize([cost], edges[1:-1])[0], 0, tiers - 1))


@dataclass
class RatePriors:
    """Fitted priors for one statistic, keyed by (position, price tier)."""

    stat: str
    by_group: dict[tuple[int, int], GammaPrior]
    by_position: dict[int, GammaPrior]
    overall: GammaPrior
    tiers: int
    costs_by_position: dict[int, np.ndarray]

    def group_for(self, element: Mapping) -> GammaPrior:
        position = int(element.get("element_type", 3))
        costs = self.costs_by_position.get(position, np.array([]))
        tier = _price_tier(float(element.get("now_cost", 0)), costs, self.tiers)
        return (
            self.by_group.get((position, tier))
            or self.by_position.get(position)
            or self.overall
        )


def _stat_value(element: Mapping, stat: str) -> float:
    raw = element.get(stat, 0.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def fit_priors(
    elements: Sequence[Mapping], stat: str, tiers: int = 4
) -> RatePriors:
    """Fit (position, price tier) priors for ``stat`` across the player pool."""
    rows: list[tuple[int, float, float, float]] = []
    for el in elements:
        nineties = float(el.get("minutes", 0) or 0) / 90.0
        if nineties <= 0:
            continue
        rows.append(
            (
                int(el.get("element_type", 3)),
                float(el.get("now_cost", 0)),
                _stat_value(el, stat),
                nineties,
            )
        )

    overall = fit_rate_prior([(r[2], r[3]) for r in rows])

    costs_by_position: dict[int, np.ndarray] = {}
    by_position: dict[int, GammaPrior] = {}
    for position in sorted({r[0] for r in rows}):
        subset = [r for r in rows if r[0] == position]
        costs_by_position[position] = np.array([r[1] for r in subset])
        by_position[position] = fit_rate_prior([(r[2], r[3]) for r in subset])

    by_group: dict[tuple[int, int], GammaPrior] = {}
    for position, costs in costs_by_position.items():
        subset = [r for r in rows if r[0] == position]
        for tier in range(tiers):
            bucket = [r for r in subset if _price_tier(r[1], costs, tiers) == tier]
            # Only trust a tier-level prior when the tier has real evidence in it.
            if sum(r[3] for r in bucket) >= 20.0:
                by_group[(position, tier)] = fit_rate_prior(
                    [(r[2], r[3]) for r in bucket]
                )

    return RatePriors(
        stat=stat,
        by_group=by_group,
        by_position=by_position,
        overall=overall,
        tiers=tiers,
        costs_by_position=costs_by_position,
    )


def shrunk_rate(
    element: Mapping,
    priors: RatePriors,
    *,
    prior_season_events: float = 0.0,
    prior_season_nineties: float = 0.0,
    prior_season_weight: float = 0.5,
) -> float:
    """Posterior mean rate per 90 for one player.

    Prior-season evidence is folded in at a discount (``prior_season_weight``):
    it is real information, but squad roles, teammates and managers change over a
    summer, so it is worth less than this season's minutes.
    """
    prior = priors.group_for(element)
    events = _stat_value(element, priors.stat)
    nineties = float(element.get("minutes", 0) or 0) / 90.0

    events += prior_season_weight * prior_season_events
    nineties += prior_season_weight * prior_season_nineties

    return prior.posterior_mean(events, nineties)


@dataclass(frozen=True)
class TeamContext:
    """Team-level output, used to convert player rates into shares."""

    expected_goals_per_match: float
    matches: int


def team_contexts(
    elements: Sequence[Mapping], teams: Sequence[Mapping], matches_played: Mapping[int, int]
) -> dict[int, TeamContext]:
    """Team expected goals per match, from the sum of its players' xG."""
    totals: dict[int, float] = {}
    for el in elements:
        team = int(el.get("team", 0))
        totals[team] = totals.get(team, 0.0) + _stat_value(el, "expected_goals")

    league_matches = max(1, max(matches_played.values(), default=1))
    league_total = sum(totals.values())
    league_mean = league_total / max(1, len(totals)) / league_matches

    contexts: dict[int, TeamContext] = {}
    for team in {int(t["id"]) for t in teams}:
        played = int(matches_played.get(team, 0))
        if played <= 0:
            contexts[team] = TeamContext(league_mean or 1.4, 0)
            continue
        raw = totals.get(team, 0.0) / played
        # Shrink team output toward the league mean early in the season, for the
        # same reason player rates are shrunk: three matches is not evidence.
        weight = played / (played + 4.0)
        contexts[team] = TeamContext(
            weight * raw + (1 - weight) * (league_mean or 1.4), played
        )
    return contexts


def output_share(rate_per_90: float, context: TeamContext) -> float:
    """Convert an absolute per-90 rate into a share of the team's output.

    The result multiplied by a fixture's team goal expectation gives the player's
    expected involvement in that specific fixture.
    """
    denominator = max(context.expected_goals_per_match, 1e-6)
    return rate_per_90 / denominator
