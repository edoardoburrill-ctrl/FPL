"""How long will he actually be on the pitch?

Every other part of the projection is conditional on this one. A 9.0 expected-point
striker who starts 60% of the time is worth less than a 6.5 expected-point striker
who never comes off, and no amount of attacking modelling recovers that if the
minutes are wrong.

The model is a three-outcome hurdle: the player is *out*, *on the bench*, or
*starting*; conditional on each, minutes have their own distribution. What FPL
actually pays for is the 60-minute threshold (2 appearance points instead of 1,
and eligibility for a clean sheet), so that threshold is modelled explicitly
rather than inferred from an expected-minutes point estimate.

Rates are shrunk with a Beta-Binomial prior whose mean comes from the player's
position and price tier, estimated empirically from the current player pool. In
August, when everyone has two or three appearances, that prior is doing most of
the work — which is correct, because two appearances tell you very little.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

# FPL availability codes on `elements`.
STATUS_AVAILABLE = "a"
STATUS_DOUBTFUL = "d"
_UNAVAILABLE_STATUSES = {"i", "s", "u", "n"}

# Shrinkage strengths, in units of "equivalent observed matches". Tuned by
# cross-validation in the backtest; these are the defaults it starts from.
START_PRIOR_STRENGTH = 2.0
DURATION_PRIOR_STRENGTH = 2.0

# Fallback league-wide behaviour, used when a player has no history at all.
PRIOR_MINUTES_IF_START = 78.0
PRIOR_P60_IF_START = 0.85
PRIOR_MINUTES_IF_BENCH = 16.0
PRIOR_P60_IF_BENCH = 0.04
PRIOR_P_BENCH_GIVEN_NOT_START = 0.55


@dataclass(frozen=True)
class MinutesForecast:
    """Distribution over a player's involvement in one fixture."""

    p_out: float
    p_bench: float
    p_start: float
    p_60plus: float
    p_1_to_59: float
    expected_minutes: float

    @property
    def p_appears(self) -> float:
        return self.p_60plus + self.p_1_to_59

    @property
    def expected_appearance_points(self) -> float:
        return 2.0 * self.p_60plus + 1.0 * self.p_1_to_59

    def scaled(self, factor: float) -> "MinutesForecast":
        """Scale involvement (e.g. for a rotation risk adjustment)."""
        factor = float(np.clip(factor, 0.0, 1.0))
        p_start = self.p_start * factor
        p_bench = self.p_bench + (self.p_start - p_start)
        return MinutesForecast(
            p_out=self.p_out,
            p_bench=p_bench,
            p_start=p_start,
            p_60plus=self.p_60plus * factor,
            p_1_to_59=self.p_1_to_59,
            expected_minutes=self.expected_minutes * factor,
        )


def availability(element: Mapping) -> float:
    """P(the player is in the matchday squad at all), from FPL's own flags.

    ``chance_of_playing_next_round`` is FPL's editorial judgement on a flagged
    player and is the best single signal available without scraping press
    conferences. ``None`` means unflagged, which is not the same as 100% — an
    unflagged player can still be rotated — but it does mean no known problem.
    """
    chance = element.get("chance_of_playing_next_round")
    if chance is not None:
        return float(chance) / 100.0
    status = element.get("status", STATUS_AVAILABLE)
    if status in _UNAVAILABLE_STATUSES:
        return 0.0
    if status == STATUS_DOUBTFUL:
        return 0.5
    return 1.0


def _weights(n: int, half_life: float) -> np.ndarray:
    """Exponential recency weights over ``n`` matches, most recent last.

    Normalised to sum to ``n`` so that recency reshapes the evidence without
    shrinking it: ten matches remain ten matches of evidence against the prior,
    they are simply not weighted equally. Without this the decayed weights sum
    to far less than ``n`` and the prior swamps genuine evidence.
    """
    if n == 0:
        return np.zeros(0)
    age = np.arange(n - 1, -1, -1, dtype=float)
    raw = np.exp(-np.log(2.0) / half_life * age)
    return raw * (n / raw.sum())


def price_tier_priors(
    elements: Sequence[Mapping], tiers: int = 5
) -> dict[tuple[int, int], float]:
    """Empirical mean start rate by (position, price tier).

    Estimated from the live player pool rather than hardcoded, so it adapts as
    the season progresses and as FPL's price distribution shifts.
    """
    by_position: dict[int, list[tuple[float, float, float]]] = {}
    for el in elements:
        pos = el.get("element_type")
        appearances = float(el.get("starts", 0) or 0)
        # `minutes` is season total; a player with no team matches yet is skipped.
        played = float(el.get("minutes", 0) or 0)
        if pos is None or played <= 0:
            continue
        by_position.setdefault(pos, []).append(
            (float(el.get("now_cost", 0)), appearances, played)
        )

    priors: dict[tuple[int, int], float] = {}
    for pos, rows in by_position.items():
        costs = np.array([r[0] for r in rows])
        starts = np.array([r[1] for r in rows])
        minutes = np.array([r[2] for r in rows])
        # Approximate matches available as the maximum minutes/90 seen in the pool.
        matches = max(1.0, float(np.max(minutes) / 90.0))
        rate = np.clip(starts / matches, 0.0, 1.0)
        edges = np.quantile(costs, np.linspace(0, 1, tiers + 1))
        bucket = np.clip(np.digitize(costs, edges[1:-1]), 0, tiers - 1)
        for t in range(tiers):
            mask = bucket == t
            if mask.any():
                priors[(pos, t)] = float(np.mean(rate[mask]))
    return priors


def price_tier(cost: float, elements: Sequence[Mapping], position: int, tiers: int = 5) -> int:
    costs = np.array(
        [float(e["now_cost"]) for e in elements if e.get("element_type") == position]
    )
    if costs.size == 0:
        return tiers // 2
    edges = np.quantile(costs, np.linspace(0, 1, tiers + 1))
    return int(np.clip(np.digitize([cost], edges[1:-1])[0], 0, tiers - 1))


def _shrunk_mean(
    values: np.ndarray, weights: np.ndarray, prior_mean: float, prior_strength: float
) -> float:
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return prior_mean
    observed = float(np.sum(weights * values) / total_weight)
    return (total_weight * observed + prior_strength * prior_mean) / (
        total_weight + prior_strength
    )


def forecast_minutes(
    history: Sequence[Mapping],
    element: Mapping,
    *,
    prior_start_rate: float | None = None,
    half_life: float = 4.0,
    lookback: int = 12,
    rotation_factor: float = 1.0,
) -> MinutesForecast:
    """Forecast involvement for the next fixture.

    ``history`` is the player's per-fixture rows (oldest first) as returned by
    ``element-summary``; each needs ``minutes`` and, where available, ``starts``.
    ``rotation_factor`` in [0, 1] downweights the start probability for known
    congestion or rotation risk.
    """
    rows = list(history)[-lookback:]
    n = len(rows)
    weights = _weights(n, half_life)

    minutes = np.array([float(r.get("minutes", 0) or 0) for r in rows])
    if n and any("starts" in r for r in rows):
        started = np.array([float(r.get("starts", 0) or 0) for r in rows])
    else:
        # Older data has no `starts` flag; 60+ minutes is the standard proxy.
        started = (minutes >= 60).astype(float)

    position = int(element.get("element_type", 3))
    if prior_start_rate is None:
        prior_start_rate = 0.5

    p_start_raw = _shrunk_mean(started, weights, prior_start_rate, START_PRIOR_STRENGTH)

    start_mask = started > 0.5
    bench_mask = (~start_mask) & (minutes > 0)
    not_start_mask = ~start_mask

    p_bench_given_not_start = _shrunk_mean(
        (minutes[not_start_mask] > 0).astype(float),
        weights[not_start_mask],
        PRIOR_P_BENCH_GIVEN_NOT_START,
        DURATION_PRIOR_STRENGTH,
    )

    minutes_if_start = _shrunk_mean(
        minutes[start_mask], weights[start_mask],
        PRIOR_MINUTES_IF_START, DURATION_PRIOR_STRENGTH,
    )
    p60_if_start = _shrunk_mean(
        (minutes[start_mask] >= 60).astype(float), weights[start_mask],
        PRIOR_P60_IF_START, DURATION_PRIOR_STRENGTH,
    )
    minutes_if_bench = _shrunk_mean(
        minutes[bench_mask], weights[bench_mask],
        PRIOR_MINUTES_IF_BENCH, DURATION_PRIOR_STRENGTH,
    )
    p60_if_bench = _shrunk_mean(
        (minutes[bench_mask] >= 60).astype(float), weights[bench_mask],
        PRIOR_P60_IF_BENCH, DURATION_PRIOR_STRENGTH,
    )

    available = availability(element)
    p_start = available * p_start_raw * float(np.clip(rotation_factor, 0.0, 1.0))
    p_bench = available * (1.0 - p_start_raw * float(np.clip(rotation_factor, 0.0, 1.0)))
    p_bench *= p_bench_given_not_start
    p_out = max(0.0, 1.0 - p_start - p_bench)

    p_60plus = p_start * p60_if_start + p_bench * p60_if_bench
    p_appears = p_start + p_bench
    p_1_to_59 = max(0.0, p_appears - p_60plus)
    expected_minutes = p_start * minutes_if_start + p_bench * minutes_if_bench

    return MinutesForecast(
        p_out=p_out,
        p_bench=p_bench,
        p_start=p_start,
        p_60plus=p_60plus,
        p_1_to_59=p_1_to_59,
        expected_minutes=expected_minutes,
    )


def synthetic_history(element: Mapping, team_matches: int) -> list[dict]:
    """Reconstruct a plausible per-match history from season aggregates.

    ``element-summary`` gives the real per-fixture log, but pulling it for 700
    players is a lot of requests for players we will never consider. For the long
    tail we approximate: ``starts`` matches at the player's average starting
    minutes, any remaining appearances as bench cameos, and the rest as absences.

    The approximation loses recency (it cannot tell a player who started the last
    three from one who started the first three), which is precisely why the deep
    sync exists for players actually under consideration.
    """
    team_matches = max(0, int(team_matches))
    starts = int(element.get("starts", 0) or 0)
    total_minutes = float(element.get("minutes", 0) or 0)
    starts = min(starts, team_matches)

    if team_matches == 0:
        return []

    # Split total minutes between starts and cameos, assuming a typical cameo.
    cameo_minutes = 18.0
    non_start_matches = team_matches - starts
    minutes_from_starts = total_minutes
    cameos = 0
    if starts > 0:
        implied_start_minutes = total_minutes / starts
        if implied_start_minutes > 90.0:
            implied_start_minutes = 90.0
        minutes_from_starts = implied_start_minutes * starts
        leftover = max(0.0, total_minutes - minutes_from_starts)
        cameos = min(non_start_matches, int(round(leftover / cameo_minutes)))
    elif total_minutes > 0:
        minutes_from_starts = 0.0
        cameos = min(non_start_matches, max(1, int(round(total_minutes / cameo_minutes))))

    per_start = minutes_from_starts / starts if starts else 0.0
    per_cameo = (total_minutes - minutes_from_starts) / cameos if cameos else 0.0

    pool = (
        [{"minutes": round(per_start), "starts": 1}] * starts
        + [{"minutes": max(1, round(per_cameo)), "starts": 0}] * cameos
        + [{"minutes": 0, "starts": 0}] * (team_matches - starts - cameos)
    )
    # Aggregates carry no ordering information, so spread the match types evenly
    # rather than in blocks. Leaving them blocked would let the recency weighting
    # read "four starts then two absences" out of data that says no such thing.
    return _interleave(pool)


def _interleave(rows: list[dict]) -> list[dict]:
    """Spread rows of each type evenly through the sequence."""
    if len(rows) <= 1:
        return list(rows)
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["starts"], row["minutes"]), []).append(row)
    ordered = sorted(groups.values(), key=lambda g: -len(g))
    total = len(rows)
    slots: list[dict | None] = [None] * total
    position = 0
    for group in ordered:
        step = total / len(group)
        for i, row in enumerate(group):
            index = int(round(i * step + position)) % total
            while slots[index] is not None:
                index = (index + 1) % total
            slots[index] = row
        position += 0.5
    return [row for row in slots if row is not None]
