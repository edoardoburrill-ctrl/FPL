"""Assembling a player's expected points for one fixture.

This is where the minutes model, the fixture goal expectations and the shrunk
player rates come together. Every term is computed as an expectation over the
relevant distribution rather than plugged in as a point estimate, because FPL
scoring is full of thresholds — 60 minutes, every 2 goals conceded, every 3
saves, 10 or 12 defensive contributions — and a threshold applied to an average
is not the average of a threshold.

The clean-sheet and goals-conceded terms are computed against the *minutes the
player is actually on the pitch*, since FPL only counts goals conceded while he
is playing.

Known limitation in this version: bonus points are estimated from an expected-BPS
normal approximation rather than by ranking every player in the match. Bonus is
inherently a within-match ranking, so the honest treatment needs the Monte Carlo
engine; this approximation is monotone and roughly calibrated but it is the
weakest term in the model, and it is replaced in Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy.stats import binom, norm, poisson

from fpl.models.minutes import MinutesForecast
from fpl.models.scoring import DEF, FWD, GK, MID, ScoringRules

# Bonus calibration: approximate BPS thresholds for 3rd, 2nd and 1st in a match,
# and the spread of those thresholds across matches. Replaced by simulation in
# Phase 3; exposed here so the backtest can tune them.
BONUS_THRESHOLDS = (24.0, 29.0, 35.0)
BONUS_THRESHOLD_SD = 6.5
BONUS_PLAYER_SD = 9.0


@dataclass
class PlayerProjection:
    """Expected points for one player in one fixture, with a full breakdown."""

    element: int
    fixture: int | None
    position: int
    minutes: MinutesForecast
    breakdown: dict[str, float] = field(default_factory=dict)
    # Variance of each breakdown term, computed from the same distributions used
    # for its mean (Binomial appearance/clean-sheet/defcon, Poisson goals/assists/
    # saves/conceded, the bonus normal approximation) — see `_bernoulli_variance`
    # and friends below. Terms are summed as if independent, the same
    # simplification the means already make of each other.
    variance: dict[str, float] = field(default_factory=dict)

    @property
    def expected_points(self) -> float:
        return float(sum(self.breakdown.values()))

    @property
    def point_variance(self) -> float:
        return float(sum(self.variance.values()))

    @property
    def point_std(self) -> float:
        return float(np.sqrt(max(self.point_variance, 0.0)))

    def explain(self) -> str:
        parts = [
            f"  {name:<22s} {value:+6.3f}"
            for name, value in sorted(
                self.breakdown.items(), key=lambda kv: -abs(kv[1])
            )
            if abs(value) > 1e-9
        ]
        header = (
            f"element {self.element} (pos {self.position}) "
            f"E[min]={self.minutes.expected_minutes:.0f} "
            f"P(start)={self.minutes.p_start:.0%} P(60+)={self.minutes.p_60plus:.0%}"
        )
        return "\n".join([header, *parts, f"  {'TOTAL':<22s} {self.expected_points:+6.3f}"])


def _conceded_while_on_pitch(
    conceded_pmf: np.ndarray, minutes_fraction: float
) -> np.ndarray:
    """Distribution of goals conceded during the player's time on the pitch.

    Goals are assumed uniformly distributed through the match, so each of the
    team's ``n`` concessions independently falls inside the player's window with
    probability ``minutes_fraction``. That makes the on-pitch count a binomial
    thinning of the match total.
    """
    fraction = float(np.clip(minutes_fraction, 0.0, 1.0))
    max_goals = len(conceded_pmf) - 1
    result = np.zeros(max_goals + 1)
    for n, p_n in enumerate(conceded_pmf):
        if p_n <= 0:
            continue
        if n == 0:
            result[0] += p_n
            continue
        k = np.arange(n + 1)
        result[: n + 1] += p_n * binom.pmf(k, n, fraction)
    return result


def clean_sheet_probability_on_pitch(
    conceded_pmf: np.ndarray, minutes_fraction: float
) -> float:
    """P(no goal conceded while the player is on the pitch)."""
    return float(_conceded_while_on_pitch(conceded_pmf, minutes_fraction)[0])


def expected_conceded_penalty(
    conceded_pmf: np.ndarray, minutes_fraction: float, per_penalty: int = 2
) -> float:
    """E[floor(conceded / 2)] over the on-pitch conceded distribution."""
    pmf = _conceded_while_on_pitch(conceded_pmf, minutes_fraction)
    counts = np.arange(len(pmf))
    return float(np.sum(pmf * (counts // per_penalty)))


def expected_saves_points(expected_saves: float, per_point: int = 3) -> float:
    """E[floor(S / 3)] for S ~ Poisson(expected_saves)."""
    if expected_saves <= 0:
        return 0.0
    max_s = int(expected_saves + 8 * np.sqrt(expected_saves) + 10)
    s = np.arange(max_s + 1)
    return float(np.sum(poisson.pmf(s, expected_saves) * (s // per_point)))


def _variance_conceded_penalty(
    conceded_pmf: np.ndarray, minutes_fraction: float, per_penalty: int = 2
) -> float:
    """Var[floor(conceded / 2)] over the on-pitch conceded distribution."""
    pmf = _conceded_while_on_pitch(conceded_pmf, minutes_fraction)
    values = np.arange(len(pmf)) // per_penalty
    mean = float(np.sum(pmf * values))
    second_moment = float(np.sum(pmf * values**2))
    return max(0.0, second_moment - mean**2)


def _variance_saves_points(expected_saves: float, per_point: int = 3) -> float:
    """Var[floor(S / 3)] for S ~ Poisson(expected_saves)."""
    if expected_saves <= 0:
        return 0.0
    max_s = int(expected_saves + 8 * np.sqrt(expected_saves) + 10)
    s = np.arange(max_s + 1)
    pmf = poisson.pmf(s, expected_saves)
    values = s // per_point
    mean = float(np.sum(pmf * values))
    second_moment = float(np.sum(pmf * values**2))
    return max(0.0, second_moment - mean**2)


def _bernoulli_variance(probability: float, value: float) -> float:
    """Var[value] for an event worth `value` points with the given probability."""
    p = float(np.clip(probability, 0.0, 1.0))
    return (value**2) * p * (1.0 - p)


def _poisson_linear_variance(rate: float, value: float) -> float:
    """Var[value * N] for N ~ Poisson(rate)."""
    return (value**2) * max(rate, 0.0)


def threshold_probability(rate_per_90: float, minutes: float, threshold: int) -> float:
    """P(count >= threshold) for a Poisson rate over a given number of minutes."""
    if minutes <= 0 or rate_per_90 <= 0:
        return 0.0
    lam = rate_per_90 * minutes / 90.0
    return float(1.0 - poisson.cdf(threshold - 1, lam))


def expected_bonus(expected_bps: float) -> float:
    """Approximate expected bonus from expected BPS.

    Bonus is awarded by rank within a match, so this models the probability of
    clearing each of the three BPS thresholds, treating both the player's BPS and
    the match's threshold as normal. Crude, monotone, and clearly labelled as the
    term most in need of the simulation-based replacement.
    """
    if expected_bps <= 0:
        return 0.0
    spread = np.sqrt(BONUS_PLAYER_SD**2 + BONUS_THRESHOLD_SD**2)
    return float(
        sum(norm.sf(threshold, loc=expected_bps, scale=spread) for threshold in BONUS_THRESHOLDS)
    )


def _bonus_variance(expected_bps: float, p_appears: float) -> float:
    """Var[bonus points], consistent with `expected_bonus`'s normal approximation.

    Bonus is a step function of a single normal draw (BPS vs. the three
    thresholds), so it takes values 0-3 with probabilities read off that
    normal's CDF at each threshold — giving an exact variance for that
    approximation, not another layer of it. `expected_bonus` gates the mean by
    `p_appears`; here that gate is treated as an independent Bernoulli on top
    of the bonus draw (`Var[A*B] = p*Var[B] + p(1-p)*E[B]^2` for independent
    `A ~ Bernoulli(p)`, `B`), which is not monotone in `p` — losing the "does
    he even appear" variance as `p -> 1` competes with the gate's own variance
    peaking at `p = 0.5`.
    """
    if expected_bps <= 0:
        return 0.0
    spread = np.sqrt(BONUS_PLAYER_SD**2 + BONUS_THRESHOLD_SD**2)
    edges = [-np.inf, *BONUS_THRESHOLDS, np.inf]
    cdfs = [norm.cdf(edge, loc=expected_bps, scale=spread) for edge in edges]
    probs = np.array([cdfs[i + 1] - cdfs[i] for i in range(4)])
    values = np.array([0.0, 1.0, 2.0, 3.0])
    mean_given_appear = float(np.sum(probs * values))
    second_given_appear = float(np.sum(probs * values**2))
    var_given_appear = max(0.0, second_given_appear - mean_given_appear**2)

    p = float(np.clip(p_appears, 0.0, 1.0))
    return p * var_given_appear + p * (1.0 - p) * mean_given_appear**2


@dataclass(frozen=True)
class PlayerRates:
    """Per-90 rates for one player, already shrunk."""

    goals_share: float  # share of team goals scored, per 90 played
    assists_share: float  # share of team goals assisted, per 90 played
    saves_per_90: float = 0.0
    defcon_per_90: float = 0.0
    bps_per_90: float = 0.0
    yellow_per_90: float = 0.0
    red_per_90: float = 0.0


def project_player(
    element_id: int,
    position: int,
    minutes: MinutesForecast,
    rates: PlayerRates,
    team_goals: float,
    opponent_goals: float,
    conceded_pmf: np.ndarray,
    rules: ScoringRules,
    *,
    fixture: int | None = None,
    league_average_conceded: float = 1.4,
) -> PlayerProjection:
    """Expected points for one player in one fixture.

    ``team_goals`` and ``opponent_goals`` are this fixture's goal expectations;
    ``conceded_pmf`` is the distribution of goals the player's team concedes,
    taken from the Dixon-Coles score matrix.
    """
    breakdown: dict[str, float] = {}
    variance: dict[str, float] = {}
    minutes_fraction = float(np.clip(minutes.expected_minutes / 90.0, 0.0, 1.0))

    breakdown["appearance"] = minutes.expected_appearance_points
    # Appearance points are 0, 1 or 2 with known probabilities straight from the
    # minutes forecast — no approximation needed for this one.
    second_moment = minutes.p_1_to_59 * 1.0 + minutes.p_60plus * 4.0
    variance["appearance"] = max(0.0, second_moment - minutes.expected_appearance_points**2)

    expected_goals = rates.goals_share * team_goals * minutes_fraction
    expected_assists = rates.assists_share * team_goals * minutes_fraction
    goal_value = rules.goal.get(position, 0)
    breakdown["goals"] = expected_goals * goal_value
    variance["goals"] = _poisson_linear_variance(expected_goals, goal_value)
    breakdown["assists"] = expected_assists * rules.assist
    variance["assists"] = _poisson_linear_variance(expected_assists, rules.assist)

    cs_points = rules.clean_sheet.get(position, 0)
    if cs_points:
        # The clean sheet needs both 60+ minutes and no concession while on.
        # Condition the concession window on a starter's typical minutes rather
        # than the unconditional mean, which is dragged down by bench outcomes.
        start_fraction = float(np.clip(minutes.expected_minutes / 90.0, 0.0, 1.0))
        if minutes.p_60plus > 1e-9:
            start_fraction = float(
                np.clip(minutes.expected_minutes / 90.0 / max(minutes.p_60plus, 1e-6), 0.0, 1.0)
            )
        cs_prob = clean_sheet_probability_on_pitch(conceded_pmf, start_fraction)
        p_cs = minutes.p_60plus * cs_prob
        breakdown["clean_sheet"] = cs_points * p_cs
        variance["clean_sheet"] = _bernoulli_variance(p_cs, cs_points)

    if position in rules.conceded_penalty_positions:
        breakdown["goals_conceded"] = -expected_conceded_penalty(
            conceded_pmf, minutes_fraction, rules.conceded_per_penalty
        )
        variance["goals_conceded"] = _variance_conceded_penalty(
            conceded_pmf, minutes_fraction, rules.conceded_per_penalty
        )

    if position == GK and rates.saves_per_90 > 0:
        # Saves scale with how much shooting the opponent does, proxied by their
        # goal expectation relative to a league-average opponent.
        intensity = opponent_goals / max(league_average_conceded, 1e-6)
        expected_saves = rates.saves_per_90 * minutes_fraction * intensity
        breakdown["saves"] = expected_saves_points(expected_saves, rules.saves_per_point)
        variance["saves"] = _variance_saves_points(expected_saves, rules.saves_per_point)

    threshold = rules.defcon_threshold.get(position)
    if threshold is not None and rates.defcon_per_90 > 0:
        # Evaluate the threshold separately for the starting and bench cases: a
        # 20-minute cameo essentially never reaches 10 or 12 actions, and
        # averaging the minutes first would wrongly suggest it sometimes does.
        minutes_if_start = (
            minutes.expected_minutes / max(minutes.p_start + minutes.p_bench, 1e-6)
        )
        p_hit = threshold_probability(rates.defcon_per_90, minutes_if_start, threshold)
        p_defcon = minutes.p_start * p_hit
        breakdown["defensive_contribution"] = rules.defcon_points * p_defcon
        variance["defensive_contribution"] = _bernoulli_variance(
            p_defcon, rules.defcon_points
        )

    if rates.yellow_per_90:
        yellow_rate = rates.yellow_per_90 * minutes_fraction
        breakdown["cards"] = rules.yellow_card * yellow_rate
        variance["cards"] = _poisson_linear_variance(yellow_rate, rules.yellow_card)
        if rates.red_per_90:
            red_rate = rates.red_per_90 * minutes_fraction
            breakdown["cards"] += rules.red_card * red_rate
            variance["cards"] += _poisson_linear_variance(red_rate, rules.red_card)

    if rates.bps_per_90:
        # Expected BPS scales with minutes and with how good the fixture is,
        # since most BPS accrues through goals, assists and clean sheets.
        fixture_multiplier = 0.5 + 0.5 * (team_goals / max(league_average_conceded, 1e-6))
        expected_bps = rates.bps_per_90 * minutes_fraction * fixture_multiplier
        breakdown["bonus"] = expected_bonus(expected_bps) * minutes.p_appears
        variance["bonus"] = _bonus_variance(expected_bps, minutes.p_appears)

    return PlayerProjection(
        element=element_id,
        fixture=fixture,
        position=position,
        minutes=minutes,
        breakdown=breakdown,
        variance=variance,
    )
