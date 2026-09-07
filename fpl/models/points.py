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
    # Variance of each term, on the same terms as `breakdown` — see
    # `project_player` for how each is derived. Used for bootstrap-resampling
    # the optimiser's inputs (`optimize/robustness.py`), not for the mean
    # projection itself.
    variance: dict[str, float] = field(default_factory=dict)

    @property
    def expected_points(self) -> float:
        return float(sum(self.breakdown.values()))

    @property
    def points_variance(self) -> float:
        """Variance of total points, treating each term as independent.

        The terms are not really independent — a player who scores is also
        more likely to pick up bonus, for instance — so this understates the
        true variance somewhat. It is still far better than treating the
        projection as certain, which is the only alternative available
        without the full joint Monte Carlo (`models/simulate.py`).
        """
        return float(sum(self.variance.values()))

    @property
    def points_sd(self) -> float:
        return float(np.sqrt(max(self.points_variance, 0.0)))

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


def threshold_probability(rate_per_90: float, minutes: float, threshold: int) -> float:
    """P(count >= threshold) for a Poisson rate over a given number of minutes."""
    if minutes <= 0 or rate_per_90 <= 0:
        return 0.0
    lam = rate_per_90 * minutes / 90.0
    return float(1.0 - poisson.cdf(threshold - 1, lam))


def _appearance_variance(minutes: MinutesForecast) -> float:
    """Variance of appearance points, a three-point discrete variable.

    Appearance points take the value 0 (didn't play enough to score any),
    1 (1-59 minutes) or 2 (60+), with probabilities the minutes model already
    gives us directly — no extra assumption needed here.
    """
    mean = minutes.expected_appearance_points
    second_moment = 1.0 * minutes.p_1_to_59 + 4.0 * minutes.p_60plus
    return max(0.0, second_moment - mean**2)


def _poisson_scaled_variance(expected_count: float, points_per_event: float) -> float:
    """Variance of ``points_per_event * X`` for ``X ~ Poisson(expected_count)``.

    Used for goals, assists and cards: a Poisson count is the same
    independence assumption the minutes-weighted rate model already makes
    for the mean, so this is not a new assumption, just its second moment.
    """
    if expected_count <= 0:
        return 0.0
    return (points_per_event**2) * expected_count


def _bernoulli_variance(probability: float, points_if_true: float) -> float:
    """Variance of a points_if_true-or-nothing outcome with the given probability."""
    p = float(np.clip(probability, 0.0, 1.0))
    return (points_if_true**2) * p * (1.0 - p)


def conceded_penalty_variance(
    conceded_pmf: np.ndarray, minutes_fraction: float, per_penalty: int = 2
) -> float:
    """Variance of ``floor(conceded / per_penalty)`` over the on-pitch window.

    Computed from the same on-pitch concession distribution the expectation
    uses (`_conceded_while_on_pitch`), not assumed — the floor makes this a
    genuinely different distribution from a scaled Poisson.
    """
    pmf = _conceded_while_on_pitch(conceded_pmf, minutes_fraction)
    counts = np.arange(len(pmf))
    units = counts // per_penalty
    mean = float(np.sum(pmf * units))
    second_moment = float(np.sum(pmf * units**2))
    return max(0.0, second_moment - mean**2)


def saves_points_variance(expected_saves: float, per_point: int = 3) -> float:
    """Variance of ``floor(S / per_point)`` for ``S ~ Poisson(expected_saves)``."""
    if expected_saves <= 0:
        return 0.0
    max_s = int(expected_saves + 8 * np.sqrt(expected_saves) + 10)
    s = np.arange(max_s + 1)
    pmf = poisson.pmf(s, expected_saves)
    units = s // per_point
    mean = float(np.sum(pmf * units))
    second_moment = float(np.sum(pmf * units**2))
    return max(0.0, second_moment - mean**2)


def bonus_variance(expected_bps: float) -> float:
    """Variance of bonus points under the same normal-threshold approximation
    ``expected_bonus`` uses.

    Bonus is a count of how many of the three (ordered) BPS thresholds a
    player's simulated BPS clears, which makes it a proper discrete random
    variable on {0, 1, 2, 3} once the thresholds are nested — clearing the
    35-BPS bar implies clearing the 29 and 24 bars too. The threshold
    probabilities are exactly the survival-function terms `expected_bonus`
    already sums for the mean; this reuses them for the second moment.
    """
    if expected_bps <= 0:
        return 0.0
    spread = np.sqrt(BONUS_PLAYER_SD**2 + BONUS_THRESHOLD_SD**2)
    survival = [
        norm.sf(threshold, loc=expected_bps, scale=spread)
        for threshold in BONUS_THRESHOLDS
    ]
    # P(count >= k) = survival[k-1] since thresholds are ascending and nested.
    p_at_least = [1.0, *survival, 0.0]  # index k -> P(count >= k), k=0..4 padded
    probs = [p_at_least[k] - p_at_least[k + 1] for k in range(4)]  # P(count == k)
    mean = sum(k * p for k, p in enumerate(probs))
    second_moment = sum((k**2) * p for k, p in enumerate(probs))
    return max(0.0, second_moment - mean**2)


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
    variance["appearance"] = _appearance_variance(minutes)

    expected_goals = rates.goals_share * team_goals * minutes_fraction
    expected_assists = rates.assists_share * team_goals * minutes_fraction
    breakdown["goals"] = expected_goals * rules.goal.get(position, 0)
    breakdown["assists"] = expected_assists * rules.assist
    variance["goals"] = _poisson_scaled_variance(expected_goals, rules.goal.get(position, 0))
    variance["assists"] = _poisson_scaled_variance(expected_assists, rules.assist)

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
        cs_probability = minutes.p_60plus * cs_prob
        breakdown["clean_sheet"] = cs_points * cs_probability
        variance["clean_sheet"] = _bernoulli_variance(cs_probability, cs_points)

    if position in rules.conceded_penalty_positions:
        breakdown["goals_conceded"] = -expected_conceded_penalty(
            conceded_pmf, minutes_fraction, rules.conceded_per_penalty
        )
        variance["goals_conceded"] = conceded_penalty_variance(
            conceded_pmf, minutes_fraction, rules.conceded_per_penalty
        )

    if position == GK and rates.saves_per_90 > 0:
        # Saves scale with how much shooting the opponent does, proxied by their
        # goal expectation relative to a league-average opponent.
        intensity = opponent_goals / max(league_average_conceded, 1e-6)
        expected_saves = rates.saves_per_90 * minutes_fraction * intensity
        breakdown["saves"] = expected_saves_points(expected_saves, rules.saves_per_point)
        variance["saves"] = saves_points_variance(expected_saves, rules.saves_per_point)

    threshold = rules.defcon_threshold.get(position)
    if threshold is not None and rates.defcon_per_90 > 0:
        # Evaluate the threshold separately for the starting and bench cases: a
        # 20-minute cameo essentially never reaches 10 or 12 actions, and
        # averaging the minutes first would wrongly suggest it sometimes does.
        minutes_if_start = (
            minutes.expected_minutes / max(minutes.p_start + minutes.p_bench, 1e-6)
        )
        p_hit = threshold_probability(rates.defcon_per_90, minutes_if_start, threshold)
        defcon_probability = minutes.p_start * p_hit
        breakdown["defensive_contribution"] = rules.defcon_points * defcon_probability
        variance["defensive_contribution"] = _bernoulli_variance(
            defcon_probability, rules.defcon_points
        )

    if rates.yellow_per_90:
        breakdown["cards"] = rules.yellow_card * rates.yellow_per_90 * minutes_fraction
        variance["cards"] = _poisson_scaled_variance(
            rates.yellow_per_90 * minutes_fraction, rules.yellow_card
        )
        if rates.red_per_90:
            breakdown["cards"] += rules.red_card * rates.red_per_90 * minutes_fraction
            variance["cards"] += _poisson_scaled_variance(
                rates.red_per_90 * minutes_fraction, rules.red_card
            )

    if rates.bps_per_90:
        # Expected BPS scales with minutes and with how good the fixture is,
        # since most BPS accrues through goals, assists and clean sheets.
        fixture_multiplier = 0.5 + 0.5 * (team_goals / max(league_average_conceded, 1e-6))
        expected_bps = rates.bps_per_90 * minutes_fraction * fixture_multiplier
        bonus_mean = expected_bonus(expected_bps)
        breakdown["bonus"] = bonus_mean * minutes.p_appears
        # Bonus only accrues if the player appears at all; treating "appears"
        # as a Bernoulli independent of the bonus count conditional on
        # appearing gives Var(A*B) = p*Var(B) + E[B]^2*p*(1-p).
        p_appears = minutes.p_appears
        variance["bonus"] = (
            p_appears * bonus_variance(expected_bps)
            + bonus_mean**2 * p_appears * (1.0 - p_appears)
        )

    return PlayerProjection(
        element=element_id,
        fixture=fixture,
        position=position,
        minutes=minutes,
        breakdown=breakdown,
        variance=variance,
    )
