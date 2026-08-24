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

    @property
    def expected_points(self) -> float:
        return float(sum(self.breakdown.values()))

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
    minutes_fraction = float(np.clip(minutes.expected_minutes / 90.0, 0.0, 1.0))

    breakdown["appearance"] = minutes.expected_appearance_points

    expected_goals = rates.goals_share * team_goals * minutes_fraction
    expected_assists = rates.assists_share * team_goals * minutes_fraction
    breakdown["goals"] = expected_goals * rules.goal.get(position, 0)
    breakdown["assists"] = expected_assists * rules.assist

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
        breakdown["clean_sheet"] = cs_points * minutes.p_60plus * cs_prob

    if position in rules.conceded_penalty_positions:
        breakdown["goals_conceded"] = -expected_conceded_penalty(
            conceded_pmf, minutes_fraction, rules.conceded_per_penalty
        )

    if position == GK and rates.saves_per_90 > 0:
        # Saves scale with how much shooting the opponent does, proxied by their
        # goal expectation relative to a league-average opponent.
        intensity = opponent_goals / max(league_average_conceded, 1e-6)
        expected_saves = rates.saves_per_90 * minutes_fraction * intensity
        breakdown["saves"] = expected_saves_points(expected_saves, rules.saves_per_point)

    threshold = rules.defcon_threshold.get(position)
    if threshold is not None and rates.defcon_per_90 > 0:
        # Evaluate the threshold separately for the starting and bench cases: a
        # 20-minute cameo essentially never reaches 10 or 12 actions, and
        # averaging the minutes first would wrongly suggest it sometimes does.
        minutes_if_start = (
            minutes.expected_minutes / max(minutes.p_start + minutes.p_bench, 1e-6)
        )
        p_hit = threshold_probability(rates.defcon_per_90, minutes_if_start, threshold)
        breakdown["defensive_contribution"] = (
            rules.defcon_points * minutes.p_start * p_hit
        )

    if rates.yellow_per_90:
        breakdown["cards"] = rules.yellow_card * rates.yellow_per_90 * minutes_fraction
        if rates.red_per_90:
            breakdown["cards"] += rules.red_card * rates.red_per_90 * minutes_fraction

    if rates.bps_per_90:
        # Expected BPS scales with minutes and with how good the fixture is,
        # since most BPS accrues through goals, assists and clean sheets.
        fixture_multiplier = 0.5 + 0.5 * (team_goals / max(league_average_conceded, 1e-6))
        expected_bps = rates.bps_per_90 * minutes_fraction * fixture_multiplier
        breakdown["bonus"] = expected_bonus(expected_bps) * minutes.p_appears

    return PlayerProjection(
        element=element_id,
        fixture=fixture,
        position=position,
        minutes=minutes,
        breakdown=breakdown,
    )
