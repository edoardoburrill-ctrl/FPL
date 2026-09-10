"""Calibration: are the projection engine's probabilities honest?

`backtest/metrics.py` scores the *mean* projection (MAE, ranking, top-pick
points). None of that says anything about the probabilistic side of the
model — a player-gameweek marked "20% to keep a clean sheet" is a different,
checkable claim from "worth 4.2 expected clean-sheet points", and PLAN.md §6
is explicit that calibration matters more than raw accuracy here: the
optimiser consumes probabilities, not just their means, so a confident but
wrong probability is worse than an honestly uncertain one.

A reliability curve answers the checkable version of that claim directly: of
every player-gameweek where the model said "X% to happen", did X% of them
actually happen? Bin predictions by their own value, plot mean-predicted
against empirical-actual per bin, and a perfectly calibrated model sits on
the `y = x` line. `brier_score` and `calibration_slope` are the same question
compressed to one number each — a low Brier score for sharpness plus
accuracy, a slope near 1 for "does X% predicted really mean X% happens"
specifically, independent of how sharp the predictions are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReliabilityBin:
    """One bucket of a reliability curve: players grouped by predicted probability."""

    n: int
    mean_predicted: float
    empirical_rate: float

    @property
    def gap(self) -> float:
        """Positive: the model is underconfident in this bucket. Negative: over."""
        return self.empirical_rate - self.mean_predicted


def reliability_curve(
    predicted: list[float], actual: list[bool], n_bins: int = 10
) -> list[ReliabilityBin]:
    """Bin player-gameweeks by predicted probability and compare to what happened.

    Bins are by rank (roughly equal-sized), not by fixed-width probability
    range: most FPL probabilities cluster well below 50%, and fixed-width
    bins would leave the upper bins nearly empty and the lower ones doing all
    the work. `n_bins` is a target, not a guarantee — fewer, larger bins are
    used automatically when there is not enough data to fill them.
    """
    predicted_arr = np.asarray(predicted, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    n = len(predicted_arr)
    if n == 0:
        return []

    order = np.argsort(predicted_arr)
    bins = min(n_bins, n)
    edges = np.array_split(order, bins)
    return [
        ReliabilityBin(
            n=len(idx),
            mean_predicted=float(predicted_arr[idx].mean()),
            empirical_rate=float(actual_arr[idx].mean()),
        )
        for idx in edges
        if len(idx) > 0
    ]


def brier_score(predicted: list[float], actual: list[bool]) -> float:
    """Mean squared error between a predicted probability and a 0/1 outcome.

    Lower is better; 0 is a clairvoyant model, 0.25 is what "always guess 50%"
    scores against a coin flip. Rewards both calibration and sharpness — a
    model that is calibrated but never confident (always predicts the base
    rate) still scores worse than one that is calibrated *and* sharp.
    """
    predicted_arr = np.asarray(predicted, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    if len(predicted_arr) == 0:
        return float("nan")
    return float(np.mean((predicted_arr - actual_arr) ** 2))


def calibration_slope(predicted: list[float], actual: list[bool]) -> float:
    """The OLS slope of the 0/1 outcome regressed on the predicted probability.

    1.0 is perfectly calibrated in the sense that matters here: on average, an
    extra point of predicted probability buys exactly an extra point of real
    frequency. Below 1 means the model's probabilities swing too far from the
    base rate (overconfident); above 1 means they do not swing far enough
    (underconfident, hedging toward the average). This is the linear-probability
    stand-in for a logistic calibration slope — simpler, and honest about not
    being a real logistic fit, which is not needed just to answer "too sharp or
    too flat".
    """
    predicted_arr = np.asarray(predicted, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    if len(predicted_arr) < 2:
        return float("nan")
    variance = float(np.var(predicted_arr))
    if variance <= 1e-12:
        return float("nan")
    covariance = float(np.cov(predicted_arr, actual_arr, ddof=0)[0, 1])
    return covariance / variance
