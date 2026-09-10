"""Tests for the pure calibration statistics — no network, no live season."""

import numpy as np
import pytest

from fpl.backtest.calibration import brier_score, calibration_slope, reliability_curve


def test_reliability_curve_is_empty_for_no_data():
    assert reliability_curve([], []) == []


def test_reliability_curve_bins_cover_every_sample():
    rng = np.random.default_rng(0)
    predicted = list(rng.uniform(0, 1, size=57))
    actual = list(rng.random(57) < 0.5)
    bins = reliability_curve(predicted, actual, n_bins=10)
    assert sum(b.n for b in bins) == 57
    assert len(bins) <= 10


def test_reliability_curve_recovers_calibration_on_well_calibrated_data():
    """Predicted probabilities that really do generate the outcomes should
    land close to the y = x line in every bin."""
    rng = np.random.default_rng(1)
    predicted = rng.uniform(0.05, 0.95, size=20000)
    actual = rng.random(20000) < predicted
    bins = reliability_curve(list(predicted), list(actual), n_bins=10)
    for b in bins:
        assert abs(b.gap) < 0.05, f"bin with n={b.n} miscalibrated by {b.gap:+.3f}"


def test_reliability_curve_uses_fewer_bins_when_data_is_scarce():
    bins = reliability_curve([0.1, 0.2, 0.3], [True, False, True], n_bins=10)
    assert len(bins) == 3
    assert sum(b.n for b in bins) == 3


def test_brier_score_is_zero_for_perfect_predictions():
    assert brier_score([1.0, 0.0, 1.0], [True, False, True]) == pytest.approx(0.0)


def test_brier_score_penalises_confident_wrong_predictions_most():
    confident_wrong = brier_score([0.95], [False])
    unsure = brier_score([0.5], [False])
    assert confident_wrong > unsure


def test_brier_score_of_coin_flip_guesses_against_random_outcomes():
    rng = np.random.default_rng(2)
    actual = rng.random(50000) < 0.5
    score = brier_score([0.5] * 50000, list(actual))
    assert score == pytest.approx(0.25, abs=0.01)


def test_brier_score_empty_is_nan():
    assert np.isnan(brier_score([], []))


def test_calibration_slope_is_one_when_outcome_equals_prediction():
    predicted = [0.0, 1.0, 0.0, 1.0, 1.0]
    actual = [False, True, False, True, True]
    assert calibration_slope(predicted, actual) == pytest.approx(1.0)


def test_calibration_slope_is_nan_with_no_variance_or_too_few_points():
    assert np.isnan(calibration_slope([0.3, 0.3, 0.3], [True, False, True]))
    assert np.isnan(calibration_slope([0.5], [True]))
    assert np.isnan(calibration_slope([], []))


def test_calibration_slope_below_one_for_an_overconfident_model():
    """Predictions that swing harder than the real, flatter relationship."""
    rng = np.random.default_rng(3)
    true_rate = rng.uniform(0.3, 0.7, size=5000)
    actual = rng.random(5000) < true_rate
    # Exaggerate every probability's distance from 0.5.
    overconfident = np.clip(0.5 + 2.0 * (true_rate - 0.5), 0.0, 1.0)
    slope = calibration_slope(list(overconfident), list(actual))
    assert slope < 1.0
