"""Tests for turning projections into optimiser inputs.

`bench_weight_table` in particular: it replaces the optimiser's flat, hardcoded
`BENCH_WEIGHT` constant with a per-gameweek estimate of how often a likely
starter at each position actually blanks — the thing that triggers an FPL
auto-sub in the first place.
"""

import pytest

from fpl.models.minutes import MinutesForecast
from fpl.models.points import PlayerProjection
from fpl.optimize.inputs import (
    MIN_STARTERS_FOR_ESTIMATE,
    bench_weight_table,
    bench_weights_for_gameweek,
)
from fpl.optimize.milp import BENCH_WEIGHT


def _projection(element: int, position: int, p_start: float, p_out: float) -> PlayerProjection:
    """A minimal projection with just the minutes fields the estimate reads."""
    forecast = MinutesForecast(
        p_out=p_out,
        p_bench=max(0.0, 1.0 - p_start - p_out),
        p_start=p_start,
        p_60plus=p_start * 0.9,
        p_1_to_59=0.0,
        expected_minutes=p_start * 80.0,
    )
    return PlayerProjection(element=element, fixture=None, position=position, minutes=forecast)


def test_estimate_averages_blank_probability_of_likely_starters():
    # 20 likely starters at 20% blank risk each, plus fringe players who should
    # be ignored regardless of their own (irrelevant) blank probability.
    projections = {
        i: _projection(i, position=2, p_start=0.8, p_out=0.2) for i in range(20)
    }
    projections.update(
        {100 + i: _projection(100 + i, position=2, p_start=0.1, p_out=0.9) for i in range(5)}
    )

    weights = bench_weights_for_gameweek(projections)

    assert weights[2] == pytest.approx(0.2)


def test_falls_back_when_too_few_likely_starters():
    # Only 5 likely starters at this position — below MIN_STARTERS_FOR_ESTIMATE.
    projections = {
        i: _projection(i, position=1, p_start=0.9, p_out=0.5)
        for i in range(MIN_STARTERS_FOR_ESTIMATE - 1)
    }

    weights = bench_weights_for_gameweek(projections)

    assert 1 not in weights


def test_table_falls_back_to_constant_where_unestimated():
    projections = {
        1: {i: _projection(i, position=1, p_start=0.9, p_out=0.5) for i in range(3)}
    }

    table = bench_weight_table(projections)

    assert table[1] == BENCH_WEIGHT


def test_table_uses_the_estimate_where_there_is_enough_signal():
    gw_projections = {i: _projection(i, position=2, p_start=0.8, p_out=0.4) for i in range(20)}
    projections = {1: gw_projections}

    table = bench_weight_table(projections)

    assert table[1][2] == pytest.approx(0.4)
    # Positions with no data at all still fall back.
    assert table[1][1] == BENCH_WEIGHT[1]
