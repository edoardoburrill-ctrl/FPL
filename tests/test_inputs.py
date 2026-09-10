"""Tests for turning a projection context into the optimiser's Player pool."""

import pytest

from fpl.models.projection import build_context
from fpl.optimize.inputs import build_players, horizon_gameweeks, project_horizon
from tests.fixtures.synthetic import build_universe


@pytest.fixture(scope="module")
def setup():
    bootstrap, fixtures = build_universe(seed=7, gameweeks_played=2)
    ctx = build_context(bootstrap, fixtures)
    gameweeks = horizon_gameweeks(ctx, 2)
    projections = project_horizon(ctx, gameweeks)
    return ctx, gameweeks, projections


def test_appearance_and_std_are_threaded_through_from_the_projection(setup):
    ctx, gameweeks, projections = setup
    players = build_players(ctx, projections)
    assert players, "the synthetic universe produced no players above the cutoff"

    for player in players:
        for gw in player.points:
            projection = projections[gw][player.element]
            assert player.appearance[gw] == pytest.approx(projection.minutes.p_appears)
            assert player.std[gw] == pytest.approx(projection.point_std)
            assert player.std[gw] >= 0.0
