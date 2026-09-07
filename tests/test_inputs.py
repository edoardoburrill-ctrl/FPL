"""Tests for turning a projection context into optimiser inputs."""

import pytest

from fpl.models.projection import build_context
from fpl.optimize.inputs import (
    AUTO_SUB_TRIGGER_PROBABILITY,
    auto_sub_weight,
    build_players,
    horizon_gameweeks,
    project_horizon,
)
from tests.fixtures.synthetic import build_universe


@pytest.fixture(scope="module")
def setup():
    bootstrap, fixtures = build_universe(seed=7, gameweeks_played=2)
    ctx = build_context(bootstrap, fixtures)
    gameweeks = horizon_gameweeks(ctx, 2)
    projections = project_horizon(ctx, gameweeks)
    return ctx, gameweeks, projections


def test_auto_sub_weight_scales_with_appearance_chance():
    """A player nailed to play, if summoned, is a likelier auto-sub than a fringe one."""
    nailed = auto_sub_weight(2, p_appears=0.95)
    fringe = auto_sub_weight(2, p_appears=0.1)
    assert nailed > fringe
    assert nailed == pytest.approx(AUTO_SUB_TRIGGER_PROBABILITY[2] * 0.95)


def test_auto_sub_weight_matches_old_constant_at_a_coin_flip():
    """The old flat constants assumed a roughly 50-50 bench player."""
    for position, old_constant in ((1, 0.06), (2, 0.13), (3, 0.13), (4, 0.13)):
        assert auto_sub_weight(position, p_appears=0.5) == pytest.approx(
            old_constant, abs=0.005
        )


def test_auto_sub_weight_is_clamped_to_a_probability():
    assert auto_sub_weight(2, p_appears=-1.0) == 0.0
    assert auto_sub_weight(2, p_appears=5.0) == AUTO_SUB_TRIGGER_PROBABILITY[2]


def test_build_players_carries_a_per_gameweek_bench_weight(setup):
    ctx, gameweeks, projections = setup
    players = build_players(ctx, projections, include_unlikely=True)

    checked = 0
    for player in players:
        for gw in gameweeks:
            projection = projections[gw].get(player.element)
            if projection is None:
                continue
            expected = auto_sub_weight(player.position, projection.minutes.p_appears)
            assert player.bench_weight[gw] == pytest.approx(expected)
            checked += 1
    assert checked > 0


def test_build_players_bench_weight_reflects_who_actually_plays(setup):
    """Two players in the same position should not share a bench weight."""
    ctx, gameweeks, projections = setup
    players = build_players(ctx, projections, include_unlikely=True)

    by_position: dict[int, set[float]] = {}
    for player in players:
        gw = gameweeks[0]
        if gw not in player.bench_weight:
            continue
        by_position.setdefault(player.position, set()).add(
            round(player.bench_weight[gw], 6)
        )

    # At least one position should show real spread across the generated pool --
    # a single shared value would mean every player is treated identically again.
    assert any(len(values) > 1 for values in by_position.values())
