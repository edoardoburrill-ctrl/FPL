"""End-to-end pipeline tests against a generated league."""

import numpy as np
import pytest

from fpl.data.odds_client import FixtureOdds
from fpl.models.projection import (
    build_context,
    defcon_season_total,
    fixture_goal_rates,
    matches_played,
    project_gameweek,
)
from tests.fixtures.synthetic import build_universe


@pytest.fixture(scope="module")
def universe():
    return build_universe(seed=7, gameweeks_played=2)


@pytest.fixture(scope="module")
def context(universe):
    bootstrap, fixtures = universe
    return build_context(bootstrap, fixtures)


def test_context_fits_ratings_and_priors(context):
    assert context.ratings is not None
    assert context.current_event == 3
    assert set(context.priors) >= {"expected_goals", "expected_assists", "bps"}
    assert 0.8 < context.league_average_goals < 2.2


def test_matches_played_counts_only_finished(universe):
    _, fixtures = universe
    counts = matches_played(fixtures)
    assert set(counts.values()) == {2}


def test_projections_cover_every_player(context, universe):
    bootstrap, _ = universe
    projections = project_gameweek(context, 3)
    assert len(projections) == len(bootstrap["elements"])
    assert all(p.expected_points >= 0 for p in projections.values())


def test_projection_magnitudes_are_sane(context):
    values = np.array(
        [p.expected_points for p in project_gameweek(context, 3).values()]
    )
    # The best player in a gameweek is worth a few points, not thirty.
    assert 4.0 < values.max() < 15.0
    assert 0.5 < values.mean() < 4.0


def test_premium_starters_outrank_squad_filler(context, universe):
    bootstrap, _ = universe
    projections = project_gameweek(context, 3)
    by_id = {int(e["id"]): e for e in bootstrap["elements"]}

    playing = [
        (p.expected_points, by_id[p.element]["now_cost"])
        for p in projections.values()
        if by_id[p.element]["minutes"] > 0
    ]
    points = np.array([row[0] for row in playing])
    cost = np.array([row[1] for row in playing])
    correlation = np.corrcoef(points, cost)[0, 1]
    assert correlation > 0.35, f"price and projection barely relate ({correlation:.2f})"


def test_injured_players_project_near_zero(context, universe):
    bootstrap, _ = universe
    projections = project_gameweek(context, 3)
    injured = [
        projections[int(e["id"])].expected_points
        for e in bootstrap["elements"]
        if e["status"] == "i"
    ]
    assert injured, "generator produced no injured players to test with"
    assert max(injured) < 0.01


def test_blank_gameweek_projects_zero(context, universe):
    """A team with no fixture must contribute nothing, not a fallback average."""
    bootstrap, fixtures = universe
    blanked_team = 1
    trimmed = [
        f for f in fixtures
        if not (f["event"] == 3 and blanked_team in (f["team_h"], f["team_a"]))
    ]
    ctx = build_context(bootstrap, trimmed)
    projections = project_gameweek(ctx, 3)
    blanked = [
        projections[int(e["id"])].expected_points
        for e in bootstrap["elements"]
        if e["team"] == blanked_team
    ]
    assert max(blanked) == 0.0


def test_double_gameweek_roughly_doubles(context, universe):
    bootstrap, fixtures = universe
    single = project_gameweek(context, 3)

    # Give team 1 a second fixture in gameweek 3 by moving one forward.
    doubled_fixtures = [dict(f) for f in fixtures]
    for fixture in doubled_fixtures:
        if fixture["event"] == 5 and 1 in (fixture["team_h"], fixture["team_a"]):
            fixture["event"] = 3
            break
    ctx = build_context(bootstrap, doubled_fixtures)
    double = project_gameweek(ctx, 3)

    team_one = [int(e["id"]) for e in bootstrap["elements"] if e["team"] == 1]
    single_total = sum(single[i].expected_points for i in team_one)
    double_total = sum(double[i].expected_points for i in team_one)
    assert 1.6 * single_total < double_total < 2.4 * single_total

    # Two independent fixtures should raise a player's point variance too, not
    # just his mean — the optimiser's bootstrap resampling depends on this.
    single_variance = sum(single[i].point_variance for i in team_one)
    double_variance = sum(double[i].point_variance for i in team_one)
    assert double_variance > single_variance


def test_odds_shift_the_goal_expectation(context, universe):
    """Market odds must actually move the fixture, not be quietly ignored."""
    bootstrap, fixtures = universe
    fixture = next(f for f in fixtures if f["event"] == 3)

    baseline = fixture_goal_rates(context, fixture)
    lopsided = FixtureOdds(
        home_team="x", away_team="y", kickoff=None,
        p_home=0.85, p_draw=0.11, p_away=0.04, p_over=0.72,
    )
    ctx = build_context(bootstrap, fixtures, odds={int(fixture["id"]): lopsided})
    with_odds = fixture_goal_rates(ctx, fixture)

    assert with_odds[0] - with_odds[1] > baseline[0] - baseline[1]


def test_market_weight_falls_away_over_the_horizon(context, universe):
    bootstrap, fixtures = universe
    near = next(f for f in fixtures if f["event"] == 3)
    lopsided = FixtureOdds(
        home_team="x", away_team="y", kickoff=None,
        p_home=0.85, p_draw=0.11, p_away=0.04,
    )
    ctx = build_context(bootstrap, fixtures, odds={int(near["id"]): lopsided})
    immediate = fixture_goal_rates(ctx, near, gameweeks_ahead=0)
    distant = fixture_goal_rates(ctx, near, gameweeks_ahead=5)
    assert immediate[0] > distant[0]


def test_defcon_totals_use_position_appropriate_components():
    defender = {"element_type": 2, "clearances_blocks_interceptions": 20,
                "tackles": 8, "recoveries": 30}
    midfielder = dict(defender, element_type=3)
    assert defcon_season_total(defender) == 28
    assert defcon_season_total(midfielder) == 58
