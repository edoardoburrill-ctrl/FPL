"""The Monte Carlo match engine.

These are statistical tests: they check that quantities converge to the right
place across many draws, with a fixed seed and enough draws that the noise
band is tight relative to what is being asserted. They are not exact-equality
tests, because nothing in a Monte Carlo engine is.
"""

import numpy as np
import pytest

from fpl.models.minutes import MinutesForecast
from fpl.models.points import PlayerRates
from fpl.models.scoring import DEF, FWD, GK, MID, ScoringRules
from fpl.models.simulate import SimPlayer, _sample_minutes, simulate_fixture

RULES = ScoringRules()
STARTER = MinutesForecast(0.03, 0.05, 0.92, 0.88, 0.09, 80.0)
CAMEO = MinutesForecast(0.30, 0.55, 0.15, 0.13, 0.57, 22.0)
N_SIMS = 6000
SEED = np.random.default_rng(42)


def make_side(specs, home: bool):
    return [
        SimPlayer(element, pos, home, minutes,
                 PlayerRates(goals_share=g, assists_share=a, **kwargs))
        for element, pos, minutes, g, a, kwargs in specs
    ]


def test_minutes_samples_track_the_expected_minutes():
    samples = _sample_minutes(STARTER, 20_000, np.random.default_rng(1))
    assert abs(samples.mean() - STARTER.expected_minutes) < 3.0


def test_minutes_samples_respect_the_appearance_split():
    samples = _sample_minutes(STARTER, 20_000, np.random.default_rng(1))
    assert abs(np.mean(samples == 0) - STARTER.p_out) < 0.02
    assert abs(np.mean(samples >= 60) - STARTER.p_60plus) < 0.02


def test_empty_fixture_returns_empty():
    result = simulate_fixture([], [], 1.5, 1.2, RULES, n_sims=100)
    assert result.players == {}


def test_a_dominant_scorer_takes_most_of_his_teams_goals():
    home = make_side([
        (1, FWD, STARTER, 0.55, 0.05, {}),
        (2, MID, STARTER, 0.03, 0.10, {}),
        (3, MID, STARTER, 0.03, 0.10, {}),
    ], home=True)
    away = make_side([(9, GK, STARTER, 0.0, 0.0, {"saves_per_90": 3.0})], home=False)

    sim = simulate_fixture(home, away, lam_home=2.2, lam_away=0.6, rules=RULES,
                           n_sims=N_SIMS, rng=np.random.default_rng(2))
    # A player taking over half the goal share should out-project teammates
    # taking a tenth each, by a wide margin — though not a full 2x, since both
    # share the same flat appearance-point floor regardless of goal share.
    assert sim.players[1].mean > 1.5 * sim.players[2].mean


def test_goalkeeper_clean_sheet_reflects_low_opponent_threat():
    weak_attack = make_side([(1, FWD, STARTER, 0.3, 0.1, {})], home=True)
    keeper = make_side([(9, GK, STARTER, 0.0, 0.0, {"saves_per_90": 2.5})], home=False)

    strong_sim = simulate_fixture(
        weak_attack, keeper, lam_home=2.4, lam_away=0.9, rules=RULES,
        n_sims=N_SIMS, rng=np.random.default_rng(3),
    )
    weak_sim = simulate_fixture(
        weak_attack, keeper, lam_home=0.4, lam_away=0.9, rules=RULES,
        n_sims=N_SIMS, rng=np.random.default_rng(3),
    )
    # A keeper facing a much weaker attack should have a materially higher
    # expected score, driven by a higher clean-sheet rate.
    assert weak_sim.players[9].mean > strong_sim.players[9].mean


def test_opposing_players_in_the_same_match_are_negatively_correlated():
    """A joint simulation should show what an independent model cannot: a home
    keeper's points and an away striker's points are driven by the same
    scoreline draw, in opposite directions."""
    home = make_side([(1, GK, STARTER, 0.0, 0.0, {"saves_per_90": 3.0})], home=True)
    away = make_side([(9, FWD, STARTER, 0.35, 0.1, {})], home=False)

    sim = simulate_fixture(home, away, lam_home=1.3, lam_away=1.3, rules=RULES,
                           n_sims=N_SIMS, rng=np.random.default_rng(4))
    assert sim.correlation(1, 9) < -0.15


def test_correlation_with_missing_player_is_zero_not_an_error():
    home = make_side([(1, FWD, STARTER, 0.3, 0.1, {})], home=True)
    sim = simulate_fixture(home, [], 1.5, 1.0, RULES, n_sims=200,
                           rng=np.random.default_rng(5))
    assert sim.correlation(1, 999) == 0.0


def test_haul_probability_is_between_zero_and_one_and_sane():
    home = make_side([(1, FWD, STARTER, 0.4, 0.1, {})], home=True)
    away = make_side([(9, GK, STARTER, 0.0, 0.0, {"saves_per_90": 3.0})], home=False)
    sim = simulate_fixture(home, away, 1.8, 0.9, RULES, n_sims=N_SIMS,
                           rng=np.random.default_rng(6))
    haul = sim.players[1].haul_probability(10.0)
    assert 0.0 <= haul <= 1.0
    # A premium attacker at 1.8 xG in a good fixture should have a real,
    # non-trivial chance of a double-figure haul, not a rounding-error one.
    assert haul > 0.05


def test_a_cameo_player_projects_far_below_a_starter_with_identical_rates():
    home = make_side([
        (1, MID, STARTER, 0.3, 0.15, {}),
        (2, MID, CAMEO, 0.3, 0.15, {}),
    ], home=True)
    away = make_side([(9, GK, STARTER, 0.0, 0.0, {"saves_per_90": 3.0})], home=False)
    sim = simulate_fixture(home, away, 1.6, 1.0, RULES, n_sims=N_SIMS,
                           rng=np.random.default_rng(7))
    assert sim.players[1].mean > 1.7 * sim.players[2].mean


def test_cards_can_be_included_and_reduce_points():
    disciplined = make_side([(1, DEF, STARTER, 0.02, 0.05, {})], home=True)
    reckless = make_side(
        [(1, DEF, STARTER, 0.02, 0.05, {"yellow_per_90": 0.9, "red_per_90": 0.05})],
        home=True,
    )
    away = make_side([(9, FWD, STARTER, 0.2, 0.1, {})], home=False)

    calm = simulate_fixture(disciplined, away, 1.3, 1.3, RULES, n_sims=N_SIMS,
                            rng=np.random.default_rng(8))
    booked = simulate_fixture(reckless, away, 1.3, 1.3, RULES, n_sims=N_SIMS,
                              rng=np.random.default_rng(8))
    assert booked.players[1].mean < calm.players[1].mean


def test_saves_scale_with_opponent_goal_threat():
    keeper = make_side([(1, GK, STARTER, 0.0, 0.0, {"saves_per_90": 3.0})], home=True)
    weak_away = make_side([(9, FWD, STARTER, 0.15, 0.05, {})], home=False)
    strong_away = make_side([(9, FWD, STARTER, 0.5, 0.15, {})], home=False)

    low = simulate_fixture(keeper, weak_away, 1.5, 0.6, RULES, n_sims=N_SIMS,
                           rng=np.random.default_rng(9))
    high = simulate_fixture(keeper, strong_away, 1.5, 2.2, RULES, n_sims=N_SIMS,
                            rng=np.random.default_rng(9))
    assert high.players[1].mean != pytest.approx(low.players[1].mean, abs=0.05)
