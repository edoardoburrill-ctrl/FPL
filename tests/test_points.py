import numpy as np
import pytest

from fpl.models.minutes import MinutesForecast
from fpl.models.points import (
    PlayerRates,
    clean_sheet_probability_on_pitch,
    expected_bonus,
    expected_conceded_penalty,
    expected_saves_points,
    project_player,
    threshold_probability,
)
from fpl.models.scoring import DEF, FWD, GK, MID, ScoringRules
from fpl.models.team_strength import conceded_distribution, score_matrix

RULES = ScoringRules()
NAILED = MinutesForecast(0.03, 0.05, 0.92, 0.88, 0.09, 80.0)
CAMEO = MinutesForecast(0.30, 0.55, 0.15, 0.13, 0.57, 22.0)


def _conceded(lam_for=1.6, lam_against=1.1, home=True):
    return conceded_distribution(score_matrix(lam_for, lam_against), home=home)


def test_clean_sheet_probability_rises_with_less_time_on_pitch():
    pmf = _conceded()
    full = clean_sheet_probability_on_pitch(pmf, 1.0)
    partial = clean_sheet_probability_on_pitch(pmf, 0.5)
    assert partial > full
    assert np.isclose(full, pmf[0])


def test_conceded_penalty_matches_a_hand_computed_case():
    # Two goals conceded with certainty costs exactly one point.
    pmf = np.array([0.0, 0.0, 1.0])
    assert np.isclose(expected_conceded_penalty(pmf, 1.0), 1.0)
    pmf = np.array([0.0, 1.0, 0.0])
    assert np.isclose(expected_conceded_penalty(pmf, 1.0), 0.0)


def test_saves_points_use_the_floor_not_the_average():
    # E[floor(S/3)] is strictly below E[S]/3 because of the floor.
    assert expected_saves_points(3.0) < 1.0
    assert expected_saves_points(0.0) == 0.0
    assert expected_saves_points(9.0) > expected_saves_points(6.0)


def test_threshold_probability_behaves():
    assert threshold_probability(10.0, 90, 10) > 0.4
    assert threshold_probability(10.0, 20, 10) < 0.02
    assert threshold_probability(0.0, 90, 10) == 0.0


def test_expected_bonus_is_monotone_and_bounded():
    values = [expected_bonus(b) for b in (0, 10, 20, 30, 45, 80)]
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert values[-1] <= 3.0


def test_a_striker_scores_more_against_a_weak_defence():
    rates = PlayerRates(goals_share=0.3, assists_share=0.1, bps_per_90=25)
    easy = project_player(1, FWD, NAILED, rates, 2.4, 0.7, _conceded(2.4, 0.7),
                          RULES).expected_points
    hard = project_player(1, FWD, NAILED, rates, 0.9, 1.9, _conceded(0.9, 1.9),
                          RULES).expected_points
    assert easy > hard + 1.0


def test_a_defender_gains_from_a_likely_clean_sheet():
    rates = PlayerRates(goals_share=0.03, assists_share=0.05, defcon_per_90=9.0,
                        bps_per_90=20)
    likely = project_player(1, DEF, NAILED, rates, 1.5, 0.6, _conceded(1.5, 0.6),
                            RULES)
    unlikely = project_player(1, DEF, NAILED, rates, 1.5, 2.2, _conceded(1.5, 2.2),
                              RULES)
    assert likely.breakdown["clean_sheet"] > unlikely.breakdown["clean_sheet"]
    assert likely.expected_points > unlikely.expected_points


def test_minutes_dominate_everything():
    """A cameo cannot out-project a starter with identical underlying rates."""
    rates = PlayerRates(goals_share=0.3, assists_share=0.15, bps_per_90=28)
    starter = project_player(1, MID, NAILED, rates, 1.8, 1.0, _conceded(), RULES)
    sub = project_player(2, MID, CAMEO, rates, 1.8, 1.0, _conceded(), RULES)
    assert starter.expected_points > 2 * sub.expected_points


def test_goalkeepers_get_saves_but_not_goals():
    rates = PlayerRates(goals_share=0.0, assists_share=0.0, saves_per_90=3.2,
                        bps_per_90=20)
    keeper = project_player(1, GK, NAILED, rates, 1.2, 1.4, _conceded(1.2, 1.4), RULES)
    assert keeper.breakdown["saves"] > 0
    assert keeper.breakdown.get("goals", 0.0) == 0.0


def test_defcon_rewards_the_high_volume_defender():
    low = PlayerRates(goals_share=0.02, assists_share=0.02, defcon_per_90=4.0)
    high = PlayerRates(goals_share=0.02, assists_share=0.02, defcon_per_90=12.0)
    pmf = _conceded()
    lo = project_player(1, DEF, NAILED, low, 1.6, 1.1, pmf, RULES)
    hi = project_player(2, DEF, NAILED, high, 1.6, 1.1, pmf, RULES)
    assert hi.breakdown["defensive_contribution"] > lo.breakdown["defensive_contribution"]
    assert hi.breakdown["defensive_contribution"] <= RULES.defcon_points


def test_projection_explains_itself():
    rates = PlayerRates(goals_share=0.3, assists_share=0.1, bps_per_90=25)
    text = project_player(1, FWD, NAILED, rates, 1.8, 1.0, _conceded(), RULES).explain()
    assert "TOTAL" in text and "appearance" in text
