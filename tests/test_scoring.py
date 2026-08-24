"""Scoring rules must reproduce known point totals exactly."""

from fpl.models.scoring import (
    DEF,
    FWD,
    GK,
    MID,
    ScoringRules,
    defensive_contribution_count,
    score_stat_line,
    verify_rules,
)

RULES = ScoringRules()


def test_no_appearance_scores_nothing():
    assert score_stat_line({"minutes": 0, "yellow_cards": 1}, MID, RULES) == 0


def test_appearance_threshold_at_60_minutes():
    assert score_stat_line({"minutes": 59}, MID, RULES) == 1
    assert score_stat_line({"minutes": 60}, MID, RULES) == 2


def test_goal_values_by_position():
    line = {"minutes": 90, "goals_scored": 1}
    assert score_stat_line(line, GK, RULES) == 2 + 6
    assert score_stat_line(line, DEF, RULES) == 2 + 6 - 0
    assert score_stat_line(line, MID, RULES) == 2 + 5
    assert score_stat_line(line, FWD, RULES) == 2 + 4


def test_clean_sheet_only_pays_defensive_positions():
    line = {"minutes": 90, "clean_sheets": 1, "goals_conceded": 0}
    assert score_stat_line(line, GK, RULES) == 2 + 4
    assert score_stat_line(line, DEF, RULES) == 2 + 4
    assert score_stat_line(line, MID, RULES) == 2 + 1
    assert score_stat_line(line, FWD, RULES) == 2


def test_goals_conceded_penalty_is_every_two():
    for conceded, expected_penalty in [(0, 0), (1, 0), (2, -1), (3, -1), (4, -2)]:
        line = {"minutes": 90, "goals_conceded": conceded}
        assert score_stat_line(line, DEF, RULES) == 2 + expected_penalty
        # Midfielders and forwards are unaffected.
        assert score_stat_line(line, MID, RULES) == 2


def test_saves_pay_every_three():
    for saves, points in [(2, 0), (3, 1), (5, 1), (6, 2)]:
        line = {"minutes": 90, "saves": saves}
        assert score_stat_line(line, GK, RULES) == 2 + points


def test_defensive_contribution_thresholds():
    # Defenders need 10 CBIT; midfielders and forwards need 12 CBIRT.
    assert score_stat_line({"minutes": 90, "defensive_contribution": 9}, DEF, RULES) == 2
    assert score_stat_line({"minutes": 90, "defensive_contribution": 10}, DEF, RULES) == 4
    assert score_stat_line({"minutes": 90, "defensive_contribution": 11}, MID, RULES) == 2
    assert score_stat_line({"minutes": 90, "defensive_contribution": 12}, MID, RULES) == 4
    # Capped at one award however many actions are made.
    assert score_stat_line({"minutes": 90, "defensive_contribution": 30}, DEF, RULES) == 4


def test_defcon_counts_recoveries_only_for_mid_and_fwd():
    stats = {"clearances_blocks_interceptions": 6, "tackles": 3, "recoveries": 5}
    assert defensive_contribution_count(stats, DEF) == 9
    assert defensive_contribution_count(stats, MID) == 14


def test_full_haul_adds_up():
    line = {
        "minutes": 90,
        "goals_scored": 2,
        "assists": 1,
        "clean_sheets": 1,
        "goals_conceded": 0,
        "bonus": 3,
        "yellow_cards": 1,
    }
    # 2 appearance + 10 goals + 3 assist + 1 clean sheet + 3 bonus - 1 card
    assert score_stat_line(line, MID, RULES) == 18


def test_verification_flags_wrong_rules():
    good = [({"minutes": 90, "goals_scored": 1, "total_points": 7}, MID)]
    assert verify_rules(good, RULES).ok

    wrong = [({"minutes": 90, "goals_scored": 1, "total_points": 99}, MID)] * 50
    report = verify_rules(wrong, RULES)
    assert not report.ok
    assert report.mismatch_rate == 1.0
    assert report.examples
