import numpy as np

from fpl.models.minutes import (
    availability,
    forecast_minutes,
    synthetic_history,
)

NAILED = [{"minutes": 90, "starts": 1} for _ in range(8)]
BENCHWARMER = [{"minutes": m, "starts": 0} for m in (0, 12, 0, 8, 0, 15, 0, 5)]
ROTATED = [{"minutes": m, "starts": s} for m, s in
           [(90, 1), (0, 0), (75, 1), (20, 0), (90, 1), (0, 0), (65, 1), (10, 0)]]

ELEMENT = {"element_type": 3, "status": "a", "chance_of_playing_next_round": None,
           "now_cost": 75}


def _forecast(history, element=None):
    return forecast_minutes(history, element or ELEMENT, prior_start_rate=0.5)


def test_probabilities_form_a_distribution():
    f = _forecast(NAILED)
    assert np.isclose(f.p_out + f.p_bench + f.p_start, 1.0)
    assert np.isclose(f.p_60plus + f.p_1_to_59, f.p_appears)


def test_nailed_player_beats_rotated_beats_benchwarmer():
    nailed, rotated, bench = _forecast(NAILED), _forecast(ROTATED), _forecast(BENCHWARMER)
    assert nailed.p_start > rotated.p_start > bench.p_start
    assert nailed.expected_minutes > rotated.expected_minutes > bench.expected_minutes


def test_evidence_overcomes_the_prior():
    """Eight consecutive 90s should read as far more than the 0.5 prior."""
    assert _forecast(NAILED).p_start > 0.85


def test_injury_flag_zeroes_the_forecast():
    injured = dict(ELEMENT, status="i", chance_of_playing_next_round=0)
    f = forecast_minutes(NAILED, injured, prior_start_rate=0.5)
    assert f.p_start == 0.0
    assert f.expected_appearance_points == 0.0


def test_doubt_scales_involvement_proportionally():
    full = _forecast(NAILED)
    half = forecast_minutes(
        NAILED, dict(ELEMENT, chance_of_playing_next_round=50), prior_start_rate=0.5
    )
    assert np.isclose(half.p_start, full.p_start * 0.5, atol=1e-9)


def test_availability_reads_fpl_flags():
    assert availability({"status": "a", "chance_of_playing_next_round": None}) == 1.0
    assert availability({"status": "d", "chance_of_playing_next_round": 75}) == 0.75
    assert availability({"status": "i", "chance_of_playing_next_round": None}) == 0.0
    assert availability({"status": "s"}) == 0.0


def test_appearance_points_are_bounded_by_two():
    assert 0.0 <= _forecast(NAILED).expected_appearance_points <= 2.0


def test_rotation_factor_moves_starters_to_the_bench():
    f = _forecast(NAILED)
    rotated = f.scaled(0.5)
    assert rotated.p_start < f.p_start
    assert rotated.p_bench > f.p_bench
    assert np.isclose(rotated.p_out, f.p_out)


def test_synthetic_history_preserves_totals():
    element = {"element_type": 3, "starts": 4, "minutes": 340}
    rows = synthetic_history(element, 6)
    assert len(rows) == 6
    assert sum(r["starts"] for r in rows) == 4
    assert abs(sum(r["minutes"] for r in rows) - 340) <= 6


def test_synthetic_history_handles_a_pure_substitute():
    rows = synthetic_history({"element_type": 4, "starts": 0, "minutes": 62}, 5)
    assert sum(r["starts"] for r in rows) == 0
    assert sum(r["minutes"] for r in rows) > 0


def test_no_history_falls_back_to_the_prior():
    f = forecast_minutes([], ELEMENT, prior_start_rate=0.7)
    assert np.isclose(f.p_start, 0.7, atol=1e-9)
