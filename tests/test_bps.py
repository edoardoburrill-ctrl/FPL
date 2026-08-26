"""BPS table and bonus allocation.

The tie-rule tests reproduce FPL's own published worked examples exactly, so a
future refactor that breaks the rank arithmetic fails loudly rather than
quietly inverting who gets bonus — which is exactly what the first draft of
`award_bonus` did (rank->bonus payouts were accidentally a pass-through instead
of the real 1->3, 2->2, 3->1 mapping, silently handing the bonus to the worst
performer). These tests exist because that bug was real.
"""

from fpl.models.bps import (
    DEFAULT_TABLE,
    PlausibilityReport,
    award_bonus,
    partial_bps,
    verify_plausibility,
)
from fpl.models.scoring import DEF, FWD, GK, MID


def test_clear_ranking_pays_three_two_one():
    assert award_bonus({1: 10, 2: 8, 3: 6, 4: 4}) == {1: 3, 2: 2, 3: 1, 4: 0}


def test_tie_for_first_both_get_three_and_third_gets_one_not_two():
    """FPL's own example: two tied at top get 3 each; next player gets 1, not 2."""
    assert award_bonus({1: 35, 2: 35, 3: 30}) == {1: 3, 2: 3, 3: 1}


def test_tie_for_second_both_get_two_and_nobody_gets_one():
    assert award_bonus({1: 40, 2: 35, 3: 35}) == {1: 3, 2: 2, 3: 2}


def test_five_way_split_from_the_official_worked_example():
    """35/35/34/34/34 -> 3/3/1/1/1, per FPL's published example."""
    result = award_bonus({1: 35, 2: 35, 3: 34, 4: 34, 5: 34})
    assert result == {1: 3, 2: 3, 3: 1, 4: 1, 5: 1}


def test_only_top_three_distinct_values_pay_anything():
    result = award_bonus({e: 10 - e for e in range(1, 8)})  # 7 distinct values
    assert sorted(result.values(), reverse=True)[:3] == [3, 2, 1]
    assert all(v == 0 for v in list(result.values())[3:])


def test_a_four_way_tie_for_top_still_only_pays_the_top_group():
    """All four tied at the top get 3; nobody is left to take 2nd or 3rd."""
    result = award_bonus({1: 20, 2: 20, 3: 20, 4: 20, 5: 15})
    assert all(result[e] == 3 for e in (1, 2, 3, 4))
    assert result[5] == 0, "rank 5 is consumed by the four-way tie, not rank 2"


def test_empty_and_singleton_do_not_crash():
    assert award_bonus({}) == {}
    assert award_bonus({7: 12}) == {7: 3}


def test_bonus_never_exceeds_three_or_goes_negative():
    import random

    rng = random.Random(0)
    for _ in range(200):
        bps = {e: rng.randint(-5, 60) for e in range(1, rng.randint(1, 25))}
        result = award_bonus(bps)
        assert all(0 <= v <= 3 for v in result.values())
        assert set(result) == set(bps)


def test_partial_bps_is_zero_for_no_appearance():
    assert partial_bps({"minutes": 0, "goals_scored": 3}, FWD) == 0


def test_partial_bps_reflects_the_appearance_threshold():
    short = partial_bps({"minutes": 45}, MID)
    long = partial_bps({"minutes": 90}, MID)
    assert short == DEFAULT_TABLE.appearance_short
    assert long == DEFAULT_TABLE.appearance_long


def test_partial_bps_scales_goal_value_by_position():
    line = {"minutes": 90, "goals_scored": 1}
    fwd = partial_bps(line, FWD) - DEFAULT_TABLE.appearance_long
    mid = partial_bps(line, MID) - DEFAULT_TABLE.appearance_long
    def_ = partial_bps(line, DEF) - DEFAULT_TABLE.appearance_long
    assert fwd > mid > def_


def test_partial_bps_only_credits_saves_to_goalkeepers():
    line = {"minutes": 90, "saves": 4}
    assert partial_bps(line, GK) > partial_bps(dict(line), DEF)


def test_plausibility_reports_a_small_residual_as_unremarkable():
    """A residual of a few points is exactly what an omitted CBI or foul term
    would produce, so it must not be flagged as extreme."""
    stats = {"minutes": 90, "goals_scored": 1, "assists": 1, "bps": 34}
    report = verify_plausibility([(stats, MID)])
    assert report.checked == 1
    assert report.extreme == []


def test_plausibility_flags_a_genuinely_extreme_residual():
    """A residual too large for any plausible number of missed fouls or CBI
    actions to explain is worth surfacing, in either direction."""
    stats = {"minutes": 90, "goals_scored": 3, "bps": 5}  # implausibly low real bps
    report = verify_plausibility([(stats, FWD)])
    assert len(report.extreme) == 1
    assert report.extreme[0]["residual"] < 0


def test_plausibility_flags_an_overshoot_in_the_other_direction_too():
    stats = {"minutes": 90, "bps": 200}  # implausibly high real bps
    report = verify_plausibility([(stats, MID)])
    assert len(report.extreme) == 1
    assert report.extreme[0]["residual"] > 0


def test_plausibility_skips_lines_with_no_bps_field_or_no_appearance():
    report = verify_plausibility([
        ({"minutes": 90, "goals_scored": 1}, FWD),  # no bps field
        ({"minutes": 0, "bps": 0}, FWD),             # did not play
    ])
    assert report.checked == 0
    assert "no stat lines" in verify_plausibility([]).summary()
