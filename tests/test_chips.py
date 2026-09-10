"""Chip decisions."""

import pulp

from fpl.optimize.chips import chip_calendar, eligible_gameweeks
from fpl.optimize.milp import optimise
from fpl.state.ledger import Holding, SquadState
from tests.test_optimise import make_pool

FAST = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
SQUAD = [1, 2] + list(range(7, 12)) + list(range(21, 26)) + list(range(35, 38))


def with_points(players, points_by_gameweek):
    for player in players:
        object.__setattr__(
            player,
            "points",
            {gw: points_by_gameweek.get(gw, {}).get(player.element, 2.0)
             for gw in points_by_gameweek},
        )
    return players


def solve(players, gameweeks, **kwargs):
    kwargs.setdefault("current_squad", SQUAD)
    kwargs.setdefault("bank", 0)
    kwargs.setdefault("free_transfers", 1)
    kwargs.setdefault("solver", FAST)
    return optimise(players, list(gameweeks), **kwargs)


def test_triple_captain_lands_on_the_biggest_week():
    """One player has a huge gameweek 2; the armband should be tripled there."""
    players = make_pool()
    with_points(players, {1: {}, 2: {8: 30.0}, 3: {}})
    plan = solve(players, (1, 2, 3), triple_captain_gameweeks=[1, 2, 3])

    assert plan.solved
    played = [d for d in plan.decisions if d.chip == "3xc"]
    assert len(played) == 1
    assert played[0].gameweek == 2


def test_bench_boost_lands_where_the_bench_is_worth_most():
    """The bench is only worth playing when the bench players actually score."""
    players = make_pool()
    # Everyone scores well in gameweek 3, so that is where a full squad pays.
    with_points(players, {1: {}, 2: {}, 3: {e: 8.0 for e in range(1, 43)}})
    plan = solve(players, (1, 2, 3), bench_boost_gameweeks=[1, 2, 3])

    assert plan.solved
    played = [d for d in plan.decisions if d.chip == "bboost"]
    assert len(played) == 1
    assert played[0].gameweek == 3


def test_each_chip_is_played_at_most_once():
    players = make_pool()
    with_points(players, {gw: {8: 25.0} for gw in (1, 2, 3)})
    plan = solve(players, (1, 2, 3), triple_captain_gameweeks=[1, 2, 3],
                 bench_boost_gameweeks=[1, 2, 3])

    assert plan.solved
    assert sum(1 for d in plan.decisions if d.chip == "3xc") <= 1
    assert sum(1 for d in plan.decisions if d.chip == "bboost") <= 1


def test_only_one_chip_per_gameweek():
    """FPL forbids stacking chips; a single-gameweek horizon proves the rule."""
    players = make_pool()
    with_points(players, {1: {e: 12.0 for e in range(1, 43)}})
    plan = solve(players, (1,), triple_captain_gameweeks=[1], bench_boost_gameweeks=[1])

    assert plan.solved
    assert plan.first().chip in ("3xc", "bboost")


def test_chip_points_are_reported_in_the_decision():
    players = make_pool()
    with_points(players, {1: {8: 20.0}})
    tripled = solve(players, (1,), triple_captain_gameweeks=[1]).first()
    plain = solve(players, (1,)).first()
    assert tripled.chip == "3xc"
    assert tripled.expected_points > plain.expected_points


def test_no_chip_is_played_when_none_is_offered():
    players = make_pool()
    with_points(players, {1: {8: 20.0}})
    assert solve(players, (1,)).first().chip is None


def test_eligibility_respects_the_two_halves():
    """A chip used in the first half is available again from gameweek 20."""
    state = SquadState(chips_used={"bboost": [8]})
    assert eligible_gameweeks(state, "bboost", [10, 15, 19]) == []
    assert eligible_gameweeks(state, "bboost", [20, 25]) == [20, 25]

    second_half_used = SquadState(chips_used={"3xc": [30]})
    assert eligible_gameweeks(second_half_used, "3xc", [10, 15]) == [10, 15]
    assert eligible_gameweeks(second_half_used, "3xc", [32]) == []


def test_unused_chips_are_eligible_everywhere():
    state = SquadState()
    assert eligible_gameweeks(state, "wildcard", [5, 25]) == [5, 25]


def test_freehit_reverts_while_wildcard_persists():
    """A fixture swing that lasts past one week should favour Wildcard over Free
    Hit — proof `chip_calendar` no longer scores them identically now that the
    Free Hit revert is modelled exactly rather than bounded by wildcard gain."""
    players = make_pool()
    for player in players:
        gw1 = 20.0 if player.element in range(28, 34) else 2.0
        gw2 = 15.0 if player.element in range(28, 34) else 2.0
        object.__setattr__(player, "points", {1: gw1, 2: gw2})

    state = SquadState(
        entry_id=1, gameweek=1,
        holdings=[Holding(e, 50, 1) for e in SQUAD],
        bank=0, free_transfers=1,
    )
    baseline, advice = chip_calendar(
        players, [1, 2], state, include_restructuring=True,
        discount=1.0, solver=FAST,
    )
    assert baseline.solved

    wildcard = next(a for a in advice if a.chip == "wildcard")
    freehit = next(a for a in advice if a.chip == "freehit")
    assert wildcard.gain > freehit.gain, (
        "wildcard keeps the upgrade into gameweek 2; a Free Hit must not"
    )


def test_a_chip_is_saved_for_a_better_week_later():
    """The classic chip mistake is burning it early; the model must not.

    A bench worth 3 points a head now and 6 a head in gameweek 4 should be
    boosted in gameweek 4, even though the discount makes the present cheaper.
    """
    from fpl.optimize.chips import CHIP_DISCOUNT

    players = make_pool()
    with_points(players, {
        1: {e: 3.0 for e in range(1, 43)},
        2: {e: 3.0 for e in range(1, 43)},
        3: {e: 3.0 for e in range(1, 43)},
        4: {e: 6.0 for e in range(1, 43)},
    })
    plan = solve(players, (1, 2, 3, 4), bench_boost_gameweeks=[1, 2, 3, 4],
                 discount=CHIP_DISCOUNT)
    played = [d for d in plan.decisions if d.chip == "bboost"]
    assert len(played) == 1
    assert played[0].gameweek == 4, (
        "the chip was burned early — check the discount used for chip timing"
    )
