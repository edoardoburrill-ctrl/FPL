"""Optimiser tests.

Deliberately small pools: the point is to assert that the *rules* are enforced
and that the economics come out right, which a 20-player league proves as well as
a 700-player one and proves in a second rather than a minute.
"""

import pulp
import pytest

from fpl.optimize.milp import (
    MAX_PER_CLUB,
    SQUAD_SHAPE,
    SQUAD_SIZE,
    XI_SIZE,
    Player,
    optimise,
    prune_pool,
)

FAST = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)


def make_pool(points_by_element=None, price=50):
    """A minimal legal universe: enough players in each position across 8 clubs."""
    points_by_element = points_by_element or {}
    players = []
    element = 1
    for position, count in ((1, 6), (2, 14), (3, 14), (4, 8)):
        for i in range(count):
            players.append(
                Player(
                    element=element,
                    position=position,
                    team=(element % 8) + 1,
                    buy_price=price,
                    sell_price=price,
                    points={1: points_by_element.get(element, 2.0)},
                    name=f"p{element}",
                )
            )
            element += 1
    return players


def solve(players, gameweeks=(1,), **kwargs):
    kwargs.setdefault("bank", 1000)
    kwargs.setdefault("free_transfers", 1)
    kwargs.setdefault("current_squad", [])
    kwargs.setdefault("solver", FAST)
    return optimise(players, list(gameweeks), **kwargs)


def positions(players, ids):
    lookup = {p.element: p.position for p in players}
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for element in ids:
        counts[lookup[element]] += 1
    return counts


def test_squad_obeys_size_shape_and_club_limits():
    players = make_pool()
    plan = solve(players)
    assert plan.solved
    decision = plan.first()

    assert len(decision.squad) == SQUAD_SIZE
    assert positions(players, decision.squad) == SQUAD_SHAPE

    teams = {p.element: p.team for p in players}
    counts = {}
    for element in decision.squad:
        counts[teams[element]] = counts.get(teams[element], 0) + 1
    assert max(counts.values()) <= MAX_PER_CLUB


def test_starting_eleven_is_a_legal_formation():
    plan = solve(make_pool())
    decision = plan.first()
    counts = positions(make_pool(), decision.starting)

    assert len(decision.starting) == XI_SIZE
    assert counts[1] == 1
    assert counts[2] >= 3
    assert counts[4] >= 1
    assert set(decision.starting) <= set(decision.squad)


def test_captain_is_the_best_available_starter():
    points = {e: 2.0 for e in range(1, 43)}
    points[20] = 12.0  # a clear standout midfielder
    players = make_pool(points)
    decision = solve(players).first()

    assert decision.captain in decision.starting
    assert decision.captain == 20
    assert decision.vice != decision.captain


def test_budget_is_respected():
    """A £60m budget cannot buy fifteen £5m players."""
    players = make_pool(price=50)
    plan = solve(players, bank=600)
    assert not plan.solved or len(plan.first().squad) == 0


def test_the_optimiser_spends_a_free_transfer_on_a_real_upgrade():
    points = {e: 2.0 for e in range(1, 43)}
    points[30] = 9.0  # a much better midfielder, currently unowned
    players = make_pool(points)

    squad = [1, 2] + list(range(7, 12)) + list(range(21, 26)) + list(range(35, 38))
    decision = solve(players, current_squad=squad, bank=0, free_transfers=1).first()

    assert 30 in decision.transfers_in
    assert decision.hits == 0


def test_no_hit_for_a_marginal_gain():
    """Four points is a high bar; a 0.5-point upgrade must not clear it."""
    points = {e: 2.0 for e in range(1, 43)}
    points[30] = 2.5
    points[31] = 2.5
    points[32] = 2.5
    players = make_pool(points)
    squad = [1, 2] + list(range(7, 12)) + list(range(21, 26)) + list(range(35, 38))

    decision = solve(players, current_squad=squad, bank=0, free_transfers=1).first()
    assert decision.hits == 0


def test_a_hit_is_taken_when_it_clearly_pays():
    points = {e: 2.0 for e in range(1, 43)}
    points[30] = 20.0
    points[31] = 20.0
    players = make_pool(points)
    squad = [1, 2] + list(range(7, 12)) + list(range(21, 26)) + list(range(35, 38))

    decision = solve(players, current_squad=squad, bank=0, free_transfers=1).first()
    assert decision.transfer_count == 2
    assert decision.hits == 1


def test_free_transfers_accumulate_when_nothing_is_worth_doing():
    """The multi-gameweek formulation should bank transfers, not spend them."""
    players = make_pool({e: 2.0 for e in range(1, 43)})
    for player in players:
        object.__setattr__(player, "points", {gw: 2.0 for gw in range(1, 5)})
    squad = [1, 2] + list(range(7, 12)) + list(range(21, 26)) + list(range(35, 38))

    plan = solve(players, gameweeks=(1, 2, 3, 4), current_squad=squad,
                 bank=0, free_transfers=1)
    free = [d.free_transfers for d in plan.decisions]
    assert free == [1, 2, 3, 4], f"free transfers did not accumulate: {free}"
    assert all(d.transfer_count == 0 for d in plan.decisions)


def test_selling_price_not_current_price_funds_the_move():
    """A player bought at 5.0 who rose to 6.0 sells for 5.5, not 6.0."""
    players = make_pool(price=50)
    target = next(p for p in players if p.element == 30)
    object.__setattr__(target, "buy_price", 56)
    object.__setattr__(target, "points", {1: 9.0})

    held = next(p for p in players if p.element == 21)
    object.__setattr__(held, "buy_price", 60)
    object.__setattr__(held, "sell_price", 55)

    squad = [1, 2] + list(range(7, 12)) + list(range(21, 26)) + list(range(35, 38))
    # 5.5 of proceeds plus 0.1 in the bank cannot fund a 5.6 buy.
    plan = solve(players, current_squad=squad, bank=0, free_transfers=1)
    decision = plan.first()
    if 30 in decision.transfers_in:
        # If it happened, it must have been funded by selling someone else.
        assert 21 not in decision.transfers_out


def test_forced_out_removes_a_player():
    players = make_pool()
    squad = [1, 2] + list(range(7, 12)) + list(range(21, 26)) + list(range(35, 38))
    decision = solve(players, current_squad=squad, bank=500,
                     free_transfers=1, forced_out=[21]).first()
    assert 21 not in decision.squad


def test_wildcard_allows_a_full_rebuild_without_hits():
    points = {e: 2.0 for e in range(1, 43)}
    for e in range(28, 34):
        points[e] = 9.0
    players = make_pool(points)
    squad = [1, 2] + list(range(7, 12)) + list(range(21, 26)) + list(range(35, 38))

    decision = solve(players, current_squad=squad, bank=0, free_transfers=1,
                     wildcard_gameweek=1).first()
    assert decision.hits == 0
    assert decision.transfer_count > 2


def test_wildcard_does_not_consume_the_free_transfer():
    players = make_pool()
    for player in players:
        object.__setattr__(player, "points", {1: 2.0, 2: 2.0})
    squad = [1, 2] + list(range(7, 12)) + list(range(21, 26)) + list(range(35, 38))

    plan = solve(players, gameweeks=(1, 2), current_squad=squad, bank=0,
                 free_transfers=1, wildcard_gameweek=1)
    assert plan.decisions[1].free_transfers == 2


def test_free_hit_allows_a_full_rebuild_without_hits():
    points = {e: 2.0 for e in range(1, 43)}
    for e in range(28, 34):
        points[e] = 9.0
    players = make_pool(points)
    squad = [1, 2] + list(range(7, 12)) + list(range(21, 26)) + list(range(35, 38))

    decision = solve(players, current_squad=squad, bank=0, free_transfers=1,
                     freehit_gameweek=1).first()
    assert decision.chip == "freehit"
    assert decision.hits == 0
    assert decision.transfer_count > 2


def test_free_hit_squad_reverts_next_gameweek():
    """The whole point of a real Free Hit: gameweek 2 starts from the squad
    gameweek 1 would have had if the chip had never been played, not from the
    one-week rebuild."""
    points = {e: 2.0 for e in range(1, 43)}
    for e in range(28, 34):
        points[e] = 9.0  # much better, but only worth taking for one week
    players = make_pool(points)
    for player in players:
        object.__setattr__(player, "points", {1: points.get(player.element, 2.0), 2: 2.0})
    squad = [1, 2] + list(range(7, 12)) + list(range(21, 26)) + list(range(35, 38))

    plan = solve(players, gameweeks=(1, 2), current_squad=squad, bank=0,
                 free_transfers=1, freehit_gameweek=1)
    gw1, gw2 = plan.decisions
    assert gw1.chip == "freehit"
    assert len(set(gw1.transfers_in) & set(range(28, 34))) >= 5
    # Nobody is worth keeping in gameweek 2 (their points drop back to 2.0),
    # so the real squad should be exactly the pre-chip one, untouched.
    assert sorted(gw2.squad) == sorted(squad)
    assert gw2.transfer_count == 0


def test_free_hit_does_not_consume_the_free_transfer_or_bank():
    players = make_pool()
    for player in players:
        object.__setattr__(player, "points", {1: 2.0, 2: 2.0})
    squad = [1, 2] + list(range(7, 12)) + list(range(21, 26)) + list(range(35, 38))

    plan = solve(players, gameweeks=(1, 2), current_squad=squad, bank=0,
                 free_transfers=1, freehit_gameweek=1)
    assert plan.decisions[0].bank == 0.0
    assert plan.decisions[1].free_transfers == 2


def test_bench_weight_scales_with_the_players_own_appearance_chance():
    """Two otherwise-identical backup keepers are told apart only by how
    likely they are to actually feature if selected. Under a flat per-position
    bench constant, the higher-scoring one always wins the bench spot; once
    the weight is scaled by the minutes model's own P(plays at all), a backup
    who almost never turns up should lose out to a lower-scoring nailed one."""
    points = {e: 2.0 for e in range(1, 43)}
    points[1] = 10.0  # the clear starting keeper
    points[2] = 5.0   # scores well *if* selected, but rarely plays at all
    players = make_pool(points)
    for player in players:
        if player.element == 2:
            object.__setattr__(player, "p_appears", {1: 0.05})

    decision = solve(players, bank=1000, free_transfers=1).first()
    assert 1 in decision.starting
    assert 2 not in decision.squad


def test_pruning_keeps_owned_players_and_cheap_options():
    players = make_pool()
    object.__setattr__(players[0], "buy_price", 40)
    pool = prune_pool(players, [1], keep={1: 2, 2: 3, 3: 3, 4: 2},
                      always_include=[players[-1].element])
    assert players[-1].element in {p.element for p in pool}
    assert min(p.buy_price for p in pool) == 40


def test_empty_horizon_is_rejected():
    with pytest.raises(ValueError):
        optimise(make_pool(), [], current_squad=[], bank=1000, free_transfers=1)


def test_unknown_player_in_squad_is_rejected():
    with pytest.raises(ValueError):
        optimise(make_pool(), [1], current_squad=[9999], bank=1000, free_transfers=1)
