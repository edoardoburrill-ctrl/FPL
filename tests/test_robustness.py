"""Bootstrap-robustness tests.

Same philosophy as test_optimise.py: a small, hand-built pool proves the
*rules* (does a real edge survive resampling, does a coin-flip not) as well
as a full player pool would, and does it in a fraction of the time — which
matters here more than usual, since robustness needs many re-solves.
"""

import numpy as np
import pulp
import pytest

from fpl.optimize.milp import Player, optimise
from fpl.optimize.robustness import bootstrap_transfer_stability, resample_points
from fpl.report.brief import pair_transfers

FAST = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)

# Mirrors test_optimise.py's squad: 2 GK, 5 DEF, 5 MID, 3 FWD, all at £5.0m.
SQUAD = [1, 2] + list(range(7, 12)) + list(range(21, 26)) + list(range(35, 38))


def make_pool(points_by_element=None, std_by_element=None, price=50):
    points_by_element = points_by_element or {}
    std_by_element = std_by_element or {}
    players = []
    element = 1
    for position, count in ((1, 6), (2, 14), (3, 14), (4, 8)):
        for _ in range(count):
            players.append(
                Player(
                    element=element,
                    position=position,
                    team=(element % 8) + 1,
                    buy_price=price,
                    sell_price=price,
                    points={1: points_by_element.get(element, 2.0)},
                    std={1: std_by_element.get(element, 0.0)},
                    name=f"p{element}",
                )
            )
            element += 1
    return players


def _baseline_moves(players):
    plan = optimise(players, [1], current_squad=SQUAD, bank=0, free_transfers=1,
                    solver=FAST)
    assert plan.solved
    decision = plan.first()
    by_id = {p.element: p for p in players}
    return pair_transfers(decision, by_id)


def test_resample_points_is_deterministic_for_a_given_seed():
    players = make_pool({30: 9.0}, {30: 2.0})
    a = resample_points(players, np.random.default_rng(0))
    b = resample_points(players, np.random.default_rng(0))
    assert [p.points[1] for p in a] == [p.points[1] for p in b]


def test_resample_points_leaves_zero_std_players_untouched():
    players = make_pool({30: 9.0})  # no std given anywhere -> defaults to 0
    resampled = resample_points(players, np.random.default_rng(0))
    for original, noisy in zip(players, resampled):
        assert noisy.points[1] == pytest.approx(original.points[1])


def test_zero_std_reproduces_the_baseline_plan_every_time():
    """With no modelled uncertainty, every resample must agree with the point
    estimate — this is the sanity floor the whole feature stands on."""
    points = {e: 2.0 for e in range(1, 43)}
    points[30] = 9.0  # a clear, noise-free upgrade over the squad's MIDs
    players = make_pool(points)  # std defaults to 0 everywhere

    baseline_moves = _baseline_moves(players)
    assert baseline_moves and baseline_moves[0].in_element == 30

    results = bootstrap_transfer_stability(
        players, [1], baseline_moves, n_resamples=10, seed=1,
        current_squad=SQUAD, bank=0, free_transfers=1, solver=FAST,
    )
    assert len(results) == len(baseline_moves)
    for result in results:
        assert result.resamples == 10
        assert result.frequency == pytest.approx(1.0)
        assert result.is_robust
        assert not result.is_coin_flip


def test_a_clear_edge_survives_resampling_more_than_a_coin_flip():
    """A big, low-noise gap should be far more stable than a hair's-breadth,
    high-noise one — the whole point of measuring this at all."""
    robust_points = {e: 2.0 for e in range(1, 43)}
    robust_points[30] = 9.0
    robust_players = make_pool(robust_points, {30: 0.5})
    robust_moves = _baseline_moves(robust_players)
    robust_result = bootstrap_transfer_stability(
        robust_players, [1], robust_moves, n_resamples=25, seed=2,
        current_squad=SQUAD, bank=0, free_transfers=1, solver=FAST,
    )[0]

    coin_flip_points = {e: 2.0 for e in range(1, 43)}
    coin_flip_points[30] = 4.0
    coin_flip_points[31] = 4.05
    coin_flip_players = make_pool(coin_flip_points, {30: 2.5, 31: 2.5})
    coin_flip_moves = _baseline_moves(coin_flip_players)
    coin_flip_result = bootstrap_transfer_stability(
        coin_flip_players, [1], coin_flip_moves, n_resamples=25, seed=2,
        current_squad=SQUAD, bank=0, free_transfers=1, solver=FAST,
    )[0]

    assert robust_result.frequency > coin_flip_result.frequency
    assert robust_result.is_robust


def test_no_baseline_moves_short_circuits_without_solving():
    """A hold recommendation has nothing to check stability of."""
    players = make_pool()
    results = bootstrap_transfer_stability(
        players, [1], [], n_resamples=5,
        current_squad=SQUAD, bank=0, free_transfers=1, solver=FAST,
    )
    assert results == []
