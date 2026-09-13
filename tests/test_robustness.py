"""Bootstrap-robust transfer recommendations.

Small pools, same style as `test_optimise.py`: the point is to prove a clear
edge survives resampling and a coin-flip does not, which a 42-player league
proves as well as a 700-player one.
"""

import pulp

from fpl.optimize.milp import Player, optimise
from fpl.optimize.robustness import bootstrap_transfers, resample_points

FAST = pulp.PULP_CBC_CMD(msg=False, timeLimit=20)


def make_pool(points_by_element=None, price=50):
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


HELD_SQUAD = [1, 2] + list(range(7, 12)) + list(range(21, 26)) + list(range(35, 38))

COMMON = dict(
    current_squad=HELD_SQUAD,
    bank=0,
    free_transfers=1,
    solver=FAST,
)


def test_resample_points_is_reproducible_given_a_seed():
    players = make_pool({30: 9.0})
    import numpy as np

    a = resample_points(players, [1], np.random.default_rng(3))
    b = resample_points(players, [1], np.random.default_rng(3))
    assert [p.points[1] for p in a] == [p.points[1] for p in b]


def test_resample_points_leaves_prices_and_shape_untouched():
    players = make_pool()
    import numpy as np

    noisy = resample_points(players, [1], np.random.default_rng(1))
    for original, perturbed in zip(players, noisy):
        assert perturbed.element == original.element
        assert perturbed.position == original.position
        assert perturbed.buy_price == original.buy_price
        assert perturbed.sell_price == original.sell_price


def test_a_clear_upgrade_survives_almost_every_resample():
    points = {e: 2.0 for e in range(1, 43)}
    points[30] = 20.0  # a blowout upgrade, not a marginal one
    players = make_pool(points)

    report = bootstrap_transfers(
        players, [1], COMMON, n_resamples=20, seed=1, time_limit=20.0
    )
    assert report.solved >= 15
    # 30 comes in, in nearly every resample, regardless of who it's paired with.
    incoming = {in_e for (_, in_e), freq in report.move_frequency.items()}
    total_in_freq = sum(
        freq for (_, in_e), freq in report.move_frequency.items() if in_e == 30
    )
    assert 30 in incoming
    assert total_in_freq > 0.8


def test_a_marginal_gain_is_not_recommended_in_most_resamples():
    points = {e: 2.0 for e in range(1, 43)}
    points[30] = 2.3  # barely above the noise floor
    players = make_pool(points)

    report = bootstrap_transfers(
        players, [1], COMMON, n_resamples=20, seed=2, time_limit=20.0
    )
    assert report.solved >= 15
    total_in_freq = sum(
        freq for (_, in_e), freq in report.move_frequency.items() if in_e == 30
    )
    # Not "never" (bootstrap noise can occasionally favour it), but nowhere
    # near the near-certainty of a real edge.
    assert total_in_freq < 0.6


def test_bootstrap_report_frequency_defaults_to_zero_for_unseen_moves():
    players = make_pool()
    report = bootstrap_transfers(players, [1], COMMON, n_resamples=3, seed=4)
    assert report.frequency(9999, 9998) == 0.0
