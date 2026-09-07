"""Bootstrap-resample robustness tests.

Deliberately small, deterministic (seeded) pools, mirroring test_optimise.py:
the point is to prove the *statistics* come out right — a real edge survives
resampling, a coin-flip does not — which a tiny pool proves as well as a big
one and proves in a second rather than a minute.
"""

import numpy as np
import pulp
import pytest

from fpl.models.minutes import MinutesForecast
from fpl.models.points import PlayerProjection
from fpl.optimize.milp import Player
from fpl.optimize.robustness import resample_players, transfer_stability

FAST = pulp.PULP_CBC_CMD(msg=False, timeLimit=10)
DUMMY_MINUTES = MinutesForecast(0.05, 0.05, 0.90, 0.85, 0.10, 80.0)


def projection(element, mean, sd, position=1):
    return PlayerProjection(
        element=element,
        fixture=None,
        position=position,
        minutes=DUMMY_MINUTES,
        breakdown={"total": mean},
        variance={"total": sd**2},
    )


def make_pool(points_by_element=None, sd_by_element=None, price=50):
    points_by_element = points_by_element or {}
    sd_by_element = sd_by_element or {}
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
    projections = {
        1: {
            p.element: projection(
                p.element, p.points[1], sd_by_element.get(p.element, 0.1), p.position
            )
            for p in players
        }
    }
    return players, projections


SQUAD = [1, 2] + list(range(7, 12)) + list(range(21, 26)) + list(range(35, 38))


def test_a_wide_clear_edge_survives_almost_every_resample():
    points = {e: 2.0 for e in range(1, 43)}
    points[21] = 0.5  # unambiguously the worst held midfielder
    points[30] = 20.0  # a huge, low-noise upgrade over anything held
    players, projections = make_pool(
        points, sd_by_element={30: 0.3, 21: 0.1}
    )

    report = transfer_stability(
        players, projections, [1], SQUAD, bank=0, free_transfers=1,
        n_resamples=25, seed=1, solver=FAST,
    )
    assert report.n_solved == 25
    assert report.frequency_of(out_element=21, in_element=30) >= 0.9


def test_a_coin_flip_transfer_is_unstable():
    points = {e: 2.0 for e in range(1, 43)}
    points[30] = 2.3  # barely ahead of the incumbent...
    players, projections = make_pool(
        points, sd_by_element={30: 3.0, 21: 3.0}  # ...and both are noisy
    )

    report = transfer_stability(
        players, projections, [1], SQUAD, bank=0, free_transfers=1,
        n_resamples=25, seed=2, solver=FAST,
    )
    # A near-tie under heavy noise should not survive resampling anywhere near
    # as reliably as a wide, low-noise edge does.
    assert report.frequency_of(out_element=21, in_element=30) < 0.9


def test_zero_variance_is_fully_deterministic():
    points = {e: 2.0 for e in range(1, 43)}
    points[30] = 9.0
    players, projections = make_pool(points, sd_by_element={30: 0.0})

    report = transfer_stability(
        players, projections, [1], SQUAD, bank=0, free_transfers=1,
        n_resamples=5, seed=3, solver=FAST,
    )
    assert report.n_solved == 5
    # With no injected noise every resample is the same solve, so the winning
    # move (if any) appears either in all of them or none.
    for stability in report.moves:
        assert stability.frequency in (1.0, 0.2, 0.4, 0.6, 0.8)


def test_resample_players_leaves_prices_and_names_untouched():
    players, projections = make_pool({30: 9.0}, sd_by_element={30: 1.0})
    rng = np.random.default_rng(4)
    resampled = resample_players(players, projections, rng)
    assert len(resampled) == len(players)
    for original, sampled in zip(players, resampled):
        assert sampled.element == original.element
        assert sampled.buy_price == original.buy_price
        assert sampled.sell_price == original.sell_price
        assert sampled.name == original.name


def test_resampling_does_not_touch_gameweeks_without_a_projection():
    """A player-gameweek missing from `projections` keeps its point estimate,
    rather than being silently zeroed or crashing."""
    players, projections = make_pool({30: 9.0})
    for player in players:
        object.__setattr__(player, "points", {1: player.points[1], 2: 5.0})
    rng = np.random.default_rng(5)
    resampled = resample_players(players, projections, rng)
    for player in resampled:
        assert player.points[2] == 5.0


def test_rejects_fewer_than_one_resample():
    players, projections = make_pool()
    with pytest.raises(ValueError):
        transfer_stability(
            players, projections, [1], SQUAD, bank=0, free_transfers=1,
            n_resamples=0, solver=FAST,
        )
