"""Tests for the weekly brief, on the generated league."""

import pytest

from fpl.models.projection import build_context
from fpl.optimize.inputs import build_players, horizon_gameweeks, project_horizon
from fpl.optimize.milp import Decision, Player, optimise, prune_pool
from fpl.report.brief import COIN_FLIP_MARGIN, build_brief, deadline_for, pair_transfers
from fpl.state.ledger import Holding, SquadState
from tests.fixtures.synthetic import build_universe

HORIZON = 2
TIME_LIMIT = 25.0


@pytest.fixture(scope="module")
def setup():
    bootstrap, fixtures = build_universe(seed=7, gameweeks_played=2)
    ctx = build_context(bootstrap, fixtures)
    gameweeks = horizon_gameweeks(ctx, HORIZON)
    projections = project_horizon(ctx, gameweeks)
    players = build_players(ctx, projections)
    return ctx, gameweeks, projections, players


@pytest.fixture(scope="module")
def optimal_state(setup):
    """A squad the optimiser itself would pick, so there is nothing to improve."""
    ctx, gameweeks, projections, players = setup
    pool = prune_pool(players, gameweeks)
    plan = optimise(pool, gameweeks[:1], current_squad=[], bank=1000,
                    free_transfers=1, time_limit=TIME_LIMIT)
    assert plan.solved
    prices = {int(e["id"]): int(e["now_cost"]) for e in ctx.elements}
    squad = plan.first().squad
    return SquadState(
        entry_id=1, gameweek=2,
        holdings=[Holding(e, prices[e], 1) for e in squad],
        bank=1000 - sum(prices[e] for e in squad),
        free_transfers=1,
    )


def test_deadline_is_parsed(setup):
    ctx, gameweeks, _, _ = setup
    deadline = deadline_for(ctx, gameweeks[0])
    assert deadline is not None and deadline.tzinfo is not None


def test_brief_on_an_optimal_squad_recommends_holding(setup, optimal_state):
    ctx, gameweeks, projections, players = setup
    brief = build_brief(ctx, optimal_state, players, projections, gameweeks,
                        time_limit=TIME_LIMIT)
    assert brief.plan.solved
    assert brief.gain < 0.5
    assert not brief.recommends_transfer or not brief.gain_is_meaningful


def test_brief_finds_an_obvious_upgrade(setup, optimal_state):
    """Replace the best player with the cheapest; the model must want him back."""
    ctx, gameweeks, projections, players = setup
    prices = {int(e["id"]): int(e["now_cost"]) for e in ctx.elements}

    premium = max(optimal_state.holdings, key=lambda h: h.purchase_price)
    position = int(ctx.element(premium.element)["element_type"])
    replacement = min(
        (
            e for e in ctx.elements
            if int(e["element_type"]) == position
            and int(e["id"]) not in optimal_state.element_ids
        ),
        key=lambda e: int(e["now_cost"]),
    )

    degraded = SquadState(
        entry_id=1, gameweek=2,
        holdings=[h for h in optimal_state.holdings if h.element != premium.element]
        + [Holding(int(replacement["id"]), int(replacement["now_cost"]), 1)],
        bank=optimal_state.bank + premium.purchase_price - int(replacement["now_cost"]),
        free_transfers=1,
    )

    brief = build_brief(ctx, degraded, players, projections, gameweeks,
                        time_limit=TIME_LIMIT)
    assert brief.recommends_transfer
    assert brief.gain_is_meaningful
    assert brief.moves[0].out_element == int(replacement["id"])
    assert brief.hits == 0, "an obvious free-transfer upgrade should not cost a hit"


def test_brief_reports_the_squad_it_actually_picked(setup, optimal_state):
    ctx, gameweeks, projections, players = setup
    brief = build_brief(ctx, optimal_state, players, projections, gameweeks,
                        time_limit=TIME_LIMIT)
    assert len(brief.starting) == 11
    assert len(brief.bench) == 4
    assert brief.captain in brief.starting
    assert brief.vice in brief.starting
    assert brief.captain != brief.vice
    assert not set(brief.starting) & set(brief.bench)


def test_captain_margin_flags_a_coin_flip(setup, optimal_state):
    ctx, gameweeks, projections, players = setup
    brief = build_brief(ctx, optimal_state, players, projections, gameweeks,
                        time_limit=TIME_LIMIT)
    assert brief.captain_margin >= 0
    assert brief.captain_is_close == (brief.captain_margin < COIN_FLIP_MARGIN)


def test_chips_are_reported(setup, optimal_state):
    ctx, gameweeks, projections, players = setup
    brief = build_brief(ctx, optimal_state, players, projections, gameweeks,
                        time_limit=TIME_LIMIT)
    assert set(brief.chips_available) == {"wildcard", "freehit", "3xc", "bboost"}


def test_stability_is_not_computed_by_default(setup, optimal_state):
    ctx, gameweeks, projections, players = setup
    brief = build_brief(ctx, optimal_state, players, projections, gameweeks,
                        time_limit=TIME_LIMIT)
    assert brief.stability is None


def test_stability_reports_a_high_frequency_for_an_obvious_upgrade(setup, optimal_state):
    """The same obvious-upgrade scenario as above, but asking for resampling:
    a real, wide edge should survive nearly every bootstrap draw."""
    ctx, gameweeks, projections, players = setup
    prices = {int(e["id"]): int(e["now_cost"]) for e in ctx.elements}

    premium = max(optimal_state.holdings, key=lambda h: h.purchase_price)
    position = int(ctx.element(premium.element)["element_type"])
    replacement = min(
        (
            e for e in ctx.elements
            if int(e["element_type"]) == position
            and int(e["id"]) not in optimal_state.element_ids
        ),
        key=lambda e: int(e["now_cost"]),
    )

    degraded = SquadState(
        entry_id=1, gameweek=2,
        holdings=[h for h in optimal_state.holdings if h.element != premium.element]
        + [Holding(int(replacement["id"]), int(replacement["now_cost"]), 1)],
        bank=optimal_state.bank + premium.purchase_price - int(replacement["now_cost"]),
        free_transfers=1,
    )

    brief = build_brief(ctx, degraded, players, projections, gameweeks,
                        time_limit=TIME_LIMIT, stability_resamples=6,
                        stability_seed=0, stability_time_limit=TIME_LIMIT)
    assert brief.stability is not None
    assert brief.stability.n_resamples == 6
    assert brief.stability.n_solved > 0
    assert all(0.0 <= m.frequency <= 1.0 for m in brief.stability.moves)
    move = brief.moves[0]
    assert brief.stability_of(move) is not None
    # Every resample bought *some* replacement for the sold player; which
    # specific cheap replacement won a given resample can vary, so check the
    # aggregate rather than pinning a single (out, in) pair.
    sold_in_every_resample = sum(
        m.frequency for m in brief.stability.moves if m.move.out_element == move.out_element
    )
    assert sold_in_every_resample >= 0.5


def test_transfers_pair_by_position():
    def player(element, position):
        return Player(element, position, team=1, buy_price=50, sell_price=50,
                      points={1: 1.0}, name=f"p{element}")

    players = {e: player(e, p) for e, p in
               [(1, 2), (2, 3), (10, 3), (11, 2)]}
    decision = Decision(
        gameweek=1, squad=[], starting=[], captain=None, vice=None,
        transfers_in=[10, 11], transfers_out=[1, 2],
        free_transfers=0, hits=0, bank=0.0, expected_points=0.0,
    )
    moves = pair_transfers(decision, players)
    assert {(m.out_element, m.in_element) for m in moves} == {(1, 11), (2, 10)}
