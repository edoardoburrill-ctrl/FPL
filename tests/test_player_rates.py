import numpy as np

from fpl.models.player_rates import (
    GammaPrior,
    TeamContext,
    fit_rate_prior,
    output_share,
    team_contexts,
)


def test_prior_recovers_a_known_rate_distribution():
    rng = np.random.default_rng(11)
    true = rng.gamma(2.0, 0.15, size=400)
    nineties = rng.uniform(1, 30, size=400)
    events = rng.poisson(true * nineties)
    prior = fit_rate_prior(list(zip(events, nineties)))
    assert abs(prior.mean - true.mean()) < 0.06


def test_small_samples_are_shrunk_hard_and_large_ones_are_not():
    prior = GammaPrior(alpha=2.0, beta=7.0)  # mean 0.29 per 90
    hot_streak = prior.posterior_mean(events=3, nineties=2)
    established = prior.posterior_mean(events=15, nineties=30)
    assert hot_streak < 0.75, "three goals in two starts must not read as elite"
    assert abs(established - 0.5) < 0.1, "30 nineties of evidence should mostly stand"


def test_posterior_is_monotone_in_evidence():
    prior = GammaPrior(alpha=2.0, beta=7.0)
    rates = [prior.posterior_mean(events=e, nineties=10) for e in range(0, 12)]
    assert all(b > a for a, b in zip(rates, rates[1:]))


def test_prior_survives_degenerate_input():
    assert fit_rate_prior([]).mean > 0
    assert fit_rate_prior([(0.0, 5.0), (0.0, 9.0)]).mean >= 0


def test_output_share_scales_with_fixture_quality():
    share = output_share(0.5, TeamContext(expected_goals_per_match=1.5, matches=10))
    easy_fixture, hard_fixture = 2.4, 0.8
    assert share * easy_fixture > share * hard_fixture
    assert np.isclose(share, 1 / 3)


def test_team_context_shrinks_early_season_output():
    elements = [
        {"team": 1, "expected_goals": "9.0"},
        {"team": 2, "expected_goals": "1.0"},
    ]
    teams = [{"id": 1}, {"id": 2}]
    contexts = team_contexts(elements, teams, {1: 2, 2: 2})
    # Two matches is not evidence of a 4.5-goal-per-game team.
    assert contexts[1].expected_goals_per_match < 4.5
    assert contexts[2].expected_goals_per_match > 0.5
