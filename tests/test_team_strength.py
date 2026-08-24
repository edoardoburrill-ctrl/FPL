import numpy as np

from fpl.models.devig import devig
from fpl.models.team_strength import (
    clean_sheet_probability,
    conceded_distribution,
    fit_ratings,
    outcome_probabilities,
    score_matrix,
    solve_from_market,
    total_goals_over,
    market_weight,
    blend_goal_rates,
)


def test_score_matrix_is_a_distribution():
    matrix = score_matrix(1.6, 1.1)
    assert np.isclose(matrix.sum(), 1.0)
    assert (matrix >= 0).all()


def test_outcomes_sum_to_one():
    home, draw, away = outcome_probabilities(score_matrix(1.6, 1.1))
    assert np.isclose(home + draw + away, 1.0)


def test_dixon_coles_lifts_draws_relative_to_independence():
    """The whole reason for the correction: low scores are not independent."""
    corrected = score_matrix(1.4, 1.2, rho=-0.08)
    independent = score_matrix(1.4, 1.2, rho=0.0)
    assert corrected[0, 0] > independent[0, 0]
    assert corrected[1, 0] < independent[1, 0]


def test_market_inversion_recovers_the_market():
    fair = devig([1.55, 4.40, 6.50])
    solution = solve_from_market(fair[0], fair[1], fair[2])
    recovered = outcome_probabilities(
        score_matrix(solution.lam_home, solution.lam_away, solution.rho)
    )
    assert np.allclose(recovered, fair, atol=0.01)


def test_stronger_favourite_gets_more_goals():
    strong = solve_from_market(*devig([1.25, 6.0, 12.0]))
    weak = solve_from_market(*devig([2.60, 3.40, 2.70]))
    assert strong.supremacy > weak.supremacy


def test_totals_market_pins_down_the_total():
    fair = devig([2.10, 3.50, 3.60])
    low = solve_from_market(fair[0], fair[1], fair[2], p_over=0.40, over_weight=5.0)
    high = solve_from_market(fair[0], fair[1], fair[2], p_over=0.70, over_weight=5.0)
    assert high.total > low.total
    assert total_goals_over(score_matrix(low.lam_home, low.lam_away)) < total_goals_over(
        score_matrix(high.lam_home, high.lam_away)
    )


def test_clean_sheet_falls_as_the_opponent_improves():
    weak_opponent = clean_sheet_probability(score_matrix(1.6, 0.7), home=True)
    strong_opponent = clean_sheet_probability(score_matrix(1.6, 2.0), home=True)
    assert weak_opponent > strong_opponent


def test_conceded_distribution_is_normalised():
    matrix = score_matrix(1.5, 1.2)
    for home in (True, False):
        assert np.isclose(conceded_distribution(matrix, home).sum(), 1.0)


def test_ratings_recover_known_team_strength():
    """Fit against a league generated from known attack/defence parameters."""
    rng = np.random.default_rng(3)
    n_teams, attack_true = 20, np.linspace(0.45, -0.45, 20)
    matches = []
    day = 0
    for _ in range(12):  # 12 rounds gives a realistic mid-season sample
        order = rng.permutation(n_teams)
        for i in range(0, n_teams, 2):
            h, a = int(order[i]), int(order[i + 1])
            lam = np.exp(0.25 + attack_true[h] - (-attack_true[a]) * 0.5 + 0.2)
            mu = np.exp(0.25 + attack_true[a] - (-attack_true[h]) * 0.5)
            matches.append(
                {
                    "home": h, "away": a, "day": float(day),
                    "home_goals": int(rng.poisson(lam)),
                    "away_goals": int(rng.poisson(mu)),
                }
            )
        day += 1

    ratings = fit_ratings(matches, half_life_days=20.0, prior_strength=1.0)
    fitted = np.array([ratings.attack[t] for t in range(n_teams)])
    correlation = np.corrcoef(fitted, attack_true)[0, 1]
    assert correlation > 0.8, f"attack ratings barely correlate with truth ({correlation:.2f})"
    assert ratings.home_advantage > 0


def test_market_weight_decays_with_horizon():
    assert market_weight(0) > market_weight(1) > market_weight(4)


def test_blend_falls_back_to_model_without_odds():
    assert blend_goal_rates(None, (1.4, 1.1)) == (1.4, 1.1)
