"""Fixture-level goal expectations.

Every player projection ultimately hangs off two numbers per fixture: the
expected goals of each team, ``(lam_home, lam_away)``. We get them two ways.

**Primary: the betting market.** Bookmakers price in team news, lineup leaks,
motivation and squad rotation faster and better than any model fitted on 380
matches a season. Given de-vigged 1X2 and Over/Under probabilities we invert a
Dixon-Coles bivariate Poisson to recover the goal rates the market is implying.

**Fallback: a fitted ratings model.** For gameweeks whose odds are not yet
posted, a time-decayed weighted-MLE Dixon-Coles on match results, shrunk toward
prior-season ratings so that three matches of noise cannot dominate in August.
Attack and defence are the two axes that carry opposition-specific strength —
every fixture's goal expectation already depends on precisely who the
opponent is, home or away, not on a league-average figure. A team-specific
home-advantage deviation (``TeamRatings.home_delta``, see below) is fit on top
of that, for the sides whose home and away form genuinely differ rather than
mirroring a single league-wide split.

The two are blended with weight shifting to the market as a fixture approaches.

The Dixon-Coles correction matters: real football is not independent Poisson at
low scores. 0-0 and 1-1 are more common than independence implies, and 1-0/0-1
less so. Since FPL pays 4 points for a clean sheet, getting the 0-0 and 1-0
corner of the distribution right is worth real points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 12
DEFAULT_RHO = -0.05  # typical fitted value for the Premier League


def _tau(matrix_shape: tuple[int, int], lam: float, mu: float, rho: float) -> np.ndarray:
    """Dixon-Coles low-score correction, as a multiplicative mask."""
    tau = np.ones(matrix_shape)
    tau[0, 0] = 1.0 - lam * mu * rho
    tau[0, 1] = 1.0 + lam * rho
    tau[1, 0] = 1.0 + mu * rho
    tau[1, 1] = 1.0 - rho
    return tau


def score_matrix(
    lam: float, mu: float, rho: float = DEFAULT_RHO, max_goals: int = MAX_GOALS
) -> np.ndarray:
    """Joint distribution of (home goals, away goals).

    Row index is home goals, column index is away goals.
    """
    if lam <= 0 or mu <= 0:
        raise ValueError("goal rates must be positive")
    goals = np.arange(max_goals + 1)
    home = poisson.pmf(goals, lam)
    away = poisson.pmf(goals, mu)
    matrix = np.outer(home, away)
    matrix *= _tau(matrix.shape, lam, mu, rho)
    matrix = np.clip(matrix, 0.0, None)
    return matrix / matrix.sum()


def outcome_probabilities(matrix: np.ndarray) -> tuple[float, float, float]:
    """(home win, draw, away win)."""
    home = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    return home, draw, away


def total_goals_over(matrix: np.ndarray, line: float = 2.5) -> float:
    """P(total goals > line)."""
    size = matrix.shape[0]
    totals = np.add.outer(np.arange(size), np.arange(size))
    return float(matrix[totals > line].sum())


def conceded_distribution(matrix: np.ndarray, home: bool) -> np.ndarray:
    """Marginal distribution of goals conceded by one side."""
    return matrix.sum(axis=0) if home else matrix.sum(axis=1)


def clean_sheet_probability(matrix: np.ndarray, home: bool) -> float:
    """P(this side concedes zero)."""
    return float(conceded_distribution(matrix, home)[0])


def _kl(target: np.ndarray, model: np.ndarray) -> float:
    model = np.clip(model, 1e-12, 1.0)
    target = np.clip(target, 1e-12, 1.0)
    return float(np.sum(target * np.log(target / model)))


@dataclass(frozen=True)
class MarketSolution:
    lam_home: float
    lam_away: float
    rho: float
    divergence: float

    @property
    def total(self) -> float:
        return self.lam_home + self.lam_away

    @property
    def supremacy(self) -> float:
        """Expected home goal margin — the market's view of who is better."""
        return self.lam_home - self.lam_away


def solve_from_market(
    p_home: float,
    p_draw: float,
    p_away: float,
    p_over: float | None = None,
    line: float = 2.5,
    rho: float = DEFAULT_RHO,
    over_weight: float = 1.0,
    max_goals: int = MAX_GOALS,
) -> MarketSolution:
    """Recover ``(lam_home, lam_away)`` from de-vigged market probabilities.

    With only 1X2 the system has two degrees of freedom and two informative
    constraints, so it is essentially exactly identified. Adding an Over/Under
    line over-determines it, which pins down the *total* far more tightly — worth
    including whenever the line is available.
    """
    target_outcomes = np.array([p_home, p_draw, p_away], dtype=float)
    target_outcomes = target_outcomes / target_outcomes.sum()

    def objective(theta: np.ndarray) -> float:
        lam, mu = np.exp(theta)
        matrix = score_matrix(lam, mu, rho, max_goals)
        loss = _kl(target_outcomes, np.array(outcome_probabilities(matrix)))
        if p_over is not None:
            over = total_goals_over(matrix, line)
            loss += over_weight * _kl(
                np.array([p_over, 1.0 - p_over]), np.array([over, 1.0 - over])
            )
        return loss

    # Start from a sensible Premier League prior (1.55 home, 1.25 away).
    start = np.log(np.array([1.55, 1.25]))
    result = minimize(objective, start, method="Nelder-Mead",
                      options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 2000})
    lam, mu = np.exp(result.x)
    return MarketSolution(float(lam), float(mu), rho, float(result.fun))


@dataclass
class TeamRatings:
    """Dixon-Coles attack/defence ratings, fitted by time-decayed weighted MLE.

    ``lam_home = exp(mu + attack[i] - defence[j] + home_advantage + home_delta[i])``
    ``lam_away = exp(mu + attack[j] - defence[i])``

    ``home_advantage`` is the league-wide home effect. ``home_delta`` is how
    much team *i* deviates from it — some sides are a genuine fortress at
    home and toothless away (or vice versa), which a single global figure
    cannot express. It is fitted with its own, stronger ridge penalty (see
    ``fit_ratings``): a team-specific home/away split is a real but usually
    small effect, estimated off a fraction of the matches attack/defence get
    (only that team's home games, not all of them), so it is shrunk harder
    toward zero to stop a small sample from being read as a fortress.
    """

    teams: list[int]
    attack: dict[int, float]
    defence: dict[int, float]
    home_advantage: float
    baseline: float
    rho: float = DEFAULT_RHO
    home_delta: dict[int, float] = field(default_factory=dict)

    def team_home_advantage(self, team: int) -> float:
        return self.home_advantage + self.home_delta.get(team, 0.0)

    def expected_goals(self, home_team: int, away_team: int) -> tuple[float, float]:
        lam = np.exp(
            self.baseline
            + self.attack.get(home_team, 0.0)
            - self.defence.get(away_team, 0.0)
            + self.team_home_advantage(home_team)
        )
        mu = np.exp(
            self.baseline
            + self.attack.get(away_team, 0.0)
            - self.defence.get(home_team, 0.0)
        )
        return float(lam), float(mu)


def fit_ratings(
    matches: list[dict],
    half_life_days: float = 180.0,
    as_of_day: float | None = None,
    prior_strength: float = 4.0,
    home_delta_prior_strength: float | None = None,
    rho: float = DEFAULT_RHO,
) -> TeamRatings:
    """Fit ratings to ``matches``.

    Each match is ``{"home": id, "away": id, "home_goals": int,
    "away_goals": int, "day": float}`` where ``day`` is a day index (larger =
    more recent).

    ``prior_strength`` is a ridge penalty pulling attack and defence toward zero
    (league average). Early in a season, when every team has three matches, this
    is what stops a 5-0 opening win from producing an absurd rating. It is the
    regularisation that makes the fallback usable in August at all.

    ``home_delta_prior_strength`` penalises each team's deviation from the
    league-wide home advantage — a "some teams are a fortress at home"
    effect. It defaults to eight times ``prior_strength``: attack and defence
    are each fit off a team's full match list (home and away), but a team's
    home-advantage deviation is fit off only half of that, its home games,
    and — unlike attack/defence, where every team's rating is pinned down by
    both its own matches and its opponents' — a home delta only ever gets
    signal from the team's own home games, so it needs the harder penalty to
    avoid soaking up noise that belongs to attack. (Confirmed empirically:
    4x measurably degraded `test_ratings_recover_known_team_strength`'s
    attack-recovery correlation; 8x does not.)
    """
    if not matches:
        raise ValueError("no matches to fit")
    if home_delta_prior_strength is None:
        home_delta_prior_strength = 8.0 * prior_strength

    teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)

    days = np.array([m["day"] for m in matches], dtype=float)
    as_of = float(as_of_day if as_of_day is not None else days.max())
    decay = np.log(2.0) / half_life_days
    weights = np.exp(-decay * (as_of - days))

    home_idx = np.array([index[m["home"]] for m in matches])
    away_idx = np.array([index[m["away"]] for m in matches])
    home_goals = np.array([m["home_goals"] for m in matches], dtype=float)
    away_goals = np.array([m["away_goals"] for m in matches], dtype=float)

    def unpack(theta: np.ndarray):
        attack = theta[:n]
        defence = theta[n : 2 * n]
        home_delta = theta[2 * n : 3 * n]
        home_adv, baseline = theta[3 * n], theta[3 * n + 1]
        # Identifiability: attack, defence and the home-delta are each only
        # defined up to a constant, which the scalar home_advantage/baseline
        # terms absorb.
        attack = attack - attack.mean()
        defence = defence - defence.mean()
        home_delta = home_delta - home_delta.mean()
        return attack, defence, home_delta, home_adv, baseline

    def negative_log_likelihood(theta: np.ndarray) -> float:
        attack, defence, home_delta, home_adv, baseline = unpack(theta)
        lam = np.exp(
            baseline + attack[home_idx] - defence[away_idx] + home_adv + home_delta[home_idx]
        )
        mu = np.exp(baseline + attack[away_idx] - defence[home_idx])
        lam = np.clip(lam, 1e-6, 20.0)
        mu = np.clip(mu, 1e-6, 20.0)

        ll = poisson.logpmf(home_goals, lam) + poisson.logpmf(away_goals, mu)

        # Dixon-Coles correction on the four low-score cells.
        tau = np.ones_like(lam)
        both_zero = (home_goals == 0) & (away_goals == 0)
        tau[both_zero] = 1.0 - lam[both_zero] * mu[both_zero] * rho
        h0a1 = (home_goals == 0) & (away_goals == 1)
        tau[h0a1] = 1.0 + lam[h0a1] * rho
        h1a0 = (home_goals == 1) & (away_goals == 0)
        tau[h1a0] = 1.0 + mu[h1a0] * rho
        both_one = (home_goals == 1) & (away_goals == 1)
        tau[both_one] = 1.0 - rho
        ll = ll + np.log(np.clip(tau, 1e-9, None))

        penalty = prior_strength * (np.sum(attack**2) + np.sum(defence**2))
        penalty += home_delta_prior_strength * np.sum(home_delta**2)
        return float(-np.sum(weights * ll) + penalty)

    start = np.zeros(3 * n + 2)
    start[3 * n] = 0.25  # home advantage
    start[3 * n + 1] = np.log(1.4)  # baseline goals
    result = minimize(negative_log_likelihood, start, method="L-BFGS-B")
    attack, defence, home_delta, home_adv, baseline = unpack(result.x)

    return TeamRatings(
        teams=teams,
        attack={t: float(attack[index[t]]) for t in teams},
        defence={t: float(defence[index[t]]) for t in teams},
        home_advantage=float(home_adv),
        baseline=float(baseline),
        rho=rho,
        home_delta={t: float(home_delta[index[t]]) for t in teams},
    )


def market_weight(gameweeks_ahead: int, weight_now: float = 0.85, decay: float = 0.45) -> float:
    """How much to trust the market versus the fitted model.

    Odds for the imminent gameweek are sharp and current. Odds for gameweeks
    further out (where they exist at all) are stale relative to team news, so the
    fitted ratings deserve more weight.
    """
    return float(weight_now * np.exp(-decay * max(0, gameweeks_ahead)))


def blend_goal_rates(
    market: tuple[float, float] | None,
    model: tuple[float, float],
    gameweeks_ahead: int = 0,
    weight_now: float = 0.85,
) -> tuple[float, float]:
    if market is None:
        return model
    w = market_weight(gameweeks_ahead, weight_now)
    return (
        w * market[0] + (1 - w) * model[0],
        w * market[1] + (1 - w) * model[1],
    )
