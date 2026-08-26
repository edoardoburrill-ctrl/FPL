"""The Monte Carlo match engine.

Everything in `points.py` treats each scoring term as an independent
expectation. That is a good approximation for goals, assists, clean sheets and
conceded — each is genuinely close to independent of the others once the
fixture's goal expectation is fixed — but it breaks down for bonus, because
bonus is a **rank** within a match: whether a player's BPS is 3rd-highest
depends on what everyone else on the pitch did, not just on his own numbers.

This module draws whole match scenarios instead. For each of `n_sims` draws:

1. Draw a scoreline from the fixture's Dixon-Coles score matrix.
2. Draw each player's minutes from his `MinutesForecast`.
3. Allocate the scoreline's goals to scorers, and a fraction of them to
   assisters, by a minutes-weighted multinomial over each player's output share.
4. Draw defensive actions, saves and cards from Poisson rates scaled by minutes.
5. Compute BPS for every player in the match from those draws.
6. Rank all of them — both teams together, since bonus is match-wide — and
   award bonus with FPL's exact tie rule (`models.bps.award_bonus`).
7. Score the resulting stat line with the real scoring rules.

The output is a sample of total points per player per draw, which is where the
payoff is: the mean recovers what the analytic path already computes (a useful
cross-check), but the samples also give a **haul probability**
(`P(points >= threshold)`) and, because the whole match is drawn jointly,
correlations between teammates — neither of which an independent-expectation
model can produce honestly.

This is deliberately the most expensive part of the system. It is not what the
optimiser prunes hundreds of players against; it is what `fpl captain` and
`fpl explain` call for a shortlist, where the extra minute of compute buys a
real answer to "how much of a coin-flip is this armband?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from fpl.models.bps import BPSTable, DEFAULT_TABLE, award_bonus, partial_bps
from fpl.models.minutes import MinutesForecast
from fpl.models.points import PlayerRates
from fpl.models.scoring import GK, ScoringRules, score_stat_line
from fpl.models.team_strength import DEFAULT_RHO, score_matrix

MAX_GOALS = 10


@dataclass(frozen=True)
class SimPlayer:
    """One player's inputs to the simulation — the same shape as the analytic
    path uses, so results are directly comparable."""

    element: int
    position: int
    is_home: bool
    minutes: MinutesForecast
    rates: PlayerRates


@dataclass
class PlayerSimulation:
    element: int
    samples: np.ndarray  # points, one entry per draw

    @property
    def mean(self) -> float:
        return float(self.samples.mean())

    @property
    def std(self) -> float:
        return float(self.samples.std())

    def haul_probability(self, threshold: float = 10.0) -> float:
        return float(np.mean(self.samples >= threshold))

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.samples, q))


@dataclass
class FixtureSimulation:
    players: dict[int, PlayerSimulation] = field(default_factory=dict)
    n_sims: int = 0

    def correlation(self, a: int, b: int) -> float:
        """Correlation between two players' points across draws.

        This is the thing an independent-expectation model cannot give you at
        all: it is why stacking three players from the same match is not three
        independent bets, and why a captain pick and a differential pick carry
        different risk even at equal expected points.
        """
        if a not in self.players or b not in self.players:
            return 0.0
        sa, sb = self.players[a].samples, self.players[b].samples
        if sa.std() == 0 or sb.std() == 0:
            return 0.0
        return float(np.corrcoef(sa, sb)[0, 1])


def _sample_minutes(
    forecast: MinutesForecast, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw minutes-played values consistent with a MinutesForecast.

    The forecast gives us `p_out`, `p_1_to_59`, `p_60plus` and an overall
    `expected_minutes`; it does not give a full distribution. We sample a
    category from those three probabilities, then a within-category minutes
    value whose mean is chosen so the overall sample mean matches
    `expected_minutes` — the one quantity the forecast is most confident about.
    """
    p_out = max(0.0, forecast.p_out)
    p_short = max(0.0, forecast.p_1_to_59)
    p_full = max(0.0, forecast.p_60plus)
    total = p_out + p_short + p_full
    if total <= 0:
        return np.zeros(n)
    probs = np.array([p_out, p_short, p_full]) / total
    category = rng.choice(3, size=n, p=probs)

    # The short-appearance mean is a fixed assumption (mirroring the general
    # cameo/early-substitution territory minutes.py's own priors use). The
    # full-match mean is *solved for*, not assumed, so that the overall sample
    # mean reproduces `expected_minutes` — the one number the forecast is most
    # confident about, since it is not itself split by category.
    short_mean = 28.0
    p_full_norm = p_full / total
    p_short_norm = p_short / total
    if p_full_norm > 1e-9:
        full_mean = (forecast.expected_minutes - p_short_norm * short_mean) / p_full_norm
        full_mean = float(np.clip(full_mean, 60.0, 90.0))
    else:
        full_mean = 85.0

    minutes = np.zeros(n)
    short_mask = category == 1
    full_mask = category == 2
    if short_mask.any():
        minutes[short_mask] = rng.uniform(1, 59, size=int(short_mask.sum()))
    if full_mask.any():
        spread = min(8.0, (full_mean - 60.0) / 2, (90.0 - full_mean) / 2 + 1)
        minutes[full_mask] = np.clip(
            rng.normal(full_mean, max(spread, 1.0), size=int(full_mask.sum())), 60, 90
        )
    return minutes


def simulate_fixture(
    home: Sequence[SimPlayer],
    away: Sequence[SimPlayer],
    lam_home: float,
    lam_away: float,
    rules: ScoringRules,
    *,
    n_sims: int = 10_000,
    rho: float = DEFAULT_RHO,
    bps_table: BPSTable = DEFAULT_TABLE,
    league_average_conceded: float = 1.4,
    rng: np.random.Generator | None = None,
) -> FixtureSimulation:
    """Simulate one fixture end to end and return per-player point samples."""
    rng = rng or np.random.default_rng()
    all_players = list(home) + list(away)
    if not all_players:
        return FixtureSimulation({}, n_sims)

    matrix = score_matrix(lam_home, lam_away, rho)
    flat = matrix.flatten()
    outcomes = rng.choice(len(flat), size=n_sims, p=flat)
    home_goals = outcomes // matrix.shape[1]
    away_goals = outcomes % matrix.shape[1]

    minutes_samples = {
        p.element: _sample_minutes(p.minutes, n_sims, rng) for p in all_players
    }
    minutes_fraction = {
        element: samples / 90.0 for element, samples in minutes_samples.items()
    }

    stats: dict[int, dict[str, np.ndarray]] = {
        p.element: {
            "minutes": minutes_samples[p.element],
            "goals_scored": np.zeros(n_sims),
            "assists": np.zeros(n_sims),
            "clean_sheets": np.zeros(n_sims),
            "goals_conceded": np.zeros(n_sims),
            "saves": np.zeros(n_sims),
            "yellow_cards": np.zeros(n_sims),
            "red_cards": np.zeros(n_sims),
        }
        for p in all_players
    }

    for side_players, side_goals, opp_goals, is_home in (
        (home, home_goals, away_goals, True),
        (away, away_goals, home_goals, False),
    ):
        conceded = opp_goals  # goals scored by the other side = conceded by this one
        for p in side_players:
            stats[p.element]["goals_conceded"] = conceded
            stats[p.element]["clean_sheets"] = (conceded == 0).astype(float)

        _allocate_goals(side_players, side_goals, minutes_fraction, stats, rng)
        _draw_saves(side_players, opp_goals, minutes_fraction, stats,
                    league_average_conceded, rng)
        _draw_cards(side_players, minutes_fraction, stats, rng)

    bps_samples = {
        p.element: np.array([
            _bps_for_draw(stats[p.element], i, p.position, bps_table)
            for i in range(n_sims)
        ])
        for p in all_players
    }

    points_samples = {p.element: np.zeros(n_sims) for p in all_players}
    for i in range(n_sims):
        bonus = award_bonus({p.element: int(bps_samples[p.element][i]) for p in all_players})
        for p in all_players:
            line = {k: v[i] for k, v in stats[p.element].items()}
            line["bonus"] = bonus[p.element]
            points_samples[p.element][i] = score_stat_line(line, p.position, rules)

    return FixtureSimulation(
        players={
            p.element: PlayerSimulation(p.element, points_samples[p.element])
            for p in all_players
        },
        n_sims=n_sims,
    )


def _allocate_goals(
    players: Sequence[SimPlayer],
    team_goals: np.ndarray,
    minutes_fraction: Mapping[int, np.ndarray],
    stats: dict[int, dict[str, np.ndarray]],
    rng: np.random.Generator,
) -> None:
    """Assign each of the team's goals to a scorer, and most to an assister.

    Weights are each player's output share scaled by his minutes fraction *in
    that draw*, renormalised per draw — a player who was subbed off in this
    particular scenario cannot score in it, which a static weighting would miss.
    """
    n = len(team_goals)
    if not players:
        return

    goal_weights = np.array([[max(p.rates.goals_share, 1e-6) for p in players]] * n)
    assist_weights = np.array([[max(p.rates.assists_share, 1e-6) for p in players]] * n)
    fractions = np.array([minutes_fraction[p.element] for p in players]).T  # n x k

    goal_weights = goal_weights * fractions
    assist_weights = assist_weights * fractions

    max_goals = int(team_goals.max()) if n else 0
    for g in range(max_goals):
        active = team_goals > g
        if not active.any():
            continue
        idx = np.where(active)[0]
        gw = goal_weights[idx]
        row_sums = gw.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        probs = gw / row_sums
        scorer = _batched_choice(probs, rng)
        for local_i, draw_i in enumerate(idx):
            stats[players[scorer[local_i]].element]["goals_scored"][draw_i] += 1

        # ~72% of goals in the Premier League have a credited assist; the rest
        # are unassisted (solo runs, deflections, direct free kicks).
        assisted = rng.random(len(idx)) < 0.72
        aw = assist_weights[idx]
        # A goal is never self-assisted.
        for local_i in range(len(idx)):
            aw[local_i, scorer[local_i]] = 0.0
        a_row_sums = aw.sum(axis=1, keepdims=True)
        a_row_sums[a_row_sums <= 0] = 1.0
        a_probs = aw / a_row_sums
        assister = _batched_choice(a_probs, rng)
        for local_i, draw_i in enumerate(idx):
            if assisted[local_i]:
                stats[players[assister[local_i]].element]["assists"][draw_i] += 1


def _batched_choice(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorised categorical draw: one choice per row of a probability matrix.

    Floating-point rounding can leave a row's cumulative sum fractionally below
    1.0, in which case a draw right at the top of the range would otherwise
    count past the last column and produce an out-of-bounds index — this
    happened in testing. Clip to the last valid column rather than let it.
    """
    cumulative = np.cumsum(probs, axis=1)
    draws = rng.random(size=(probs.shape[0], 1))
    index = (draws > cumulative).sum(axis=1)
    return np.clip(index, 0, probs.shape[1] - 1)


def _draw_saves(
    players: Sequence[SimPlayer],
    opponent_goals: np.ndarray,
    minutes_fraction: Mapping[int, np.ndarray],
    stats: dict[int, dict[str, np.ndarray]],
    league_average_conceded: float,
    rng: np.random.Generator,
) -> None:
    for p in players:
        if p.position != GK or p.rates.saves_per_90 <= 0:
            continue
        intensity = np.clip(opponent_goals, 0, None) / max(league_average_conceded, 1e-6)
        intensity = np.clip(intensity, 0.2, 4.0)  # a shutout does not imply zero shots
        lam = p.rates.saves_per_90 * minutes_fraction[p.element] * intensity
        stats[p.element]["saves"] = rng.poisson(np.clip(lam, 0, None))


def _draw_cards(
    players: Sequence[SimPlayer],
    minutes_fraction: Mapping[int, np.ndarray],
    stats: dict[int, dict[str, np.ndarray]],
    rng: np.random.Generator,
) -> None:
    for p in players:
        fraction = minutes_fraction[p.element]
        if p.rates.yellow_per_90:
            lam = p.rates.yellow_per_90 * fraction
            stats[p.element]["yellow_cards"] = rng.poisson(np.clip(lam, 0, None))
        if p.rates.red_per_90:
            lam = p.rates.red_per_90 * fraction
            stats[p.element]["red_cards"] = rng.poisson(np.clip(lam, 0, None))


def _bps_for_draw(
    player_stats: Mapping[str, np.ndarray], i: int, position: int, table: BPSTable
) -> int:
    line = {k: v[i] for k, v in player_stats.items()}
    return partial_bps(line, position, table)
