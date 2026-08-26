"""Scoring the backtest: is the projection engine actually any good?

Two baselines, both real. `hold` — literally guessing the season average for
everyone — establishes the floor: any model that cannot beat it is worthless.
FPL's own `xP` column establishes the bar that matters, since it is the
figure PLAN.md named as the one to beat, and it is what every manager already
sees for free without this tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr


@dataclass
class GameweekResult:
    gameweek: int
    predicted: list[float] = field(default_factory=list)
    actual: list[float] = field(default_factory=list)
    xp_baseline: list[float] = field(default_factory=list)
    elements: list[int] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.actual)


def _mae(pred, actual) -> float:
    return float(np.mean(np.abs(np.array(pred) - np.array(actual))))


def _spearman(pred, actual) -> float:
    if len(pred) < 3 or np.std(pred) == 0 or np.std(actual) == 0:
        return float("nan")
    rho, _ = spearmanr(pred, actual)
    return float(rho)


@dataclass
class SeasonReport:
    season: str
    gameweeks: list[GameweekResult]

    @property
    def n(self) -> int:
        return sum(g.n for g in self.gameweeks)

    def _pooled(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred = np.concatenate([g.predicted for g in self.gameweeks]) if self.gameweeks else np.array([])
        actual = np.concatenate([g.actual for g in self.gameweeks]) if self.gameweeks else np.array([])
        xp = np.concatenate([g.xp_baseline for g in self.gameweeks]) if self.gameweeks else np.array([])
        return pred, actual, xp

    def mae(self) -> float:
        pred, actual, _ = self._pooled()
        return _mae(pred, actual)

    def mae_baseline(self) -> float:
        _, actual, xp = self._pooled()
        return _mae(xp, actual)

    def mae_naive(self) -> float:
        """A model with no fixtures, no minutes, no rates — just the season mean."""
        _, actual, _ = self._pooled()
        if len(actual) == 0:
            return float("nan")
        return _mae(np.full_like(actual, actual.mean()), actual)

    def spearman_per_gameweek(self) -> tuple[float, float]:
        """Mean Spearman correlation within each gameweek — ranking, not scale."""
        ours = [_spearman(g.predicted, g.actual) for g in self.gameweeks if g.n >= 5]
        theirs = [_spearman(g.xp_baseline, g.actual) for g in self.gameweeks if g.n >= 5]
        ours = [v for v in ours if not np.isnan(v)]
        theirs = [v for v in theirs if not np.isnan(v)]
        return (
            float(np.mean(ours)) if ours else float("nan"),
            float(np.mean(theirs)) if theirs else float("nan"),
        )

    def top_pick_points(self, min_start_pool: int = 3) -> tuple[float, float]:
        """If each gameweek you'd started only the #1 projected player, how many
        real points would you have captured — versus doing the same with xP?

        A direct, concrete stand-in for captaincy quality: it rewards a model
        for being right about *who is best*, not just accurate on average.
        """
        mine = theirs = 0.0
        for g in self.gameweeks:
            if g.n < min_start_pool:
                continue
            best_mine = g.elements[int(np.argmax(g.predicted))]
            best_theirs = g.elements[int(np.argmax(g.xp_baseline))]
            actual_by_element = dict(zip(g.elements, g.actual))
            mine += actual_by_element[best_mine]
            theirs += actual_by_element[best_theirs]
        return mine, theirs

    def summary(self) -> str:
        mae, base, naive = self.mae(), self.mae_baseline(), self.mae_naive()
        rho_mine, rho_theirs = self.spearman_per_gameweek()
        top_mine, top_theirs = self.top_pick_points()
        lines = [
            f"Season {self.season} — {len(self.gameweeks)} gameweeks, {self.n} player-gameweeks",
            f"  MAE          ours={mae:.3f}  FPL xP={base:.3f}  season-mean baseline={naive:.3f}",
            f"  Spearman rho ours={rho_mine:.3f}  FPL xP={rho_theirs:.3f}  (per-gameweek, averaged)",
            f"  Top-pick pts ours={top_mine:.0f}  FPL xP={top_theirs:.0f}  "
            f"(sum of real points from each gameweek's #1 projected player)",
        ]
        return "\n".join(lines)
