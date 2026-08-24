"""Removing the bookmaker margin from odds.

Naive normalisation (divide each implied probability by their sum) assumes the
margin is spread proportionally. It is not: bookmakers load more margin onto
longshots, so proportional de-vigging systematically *overstates* unlikely
outcomes. That bias lands directly on clean-sheet and low-scoring estimates,
which are exactly the quantities FPL defenders live on.

Shin's method models the margin as arising from a proportion ``z`` of
insider-informed money and backs it out, which corrects the longshot bias. It is
the default here; the alternatives are kept for comparison and as fallbacks.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import brentq


def implied_probabilities(odds: Sequence[float]) -> np.ndarray:
    """Decimal odds -> raw (over-round) implied probabilities."""
    arr = np.asarray(odds, dtype=float)
    if np.any(arr <= 1.0):
        raise ValueError("decimal odds must all exceed 1.0")
    return 1.0 / arr


def multiplicative(book: Sequence[float]) -> np.ndarray:
    """Proportional normalisation. Simple, biased, useful as a baseline."""
    arr = np.asarray(book, dtype=float)
    return arr / arr.sum()


def power(book: Sequence[float], tol: float = 1e-10) -> np.ndarray:
    """Solve for k with sum(p_i^k) = 1. Corrects longshot bias more gently."""
    arr = np.asarray(book, dtype=float)

    def excess(k: float) -> float:
        return float(np.sum(arr**k) - 1.0)

    if abs(excess(1.0)) < tol:
        return arr.copy()
    k = brentq(excess, 0.5, 3.0, xtol=tol)
    return arr**k


def shin(book: Sequence[float], tol: float = 1e-10) -> np.ndarray:
    """Shin (1993) de-vigging.

    Returns fair probabilities summing to 1. ``book`` are the raw implied
    probabilities (which sum to more than 1).
    """
    arr = np.asarray(book, dtype=float)
    total = arr.sum()
    if total <= 1.0 + tol:
        # No margin to remove (or a rounding artefact); just normalise.
        return arr / total

    def p_of_z(z: float) -> np.ndarray:
        root = np.sqrt(z * z + 4.0 * (1.0 - z) * arr * arr / total)
        return (root - z) / (2.0 * (1.0 - z))

    def excess(z: float) -> float:
        return float(p_of_z(z).sum() - 1.0)

    lo, hi = 1e-12, 0.9
    if excess(lo) * excess(hi) > 0:
        # Shin has no root in range (pathological book); fall back to power.
        return power(arr, tol=tol)

    z = brentq(excess, lo, hi, xtol=tol)
    fair = p_of_z(z)
    return fair / fair.sum()


def devig(odds: Sequence[float], method: str = "shin") -> np.ndarray:
    """Decimal odds -> fair probabilities."""
    book = implied_probabilities(odds)
    if method == "shin":
        return shin(book)
    if method == "power":
        return power(book)
    if method == "multiplicative":
        return multiplicative(book)
    raise ValueError(f"unknown de-vig method {method!r}")


def overround(odds: Sequence[float]) -> float:
    """Bookmaker margin, e.g. 0.05 for a 5% book."""
    return float(implied_probabilities(odds).sum() - 1.0)
