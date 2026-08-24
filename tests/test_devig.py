import numpy as np
import pytest

from fpl.models.devig import devig, implied_probabilities, overround, shin


def test_fair_book_is_unchanged():
    fair = np.array([0.5, 0.3, 0.2])
    odds = 1.0 / fair
    assert np.allclose(devig(odds, "shin"), fair, atol=1e-6)


@pytest.mark.parametrize("method", ["shin", "power", "multiplicative"])
def test_probabilities_sum_to_one(method):
    assert np.isclose(devig([1.55, 4.40, 6.50], method).sum(), 1.0)


def test_shin_shrinks_longshots_relative_to_proportional():
    """The point of Shin's method: longshots carry more margin."""
    odds = [1.25, 6.5, 13.0]
    naive = devig(odds, "multiplicative")
    fair = devig(odds, "shin")
    assert fair[-1] < naive[-1]  # longshot down
    assert fair[0] > naive[0]  # favourite up


def test_overround_is_positive_for_a_real_book():
    assert overround([1.55, 4.40, 6.50]) > 0


def test_rejects_impossible_odds():
    with pytest.raises(ValueError):
        implied_probabilities([1.0, 2.0])
