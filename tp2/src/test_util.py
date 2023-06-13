import numpy as np

from src.util import Stump, calculate_error, get_predictions


def test_get_predictions():
    s1 = Stump([2, 9], "Sparkly = Y", "")
    s1.alpha = 0.2
    labels = np.array([1, 1, 1, 1, 1, 1, 1, -1, -1, -1])

    actual = get_predictions([s1], labels)
    expected = np.array([1, 1, -1, 1, 1, 1, 1, -1, -1, 1])

    assert np.array_equal(actual, expected)


def test_calculate_error():
    s1 = Stump([2, 9], "Sparkly = Y", "")
    s1.alpha = 0.2
    labels = np.array([1, 1, 1, 1, 1, 1, 1, -1, -1, -1])

    actual = calculate_error([s1], labels)
    expected = 0.2

    assert np.array_equal(actual, expected)
