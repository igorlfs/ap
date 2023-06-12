import numpy as np
from numpy.testing import assert_equal
from src.dataset import DataSet


def test_boost():
    SIZE = 4
    dataset = DataSet("Vampire", ["Y", "N"], "N", "Y", "sample.csv")

    expected = [dataset.boost(i, True) for i in range(SIZE)]
    actual = np.array([0.3, 0.2, 0.2, 0.1])

    assert_equal(actual, expected)
