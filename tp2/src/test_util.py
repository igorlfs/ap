import numpy as np
from numpy.testing import assert_array_almost_equal

from src.dataset import DataSet
from src.util import Stump, boosting, get_predictions


def test_get_predictions():
    s1 = Stump([2, 9], "Sparkly = Y", "")
    s1.alpha = 0.2
    labels = np.array([1, 1, 1, 1, 1, 1, 1, -1, -1, -1])

    actual = get_predictions([s1], labels)
    expected = np.array([1, 1, -1, 1, 1, 1, 1, -1, -1, 1])

    assert np.array_equal(actual, expected)


def test_boosting():
    dataset = DataSet("Vampire", ["Y", "N"], "N", "Y", "sample.csv")
    stumps = dataset.gen_stumps(dataset.df)

    boost = boosting(2, stumps, dataset.y)

    actual_names = []
    for stump in boost:
        actual_names.append(stump.name)

    expected_names = ["Sparkly == Y", "Evil == Y"]

    assert actual_names == expected_names

    actual_alphas = []
    for stump in boost:
        actual_alphas.append(stump.alpha)

    # TODO: fazer a conta manual para o n=3
    expected_alphas = 0.5 * np.array([np.log(4), np.log(3)])

    assert_array_almost_equal(actual_alphas, expected_alphas)
