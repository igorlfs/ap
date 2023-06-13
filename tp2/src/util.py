import numpy as np
import pandas as pd


class Stump:
    def __init__(self, indexes: np.ndarray, name: str, query: str):
        self.indexes = indexes
        self.name = name
        self.query = query
        self.alpha = np.inf

    def __str__(self):
        return f"{self.name}, ({len(self.indexes)}) {self.indexes}"

    def __repr__(self):
        return str(self)


class Boost:
    def __init__(self, weights: np.ndarray, stumps: list[Stump], y: np.ndarray):
        self.weights = weights.copy()
        self.stumps = stumps.copy()
        self.y = y
        self.boost: list[Stump] = []

    def iteration(self):
        best_stump = min(self.stumps, key=lambda s: np.sum(self.weights[s.indexes]))

        error = np.sum(self.weights[best_stump.indexes])
        alpha: float = 0.5 * (np.log(1 - error) - np.log(error))

        best_stump.alpha = alpha
        self.boost.append(best_stump)

        # update weights
        predictions = get_predictions([best_stump], self.y)
        exponent = -alpha * predictions * self.y
        self.weights = self.weights * np.exp(exponent)
        normalizer = np.sum(self.weights)
        self.weights = self.weights / normalizer

        self.stumps.remove(best_stump)
        return calculate_error(self.boost, self.y)


def get_predictions(boost: list[Stump], y: np.ndarray) -> np.ndarray:
    predictions_matrix = np.zeros((len(y), np.size(boost)))

    for i, s in enumerate(boost):
        assert s.alpha != np.inf
        for j in range(len(y)):
            predictions_matrix[j][i] = y[j] * s.alpha

    for i, s in enumerate(boost):
        predictions_matrix[boost[i].indexes, i] *= -1

    return np.array(
        [-1 if i == -1 else 1 for i in np.sign(np.sum(predictions_matrix, axis=1))]
    )


def calculate_error(boost: list[Stump], y: np.ndarray):
    predictions = get_predictions(boost, y)
    error_count = 0
    for i in range(len(y)):
        if predictions[i] != y[i]:
            error_count += 1
    return error_count / len(y)


def split_dataset(dataset: pd.DataFrame, test_ratio=0.30):
    """Splits a pandas dataframe in two."""
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices].reset_index(), dataset[test_indices].reset_index()
