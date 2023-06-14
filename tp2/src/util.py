import numpy as np


class Stump:
    def __init__(self, indexes: np.ndarray, name: str, query: str):
        self.train_fail_indexes = indexes
        self.name = name
        self.query = query
        self.alpha = np.inf

    def __str__(self):
        return f"{self.name}, ({len(self.train_fail_indexes)}) {self.train_fail_indexes}"

    def __repr__(self):
        return str(self)


def get_predictions(boost: list[Stump], y: np.ndarray) -> np.ndarray:
    predictions_matrix = np.zeros((len(y), np.size(boost)))

    for i, s in enumerate(boost):
        assert s.alpha != np.inf
        for j in range(len(y)):
            predictions_matrix[j][i] = y[j] * s.alpha

    for i in range(len(boost)):
        predictions_matrix[boost[i].train_fail_indexes, i] *= -1

    return np.array(
        [-1 if i == -1 else 1 for i in np.sign(np.sum(predictions_matrix, axis=1))]
    )


def boosting(iterations: int, stumps: list[Stump], y: np.ndarray):
    weights = np.full(len(y), 1 / len(y))
    stumps_copy = stumps.copy()
    boost: list[Stump] = []
    for _ in range(iterations):
        best_stump = min(stumps_copy, key=lambda s: np.sum(weights[s.train_fail_indexes]))

        error = np.sum(weights[best_stump.train_fail_indexes])
        alpha = 0.5 * (np.log(1 - error) - np.log(error))

        best_stump.alpha = alpha
        boost.append(best_stump)

        # update weights
        predictions = get_predictions([best_stump], y)
        exponent = -alpha * predictions * y
        weights = weights * np.exp(exponent)
        normalizer = np.sum(weights)
        weights = weights / normalizer

        stumps_copy.remove(best_stump)
    return boost
