import numpy as np
import pandas as pd
from src.util import Boost, Stump, calculate_error, split_dataset


class DataSet:
    def __init__(self, label: str, values: list[str], false: str, true: str, path: str):
        self.label = label
        self.values = values
        self.false = false
        self.true = true
        # Ignore o ID dos vampiros
        df = pd.read_csv(path, usecols=lambda x: x != "id")
        # Shuffle
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.x_train, self.x_test = split_dataset(self.df)
        self.y_train = (
            self.x_train[self.label]
            .apply(lambda x: -1 if x == self.false else 1)
            .to_numpy()
        )
        self.y_test = (
            self.x_test[self.label]
            .apply(lambda x: -1 if x == self.false else 1)
            .to_numpy()
        )
        self.y = (
            self.df[self.label].apply(lambda x: -1 if x == self.false else 1).to_numpy()
        )
        self.weights = np.full(len(self.x_train), 1 / len(self.x_train))
        self._gen_stumps()

    def _gen_stumps(self):
        stumps: list[Stump] = []
        for column in self.x_train.columns:
            if column != self.label:
                for v in self.values:
                    query = f"({column} == '{v}' and {self.label} == '{self.false}') or ({column} != '{v}' and {self.label} == '{self.true}')"
                    fail_indexes = self.x_train.query(query).index.to_numpy()
                    stumps.append(Stump(fail_indexes, f"{column} == {v}", query))

        QUERY_TRUE = f"{self.label} == '{self.false}'"
        stumps.append(
            Stump(self.x_train.query(QUERY_TRUE).index.to_numpy(), "TRUE", QUERY_TRUE)
        )
        QUERY_FALSE = f"{self.label} != '{self.false}'"
        stumps.append(
            Stump(
                self.x_train.query(QUERY_FALSE).index.to_numpy(), "FALSE", QUERY_FALSE
            )
        )

        self.stumps = stumps

    def calculate_test_error(self, boost: list[Stump]):
        new_stuff = []
        for stump in boost:
            new_stuff.append(self.x_test.query(stump.query).index.to_numpy())
        predictions_matrix = np.zeros((len(self.y_test), np.size(boost)))

        for i, s in enumerate(boost):
            assert s.alpha != np.inf
            for j in range(len(self.y_test)):
                predictions_matrix[j][i] = self.y_test[j] * s.alpha

        for i, s in enumerate(boost):
            predictions_matrix[new_stuff[i], i] *= -1

        predictions = np.array(
            [-1 if i == -1 else 1 for i in np.sign(np.sum(predictions_matrix, axis=1))]
        )

        error_count = 0
        for i in range(len(self.y_test)):
            if predictions[i] != self.y_test[i]:
                error_count += 1
        return error_count / len(self.y_test)

    def boost(self, iterations: int, verbose: bool = False):
        boost = Boost(self.weights, self.stumps, self.y_train)
        for _ in range(iterations):
            boost.iteration()
        if verbose:
            for stump in boost.boost:
                print(stump.name, stump.indexes)
        return boost
