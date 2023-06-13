import numpy as np
import pandas as pd

from src.util import Boost, Stump, split_dataset


class DataSet:
    def __init__(
        self, label_column: str, x_values: list[str], false: str, true: str, path: str
    ):
        self.label_column = label_column
        self.x_values = x_values
        self.false = false
        self.true = true

        # Ignore o ID dos vampiros
        df = pd.read_csv(path, usecols=lambda x: x != "id")
        # Shuffle
        self.df = df.sample(frac=1).reset_index(drop=True)

        self.x_train, self.x_test = split_dataset(self.df)
        self.y_train = (
            self.x_train[self.label_column]
            .apply(lambda x: -1 if x == self.false else 1)
            .to_numpy()
        )
        self.y_test = (
            self.x_test[self.label_column]
            .apply(lambda x: -1 if x == self.false else 1)
            .to_numpy()
        )
        self.y = (
            self.df[self.label_column]
            .apply(lambda x: -1 if x == self.false else 1)
            .to_numpy()
        )
        self.weights = np.full(len(self.x_train), 1 / len(self.x_train))
        self._gen_stumps()

    def _gen_stumps(self):
        stumps: list[Stump] = []
        for column in self.x_train.columns:
            if column != self.label_column:
                for v in self.x_values:
                    query = f"({column} == '{v}' and {self.label_column} == '{self.false}') or ({column} != '{v}' and {self.label_column} == '{self.true}')"
                    fail_indexes = self.x_train.query(query).index.to_numpy()
                    stumps.append(Stump(fail_indexes, f"{column} == {v}", query))

        query_true = f"{self.label_column} == '{self.false}'"
        stumps.append(
            Stump(self.x_train.query(query_true).index.to_numpy(), "TRUE", query_true)
        )
        query_false = f"{self.label_column} != '{self.false}'"
        stumps.append(
            Stump(
                self.x_train.query(query_false).index.to_numpy(), "FALSE", query_false
            )
        )

        self.stumps = stumps

    def calculate_test_error(self, boost: list[Stump]):
        """Calcula o erro de teste com base na divisão do construtor."""
        new_stuff = []
        for stump in boost:
            new_stuff.append(self.x_test.query(stump.query).index.to_numpy())
        predictions_matrix = np.zeros((len(self.y_test), np.size(boost)))

        for i, s in enumerate(boost):
            for j in range(len(self.y_test)):
                predictions_matrix[j][i] = self.y_test[j] * s.alpha

        for i in range(len(boost)):
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
        """Roda um Boost por com `iterations`."""
        boost = Boost(self.weights, self.stumps, self.y_train)
        for _ in range(iterations):
            boost.iteration()
        if verbose:
            for stump in boost.boost:
                print(stump.name, stump.train_fail_indexes)
        return boost
