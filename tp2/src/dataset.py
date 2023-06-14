import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.util import Stump, boosting


class DataSet:
    def __init__(  # noqa: PLR0913
        self, label_column: str, x_values: list[str], false: str, true: str, path: str
    ):
        self.label_column = label_column
        self.x_values = x_values
        self.false = false
        self.true = true

        self.df = pd.read_csv(path)
        self.y = (
            self.df[self.label_column]
            .apply(lambda x: -1 if x == self.false else 1)
            .to_numpy()
        )

    def cross_validation(self, num_folds: int, iterations: int):
        kfold = KFold(n_splits=num_folds, shuffle=True)
        errors = []

        for train_index, test_index in kfold.split(self.df):
            x_train, x_test = self.df.iloc[train_index], self.df.iloc[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            x_train = x_train.reset_index(drop=True)
            x_test = x_test.reset_index(drop=True)

            stumps = self.gen_stumps(x_train)

            boost = boosting(iterations, stumps, y_train)
            error = calculate_test_error(boost, x_test, y_test)
            errors.append(error)
        return errors

    def gen_stumps(self, x: pd.DataFrame):
        stumps: list[Stump] = []

        for column in x.columns:
            if column != self.label_column:
                for v in self.x_values:
                    query = f"({column} == '{v}' and {self.label_column} == '{self.false}') or ({column} != '{v}' and {self.label_column} == '{self.true}')"
                    fail_indexes = x.query(query).index.to_numpy()
                    stumps.append(Stump(fail_indexes, f"{column} == {v}", query))

        query_true = f"{self.label_column} == '{self.false}'"
        stumps.append(Stump(x.query(query_true).index.to_numpy(), "TRUE", query_true))
        query_false = f"{self.label_column} != '{self.false}'"
        stumps.append(Stump(x.query(query_false).index.to_numpy(), "FALSE", query_false))

        return stumps


def calculate_test_error(boost: list[Stump], x: pd.DataFrame, y: pd.DataFrame):
    fail_indexes = []
    for stump in boost:
        fail_indexes.append(x.query(stump.query).index.to_numpy())
    predictions_matrix = np.zeros((len(y), np.size(boost)))

    for i, s in enumerate(boost):
        for j in range(len(y)):
            predictions_matrix[j][i] = y[j] * s.alpha

    for i in range(len(boost)):
        predictions_matrix[fail_indexes[i], i] *= -1

    predictions = np.array(
        [-1 if i == -1 else 1 for i in np.sign(np.sum(predictions_matrix, axis=1))]
    )

    error_count = 0
    for i in range(len(y)):
        if predictions[i] != y[i]:
            error_count += 1
    return error_count / len(y)
