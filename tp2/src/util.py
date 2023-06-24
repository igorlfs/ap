import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


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


class DataSet:
    def __init__(self, label_column: str, negative: str, path: str):
        self.label_column = label_column

        self.df = pd.read_csv(path)
        self.df[label_column] = self.df[label_column].apply(
            lambda x: -1 if x == negative else 1
        )
        self.y = self.df[self.label_column].to_numpy()

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
                for v in x[column].unique():
                    query_true = f"(`{column}` == '{v}' and {self.label_column} == -1) or (`{column}` != '{v}' and {self.label_column} == 1)"
                    fail_indexes = x.query(query_true).index.to_numpy()
                    stumps.append(Stump(fail_indexes, f"`{column}` == {v}", query_true))

                    query_false = f"(`{column}` == '{v}' and {self.label_column} != -1) or (`{column}` != '{v}' and {self.label_column} != 1)"
                    fail_indexes = x.query(query_false).index.to_numpy()
                    stumps.append(Stump(fail_indexes, f"`{column}` != {v}", query_false))

        all_false = f"{self.label_column} == -1"
        stumps.append(Stump(x.query(all_false).index.to_numpy(), "FALSE", all_false))
        all_true = f"{self.label_column} == 1"
        stumps.append(Stump(x.query(all_true).index.to_numpy(), "TRUE", all_true))

        return stumps


def get_predictions(boost: list[Stump], fail_indexes: list[np.ndarray], y: np.ndarray):
    predictions_matrix = np.zeros((len(y), np.size(boost)))

    for i, s in enumerate(boost):
        assert s.alpha != np.inf
        for j in range(len(y)):
            predictions_matrix[j][i] = y[j] * s.alpha

    for i in range(len(boost)):
        predictions_matrix[fail_indexes[i], i] *= -1

    return np.array(
        [-1 if i == -1 else 1 for i in np.sign(np.sum(predictions_matrix, axis=1))]
    )


def boosting(iterations: int, stumps: list[Stump], y: np.ndarray):
    weights = np.full(len(y), 1 / len(y), dtype=np.float64)
    boost: list[Stump] = []
    for _ in range(iterations):
        best_stump = min(stumps, key=lambda s: np.sum(weights[s.train_fail_indexes]))

        error: float = np.sum(weights[best_stump.train_fail_indexes])
        alpha: float = 0.5 * (np.log(1 - error) - np.log(error))

        best_stump.alpha = alpha
        boost.append(best_stump)

        # update weights
        predictions = get_predictions([best_stump], [best_stump.train_fail_indexes], y)
        exponent = -alpha * predictions * y
        weights_aux = weights * np.exp(exponent)
        weights = weights_aux / np.sum(weights_aux)

    return boost


def calculate_test_error(boost: list[Stump], x: pd.DataFrame, y: pd.DataFrame):
    fail_indexes = []
    for stump in boost:
        fail_indexes.append(x.query(stump.query).index.to_numpy())
    predictions = get_predictions(boost, fail_indexes, y)

    error_count = 0
    for i in range(len(y)):
        if predictions[i] != y[i]:
            error_count += 1
    return error_count / len(y)
