import numpy as np
import pandas as pd
from src.util import Boost, Stump, calculate_error


class DataSet:
    def __init__(self, label: str, values: list[str], false: str, true: str, path: str):
        self.label = label
        self.values = values
        self.false = false
        self.true = true
        # Ignore o ID dos vampiros
        self.df = pd.read_csv(path, usecols=lambda x: x != "id")
        self.y = (
            self.df[self.label].apply(lambda x: -1 if x == self.false else 1).to_numpy()
        )
        self.weights = np.full(len(self.df), 1 / len(self.df))
        self._gen_stumps()

    def _gen_stumps(self):
        stumps: list[Stump] = []
        for column in self.df.columns:
            if column != self.label:
                for v in self.values:
                    fail_indexes = self.df.query(
                        f"({column} == '{v}' and {self.label} == '{self.false}') or ({column} != '{v}' and {self.label} == '{self.true}')"
                    ).index.to_numpy()
                    stumps.append(Stump(fail_indexes, f"{column} == {v}"))

        QUERY_TRUE = self.df[self.label] == self.false
        stumps.append(Stump(self.df.index[QUERY_TRUE].to_numpy(), "TRUE"))
        stumps.append(Stump(self.df.index[~QUERY_TRUE].to_numpy(), "FALSE"))

        self.stumps = stumps

    def boost(self, iterations: int, verbose: bool = False):
        boost = Boost(self.weights, self.stumps, self.y)
        for _ in range(iterations):
            boost.iteration()
        if verbose:
            for stump in boost.boost:
                print(stump.name, stump.indexes)
        return calculate_error(boost.boost, self.y)
