import numpy as np
import pandas as pd

# |%%--%%| <ObC6RCKtJA|aueYcM7o6y>

LABEL = "Vampire"
DATA_VALUES = ["Y", "N"]
LABEL_FALSE = "N"
LABEL_TRUE = "Y"
DATASET_PATH = "sample.csv"
# DATASET_PATH = "../sample.csv"
LABEL = "label"
DATA_VALUES = ["x", "o", "b"]
LABEL_FALSE = "negative"
LABEL_TRUE = "positive"
DATASET_PATH = "data/tic-tac-toe.data"
# DATASET_PATH = "../data/tic-tac-toe.data"

# |%%--%%| <aueYcM7o6y|9ghqUTyxtX>

# Ignore o ID dos vampiros
df = pd.read_csv(DATASET_PATH, usecols=lambda x: x != "id")

# |%%--%%| <9ghqUTyxtX|53lof3gjkn>

labels = df[LABEL].apply(lambda x: -1 if x == LABEL_FALSE else 1).to_numpy()

# |%%--%%| <53lof3gjkn|O7kKLi5ZzZ>


class Stump:
    def __init__(self, indexes: np.ndarray, name: str) -> None:
        self.indexes = indexes
        self.name = name
        self.alpha = np.inf

    def __str__(self) -> str:
        return f"{self.name}, ({len(self.indexes)}) {self.indexes}"

    def __repr__(self) -> str:
        return str(self)


# |%%--%%| <O7kKLi5ZzZ|VRKRm18QMq>


def get_predictions(boost: list[Stump], labels: np.ndarray) -> np.ndarray:
    predictions_matrix = np.zeros((len(labels), np.size(boost)))

    for i, s in enumerate(boost):
        for j in range(len(labels)):
            predictions_matrix[j][i] = labels[j] * s.alpha

    for i, s in enumerate(boost):
        predictions_matrix[boost[i].indexes, i] *= -1

    return np.array(
        [-1 if i == -1 else 1 for i in np.sign(np.sum(predictions_matrix, axis=1))]
    )


# |%%--%%| <VRKRm18QMq|hKIkwO6pDL>


def calculate_error(boost: list[Stump], labels: np.ndarray):
    predictions = get_predictions(boost, labels)
    error_count = 0
    for i in range(len(labels)):
        if predictions[i] != labels[i]:
            error_count += 1
    return error_count / len(labels)


# |%%--%%| <hKIkwO6pDL|UG5vBNDXmt>

stumps: list[Stump] = []
for column in df.columns:
    if column != LABEL:
        for v in DATA_VALUES:
            fail_indexes = df.query(
                f"({column} == '{v}' and {LABEL} == '{LABEL_FALSE}') or ({column} != '{v}' and {LABEL} == '{LABEL_TRUE}')"
            ).index.to_numpy()
            stumps.append(Stump(fail_indexes, f"{column} == {v}"))

# |%%--%%| <UG5vBNDXmt|kof1HGEVwQ>

QUERY_TRUE = df[LABEL] == LABEL_FALSE
stumps.append(Stump(df.index[QUERY_TRUE].to_numpy(), "TRUE"))
stumps.append(Stump(df.index[~QUERY_TRUE].to_numpy(), "FALSE"))

# |%%--%%| <kof1HGEVwQ|crD6w61NRc>

DF_SIZE = len(df)
weights = np.full(DF_SIZE, 1 / DF_SIZE)

# |%%--%%| <crD6w61NRc|1g88qWcOq5>

boost: list[Stump] = []
ITERATIONS = 8
stumps_copy = stumps.copy()
weights_copy = weights.copy()
for n in range(ITERATIONS):
    best_stump = min(stumps_copy, key=lambda s: np.sum(weights_copy[s.indexes]))

    error = np.sum(weights_copy[best_stump.indexes])
    alpha: float = 0.5 * (np.log(1 - error) - np.log(error))

    best_stump.alpha = alpha
    boost.append(best_stump)

    # update weights
    predictions = get_predictions([best_stump], labels)
    exponent = -alpha * predictions * labels
    weights_copy = weights_copy * np.exp(exponent)
    normalizer = np.sum(weights_copy)
    weights_copy = weights_copy / normalizer

    stumps_copy.remove(best_stump)

for b in boost:
    print(b.name)


# |%%--%%| <1g88qWcOq5|euA2EU3vJL>

calculate_error(boost, labels)

# |%%--%%| <euA2EU3vJL|WJqpCuEgnT>

for stump in stumps:
    print(stump.name, len(stump.indexes))

# |%%--%%| <WJqpCuEgnT|AwFMm8Aebq>
