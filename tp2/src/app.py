import matplotlib.pyplot as plt
import numpy as np
from src.dataset import DataSet

# |%%--%%| <ObC6RCKtJA|kIfJYDnDfv>

dataset = DataSet(
    "label", ["x", "o", "b"], "negative", "positive", "data/tic-tac-toe.data"
)

# |%%--%%| <kIfJYDnDfv|1g88qWcOq5>

SIZE = 10
errors_per_stumps = np.zeros(SIZE)

for i in range(SIZE):
    errors_per_stumps[i] = dataset.boost(i)

plt.plot(np.linspace(1, SIZE + 1, SIZE), errors_per_stumps)


# |%%--%%| <1g88qWcOq5|AwFMm8Aebq>
