import matplotlib.pyplot as plt
import numpy as np

from src.util import DataSet

# |%%--%%| <ObC6RCKtJA|kIfJYDnDfv>

dataset = DataSet(
    "label", ["x", "o", "b"], "negative", "positive", "data/tic-tac-toe.data"
)

# |%%--%%| <kIfJYDnDfv|VSzvvd3cZ9>

N = 29
K = 5

test_error = np.array(
    [dataset.cross_validation(K, i) for i in np.arange(1, N + 1)], dtype=float
)

x_range = np.linspace(1, N, N)
avg = test_error.mean(axis=1)

for i in range(K):
    plt.plot(x_range, test_error[:, i], label=f"Split {i+1}", linewidth=0.8)

plt.plot(
    x_range,
    avg,
    label="Average",
    color="magenta",
    linestyle="dashed",
    linewidth=2,
)

plt.legend()

plt.title("Error over number of Stumps")

plt.axhline(y=0.2, color="black", linestyle="dashed")

# |%%--%%| <VSzvvd3cZ9|AwFMm8Aebq>
