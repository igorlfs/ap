import matplotlib.pyplot as plt
import numpy as np

from src.dataset import DataSet

# |%%--%%| <ObC6RCKtJA|kIfJYDnDfv>

dataset = DataSet(
    "label", ["x", "o", "b"], "negative", "positive", "data/tic-tac-toe.data"
)

# |%%--%%| <kIfJYDnDfv|vY4XuQBXV3>
r"""°°°
TODO:
- Merge `src/dataset.py` and `src/util.py`
- More tests
°°°"""
# |%%--%%| <vY4XuQBXV3|VSzvvd3cZ9>

N = 29
K = 5
test_error = np.zeros((N, K))
for i in np.arange(1, N + 1):
    test_error[i - 1] = dataset.cross_validation(K, i)
plt.plot(np.linspace(1, N, N), test_error, scaley=False)
plt.axhline(y=0.2, color="black", linestyle="dashed")

# |%%--%%| <VSzvvd3cZ9|AwFMm8Aebq>
