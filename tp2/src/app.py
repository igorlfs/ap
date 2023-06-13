import matplotlib.pyplot as plt
import numpy as np

from src.dataset import DataSet

# |%%--%%| <ObC6RCKtJA|kIfJYDnDfv>

dataset = DataSet(
    "label", ["x", "o", "b"], "negative", "positive", "data/tic-tac-toe.data"
)

# |%%--%%| <kIfJYDnDfv|0DPZ6Su9EB>

# dataset = DataSet("Vampire", ["Y", "N"], "N", "Y", "sample.csv")

# |%%--%%| <0DPZ6Su9EB|46DLSq5vrV>
r"""°°°
# TODO
## Cross-Validation
### Shuffle
- Shuffle and use the first 20% as test, then ...
- Calculate test error normally (we will need a matrix and then we get the avg)
### update_train_and_test
- Handles going to the next train/test split
°°°"""
# |%%--%%| <46DLSq5vrV|4TeWIqUUK2>

SIZE = 20
errors = np.zeros(SIZE)
for i in np.arange(1, SIZE + 1):
    boost = dataset.boost(i)
    errors[i - 1] = dataset.calculate_test_error(boost.boost)

plt.plot(np.linspace(1, SIZE, SIZE), errors, scaley=False)
plt.axhline(y=0.2, color="r", linestyle="dashed")
# for b in boost.boost:
#     print(b.alpha)
# boost.boost

# |%%--%%| <4TeWIqUUK2|AwFMm8Aebq>
