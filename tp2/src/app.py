import matplotlib.pyplot as plt
import numpy as np
from src.dataset import DataSet

# |%%--%%| <ObC6RCKtJA|kIfJYDnDfv>

dataset = DataSet(
    "label", ["x", "o", "b"], "negative", "positive", "data/tic-tac-toe.data"
)

# |%%--%%| <kIfJYDnDfv|0DPZ6Su9EB>

# dataset = DataSet("Vampire", ["Y", "N"], "N", "Y", "sample.csv")

# |%%--%%| <0DPZ6Su9EB|1g88qWcOq5>

# SIZE = 10
# errors_per_stumps = np.zeros(SIZE)
#
# for i in range(SIZE):
#     errors_per_stumps[i] = dataset.boost(i)
#
# plt.plot(np.linspace(1, SIZE + 1, SIZE), errors_per_stumps)


# |%%--%%| <1g88qWcOq5|4TeWIqUUK2>

SIZE = 15
errors = np.zeros(SIZE)
for i in np.arange(1, SIZE + 1):
    boost = dataset.boost(i)
    errors[i - 1] = dataset.calculate_test_error(boost.boost)

plt.plot(np.linspace(1, SIZE, SIZE), errors, scaley=False)
# for b in boost.boost:
#     print(b.alpha)
# boost.boost

# |%%--%%| <4TeWIqUUK2|Ug5Z54BJ98>

dataset.calculate_test_error(boost.boost)

# |%%--%%| <Ug5Z54BJ98|AwFMm8Aebq>
