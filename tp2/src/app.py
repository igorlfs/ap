import matplotlib.pyplot as plt
import numpy as np
from src.util import DataSet

# |%%--%%| <ObC6RCKtJA|8JBzAiCnAw>

dataset = DataSet("label", "negative", "data/tic-tac-toe.data")

# |%%--%%| <8JBzAiCnAw|VSzvvd3cZ9>

N = 100  # Máximo de Stumps
K = 5  # Número de folds

test_error = np.array([dataset.cross_validation(K, i) for i in np.arange(1, N + 1)])
average = test_error.mean(axis=1)
plt.plot(np.linspace(1, N, N), average)
plt.title(r"Erro médio por número de $\it{Stumps}$")
minimum = min(average)
plt.axhline(y=minimum, linestyle="dashed", label=f"Mínimo: ({round(minimum, 3)})")
plt.legend()

# |%%--%%| <VSzvvd3cZ9|AwFMm8Aebq>
