import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow import keras

matplotlib.use("GTK3Agg")


def model(hidden: int, rate: float, batch: int):
    # Get data
    mnist = keras.datasets.mnist
    # Split
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # Model
    model = keras.models.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(hidden, activation="relu"),
            keras.layers.Dense(10),
        ]
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.SGD(learning_rate=rate)
    metrics = keras.metrics.SparseCategoricalAccuracy()
    # Compile
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # Epochs
    if batch == 1:
        epochs = 3
    elif batch == len(x_train):  # 60000
        epochs = 100
    else:
        epochs = 10
    # Training loop
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch,
        epochs=epochs,
    )
    # Evaluate
    print(model.evaluate(x_test, y_test))
    return history


def title_and_labels(xlabel: str, ylabel: str):
    if ylabel == "Sparse_categorical_accuracy":
        ylabel = "Accuracy"
    plt.title(f"{ylabel} over {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(f"{ylabel}_over_{xlabel}.png")
    plt.show()


def plot_var(var: list[int], history, metric: str, var_name: str):
    for i, j in zip(var, history):
        sns.lineplot(
            x=range(1, len(j.history[metric]) + 1),
            y=j.history[metric],
            label=f"{var_name}: {i}",
        )
    title_and_labels("Epoch", metric.capitalize())


def generate_data_var_hidden(hidden_layer_sizes: list[int]):
    history = []
    for size in hidden_layer_sizes:
        history.append(model(size, 1, 50))
    return history


def generate_data_var_rate(learning_rates: list[int]):
    history = []
    for rate in learning_rates:
        history.append(model(50, rate, 50))
    return history


def generate_data_var_batches(batch_sizes: list[int]):
    history = []
    for size in batch_sizes:
        history.append(model(50, 1, size))
    return history


def plot_var_batch(batch_sizes: list[int], history, metric: str):
    num_of_epochs = [3, 10, 10, 100]
    for i in range(len(batch_sizes)):
        sns.lineplot(
            x=np.linspace(0, 1, num_of_epochs[i]),
            y=history[i].history[metric],
            label=f"Batch Size: {batch_sizes[i]}",
        )
    title_and_labels("Epoch Progress", metric.capitalize())
