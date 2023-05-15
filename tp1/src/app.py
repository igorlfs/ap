import seaborn as sns

from .util import (generate_data_var_batches, generate_data_var_hidden,
                   generate_data_var_rate, plot_var, plot_var_batch)


def run():
    sns.set_style("darkgrid")

    metrics = ["loss", "sparse_categorical_accuracy"]

    hidden_layer_sizes = [25, 50, 100]
    history = generate_data_var_hidden(hidden_layer_sizes)
    for metric in metrics:
        plot_var(hidden_layer_sizes, history, metric, "Hidden Layer Size")

    learning_rates = [0.5, 1, 10]
    history = generate_data_var_rate(learning_rates)
    for metric in metrics:
        plot_var(learning_rates, history, metric, "Learning Rate")

    batch_sizes = [1, 20, 50, 60000]
    history = generate_data_var_batches(batch_sizes)
    for metric in metrics:
        plot_var_batch(batch_sizes, history, metric)
