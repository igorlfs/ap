import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score


def pretty_df(df: pd.DataFrame | pd.Series):
    """Plota o dataframe usando a ferramenta `visidata`."""
    try:
        subprocess.run(["vd", "-f", "csv", "-"], input=df.to_csv(index=False), text=True)
    except subprocess.CalledProcessError:
        print("Visidata not installed, skipping")


def split_df(df: pd.DataFrame, test_ratio=0.30):
    """Divide um dataframe em dois."""
    test_indices = np.random.rand(len(df)) < test_ratio
    return df[~test_indices], df[test_indices]


def train_test_split_df(df: pd.DataFrame, label: str):
    """Divide um dataframe em teste e treino, com uma `label`"""
    train_ds_pd, test_ds_pd = split_df(df)
    x_train = train_ds_pd.drop(columns=[label])
    y_train = train_ds_pd[label]
    x_test = test_ds_pd.drop(columns=[label])
    y_test = test_ds_pd[label]
    return (x_train, y_train), (x_test, y_test)


def f1_score(precision: float, recall: float) -> float:
    """Calcula o f1-score com base na precisão e no recall."""
    return (2 * precision * recall) / (precision + recall) if recall != 0 else -1


def print_metrics(y_test: pd.Series, y_pred: list[int] | np.ndarray):
    """Imprime métricas para os modelos baseados no scikit-learn."""
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    print(f"Precision: {pre}")
    print(f"Recall: {rec}")
    print(f"F1-Score: {f1_score(pre,rec)}")
