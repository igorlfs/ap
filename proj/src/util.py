import subprocess

import numpy as np
import pandas as pd


def pretty_df(df: pd.DataFrame | pd.Series):
    """Plota o dataframe usando a ferramenta `visidata`."""
    try:
        subprocess.run(["vd", "-f", "csv", "-"], input=df.to_csv(index=False), text=True)
    except subprocess.CalledProcessError:
        print("Visidata not installed, skipping")


def merge_df(paths: list[str], droppable_first: list[str]) -> pd.DataFrame:
    """Une uma lista de CSVs com base em seus paths.

    Exclui colunas em comum, exceto no primeiro dataframe, que recebe uma lista de quais
    colunas excluir. Isso permite reter colunas que são úteis, apesar de aparecem várias vezes.
    """

    dataframes: list[pd.DataFrame] = []
    for path in paths:
        dataframes.append(pd.read_csv(path))

    common_columns = set(dataframes[0].columns)
    for df in dataframes[1:]:
        common_columns = common_columns.intersection(set(df.columns))

    # Merge dataframes, excluding common columns
    return pd.concat(
        [dataframes[0].drop(columns=droppable_first)]
        + [df.drop(columns=common_columns) for df in dataframes[1:]],
        axis=1,
    )


def split_dataset(dataset: pd.DataFrame, test_ratio=0.30):
    """Divide um dataframe em dois."""
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


def f1_score(precision: float, recall: float) -> float:
    """Calcula o f1_score com base na precisão e no recall."""
    return (2 * precision * recall) / (precision + recall) if recall != 0 else -1
