import pandas as pd


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

    return pd.concat(
        [dataframes[0].drop(columns=droppable_first)]
        + [df.drop(columns=common_columns) for df in dataframes[1:]],
        axis=1,
    )


def sane_df(paths: list[str], is_tensorflow: bool = False):
    # Colunas comuns a todos os datasets mas que não possuem semântica para o nosso problema
    useless_columns = [
        "SuiteName",
        "HospitalCode",
        "HospitalTypeCode",
        "HospitalTypeName",
        "CountryName",
        "UnitCode",
        "Beds",
        "EpimedCode",
        "MedicalRecord",
    ]

    df = merge_df(paths, useless_columns)

    # Colunas que
    # 1. São inúteis (e.g., datas)
    # 2. Exigem domínio da área para serem mapeadas em números de forma semântica
    # 3. Não estão abudantemente preeenchidas
    droppable_columns = [
        "HematologicalMalignancyTypeCode",
        "HematologicalMalignancyTypeName",
        "Anatomic Tumor Site Name",
        "HospitalAdmissionDate",
        "HospitalDischargeDate",
        "HospitalIsClosed",
        "SCCMClass Name2016",
        "HealthInsuranceName",
        "AdmissionSourceName",
        "AdmissionReasonName",
        "AdmissionMainDiagnosisName",
        "MFI points",
        "MFI Score",
    ]

    for col in droppable_columns:
        if col in df.columns:
            df = df.drop(columns=col)

    # Preencher os dados deve vir antes do tratamento
    _fill_data(df)

    if not is_tensorflow:
        df = _handle_non_tensorflow(df)

    # O Adaboost não gosta de dados nulos
    return df.dropna()


def _handle_non_tensorflow(df: pd.DataFrame):
    """Mapeia colunas categóricas para números ou booleanos."""

    # A escolha dos pesos é arbitrária
    if "AdmissionTypeName" in df.columns:
        df["AdmissionTypeName"] = df["AdmissionTypeName"].map(
            {"Clínica": 0, "Cirurgia eletiva": 2, "Cirurgia de urgência / emergência": 10}
        )
    if "Chronic Health Status Name" in df.columns:
        df["Chronic Health Status Name"] = df["Chronic Health Status Name"].map(
            {"Independente": 0, "Necessidade de assistência": 1, "Restrito / acamado": 2}
        )

    if "Decision Palliative Care Name" in df.columns:
        df["Decision Palliative Care Name"] = df["Decision Palliative Care Name"] == "Não"

    if "ProductName" in df.columns:
        df["ProductName"] = df["ProductName"] == "Cardiológica"

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"] == "M"
    for col in df.columns:
        if "Falso" in df[col].unique():
            # Se não possuir "Verdadeiro" então a classe não agrega em nada no modelo
            if "Verdadeiro" in df[col].unique():
                df[col] = df[col] == "Verdadeiro"
            else:
                df = df.drop(columns=col)
    for col in df.columns:
        if "Sim" in df[col].unique():
            df[col] = df[col] == "Sim"

    return df


def _fill_data(df: pd.DataFrame):
    for col in ["Mechanical Ventilation Duration", "Renal Replacement Therapy Duration"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    sepsis_columns = [
        "Is Sepsis D1",
        "Is Septicshock D1",
        "Is Sepsis septicshock D1",
        "Is Sepsis ICUStay",
        "Is Septicshock ICUStay",
        "Is Sepsis septicshock ICUStay",
    ]
    for col in sepsis_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Falso")
