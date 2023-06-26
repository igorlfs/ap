import pandas as pd
from src.util import merge_df, pretty_df

# |%%--%%| <zYifu2uHrk|wqBXaSf69y>

useless_columns = [
    "SuiteName",
    "ProductName",
    "HospitalCode",
    "HospitalTypeCode",
    "HospitalTypeName",
    "CountryName",
    "UnitCode",
    "Beds",
    "EpimedCode",
    "MedicalRecord",
]

df_paths = ["data/3-comorbidity.csv", "data/5-complication.csv", "data/7-icu.csv"]


df = merge_df(df_paths, useless_columns)

# |%%--%%| <wqBXaSf69y|XYKuPfnkRQ>

LABEL = "UnitDischargeName"
COMPLEMENTARY = "HospitalDischargeName"

# LABEL, COMPLEMENTARY = COMPLEMENTARY, LABEL

# |%%--%%| <XYKuPfnkRQ|oz8u0YhtZv>

# NOTE: Parece não ter muitos dados
# df = df.drop(columns=["Mechanical Ventilation Duration"])
# df = df.drop(columns=["Renal Replacement Therapy Duration"])

# |%%--%%| <oz8u0YhtZv|1UglUEq3ui>

# TODO: investigar a influência dessa coluna
df = df.drop(columns=["HospitalLengthStay"])

# |%%--%%| <1UglUEq3ui|ejWhdaskZn>

len(df.query(f"{LABEL} == 'Óbito'"))

# |%%--%%| <ejWhdaskZn|PftyYbkS90>

data = []
for col in df.columns:
    if True in df[col].unique():
        lives = len(df.query(f"`{col}` == True and {LABEL} == 'Óbito'"))
        dies = len(df.query(f"`{col}` == True and {LABEL} != 'Óbito'"))
        data.append([col, lives, dies, round(lives / (lives + dies), 2)])
df_col = pd.DataFrame(data=data, columns=["Col", "Óbito", "Alta", "Razão"])

# |%%--%%| <PftyYbkS90|zWLzAXZUpU>

dx = df_col.sort_values(by=["Razão", "Óbito"], ascending=False)

# |%%--%%| <zWLzAXZUpU|2w1Q9BjpCj>

df_1h = df.query("UnitLengthStay <= 1")

# |%%--%%| <2w1Q9BjpCj|O4a8voLSaG>

data = []
for col in df_1h.columns:
    if True in df_1h[col].unique():
        lives = len(df_1h.query(f"`{col}` == True and {LABEL} == 'Óbito'"))
        dies = len(df_1h.query(f"`{col}` == True and {LABEL} != 'Óbito'"))
        data.append([col, lives, dies, round(lives / (lives + dies), 2)])
df_col = pd.DataFrame(data=data, columns=["Col", "Óbito", "Alta", "Razão"])

# |%%--%%| <O4a8voLSaG|5THVc96vqH>

dx = df_col.sort_values(by=["Razão", "Óbito"], ascending=False)
pretty_df(dx)
