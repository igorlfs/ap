import pandas as pd
from src.preprocessing import sane_df
from src.util import pretty_df

# |%%--%%| <zYifu2uHrk|wqBXaSf69y>

df_paths = [
    "data/2-admission.csv",
    "data/3-comorbidity.csv",
    "data/5-complication.csv",
    "data/7-icu.csv",
]

df = sane_df(df_paths)

# |%%--%%| <wqBXaSf69y|XYKuPfnkRQ>

LABEL = "UnitDischargeName"
COMPLEMENTARY = "HospitalDischargeName"

# LABEL, COMPLEMENTARY = COMPLEMENTARY, LABEL

# |%%--%%| <XYKuPfnkRQ|vdwiMXkqtd>

df = df.drop(columns=[COMPLEMENTARY])

# |%%--%%| <vdwiMXkqtd|1UglUEq3ui>

df = df.drop(columns=["HospitalLengthStay"])

# |%%--%%| <1UglUEq3ui|ejWhdaskZn>

len(df.query(f"{LABEL} == 'Óbito'"))

# |%%--%%| <ejWhdaskZn|h1NXqQWYBc>

for col in df.columns:
    if "Falso" in df[col].unique():
        df[col] = df[col].map({"Verdadeiro": True, "Falso": False})

# |%%--%%| <h1NXqQWYBc|PftyYbkS90>

data = []
for col in df.columns:
    if df[col].dtype == bool:
        lives = len(df.query(f"`{col}` == True and {LABEL} == 'Óbito'"))
        dies = len(df.query(f"`{col}` == True and {LABEL} != 'Óbito'"))
        data.append([col, lives, dies, round(lives / (lives + dies), 2)])
df_col = pd.DataFrame(data=data, columns=["Col", "Óbito", "Alta", "Razão"])

# |%%--%%| <PftyYbkS90|zWLzAXZUpU>

dx = df_col.sort_values(by=["Razão", "Óbito"], ascending=False)
pretty_df(dx)

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
