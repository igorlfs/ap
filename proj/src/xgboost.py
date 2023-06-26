import xgboost as xgb
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from src.util import merge_df, split_dataset

# |%%--%%| <CHegZyhACA|wqBXaSf69y>

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

# |%%--%%| <wqBXaSf69y|KtrGFZtI9M>

LABEL = "UnitDischargeName"
COMPLEMENTARY = "HospitalDischargeName"

# LABEL, COMPLEMENTARY = COMPLEMENTARY, LABEL

# |%%--%%| <KtrGFZtI9M|f7EtOyzatd>

df = df.drop(columns=[COMPLEMENTARY])

# |%%--%%| <f7EtOyzatd|FneSK9lFbf>

# O xgboost não suporta colunas não numéricas (categóricas)
df = df.drop(columns=["AdmissionTypeName", "Decision Palliative Care Name"])

# |%%--%%| <FneSK9lFbf|JChdWwmWEF>

# Outras colunas que o xgboost não curte
df = df.drop(
    columns=[
        "Chronic Health Status Name",
        "Anatomic Tumor Site Name",
        "HematologicalMalignancyTypeName",
    ]
)

# |%%--%%| <JChdWwmWEF|2w1Q9BjpCj>

df = df.query("UnitLengthStay <= 1")

# |%%--%%| <2w1Q9BjpCj|7dkV7HYmCr>

# NOTE: para as labels invertidas, uma está faltando
df = df.dropna(axis=0, subset=LABEL)

# |%%--%%| <7dkV7HYmCr|1gTIoZWiFB>

# o xgboost exige um pipeline de dados mais chatinho
for col in df.columns:
    if "Falso" in df[col].unique():
        df[col] = df[col].map({"Verdadeiro": True, "Falso": False})

# |%%--%%| <1gTIoZWiFB|wMQMB5OxRS>

for col in df.columns:
    if False in df[col].unique():
        df[col] = df[col].map({True: 1, False: 0})

# |%%--%%| <wMQMB5OxRS|J2Si5blE7a>

for col in [LABEL]:
    df[col] = df[col].map({"Alta": 0, "Óbito": 1})

# |%%--%%| <J2Si5blE7a|MG17maRBid>

train_ds_pd, test_ds_pd = split_dataset(df)

# |%%--%%| <MG17maRBid|QvwFdx84Bw>

X_train = train_ds_pd.drop(columns=[LABEL])
Y_train = train_ds_pd[LABEL]

# |%%--%%| <QvwFdx84Bw|QdnqbxSt3C>

X_test = test_ds_pd.drop(columns=[LABEL])
Y_test = test_ds_pd[LABEL]

# |%%--%%| <QdnqbxSt3C|0xBwsGWL3s>

dtrain = xgb.DMatrix(X_train, label=Y_train)
dtest = xgb.DMatrix(X_test, label=Y_test)

# |%%--%%| <0xBwsGWL3s|wW7EWuwOQD>

param = {"max_depth": 5, "eta": 1, "objective": "binary:logistic"}
# param = {}
param["nthread"] = 8
param["eval_metric"] = "auc"

# |%%--%%| <wW7EWuwOQD|YmQkllg1oU>

evallist = [(dtrain, "train"), (dtest, "eval")]

# |%%--%%| <YmQkllg1oU|Nz25OmMlx6>

num_round = 10
bst = xgb.train(param, dtrain, num_round, evallist)

# |%%--%%| <Nz25OmMlx6|4zjkxEAFMH>

ypred = bst.predict(dtest)

# |%%--%%| <4zjkxEAFMH|2yAgzcC3Mx>

Y_pred = [1 if y > 0.5 else 0 for y in ypred]  # noqa: PLR2004

# |%%--%%| <2yAgzcC3Mx|GHL4QCp0Ap>

f1_score(Y_test, Y_pred)

# |%%--%%| <GHL4QCp0Ap|cKxKZ0E110>

cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# |%%--%%| <cKxKZ0E110|R06kJmXSDZ>

xgb.plot_importance(bst)

# |%%--%%| <R06kJmXSDZ|lrbo2AVSOB>

xgb.plot_tree(bst, num_trees=2)

# |%%--%%| <lrbo2AVSOB|rVapEzYQL8>
