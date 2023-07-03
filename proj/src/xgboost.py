import xgboost as xgb
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from src.preprocessing import sane_df
from src.util import print_metrics, train_test_split_df

# |%%--%%| <CHegZyhACA|wqBXaSf69y>

df_paths = [
    "data/2-admission.csv",
    "data/3-comorbidity.csv",
    "data/4-admissionDiagnosis.csv",
    "data/5-complication.csv",
    "data/7-icu.csv",
]

df = sane_df(df_paths)

# |%%--%%| <wqBXaSf69y|nYO1W3aICF>

LABEL = "UnitDischargeName"
COMPLEMENTARY = "HospitalDischargeName"

df = df.drop(columns=[COMPLEMENTARY])

# |%%--%%| <nYO1W3aICF|J2Si5blE7a>

for col in [LABEL]:
    df[col] = df[col].map({"Alta": 0, "Óbito": 1})

# |%%--%%| <J2Si5blE7a|cPmIaZLWaV>

(x_train, y_train), (x_test, y_test) = train_test_split_df(df, LABEL)

# |%%--%%| <cPmIaZLWaV|0xBwsGWL3s>

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

# |%%--%%| <0xBwsGWL3s|wW7EWuwOQD>

param = {"max_depth": 3, "eta": 1, "objective": "binary:logistic", "eval_metric": "auc"}

# |%%--%%| <wW7EWuwOQD|YmQkllg1oU>

evallist = [(dtrain, "train"), (dtest, "eval")]

# |%%--%%| <YmQkllg1oU|Nz25OmMlx6>

num_round = 10
bst = xgb.train(param, dtrain, num_round, evallist)

# |%%--%%| <Nz25OmMlx6|4zjkxEAFMH>

predictions = bst.predict(dtest)

# |%%--%%| <4zjkxEAFMH|2yAgzcC3Mx>

THRESHOLD = 0.3
y_pred = [1 if y > THRESHOLD else 0 for y in predictions]

# |%%--%%| <2yAgzcC3Mx|GHL4QCp0Ap>

print_metrics(y_test, y_pred)

# |%%--%%| <GHL4QCp0Ap|cKxKZ0E110>

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# |%%--%%| <cKxKZ0E110|R06kJmXSDZ>

xgb.plot_importance(bst)

# |%%--%%| <R06kJmXSDZ|lrbo2AVSOB>

xgb.plot_tree(bst, num_trees=2, rankdir="LR")

# |%%--%%| <lrbo2AVSOB|rVapEzYQL8>
