from sklearn import ensemble
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from src.preprocessing import sane_df
from src.util import print_metrics, train_test_split_df

# |%%--%%| <9XB0l0zI3I|CM5QaujlNd>

df_paths = [
    "data/2-admission.csv",
    "data/3-comorbidity.csv",
    "data/4-admissionDiagnosis.csv",
    "data/5-complication.csv",
    "data/7-icu.csv",
]


df = sane_df(df_paths)

# |%%--%%| <CM5QaujlNd|1lqexRvVhN>

LABEL = "UnitDischargeName"
COMPLEMENTARY = "HospitalDischargeName"

df = df.drop(columns=[COMPLEMENTARY])

# |%%--%%| <1lqexRvVhN|8RnbCc7hrp>

for col in [LABEL]:
    df[col] = df[col].map({"Alta": 0, "Óbito": 1})

# |%%--%%| <8RnbCc7hrp|3D1n2BWeET>

# df = df.query("UnitLengthStay <= 1")

# |%%--%%| <3D1n2BWeET|89a29JVTe0>

(x_train, y_train), (x_test, y_test) = train_test_split_df(df, LABEL)

# |%%--%%| <89a29JVTe0|sbcMY5NpUv>

clf = ensemble.AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=200
)

# |%%--%%| <sbcMY5NpUv|fBLvhdZnYJ>

model = clf.fit(x_train, y_train)

# |%%--%%| <fBLvhdZnYJ|uMbgXriEMr>

y_pred = model.predict(x_test)

# |%%--%%| <uMbgXriEMr|fghlXMIPfq>

print_metrics(y_test, y_pred)

# |%%--%%| <fghlXMIPfq|okRjGnDp8G>

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
