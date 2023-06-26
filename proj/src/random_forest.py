import matplotlib.pyplot as plt
import tensorflow_decision_forests as tfdf
from keras import metrics
from src.util import f1_score, merge_df, split_dataset

# |%%--%%| <nFyzjNUDgA|QxMwp8iBc5>

tfdf.keras.set_training_logs_redirection(False)

# |%%--%%| <QxMwp8iBc5|8zEsQiE4XY>

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

df_paths = [
    # "data/2-admission.csv",
    "data/3-comorbidity.csv",
    "data/4-admissionDiagnosis.csv",
    "data/5-complication.csv",
    # "data/6-lab1h.csv",
    "data/7-icu.csv",
]


df = merge_df(df_paths, useless_columns)

# |%%--%%| <8zEsQiE4XY|vY5bf2P8th>

LABEL = "UnitDischargeName"
COMPLEMENTARY = "HospitalDischargeName"

# LABEL, COMPLEMENTARY = COMPLEMENTARY, LABEL

# |%%--%%| <vY5bf2P8th|WF7AXd6CQs>

# HACK
df = df.query("UnitLengthStay <= 1")

# |%%--%%| <WF7AXd6CQs|4nOBMphAN1>

df = df.drop(columns=[COMPLEMENTARY])

# |%%--%%| <4nOBMphAN1|ldiMzVIvG8>

train_ds_pd, test_ds_pd = split_dataset(df)

# |%%--%%| <ldiMzVIvG8|VKRbwPXIXR>

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=LABEL)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=LABEL)

# |%%--%%| <VKRbwPXIXR|sLbeDXH0G0>

model = tfdf.keras.RandomForestModel(num_trees=250)
# model = tfdf.keras.GradientBoostedTreesModel()

# |%%--%%| <sLbeDXH0G0|lWK5FWKJLW>

model.compile(
    metrics=[
        metrics.Accuracy(name="accuracy"),
        metrics.Precision(name="precision"),
        metrics.Recall(name="recall"),
    ],
)

# |%%--%%| <lWK5FWKJLW|nOu9qgBiWj>

model.fit(train_ds, validation_data=test_ds)

# |%%--%%| <nOu9qgBiWj|eVYYn47LA9>

evaluation = model.evaluate(test_ds, return_dict=True)

# |%%--%%| <eVYYn47LA9|Urvmza5dsw>

for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

f1 = f1_score(evaluation["precision"], evaluation["recall"])
print(f"f1-score: {f1:.4f}")

# |%%--%%| <Urvmza5dsw|yU9gaN9DTZ>

with open("perfect.html", "w") as file:
    file.write(tfdf.model_plotter.plot_model(model, max_depth=16))

# |%%--%%| <yU9gaN9DTZ|JwjEYFfek3>

model.summary()

# |%%--%%| <JwjEYFfek3|PDVJIMLfYH>

logs = model.make_inspector().training_logs()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")

plt.subplot(1, 2, 2)
plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Logloss (out-of-bag)")

plt.show()

# |%%--%%| <PDVJIMLfYH|0v3BMnHz4D>
