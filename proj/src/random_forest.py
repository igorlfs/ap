import matplotlib.pyplot as plt
import tensorflow_decision_forests as tfdf
from keras import metrics
from src.preprocessing import sane_df
from src.util import f1_score, pretty_df, split_df

# |%%--%%| <nFyzjNUDgA|QxMwp8iBc5>

tfdf.keras.set_training_logs_redirection(False)

# |%%--%%| <QxMwp8iBc5|upptfJQ6Xm>

df_paths = [
    "data/2-admission.csv",
    "data/3-comorbidity.csv",
    "data/4-admissionDiagnosis.csv",
    "data/5-complication.csv",
    # "data/6-lab1h.csv",
    "data/7-icu.csv",
    # "data/8-secondaryDiagnosis.csv",
]


df = sane_df(df_paths, True)

# |%%--%%| <upptfJQ6Xm|RR6ag6iR5g>

pretty_df(df)

# |%%--%%| <RR6ag6iR5g|9EcxWKJaSb>

LABEL = "UnitDischargeName"
COMPLEMENTARY = "HospitalDischargeName"

# LABEL, COMPLEMENTARY = COMPLEMENTARY, LABEL

df = df.drop(columns=[COMPLEMENTARY])

# |%%--%%| <9EcxWKJaSb|ldiMzVIvG8>

train_ds_pd, test_ds_pd = split_df(df)

# |%%--%%| <ldiMzVIvG8|VKRbwPXIXR>

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=LABEL)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=LABEL)

# |%%--%%| <VKRbwPXIXR|sLbeDXH0G0>

model = tfdf.keras.RandomForestModel(num_trees=200)

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

# |%%--%%| <Urvmza5dsw|l66fcT2Fdj>

model.make_inspector().variable_importances()["SUM_SCORE"][:10]

# |%%--%%| <l66fcT2Fdj|yU9gaN9DTZ>

with open("random-forest.html", "w") as file:
    file.write(tfdf.model_plotter.plot_model(model, max_depth=16))

# |%%--%%| <yU9gaN9DTZ|PDVJIMLfYH>

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
