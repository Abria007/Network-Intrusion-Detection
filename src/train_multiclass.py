
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)
from xgboost import XGBClassifier

from common import (
    DATA_DIR,
    ARTIFACTS_DIR,
    MODELS_DIR,
    REPORTS_DIR
)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load original dataset with multiclass labels
frames = []

for file in sorted(DATA_DIR.glob("*.csv")):
    print("Loading:", file.name)

    df = pd.read_csv(file, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype("string").str.strip()
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

if data["Label"].isna().any():
    raise ValueError("Missing labels found.")

feature_columns = [
    column for column in data.columns if column != "Label"
]

X = data[feature_columns].copy()
y = data["Label"].astype(str)

# Ensure features are numeric and handle infinities as missing values
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# 2. Load the dedicated multiclass split
split = np.load(ARTIFACTS_DIR / "multiclass_split_indices.npz")
train_idx = split["train_idx"]
test_idx = split["test_idx"]

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# 3. Encode original class names as integers
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)

# Verify every test class was seen during training
unknown_test_labels = set(y_test.unique()) - set(encoder.classes_)

if unknown_test_labels:
    raise ValueError(
        f"Test classes missing from training: {unknown_test_labels}"
    )

y_test_encoded = encoder.transform(y_test)

print("\nNumber of classes:", len(encoder.classes_))
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# 4. Build the baseline model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", XGBClassifier(
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    ))
])

# 5. Train
print("\nTraining XGBoost multiclass baseline...")
model.fit(X_train, y_train_encoded)

# 6. Evaluate on the held-out test set
print("\nPredicting test labels...")
predictions = model.predict(X_test).astype(int)

accuracy = accuracy_score(y_test_encoded, predictions)
macro_f1 = f1_score(
    y_test_encoded, predictions, average="macro", zero_division=0
)
weighted_f1 = f1_score(
    y_test_encoded, predictions, average="weighted", zero_division=0
)

print("\n===== MULTICLASS BASELINE RESULTS =====")
print(f"Accuracy:    {accuracy:.6f}")
print(f"Macro-F1:    {macro_f1:.6f}")
print(f"Weighted-F1: {weighted_f1:.6f}")

# 7. Save detailed per-class evaluation
report = classification_report(
    y_test_encoded,
    predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()
report_path = REPORTS_DIR / "xgboost_multiclass_classification_report.csv"
report_df.to_csv(report_path)

metrics = pd.DataFrame([{
    "Model": "XGBoost multiclass baseline",
    "Accuracy": accuracy,
    "Macro_F1": macro_f1,
    "Weighted_F1": weighted_f1
}])

metrics_path = REPORTS_DIR / "xgboost_multiclass_metrics.csv"
metrics.to_csv(metrics_path, index=False)

# 8. Save model and class encoder
model_path = MODELS_DIR / "xgboost_multiclass.joblib"

joblib.dump({
    "model": model,
    "label_encoder": encoder,
    "feature_columns": feature_columns
}, model_path)

print("\nSaved model:", model_path)
print("Saved metrics:", metrics_path)
print("Saved classification report:", report_path)