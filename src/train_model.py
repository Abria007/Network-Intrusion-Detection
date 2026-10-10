
import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from common import (
    load_dataset,
    get_split,
    create_folders,
    MODEL_FILE,
    REPORTS_DIR
)

create_folders()

df, X, y, feature_cols = load_dataset()
train_idx, test_idx = get_split(X, y)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    ))
])

print("Training XGBoost. Please wait...")

pipeline.fit(X_train, y_train)

# Save the trained model immediately
joblib.dump(pipeline, MODEL_FILE)

print("Model saved successfully:")
print(MODEL_FILE)

pred = pipeline.predict(X_test)

metrics = {
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(
        y_test, pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, pred, zero_division=0
    ),
    "F1-score": f1_score(
        y_test, pred, zero_division=0
    )
}

results = pd.DataFrame([metrics])

output_file = REPORTS_DIR / "xgboost_retrained_metrics.csv"
results.to_csv(output_file, index=False)

print("\nNew model metrics:")
print(results.to_string(index=False))
print("\nMetrics saved to:", output_file)