
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from common import (
    load_dataset,
    get_split,
    MODEL_FILE,
    REPORTS_DIR,
    create_folders
)

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "Saved model not found. Run src/train_model.py first."
    )

create_folders()

df, X, y, feature_cols = load_dataset()
train_idx, test_idx = get_split(X, y)

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# Load saved model. No fitting or retraining here.
pipeline = joblib.load(MODEL_FILE)

pred = pipeline.predict(X_test)
prob = pipeline.predict_proba(X_test)[:, 1]

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
    ),
    "ROC-AUC": roc_auc_score(y_test, prob),
    "Average Precision": average_precision_score(
        y_test, prob
    )
}

results = pd.DataFrame([metrics])

metrics_path = REPORTS_DIR / "xgboost_evaluation_metrics.csv"
results.to_csv(metrics_path, index=False)

print(results.to_string(index=False))

print("\nClassification report:")
print(classification_report(
    y_test,
    pred,
    labels=[0, 1],
    target_names=["BENIGN", "ATTACK"],
    zero_division=0
))

cm = confusion_matrix(y_test, pred, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["BENIGN", "ATTACK"]
)

display.plot(values_format=",")
plt.title("XGBoost Confusion Matrix")
plt.tight_layout()

plot_path = REPORTS_DIR / "xgboost_confusion_matrix.png"
plt.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close()

print("\nSaved metrics:", metrics_path)
print("Saved confusion matrix:", plot_path)