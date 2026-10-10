
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load the processed dataset and retain original multiclass labels
frames = []

for file in sorted(DATA_DIR.glob("*.csv")):
    print("Loading:", file.name)

    df = pd.read_csv(file, low_memory=False)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype("string").str.strip()
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

if data["Label"].isna().any():
    raise ValueError("Missing labels found. Investigate before splitting.")

y = data["Label"].astype(str).to_numpy()
X = data.drop(columns=["Label"])

print("\nDataset shape:", data.shape)
print("Number of classes:", len(np.unique(y)))

# Keep identical feature vectors in the same group
groups = pd.util.hash_pandas_object(
    X, index=False
).to_numpy()

# Five folds: approximately 80% train and 20% test
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

dummy_X = np.zeros((len(y), 1), dtype=np.uint8)

train_idx, test_idx = next(
    splitter.split(dummy_X, y, groups=groups)
)

# Verify that no group crosses the split boundary
train_groups = np.unique(groups[train_idx])
test_groups = np.unique(groups[test_idx])

overlap = np.intersect1d(train_groups, test_groups)

if len(overlap) > 0:
    raise ValueError("Identical feature groups cross train/test!")

# Summarize class counts
all_counts = pd.Series(y).value_counts()
train_counts = pd.Series(y[train_idx]).value_counts()
test_counts = pd.Series(y[test_idx]).value_counts()

summary = pd.DataFrame({
    "All": all_counts,
    "Train": train_counts,
    "Test": test_counts
}).fillna(0).astype(int)

summary["Train %"] = (
    summary["Train"] / summary["All"] * 100
).round(3)

summary["Test %"] = (
    summary["Test"] / summary["All"] * 100
).round(3)

summary = summary.sort_values("All")

print("\nMulticlass split distribution:")
print(summary.to_string())

print("\nTraining records:", len(train_idx))
print("Testing records:", len(test_idx))
print("Shared feature groups:", len(overlap))

print("\nClasses missing from training:",
      summary[summary["Train"] == 0].index.tolist())

print("Classes missing from testing:",
      summary[summary["Test"] == 0].index.tolist())

print("\nRare classes (fewer than 100 records):")
print(
    summary[summary["All"] < 100][
        ["All", "Train", "Test"]
    ].to_string()
)

# Save a separate multiclass split
split_path = ARTIFACTS_DIR / "multiclass_split_indices.npz"

np.savez_compressed(
    split_path,
    train_idx=train_idx,
    test_idx=test_idx
)

report_path = REPORTS_DIR / "multiclass_split_class_distribution.csv"
summary.to_csv(report_path)

print("\nSaved multiclass split:", split_path)
print("Saved class distribution:", report_path)