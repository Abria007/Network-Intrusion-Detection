
import numpy as np
import pandas as pd

from common import DATA_DIR, SPLIT_FILE

# Load original labels without converting them to binary labels
all_labels = []

for file in sorted(DATA_DIR.glob("*.csv")):
    header = pd.read_csv(file, nrows=0)
    label_column = next(
        (c for c in header.columns if c.strip() == "Label"),
        None
    )

    if label_column is None:
        raise ValueError(f"Label column missing in {file.name}")

    part = pd.read_csv(
        file, usecols=[label_column], low_memory=False
    )
    part.columns = ["Label"]
    part["Label"] = part["Label"].astype("string").str.strip()
    all_labels.append(part)

labels = pd.concat(all_labels, ignore_index=True)["Label"]

if labels.isna().any():
    raise ValueError("Missing original labels found. Investigate before splitting.")

# Load the existing split indices; do not modify the saved file
split = np.load(SPLIT_FILE)

print("Keys in saved split:", split.files)

# The existing workflow stores these keys
train_idx = split["train_idx"]
test_idx = split["test_idx"]

if len(train_idx) + len(test_idx) != len(labels):
    raise ValueError(
        "Saved split row count does not match the original labels."
    )

if np.intersect1d(train_idx, test_idx).size:
    raise ValueError("Train/test indices overlap!")

train_labels = labels.iloc[train_idx]
test_labels = labels.iloc[test_idx]

summary = pd.DataFrame({
    "All": labels.value_counts(),
    "Train": train_labels.value_counts(),
    "Test": test_labels.value_counts()
}).fillna(0).astype(int)

summary["Train %"] = (
    summary["Train"] / summary["All"] * 100
).round(3)

summary["Test %"] = (
    summary["Test"] / summary["All"] * 100
).round(3)

summary = summary.sort_values("All")

print("\nClass distribution in existing split:")
print(summary.to_string())

print("\nClasses missing from training:")
print(summary[summary["Train"] == 0].index.tolist())

print("\nClasses missing from testing:")
print(summary[summary["Test"] == 0].index.tolist())


print("\nRare-class split check (classes with fewer than 100 records):")
print(
    summary[summary["All"] < 100][
        ["All", "Train", "Test", "Train %", "Test %"]
    ].to_string()
)

print("\nTotal training records:", len(train_idx))
print("Total testing records:", len(test_idx))
print("Training fraction:", round(len(train_idx) / len(labels), 4))
print("Testing fraction:", round(len(test_idx) / len(labels), 4))