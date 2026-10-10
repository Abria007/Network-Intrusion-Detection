
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from common import DATA_DIR, REPORTS_DIR


# Create reports folder if it does not exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No processed CSV files found in {DATA_DIR}"
    )

all_labels = []

# Read only the Label column to save memory
for file in csv_files:
    print("Reading:", file.name)

    header = pd.read_csv(file, nrows=0)

    label_column = next(
        (column for column in header.columns
         if column.strip() == "Label"),
        None
    )

    if label_column is None:
        raise ValueError(f"Label column not found in {file.name}")

    labels = pd.read_csv(
        file,
        usecols=[label_column],
        low_memory=False
    )

    labels.columns = ["Label"]
    labels["Label"] = labels["Label"].astype("string").str.strip()

    all_labels.append(labels)

# Combine labels from all processed files
label_data = pd.concat(all_labels, ignore_index=True)

# Report missing labels separately
label_data["Label"] = label_data["Label"].fillna("<MISSING LABEL>")

# Calculate class counts and percentages
class_counts = (
    label_data["Label"]
    .value_counts()
    .rename_axis("Label")
    .reset_index(name="Count")
)

class_counts["Percentage"] = (
    class_counts["Count"] / len(label_data) * 100
).round(4)

# Save the distribution
csv_path = REPORTS_DIR / "multiclass_class_distribution.csv"
class_counts.to_csv(csv_path, index=False)

# Print results
print("\nTotal rows:", len(label_data))
print("Total classes:", label_data["Label"].nunique())
print("\nClass distribution:")
print(class_counts.to_string(index=False))

# Plot class imbalance
plot_data = class_counts.sort_values("Count", ascending=True)

plt.figure(figsize=(12, 8))
plt.barh(plot_data["Label"], plot_data["Count"])
plt.xscale("log")
plt.xlabel("Number of records (log scale)")
plt.ylabel("Traffic class")
plt.title("Multiclass Network Intrusion Dataset: Class Distribution")
plt.tight_layout()

plot_path = REPORTS_DIR / "multiclass_class_distribution.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved class distribution:", csv_path)
print("Saved class distribution plot:", plot_path)