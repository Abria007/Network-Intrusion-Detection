
import numpy as np
import pandas as pd

from common import DATA_DIR, SPLIT_FILE

# Load all original features and labels
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

split = np.load(SPLIT_FILE)
train_idx = split["train_idx"]
test_idx = split["test_idx"]

if len(data) != len(train_idx) + len(test_idx):
    raise ValueError("Split indices do not match dataset size.")

feature_columns = [c for c in data.columns if c != "Label"]
X = data[feature_columns]

# Hash feature rows; identical feature vectors get the same hash
hashes = pd.util.hash_pandas_object(
    X, index=False
).to_numpy()

train_hashes = set(hashes[train_idx])
test_hashes = set(hashes[test_idx])

overlap = train_hashes.intersection(test_hashes)

print("\nUnique train feature hashes:", len(train_hashes))
print("Unique test feature hashes:", len(test_hashes))
print("Feature hashes shared across train/test:", len(overlap))

print("\nThis checks exact feature-vector overlap.")
print("It does not measure similarity between different feature vectors.")