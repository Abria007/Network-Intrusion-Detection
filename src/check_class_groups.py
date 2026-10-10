
import pandas as pd

from common import DATA_DIR

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

feature_columns = [c for c in data.columns if c != "Label"]

# Group exact duplicate feature vectors
data["feature_hash"] = pd.util.hash_pandas_object(
    data[feature_columns],
    index=False
).to_numpy()

# Count records and distinct feature groups per class
class_stats = (
    data.groupby("Label")
    .agg(
        Records=("feature_hash", "size"),
        Distinct_feature_groups=("feature_hash", "nunique")
    )
    .sort_values("Records")
)

print("\nRecords and distinct feature groups by class:")
print(class_stats.to_string())

print("\nSaved no files; this is a diagnostic only.")