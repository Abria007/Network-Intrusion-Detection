
import pandas as pd

from common import DATA_DIR, REPORTS_DIR

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No processed CSV files found.")

chunk_size = 100_000
group_results = []

for file in csv_files:
    print("Auditing:", file.name)

    header = pd.read_csv(file, nrows=0)
    columns = {c.strip(): c for c in header.columns}

    if "Label" not in columns:
        raise ValueError(f"Label column missing in {file.name}")

    label_column = columns["Label"]

    for chunk in pd.read_csv(
        file, chunksize=chunk_size, low_memory=False
    ):
        chunk.columns = chunk.columns.str.strip()
        chunk["Label"] = chunk["Label"].astype("string").str.strip()

        # Keep missing labels visible in the audit
        chunk["Label"] = chunk["Label"].fillna("<MISSING LABEL>")

        feature_columns = [
            c for c in chunk.columns if c != "Label"
        ]

        # Hash identical feature rows consistently
        hashes = pd.util.hash_pandas_object(
            chunk[feature_columns], index=False
        )

        pairs = pd.DataFrame({
            "hash": hashes.to_numpy(),
            "Label": chunk["Label"].to_numpy()
        })

        # Aggregate within each chunk to reduce memory usage
        grouped = (
            pairs.groupby(["hash", "Label"], dropna=False)
            .size()
            .reset_index(name="Rows")
        )

        group_results.append(grouped)

# Combine the aggregated results from all chunks
combined = pd.concat(group_results, ignore_index=True)

combined = (
    combined.groupby(["hash", "Label"], as_index=False)["Rows"]
    .sum()
)

# A feature hash is conflicting if it has multiple original labels
label_counts = combined.groupby("hash")["Label"].nunique()
conflicting_hashes = label_counts[label_counts > 1].index

conflicts = combined[combined["hash"].isin(conflicting_hashes)].copy()

conflict_summary = (
    conflicts.groupby("Label")["Rows"]
    .sum()
    .sort_values(ascending=False)
)

total_conflict_rows = int(conflicts["Rows"].sum())

print("\nUnique feature groups with conflicting labels:",
      len(conflicting_hashes))
print("Rows belonging to conflicting groups:", total_conflict_rows)

print("\nLabels involved in conflicting groups:")
print(conflict_summary.to_string())

output_path = REPORTS_DIR / "multiclass_duplicate_label_audit.csv"
conflicts.to_csv(output_path, index=False)

print("\nSaved detailed audit:", output_path)