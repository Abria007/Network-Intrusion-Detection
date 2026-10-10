
import pandas as pd

from common import (
    load_dataset,
    get_split,
    REPORTS_DIR,
    create_folders
)

create_folders()

df, X, y, feature_cols = load_dataset()
train_idx, test_idx = get_split(X, y)


def audit_split(X_part, y_part, split_name):
    data = X_part.copy()
    data["_target"] = y_part.to_numpy()

    grouped = data.groupby(
        list(X_part.columns),
        dropna=False,
        sort=False
    )["_target"]

    label_counts = grouped.nunique()
    group_sizes = grouped.size()

    conflicting_groups = label_counts[label_counts > 1]

    conflicting_rows = int(
        group_sizes.loc[conflicting_groups.index].sum()
    )

    return {
        "Split": split_name,
        "Rows": len(X_part),
        "Unique feature groups": len(label_counts),
        "Conflicting groups": len(conflicting_groups),
        "Rows in conflicting groups": conflicting_rows,
        "Percent affected": round(
            100 * conflicting_rows / len(X_part), 5
        )
    }


results = pd.DataFrame([
    audit_split(X.iloc[train_idx], y.iloc[train_idx], "Training"),
    audit_split(X.iloc[test_idx], y.iloc[test_idx], "Testing")
])

output_file = REPORTS_DIR / "label_conflict_audit.csv"
results.to_csv(output_file, index=False)

print(results.to_string(index=False))
print("\nSaved audit:", output_file)