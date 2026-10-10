
from common import load_dataset, get_split
import pandas as pd

df, X, y, feature_cols = load_dataset()

train_idx, test_idx = get_split(X, y)

train_hashes = pd.util.hash_pandas_object(
    X.iloc[train_idx], index=False
)

test_hashes = pd.util.hash_pandas_object(
    X.iloc[test_idx], index=False
)

overlap = len(
    set(train_hashes).intersection(set(test_hashes))
)

print("\nDataset shape:", df.shape)
print("Feature count:", len(feature_cols))
print("Training shape:", X.iloc[train_idx].shape)
print("Testing shape:", X.iloc[test_idx].shape)
print("Training label counts:")
print(y.iloc[train_idx].value_counts().sort_index())
print("Testing label counts:")
print(y.iloc[test_idx].value_counts().sort_index())
print("Cross-split feature-hash overlap:", overlap)

if overlap != 0:
    raise ValueError("Train/test feature overlap detected.")