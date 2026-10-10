
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SPLIT_FILE = ARTIFACTS_DIR / "group_split_indices.npz"
MODEL_FILE = MODELS_DIR / "xgboost_intrusion.joblib"


def create_folders():
    for folder in [ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def load_dataset():
    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))

    if not files:
        raise FileNotFoundError(
            f"No processed CSV files found in {DATA_DIR}"
        )

    frames = []

    for file in files:
        print("Loading:", Path(file).name)
        frames.append(pd.read_csv(file, low_memory=False))

    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()

    feature_cols = [
        col for col in df.columns if col != "Label"
    ]

    X = df[feature_cols].copy()
    y = (df["Label"] != "BENIGN").astype(int)

    return df, X, y, feature_cols


def get_split(X, y, rebuild=False):
    create_folders()

    if SPLIT_FILE.exists() and not rebuild:
        saved = np.load(SPLIT_FILE)
        train_idx = saved["train_idx"]
        test_idx = saved["test_idx"]

        if len(train_idx) + len(test_idx) == len(X):
            print("Reusing saved split.")
            return train_idx, test_idx

        print("Dataset size changed. Creating a new split.")

    groups = pd.util.hash_pandas_object(
        X, index=False
    ).to_numpy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    np.savez_compressed(
        SPLIT_FILE,
        train_idx=train_idx,
        test_idx=test_idx
    )

    print("Split indices saved to:", SPLIT_FILE)

    return train_idx, test_idx