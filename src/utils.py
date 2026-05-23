"""Shared helpers: official label mapping, CV splits, metrics, seeding."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

TRAIN_CSV = DATA_DIR / "kaggle_trainset.csv"
TEST_CSV = DATA_DIR / "kaggle_testset.csv"
SUBMISSION_TEMPLATE_CSV = DATA_DIR / "kaggle_testset_submission.csv"

# Official mapping (from competition description). DO NOT change.
LABEL2ID = {
    "neoplasms": 1,
    "digestive system diseases": 2,
    "nervous system diseases": 3,
    "cardiovascular diseases": 4,
    "general pathological conditions": 5,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Internal contiguous labels for model training (0..4), separated from submission labels (1..5).
LABEL_LIST = list(LABEL2ID.keys())
LABEL2IDX = {name: i for i, name in enumerate(LABEL_LIST)}
IDX2LABEL = {i: name for name, i in LABEL2IDX.items()}
NUM_CLASSES = len(LABEL2ID)

N_SPLITS = 5
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_train() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV)
    df = df.dropna(subset=["label", "condition"]).reset_index(drop=True)
    df["label"] = df["label"].str.strip().str.lower()
    unknown = set(df["label"]) - set(LABEL2ID)
    if unknown:
        raise ValueError(f"Unknown labels in trainset: {unknown}")
    df["label_idx"] = df["label"].map(LABEL2IDX).astype(int)
    df["label_id"] = df["label"].map(LABEL2ID).astype(int)
    return df


def load_test() -> pd.DataFrame:
    df = pd.read_csv(TEST_CSV)
    df = df.reset_index(drop=True)
    df["index"] = df.index.astype(int)
    return df


def make_folds(df: pd.DataFrame, n_splits: int = N_SPLITS, seed: int = SEED) -> pd.DataFrame:
    """Return df with an added `fold` column (0..n_splits-1)."""
    df = df.copy()
    df["fold"] = -1
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(skf.split(df, df["label_idx"])):
        df.loc[val_idx, "fold"] = fold
    assert (df["fold"] >= 0).all()
    return df


def make_folds_grouped(df: pd.DataFrame, n_splits: int = N_SPLITS, seed: int = SEED) -> pd.DataFrame:
    """GroupKFold by text hash — all rows with the same text go to the SAME fold.

    Used for multi-label BCE training to prevent cross-fold target sharing:
    if text T appears as (T, A) and (T, B) in train, both rows must be in the
    same fold so that BCE multi-hot targets don't leak val labels into train.

    Uses StratifiedGroupKFold to maintain class balance per fold.
    """
    import hashlib
    from sklearn.model_selection import StratifiedGroupKFold

    df = df.copy()
    df["_text_hash"] = df["condition"].map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(
        sgkf.split(df, df["label_idx"], df["_text_hash"])
    ):
        df.loc[val_idx, "fold"] = fold
    assert (df["fold"] >= 0).all()
    df = df.drop(columns=["_text_hash"])
    return df


def save_fold_assignment(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or OUTPUTS_DIR / "fold_assignment.csv"
    df[["fold"]].to_csv(path, index_label="row_id")
    return path


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def report(y_true, y_pred) -> str:
    return classification_report(
        y_true, y_pred, target_names=LABEL_LIST, digits=4, zero_division=0
    )


def idx_to_submission_label(idx_array: np.ndarray) -> np.ndarray:
    """Map internal indices (0..4) to official submission ids (1..5)."""
    return np.vectorize(lambda i: LABEL2ID[IDX2LABEL[int(i)]])(idx_array).astype(int)


def write_submission(pred_idx: np.ndarray, tag: str) -> Path:
    """Write a Kaggle submission CSV from internal indices and run sanity checks."""
    template = pd.read_csv(SUBMISSION_TEMPLATE_CSV)
    if len(pred_idx) != len(template):
        raise ValueError(f"pred length {len(pred_idx)} != template length {len(template)}")
    sub = template.copy()
    sub["label"] = idx_to_submission_label(pred_idx)

    # Sanity checks.
    assert len(sub) == 1444, f"expected 1444 rows, got {len(sub)}"
    assert sub["index"].tolist() == list(range(len(sub))), "index column corrupted"
    assert sub["label"].isin([1, 2, 3, 4, 5]).all(), "label contains values outside 1..5"
    assert sub["label"].isna().sum() == 0, "label contains NaN"

    out_path = OUTPUTS_DIR / f"submission_{tag}.csv"
    sub.to_csv(out_path, index=False)
    return out_path
