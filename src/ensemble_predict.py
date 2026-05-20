"""Aggregate predictions from one or more BERT runs (and optionally the TF-IDF baseline),
then write the final Kaggle submission with overlap-constraint applied.

Each BERT run directory must contain:
- args.json   (with `fold` and `model`)
- val_probs.npy   shape (n_fold_val, 5)  internal idx order
- test_probs.npy  shape (1444, 5)        internal idx order
- val_index.csv   row_id -> original train row index (so we can reconstruct OOF)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    LABEL_LIST,
    NUM_CLASSES,
    OUTPUTS_DIR,
    SEED,
    idx_to_submission_label,
    load_test,
    load_train,
    macro_f1,
    make_folds,
    report,
    write_submission,
)


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def collect_run_oof(run_dirs: list[Path], train: pd.DataFrame) -> tuple[np.ndarray, list[Path]]:
    """Stitch val_probs across folds into a single OOF tensor aligned with train rows.

    Assumes all runs share the same fold assignment (seed=42, n_splits=5) so we can
    place each fold's val_probs back into train order.
    """
    oof = np.zeros((len(train), NUM_CLASSES), dtype=np.float64)
    filled = np.zeros(len(train), dtype=bool)
    used: list[Path] = []
    for d in run_dirs:
        args_path = d / "args.json"
        if not args_path.exists():
            continue
        args = json.loads(args_path.read_text())
        fold = args["fold"]
        val_probs = np.load(d / "val_probs.npy")
        fold_idx = train.index[train["fold"] == fold].to_numpy()
        if len(fold_idx) != len(val_probs):
            raise ValueError(
                f"{d}: val_probs len {len(val_probs)} != fold {fold} size {len(fold_idx)}"
            )
        oof[fold_idx] += val_probs
        filled[fold_idx] = True
        used.append(d)
    if not filled.all():
        missing = (~filled).sum()
        raise ValueError(f"OOF coverage incomplete: {missing} train rows unfilled")
    return oof, used


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bert-runs",
        nargs="*",
        default=[],
        help="Run directories or glob patterns containing val_probs/test_probs.",
    )
    p.add_argument(
        "--include-tfidf",
        action="store_true",
        help="Include TF-IDF baseline probs in the ensemble (small weight by default).",
    )
    p.add_argument("--tfidf-weight", type=float, default=0.1)
    p.add_argument("--tag", default="final")
    p.add_argument(
        "--no-overlap-constraint",
        action="store_true",
        help="Disable the train/test overlap label-set constraint.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train = make_folds(load_train(), seed=SEED).reset_index(drop=True)
    test = load_test()
    train["h"] = train["condition"].map(md5)
    test["h"] = test["condition"].map(md5)

    # Resolve --bert-runs: accept dirs or glob patterns.
    resolved: list[Path] = []
    for pattern in args.bert_runs:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        if not matches:
            p = Path(pattern)
            if p.is_dir():
                matches = [p]
        resolved.extend(matches)
    resolved = sorted(set(resolved))
    print(f"Found {len(resolved)} BERT run dirs:")
    for d in resolved:
        print(f"  {d}")

    bert_oof = bert_test = None
    if resolved:
        # Group runs by fold and average; if there are multiple seeds per fold, mean over them.
        runs_by_fold: dict[int, list[Path]] = {}
        for d in resolved:
            cfg = json.loads((d / "args.json").read_text())
            runs_by_fold.setdefault(cfg["fold"], []).append(d)

        oof = np.zeros((len(train), NUM_CLASSES), dtype=np.float64)
        test_acc = np.zeros((len(test), NUM_CLASSES), dtype=np.float64)
        n_test_runs = 0
        for fold, dirs in sorted(runs_by_fold.items()):
            fold_oof = np.mean([np.load(d / "val_probs.npy") for d in dirs], axis=0)
            fold_test = np.mean([np.load(d / "test_probs.npy") for d in dirs], axis=0)
            fold_idx = train.index[train["fold"] == fold].to_numpy()
            oof[fold_idx] = fold_oof
            test_acc += fold_test
            n_test_runs += 1
            print(
                f"  fold {fold}: {len(dirs)} run(s), "
                f"val macro F1 = {macro_f1(train.loc[fold_idx, 'label_idx'], fold_oof.argmax(1)):.4f}"
            )
        bert_oof = oof
        bert_test = test_acc / max(n_test_runs, 1)

        bert_oof_pred = bert_oof.argmax(1)
        print(f"\nBERT OOF Macro F1: {macro_f1(train['label_idx'], bert_oof_pred):.4f}")
        print(report(train["label_idx"], bert_oof_pred))

    tfidf_oof = tfidf_test = None
    if args.include_tfidf:
        tfidf_oof = np.load(OUTPUTS_DIR / "oof_tfidf_logreg.npy")
        tfidf_test = np.load(OUTPUTS_DIR / "test_tfidf_logreg.npy")
        print(f"TF-IDF OOF Macro F1: {macro_f1(train['label_idx'], tfidf_oof.argmax(1)):.4f}")

    # Combine.
    if bert_oof is not None and tfidf_oof is not None:
        w = args.tfidf_weight
        oof = (1 - w) * bert_oof + w * tfidf_oof
        test_probs = (1 - w) * bert_test + w * tfidf_test
        print(f"\nEnsemble (BERT*{1-w:.2f} + TFIDF*{w:.2f})")
    elif bert_oof is not None:
        oof = bert_oof
        test_probs = bert_test
        print("\nUsing BERT only")
    elif tfidf_oof is not None:
        oof = tfidf_oof
        test_probs = tfidf_test
        print("\nUsing TF-IDF only (no BERT runs found)")
    else:
        raise SystemExit("No predictions to ensemble — pass --bert-runs and/or --include-tfidf")

    ens_pred = oof.argmax(1)
    print(f"Ensemble OOF Macro F1: {macro_f1(train['label_idx'], ens_pred):.4f}")
    print(report(train["label_idx"], ens_pred))

    # Test prediction with overlap constraint.
    overlap: dict[str, set[int]] = {}
    for h, lbl in zip(train["h"], train["label_idx"]):
        overlap.setdefault(h, set()).add(int(lbl))

    test_pred = test_probs.argmax(1).copy()
    if not args.no_overlap_constraint:
        n_changed = 0
        for i, h in enumerate(test["h"]):
            allowed = overlap.get(h)
            if allowed and len(allowed) < NUM_CLASSES:
                masked = np.full(NUM_CLASSES, -np.inf)
                for j in allowed:
                    masked[j] = test_probs[i, j]
                new_pred = int(np.argmax(masked))
                if new_pred != test_pred[i]:
                    n_changed += 1
                test_pred[i] = new_pred
        print(f"\nOverlap-constraint changed {n_changed} test predictions.")

    out_path = write_submission(test_pred, tag=args.tag)
    print(f"\nWrote final submission: {out_path}")

    pred_dist = pd.Series(idx_to_submission_label(test_pred)).value_counts().sort_index()
    train_dist = train["label_id"].value_counts(normalize=True).sort_index().round(3)
    print("\nSubmission label distribution (counts):")
    print(pred_dist)
    print("\nSubmission distribution vs train (proportions):")
    pred_prop = (pred_dist / pred_dist.sum()).round(3)
    print(pd.DataFrame({"submission": pred_prop, "train": train_dist}))


if __name__ == "__main__":
    main()
