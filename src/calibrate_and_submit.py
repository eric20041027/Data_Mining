"""Per-class logit calibration to fix balanced-class-weight overcorrection.

Searches a 5-dim bias vector b such that (log_probs + b).argmax maximises OOF
Macro F1. Then applies the same bias to test predictions and writes the
submission with overlap constraint.

Run inside the project root (where outputs/bert_runs/ lives).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

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


def collect_probs(pattern: str, train: pd.DataFrame, test: pd.DataFrame):
    """Returns oof (N,5) and test_mean (M,5). Averages duplicate runs per fold."""
    dirs = sorted(Path(p) for p in glob.glob(pattern))
    by_fold: dict[int, list[Path]] = {}
    for d in dirs:
        cfg = json.loads((d / "args.json").read_text())
        by_fold.setdefault(cfg["fold"], []).append(d)

    oof = np.zeros((len(train), NUM_CLASSES), dtype=np.float64)
    test_acc = np.zeros((len(test), NUM_CLASSES), dtype=np.float64)
    n_test = 0
    for fold, ds in sorted(by_fold.items()):
        fold_oof = np.mean([np.load(d / "val_probs.npy") for d in ds], axis=0)
        fold_test = np.mean([np.load(d / "test_probs.npy") for d in ds], axis=0)
        fold_idx = train.index[train["fold"] == fold].to_numpy()
        oof[fold_idx] = fold_oof
        test_acc += fold_test
        n_test += 1
        print(f"  fold {fold}: {len(ds)} run(s)")
    return oof, test_acc / max(n_test, 1)


def coord_descent_bias(
    oof_logp: np.ndarray,
    y: np.ndarray,
    rounds: int = 5,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Coordinate-descent search for additive bias b[c] maximising macro F1."""
    if grid is None:
        grid = np.linspace(-3.0, 3.0, 121)  # step 0.05
    K = oof_logp.shape[1]
    bias = np.zeros(K)
    base_pred = oof_logp.argmax(1)
    best_f1 = f1_score(y, base_pred, average="macro")
    print(f"  start macro F1 = {best_f1:.4f}")
    for r in range(rounds):
        improved = False
        for c in range(K):
            best_b_c = bias[c]
            for b in grid:
                trial = bias.copy()
                trial[c] = b
                pred = (oof_logp + trial).argmax(1)
                f1 = f1_score(y, pred, average="macro")
                if f1 > best_f1 + 1e-6:
                    best_f1 = f1
                    best_b_c = b
                    improved = True
            bias[c] = best_b_c
        print(f"  round {r}: macro F1 = {best_f1:.4f}, bias = {bias.round(3).tolist()}")
        if not improved:
            break
    return bias


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bert-runs",
        default="outputs/bert_runs/pubmedbert_base_seed42_fold*",
        help="Glob pattern for fold run directories.",
    )
    parser.add_argument("--tag", default="pubmedbert_v2_calibrated")
    parser.add_argument(
        "--no-overlap-constraint", action="store_true", help="Disable overlap constraint."
    )
    parser.add_argument(
        "--include-tfidf",
        action="store_true",
        help="Also ensemble in TF-IDF baseline probs (small weight).",
    )
    parser.add_argument("--tfidf-weight", type=float, default=0.1)
    args = parser.parse_args()

    train = make_folds(load_train(), seed=SEED).reset_index(drop=True)
    test = load_test()
    train["h"] = train["condition"].map(md5)
    test["h"] = test["condition"].map(md5)

    print(f"Collecting probs from: {args.bert_runs}")
    oof_bert, test_bert = collect_probs(args.bert_runs, train, test)

    if args.include_tfidf:
        oof_tfidf = np.load(OUTPUTS_DIR / "oof_tfidf_logreg.npy")
        test_tfidf = np.load(OUTPUTS_DIR / "test_tfidf_logreg.npy")
        w = args.tfidf_weight
        oof = (1 - w) * oof_bert + w * oof_tfidf
        test_probs = (1 - w) * test_bert + w * test_tfidf
        print(f"Mixed BERT*{1-w:.2f} + TFIDF*{w:.2f}")
    else:
        oof = oof_bert
        test_probs = test_bert

    print(f"\nUncalibrated OOF Macro F1: {macro_f1(train['label_idx'], oof.argmax(1)):.4f}")

    # Work in log-prob space so additive bias acts like a multiplicative scaler on probs.
    oof_logp = np.log(np.clip(oof, 1e-12, 1.0))
    test_logp = np.log(np.clip(test_probs, 1e-12, 1.0))

    print("\nCoarse search...")
    bias = coord_descent_bias(oof_logp, train["label_idx"].to_numpy())
    # Fine refinement.
    fine_grid = np.linspace(-0.4, 0.4, 81)
    print("\nFine refinement (anchored to coarse solution)...")
    K = NUM_CLASSES
    best_f1 = f1_score(train["label_idx"], (oof_logp + bias).argmax(1), average="macro")
    for r in range(5):
        improved = False
        for c in range(K):
            for d in fine_grid:
                trial = bias.copy()
                trial[c] = bias[c] + d
                pred = (oof_logp + trial).argmax(1)
                f1 = f1_score(train["label_idx"], pred, average="macro")
                if f1 > best_f1 + 1e-6:
                    best_f1 = f1
                    bias = trial
                    improved = True
        print(f"  fine round {r}: F1={best_f1:.4f}, bias={bias.round(3).tolist()}")
        if not improved:
            break

    oof_pred = (oof_logp + bias).argmax(1)
    print(f"\nCalibrated OOF Macro F1: {macro_f1(train['label_idx'], oof_pred):.4f}")
    print(report(train["label_idx"], oof_pred))

    # Apply bias to test.
    test_calibrated = test_logp + bias
    test_pred = test_calibrated.argmax(1).copy()

    if not args.no_overlap_constraint:
        overlap: dict[str, set[int]] = {}
        for h, lbl in zip(train["h"], train["label_idx"]):
            overlap.setdefault(h, set()).add(int(lbl))
        n_changed = 0
        for i, h in enumerate(test["h"]):
            allowed = overlap.get(h)
            if allowed and len(allowed) < NUM_CLASSES:
                masked = np.full(NUM_CLASSES, -np.inf)
                for j in allowed:
                    masked[j] = test_calibrated[i, j]
                new_pred = int(np.argmax(masked))
                if new_pred != test_pred[i]:
                    n_changed += 1
                test_pred[i] = new_pred
        print(f"\nOverlap-constraint changed {n_changed} test predictions.")

    out = write_submission(test_pred, tag=args.tag)
    print(f"\nWrote: {out}")

    pred_dist = pd.Series(idx_to_submission_label(test_pred)).value_counts().sort_index()
    train_prop = train["label_id"].value_counts(normalize=True).sort_index().round(3)
    sub_prop = (pred_dist / pred_dist.sum()).round(3)
    print("\nSubmission vs train distribution:")
    print(pd.DataFrame({"submission_count": pred_dist, "submission_prop": sub_prop, "train_prop": train_prop}))

    np.save(OUTPUTS_DIR / f"calibration_bias_{args.tag}.npy", bias)
    json.dump(
        {
            "tag": args.tag,
            "bias": bias.tolist(),
            "calibrated_oof_macro_f1": float(macro_f1(train["label_idx"], oof_pred)),
            "uncalibrated_oof_macro_f1": float(macro_f1(train["label_idx"], oof.argmax(1))),
        },
        open(OUTPUTS_DIR / f"calibration_meta_{args.tag}.json", "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
