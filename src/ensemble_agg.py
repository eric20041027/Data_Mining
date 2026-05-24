"""Try multiple probability-aggregation methods on the same BERT runs.

Generates four submission CSVs in one pass and prints an OOF comparison table:
  arith  - arithmetic mean (same as current ensemble_predict.py)
  geom   - geometric mean  (log-space average, then renormalise)
  pow2   - power mean p=2  (amplifies high-confidence predictions)
  pow05  - power mean p=0.5 (softens high-confidence predictions)

Usage (same flag syntax as ensemble_predict.py):
    python src/ensemble_agg.py \
        --bert-runs 'outputs/bert_runs/pubmedbert_noweight_*' \
        --tag v_agg
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

EPS = 1e-9  # guard log(0)


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _power_mean(arrays: list[np.ndarray], p: float) -> np.ndarray:
    """Generalised power mean of a list of (N, C) probability arrays.

    p=1  -> arithmetic mean
    p->0 -> geometric mean
    p=2  -> quadratic mean (amplifies dominant class)
    p=0.5-> square-root mean (softens dominant class)
    """
    stacked = np.stack(arrays, axis=0)  # (K, N, C)
    powered = np.power(np.clip(stacked, EPS, 1.0), p)
    mean_powered = powered.mean(axis=0)            # (N, C)
    result = np.power(mean_powered, 1.0 / p)       # (N, C)
    # Renormalise so each row sums to 1 (argmax is unchanged, but keeps semantics clean)
    result = result / result.sum(axis=1, keepdims=True)
    return result


def _geom_mean(arrays: list[np.ndarray]) -> np.ndarray:
    """Geometric mean across K models for each (N, C) probability matrix."""
    log_sum = np.zeros_like(arrays[0])
    for arr in arrays:
        log_sum += np.log(np.clip(arr, EPS, 1.0))
    log_mean = log_sum / len(arrays)
    result = np.exp(log_mean)
    result = result / result.sum(axis=1, keepdims=True)
    return result


def aggregate(fold_arrays: list[np.ndarray], method: str) -> np.ndarray:
    """Aggregate a list of (N, C) arrays (one per fold) into a single (N, C) array."""
    if method == "arith":
        return np.stack(fold_arrays, axis=0).mean(axis=0)
    if method == "geom":
        return _geom_mean(fold_arrays)
    if method == "pow2":
        return _power_mean(fold_arrays, p=2.0)
    if method == "pow05":
        return _power_mean(fold_arrays, p=0.5)
    raise ValueError(f"Unknown method: {method}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bert-runs", nargs="*", default=[],
                   help="Run directories or glob patterns.")
    p.add_argument("--tag", default="agg")
    p.add_argument("--no-overlap-constraint", action="store_true")
    p.add_argument("--prefer-tta", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train = make_folds(load_train(), seed=SEED).reset_index(drop=True)
    test = load_test()
    train["h"] = train["condition"].map(md5)
    test["h"] = test["condition"].map(md5)

    # Resolve run dirs
    resolved: list[Path] = []
    for pattern in args.bert_runs:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        if not matches:
            p2 = Path(pattern)
            if p2.is_dir():
                matches = [p2]
        resolved.extend(matches)
    resolved = sorted(set(resolved))

    if not resolved:
        raise SystemExit("No BERT run dirs found — pass --bert-runs")

    print(f"Found {len(resolved)} BERT run dirs:")
    for d in resolved:
        print(f"  {d}")

    # Group by fold
    runs_by_fold: dict[int, list[Path]] = {}
    for d in resolved:
        cfg = json.loads((d / "args.json").read_text())
        runs_by_fold.setdefault(cfg["fold"], []).append(d)

    def _test_probs(run_dir: Path) -> np.ndarray:
        tta = run_dir / "test_probs_tta.npy"
        if args.prefer_tta and tta.exists():
            return np.load(tta)
        return np.load(run_dir / "test_probs.npy")

    # Per-fold averaged probs (arithmetic within fold, same as original script)
    fold_val_probs: dict[int, np.ndarray] = {}   # fold -> (n_fold_val, 5)
    fold_test_probs: list[np.ndarray] = []        # list of (1444, 5), one per fold
    fold_val_idx: dict[int, np.ndarray] = {}

    for fold, dirs in sorted(runs_by_fold.items()):
        vp = np.mean([np.load(d / "val_probs.npy") for d in dirs], axis=0)
        tp = np.mean([_test_probs(d) for d in dirs], axis=0)
        fold_val_probs[fold] = vp
        fold_test_probs.append(tp)
        fold_val_idx[fold] = train.index[train["fold"] == fold].to_numpy()
        print(f"  fold {fold}: {len(dirs)} run(s), "
              f"val macro F1 = {macro_f1(train.loc[fold_val_idx[fold], 'label_idx'], vp.argmax(1)):.4f}")

    # Overlap map (for optional constraint)
    overlap: dict[str, set[int]] = {}
    for h, lbl in zip(train["h"], train["label_idx"]):
        overlap.setdefault(h, set()).add(int(lbl))

    methods = ["arith", "geom", "pow2", "pow05"]
    results = []

    for method in methods:
        # OOF: each fold uses its own val_probs (aggregation across folds doesn't apply to OOF
        # since each sample is only in one fold's val set)
        oof = np.zeros((len(train), NUM_CLASSES), dtype=np.float64)
        for fold, vp in fold_val_probs.items():
            oof[fold_val_idx[fold]] = vp   # per-fold val probs unchanged
        oof_f1 = macro_f1(train["label_idx"], oof.argmax(1))

        # Test: aggregate across folds
        test_probs = aggregate(fold_test_probs, method)

        # Optional overlap constraint
        test_pred = test_probs.argmax(1).copy()
        if not args.no_overlap_constraint:
            for i, h in enumerate(test["h"]):
                allowed = overlap.get(h)
                if allowed and len(allowed) < NUM_CLASSES:
                    masked = np.full(NUM_CLASSES, -np.inf)
                    for j in allowed:
                        masked[j] = test_probs[i, j]
                    test_pred[i] = int(np.argmax(masked))

        tag = f"{args.tag}_{method}"
        out = write_submission(test_pred, tag=tag)

        dist = pd.Series(idx_to_submission_label(test_pred)).value_counts().sort_index()
        results.append({"method": method, "oof_f1": oof_f1, "submission": out.name,
                         "dist": dist.to_dict()})
        print(f"\n[{method}] OOF Macro F1 = {oof_f1:.4f} | submission -> {out.name}")
        print(report(train["label_idx"], oof.argmax(1)))

    # Summary table
    print("\n" + "=" * 70)
    print("AGGREGATION METHOD COMPARISON")
    print("=" * 70)
    header = f"{'Method':<8} {'OOF F1':>8}  " + "  ".join(f"cls{k}" for k in range(1, 6))
    print(header)
    print("-" * 70)
    for r in results:
        dist_str = "  ".join(f"{r['dist'].get(k, 0):>4}" for k in range(1, 6))
        print(f"{r['method']:<8} {r['oof_f1']:>8.4f}  {dist_str}   <- {r['submission']}")
    print("=" * 70)
    print("Note: OOF F1 is identical across methods (each sample in exactly one fold val).")
    print("Differences will only show up on the LB. Submit the one with most balanced dist.")


if __name__ == "__main__":
    main()
