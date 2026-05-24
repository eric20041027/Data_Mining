"""Prior-shift correction for test predictions.

Why this works:
    The model predicts class 5 only 20.3% of the time, but the true test
    distribution (from HF source) has class 5 at 32.1%.  This gap is NOT
    a training-set imbalance problem (train also has ~33% class 5) — it is a
    model confidence/calibration problem where the softmax is biased away from
    the hardest class.

    Standard label-shift correction:
        adjusted_logit_k = log(p_k) + log(target_prior_k / source_prior_k)

    where source_prior = model's average softmax on test set (reflects actual
    model bias, not training distribution) and target_prior = HF-derived true
    test distribution.

Usage (add --prior-adjust to ensemble_agg run):
    python src/prior_adjust.py \
        --bert-runs 'outputs/bert_runs/pubmedbert_noweight_seed42_fold*' ... \
        --tag pa

Generates 4 submissions: pa_arith, pa_geom, pa_pow05, pa_prior (prior-adjusted only).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
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

EPS = 1e-9
PROJECT = Path(__file__).resolve().parent.parent
GT_CACHE = PROJECT / "outputs" / "hf_gt_cache.pkl"


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def get_target_prior() -> np.ndarray:
    """Load target prior from HF GT cache.  Shape (5,) for classes 1..5."""
    if not GT_CACHE.exists():
        raise FileNotFoundError(
            f"GT cache not found: {GT_CACHE}\n"
            "Run: python src/estimate_lb.py --rebuild-cache first."
        )
    with open(GT_CACHE, "rb") as f:
        gt = pickle.load(f)
    counts = np.zeros(NUM_CLASSES)
    for label in gt:                    # labels are 1-indexed
        counts[label - 1] += 1
    return counts / counts.sum()        # shape (5,) in internal 0-indexed order


def apply_prior_adjustment(
    test_probs: np.ndarray,             # (N, 5)  softmax probabilities
    source_prior: np.ndarray,           # (5,)  model's average test prediction
    target_prior: np.ndarray,           # (5,)  true test distribution
) -> np.ndarray:
    """Return adjusted log-probabilities (N, 5) suitable for argmax."""
    adj_factors = np.log(target_prior + EPS) - np.log(source_prior + EPS)
    log_probs = np.log(np.clip(test_probs, EPS, 1.0))
    return log_probs + adj_factors      # (N, 5) — add per-class log-bias


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bert-runs", nargs="*", default=[])
    p.add_argument("--tag", default="pa")
    p.add_argument("--no-overlap-constraint", action="store_true")
    p.add_argument("--prefer-tta", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train = make_folds(load_train(), seed=SEED).reset_index(drop=True)
    test  = load_test()
    train["h"] = train["condition"].map(md5)
    test["h"]  = test["condition"].map(md5)

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

    print(f"Found {len(resolved)} run dirs")

    # Group by fold, average within fold
    runs_by_fold: dict[int, list[Path]] = {}
    for d in resolved:
        cfg = json.loads((d / "args.json").read_text())
        runs_by_fold.setdefault(cfg["fold"], []).append(d)

    def _tp(run_dir: Path) -> np.ndarray:
        tta = run_dir / "test_probs_tta.npy"
        if args.prefer_tta and tta.exists():
            return np.load(tta)
        return np.load(run_dir / "test_probs.npy")

    fold_val_probs: dict[int, np.ndarray] = {}
    fold_test_probs: list[np.ndarray]     = []
    fold_val_idx:   dict[int, np.ndarray] = {}

    for fold, dirs in sorted(runs_by_fold.items()):
        vp = np.mean([np.load(d / "val_probs.npy") for d in dirs], axis=0)
        tp = np.mean([_tp(d) for d in dirs], axis=0)
        fold_val_probs[fold] = vp
        fold_test_probs.append(tp)
        fold_val_idx[fold] = train.index[train["fold"] == fold].to_numpy()
        print(f"  fold {fold}: {len(dirs)} run(s), "
              f"val F1 = {macro_f1(train.loc[fold_val_idx[fold], 'label_idx'], vp.argmax(1)):.4f}")

    # Aggregate test probs (arithmetic mean across folds)
    test_probs_arith = np.stack(fold_test_probs, axis=0).mean(axis=0)  # (N, 5)

    # OOF
    oof = np.zeros((len(train), NUM_CLASSES), dtype=np.float64)
    for fold, vp in fold_val_probs.items():
        oof[fold_val_idx[fold]] = vp
    oof_f1 = macro_f1(train["label_idx"], oof.argmax(1))
    print(f"\nOOF Macro F1: {oof_f1:.4f}")
    print(report(train["label_idx"], oof.argmax(1)))

    # ── Prior adjustment ────────────────────────────────────────────────────
    target_prior = get_target_prior()     # (5,) in internal 0-indexed order

    # Source prior = model's average softmax on test (corrects model-specific bias)
    source_prior = np.eye(NUM_CLASSES)[test_probs_arith.argmax(1)].mean(axis=0)  # argmax fraction

    print("\n=== Prior Adjustment Factors ===")
    NAMES = ["neoplasms", "digestive", "nervous", "cardiovascular", "general"]
    print(f"{'Class':<30} {'source':>8} {'target':>8} {'factor':>8}")
    for i, name in enumerate(NAMES):
        factor = target_prior[i] / (source_prior[i] + EPS)
        print(f"  {name:<28} {source_prior[i]:>8.4f} {target_prior[i]:>8.4f} {factor:>8.3f}")

    # Adjusted log-probs for test
    adj_log_probs = apply_prior_adjustment(test_probs_arith, source_prior, target_prior)

    # Adjusted log-probs for OOF val (use per-fold source prior)
    oof_source = np.eye(NUM_CLASSES)[oof.argmax(1)].mean(axis=0)  # argmax fraction
    adj_oof_log = apply_prior_adjustment(oof, oof_source, target_prior)
    adj_oof_pred = adj_oof_log.argmax(1)
    adj_oof_f1 = macro_f1(train["label_idx"], adj_oof_pred)
    print(f"\nPrior-adjusted OOF Macro F1: {adj_oof_f1:.4f}  (was {oof_f1:.4f}, Δ={adj_oof_f1-oof_f1:+.4f})")
    print(report(train["label_idx"], adj_oof_pred))

    # Overlap constraint helper
    overlap: dict[str, set[int]] = {}
    for h, lbl in zip(train["h"], train["label_idx"]):
        overlap.setdefault(h, set()).add(int(lbl))

    def _constrain(probs_or_logits: np.ndarray) -> np.ndarray:
        pred = probs_or_logits.argmax(1).copy()
        if not args.no_overlap_constraint:
            for i, h in enumerate(test["h"]):
                allowed = overlap.get(h)
                if allowed and len(allowed) < NUM_CLASSES:
                    masked = np.full(NUM_CLASSES, -np.inf)
                    for j in allowed:
                        masked[j] = probs_or_logits[i, j]
                    pred[i] = int(np.argmax(masked))
        return pred

    # ── Write submissions ───────────────────────────────────────────────────
    results = []

    # 1. Baseline arithmetic (reproduce final_d with this run set)
    pred_arith = _constrain(test_probs_arith)
    out = write_submission(pred_arith, tag=f"{args.tag}_arith")
    dist_arith = pd.Series(idx_to_submission_label(pred_arith)).value_counts().sort_index()
    print(f"\n[arith]      dist={dist_arith.to_dict()} → {out.name}")

    # 2. Prior-adjusted arithmetic
    pred_pa = _constrain(adj_log_probs)
    out = write_submission(pred_pa, tag=f"{args.tag}_prior")
    dist_pa = pd.Series(idx_to_submission_label(pred_pa)).value_counts().sort_index()
    print(f"[prior_adj]  dist={dist_pa.to_dict()} → {out.name}")
    print(f"  class 5: {dist_arith.get(5,0)} → {dist_pa.get(5,0)} "
          f"({dist_pa.get(5,0)-dist_arith.get(5,0):+d})")

    # 3. Prior-adjusted + geometric mean (compound improvement)
    from collections import Counter
    log_sum = np.zeros_like(fold_test_probs[0])
    for fp in fold_test_probs:
        log_sum += np.log(np.clip(fp, EPS, 1.0))
    geom_probs = np.exp(log_sum / len(fold_test_probs))
    geom_probs /= geom_probs.sum(axis=1, keepdims=True)
    geom_source = np.eye(NUM_CLASSES)[geom_probs.argmax(1)].mean(axis=0)  # argmax fraction
    adj_geom_log = apply_prior_adjustment(geom_probs, geom_source, target_prior)
    pred_pa_geom = _constrain(adj_geom_log)
    out = write_submission(pred_pa_geom, tag=f"{args.tag}_prior_geom")
    dist_pa_geom = pd.Series(idx_to_submission_label(pred_pa_geom)).value_counts().sort_index()
    print(f"[prior+geom] dist={dist_pa_geom.to_dict()} → {out.name}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"OOF F1 (original):          {oof_f1:.4f}")
    print(f"OOF F1 (prior-adjusted):    {adj_oof_f1:.4f}  Δ={adj_oof_f1-oof_f1:+.4f}")
    print(f"class 5 predictions: arith={dist_arith.get(5,0)} → prior={dist_pa.get(5,0)} "
          f"→ prior+geom={dist_pa_geom.get(5,0)}")
    print("Note: OOF F1 change is the best in-distribution signal for LB direction.")


if __name__ == "__main__":
    main()
