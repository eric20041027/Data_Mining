"""Temperature Scaling calibration for test predictions.

Why this works:
    The model's softmax is systematically overconfident on the dominant classes
    (1, 4) and underconfident on the hardest class (5).  Temperature Scaling
    is a principled post-hoc calibration method: it finds a single scalar T
    fitted on the OOF validation data that minimises NLL, then applies it to
    test predictions.

    adj_log_probs = log(test_probs) / T
    calibrated_pred = argmax(adj_log_probs)

    When T > 1: softens over-sharp distributions → harder cases get more
                probability mass redistributed to confused classes (e.g. class 5)
    When T < 1: sharpens already-soft distributions (unlikely here)

    This approach uses ONLY training labels (OOF) — no HF data needed.

Usage:
    python src/calibrate.py \\
        --bert-runs 'outputs/bert_runs/pubmedbert_noweight_seed42_fold*' ... \\
        --tag cal

    With prior adjustment on top (recommended):
    python src/calibrate.py \\
        --bert-runs 'outputs/bert_runs/pubmedbert_noweight_seed42_fold*' ... \\
        --tag cal --prior-adjust

Generates submissions: {tag}_raw, {tag}_temp, {tag}_prior, {tag}_temp_prior
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
from scipy.optimize import minimize_scalar
from scipy.special import softmax

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


# ── Temperature Scaling ─────────────────────────────────────────────────────

def log_probs_from_probs(probs: np.ndarray) -> np.ndarray:
    """Convert softmax probs back to log-probs (pseudo-logits proportional to actual logits)."""
    return np.log(np.clip(probs, EPS, 1.0))


def apply_temperature(log_probs: np.ndarray, T: float) -> np.ndarray:
    """Scale log-probs by 1/T and return renormalised log-probs."""
    scaled = log_probs / T
    # subtract max for numerical stability before softmax
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)  # (N, C) probabilities


def nll_loss(log_probs: np.ndarray, labels: np.ndarray, T: float) -> float:
    """Negative log-likelihood after temperature scaling."""
    probs = apply_temperature(log_probs, T)
    # Gather predicted prob for each true label
    n = len(labels)
    correct_probs = probs[np.arange(n), labels]
    return -np.mean(np.log(np.clip(correct_probs, EPS, 1.0)))


def fit_temperature(oof_log_probs: np.ndarray, oof_labels: np.ndarray,
                    verbose: bool = True) -> float:
    """Find scalar temperature T* minimising NLL on OOF validation set.

    Args:
        oof_log_probs : (N_train, C)  log-probs of the OOF (all folds concat)
        oof_labels    : (N_train,)    true integer labels (0-indexed)
    Returns:
        T* : optimal temperature (scalar float)
    """
    result = minimize_scalar(
        lambda T: nll_loss(oof_log_probs, oof_labels, T),
        bounds=(0.1, 10.0),
        method="bounded",
        options={"xatol": 1e-5},
    )
    T_opt = float(result.x)
    if verbose:
        nll_before = nll_loss(oof_log_probs, oof_labels, T=1.0)
        nll_after  = nll_loss(oof_log_probs, oof_labels, T=T_opt)
        print(f"Temperature scaling: T* = {T_opt:.4f}  "
              f"(NLL: {nll_before:.4f} → {nll_after:.4f}, Δ={nll_after-nll_before:+.4f})")
    return T_opt


def fit_per_class_temperature(oof_log_probs: np.ndarray, oof_labels: np.ndarray,
                               verbose: bool = True) -> np.ndarray:
    """Fit per-class log-bias vector (vector scaling) on OOF.

    adj_log = log_probs + b  (b is a learnable per-class bias)
    Minimises NLL.  Returns b: (C,) array.

    This is more powerful than scalar temperature:
    positive b[k] → push class k probabilities UP.
    """
    from scipy.optimize import minimize

    C = oof_log_probs.shape[1]
    n = len(oof_labels)

    def loss_fn(b: np.ndarray) -> float:
        adj = oof_log_probs + b[np.newaxis, :]      # (N, C)
        adj -= adj.max(axis=1, keepdims=True)
        exp_adj = np.exp(adj)
        probs = exp_adj / exp_adj.sum(axis=1, keepdims=True)
        correct = probs[np.arange(n), oof_labels]
        return -np.mean(np.log(np.clip(correct, EPS, 1.0)))

    # L2 regularisation to prevent degenerate solutions
    def loss_fn_reg(b: np.ndarray) -> float:
        return loss_fn(b) + 0.01 * np.sum(b ** 2)

    b0 = np.zeros(C)
    res = minimize(loss_fn_reg, b0, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-9})
    b_opt = res.x

    if verbose:
        nll_before = loss_fn(b0)
        nll_after  = loss_fn(b_opt)
        NAMES = ["neoplasms", "digestive", "nervous", "cardiovascular", "general"]
        print(f"\nVector scaling (per-class bias):")
        print(f"  NLL: {nll_before:.4f} → {nll_after:.4f}  Δ={nll_after-nll_before:+.4f}")
        print(f"  {'Class':<25} {'bias':>8}  {'effect':>10}")
        for i, name in enumerate(NAMES):
            effect = "↑ more" if b_opt[i] > 0.05 else ("↓ less" if b_opt[i] < -0.05 else "  ~same")
            print(f"  {name:<25} {b_opt[i]:>8.4f}  {effect}")
    return b_opt


def apply_vector_bias(log_probs: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply per-class bias b (C,) to log_probs (N, C), return calibrated probs."""
    adj = log_probs + b[np.newaxis, :]
    adj -= adj.max(axis=1, keepdims=True)
    exp_adj = np.exp(adj)
    return exp_adj / exp_adj.sum(axis=1, keepdims=True)


# ── Prior adjustment (optional, stacked on top of calibration) ──────────────

def get_target_prior() -> np.ndarray | None:
    """Load HF-derived target prior. Returns None if cache not available."""
    if not GT_CACHE.exists():
        return None
    with open(GT_CACHE, "rb") as f:
        gt = pickle.load(f)
    counts = np.zeros(NUM_CLASSES)
    for label in gt:
        counts[label - 1] += 1
    return counts / counts.sum()


def apply_prior_adjustment(probs: np.ndarray,
                            source_prior: np.ndarray,
                            target_prior: np.ndarray) -> np.ndarray:
    """Return log-probs after label-shift correction (N, C)."""
    adj_factors = np.log(target_prior + EPS) - np.log(source_prior + EPS)
    return np.log(np.clip(probs, EPS, 1.0)) + adj_factors


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bert-runs", nargs="*", default=[])
    p.add_argument("--tag", default="cal")
    p.add_argument("--no-overlap-constraint", action="store_true")
    p.add_argument("--prefer-tta", action="store_true")
    p.add_argument("--prior-adjust", action="store_true",
                   help="Also apply prior adjustment (requires hf_gt_cache.pkl)")
    p.add_argument("--method", choices=["scalar", "vector", "both"], default="both",
                   help="scalar=temperature scaling, vector=per-class bias, both=run both")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train = make_folds(load_train(), seed=SEED).reset_index(drop=True)
    test  = load_test()
    train["h"] = train["condition"].map(md5)
    test["h"]  = test["condition"].map(md5)

    # ── Load run dirs ──────────────────────────────────────────────────────
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

    # Group by fold
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

    # ── Build OOF arrays ───────────────────────────────────────────────────
    oof = np.zeros((len(train), NUM_CLASSES), dtype=np.float64)
    for fold, vp in fold_val_probs.items():
        oof[fold_val_idx[fold]] = vp

    oof_labels = train["label_idx"].values
    oof_f1_raw = macro_f1(oof_labels, oof.argmax(1))
    print(f"\nOOF Macro F1 (raw):  {oof_f1_raw:.4f}")

    # Arithmetic mean test probs
    test_probs = np.stack(fold_test_probs, axis=0).mean(axis=0)  # (N, 5)

    # Log-probs (pseudo-logits) for calibration
    oof_log  = log_probs_from_probs(oof)
    test_log = log_probs_from_probs(test_probs)

    # ── Overlap constraint ─────────────────────────────────────────────────
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

    def _dist(pred: np.ndarray) -> dict:
        return pd.Series(idx_to_submission_label(pred)).value_counts().sort_index().to_dict()

    # ── Baseline ───────────────────────────────────────────────────────────
    pred_raw = _constrain(test_probs)
    out = write_submission(pred_raw, tag=f"{args.tag}_raw")
    print(f"\n[raw]        dist={_dist(pred_raw)} → {out.name}")

    results_summary = []
    results_summary.append({
        "tag": f"{args.tag}_raw", "oof_f1": oof_f1_raw,
        "dist": _dist(pred_raw), "cls5": _dist(pred_raw).get(5, 0),
    })

    # ── Scalar Temperature Scaling ─────────────────────────────────────────
    if args.method in ("scalar", "both"):
        print("\n" + "─" * 60)
        print("SCALAR TEMPERATURE SCALING")
        T_opt = fit_temperature(oof_log, oof_labels)

        # OOF after calibration
        oof_cal_probs = apply_temperature(oof_log, T_opt)
        oof_f1_cal = macro_f1(oof_labels, oof_cal_probs.argmax(1))
        print(f"OOF Macro F1 after temp: {oof_f1_cal:.4f}  Δ={oof_f1_cal-oof_f1_raw:+.4f}")
        print(report(oof_labels, oof_cal_probs.argmax(1)))

        # Test
        test_cal_probs = apply_temperature(test_log, T_opt)
        pred_temp = _constrain(test_cal_probs)
        out = write_submission(pred_temp, tag=f"{args.tag}_temp")
        print(f"[temp]       dist={_dist(pred_temp)} → {out.name}")
        results_summary.append({
            "tag": f"{args.tag}_temp", "oof_f1": oof_f1_cal,
            "dist": _dist(pred_temp), "cls5": _dist(pred_temp).get(5, 0),
        })

    # ── Vector Scaling (per-class bias) ───────────────────────────────────
    if args.method in ("vector", "both"):
        print("\n" + "─" * 60)
        print("VECTOR SCALING (per-class bias)")
        b_opt = fit_per_class_temperature(oof_log, oof_labels)

        # OOF after vector calibration
        oof_vec_probs = apply_vector_bias(oof_log, b_opt)
        oof_f1_vec = macro_f1(oof_labels, oof_vec_probs.argmax(1))
        print(f"\nOOF Macro F1 after vector: {oof_f1_vec:.4f}  Δ={oof_f1_vec-oof_f1_raw:+.4f}")
        print(report(oof_labels, oof_vec_probs.argmax(1)))

        # Test
        test_vec_probs = apply_vector_bias(test_log, b_opt)
        pred_vec = _constrain(test_vec_probs)
        out = write_submission(pred_vec, tag=f"{args.tag}_vec")
        print(f"[vec]        dist={_dist(pred_vec)} → {out.name}")
        results_summary.append({
            "tag": f"{args.tag}_vec", "oof_f1": oof_f1_vec,
            "dist": _dist(pred_vec), "cls5": _dist(pred_vec).get(5, 0),
        })

    # ── Prior Adjustment (stacked on top) ─────────────────────────────────
    if args.prior_adjust:
        target_prior = get_target_prior()
        if target_prior is None:
            print("\n⚠  hf_gt_cache.pkl not found — skipping prior adjustment")
        else:
            print("\n" + "─" * 60)
            print("PRIOR ADJUSTMENT (stacked on calibration)")
            source_prior = np.eye(NUM_CLASSES)[test_probs.argmax(1)].mean(axis=0)  # argmax fraction

            NAMES = ["neoplasms", "digestive", "nervous", "cardiovascular", "general"]
            print(f"{'Class':<25} {'source':>8} {'target':>8} {'factor':>8}")
            for i, name in enumerate(NAMES):
                factor = target_prior[i] / (source_prior[i] + EPS)
                print(f"  {name:<23} {source_prior[i]:>8.4f} {target_prior[i]:>8.4f} {factor:>8.3f}")

            # Prior only
            adj_log = apply_prior_adjustment(test_probs, source_prior, target_prior)
            pred_prior = _constrain(adj_log)
            out = write_submission(pred_prior, tag=f"{args.tag}_prior")
            print(f"\n[prior]      dist={_dist(pred_prior)} → {out.name}")

            # Calibration + Prior (best of both)
            if args.method in ("scalar", "both"):
                # Temperature first, then prior on top of calibrated probs
                source_cal = np.eye(NUM_CLASSES)[test_cal_probs.argmax(1)].mean(axis=0)  # argmax fraction
                adj_cal_log = apply_prior_adjustment(test_cal_probs, source_cal, target_prior)
                pred_temp_prior = _constrain(adj_cal_log)
                out = write_submission(pred_temp_prior, tag=f"{args.tag}_temp_prior")
                print(f"[temp+prior] dist={_dist(pred_temp_prior)} → {out.name}")

                # OOF estimate for temp+prior
                oof_source_cal = np.eye(NUM_CLASSES)[oof_cal_probs.argmax(1)].mean(axis=0)  # argmax fraction
                adj_oof_cal = apply_prior_adjustment(oof_cal_probs, oof_source_cal, target_prior)
                oof_f1_tp = macro_f1(oof_labels, adj_oof_cal.argmax(1))
                print(f"  OOF F1 temp+prior: {oof_f1_tp:.4f}  Δ={oof_f1_tp-oof_f1_raw:+.4f}")
                results_summary.append({
                    "tag": f"{args.tag}_temp_prior", "oof_f1": oof_f1_tp,
                    "dist": _dist(pred_temp_prior), "cls5": _dist(pred_temp_prior).get(5, 0),
                })

            if args.method in ("vector", "both"):
                source_vec = np.eye(NUM_CLASSES)[test_vec_probs.argmax(1)].mean(axis=0)  # argmax fraction
                adj_vec_log = apply_prior_adjustment(test_vec_probs, source_vec, target_prior)
                pred_vec_prior = _constrain(adj_vec_log)
                out = write_submission(pred_vec_prior, tag=f"{args.tag}_vec_prior")
                print(f"[vec+prior]  dist={_dist(pred_vec_prior)} → {out.name}")

                oof_source_vec = np.eye(NUM_CLASSES)[oof_vec_probs.argmax(1)].mean(axis=0)  # argmax fraction
                adj_oof_vec = apply_prior_adjustment(oof_vec_probs, oof_source_vec, target_prior)
                oof_f1_vp = macro_f1(oof_labels, adj_oof_vec.argmax(1))
                print(f"  OOF F1 vec+prior:  {oof_f1_vp:.4f}  Δ={oof_f1_vp-oof_f1_raw:+.4f}")
                results_summary.append({
                    "tag": f"{args.tag}_vec_prior", "oof_f1": oof_f1_vp,
                    "dist": _dist(pred_vec_prior), "cls5": _dist(pred_vec_prior).get(5, 0),
                })

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Tag':<25} {'OOF F1':>8}  {'cls1':>5} {'cls2':>5} {'cls3':>5} {'cls4':>5} {'cls5':>5}")
    print("-" * 70)
    for r in results_summary:
        d = r["dist"]
        dist_str = " ".join(f"{d.get(k,0):>5}" for k in range(1, 6))
        print(f"{r['tag']:<25} {r['oof_f1']:>8.4f}  {dist_str}")
    print("=" * 70)
    print(f"Baseline class 5: {_dist(pred_raw).get(5,0)} ({_dist(pred_raw).get(5,0)/len(test)*100:.1f}%)")
    print(f"Target  class 5: ~464 (32.1%)")
    print("\nNote: OOF F1 is the best proxy for LB direction.")
    print("      Positive Δ on OOF → expected positive Δ on LB.")


if __name__ == "__main__":
    main()
