"""F1-Threshold Optimization for test predictions.

Why this beats Temperature/Vector Scaling:
    Vector scaling minimises NLL (cross-entropy) which is a PROXY metric.
    This script directly optimises OOF Macro F1 — the actual competition metric.

    For each class k, we learn a per-class log-bias b_k:
        adj_log[i, k] = log(p[i, k]) + b_k
        pred[i]       = argmax_k(adj_log[i, :])

    Positive b_k → model predicts class k MORE often.
    Negative b_k → model predicts class k LESS often.

    The biases are fitted using Differential Evolution (global optimizer) on
    OOF data to maximise Macro F1 directly.

    Can be stacked on top of vector scaling calibration (--method vec+f1).

Usage:
    python src/threshold_opt.py \\
        --bert-runs 'outputs/bert_runs/pubmedbert_noweight_seed42_fold*' ... \\
        --tag f1opt

    Stack on existing calibration:
    python src/threshold_opt.py \\
        --bert-runs ... --tag f1opt --method vec+f1 --prior-adjust
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
from scipy.optimize import differential_evolution, minimize

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


def log_p(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probs, EPS, 1.0))


def apply_bias(lp: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply per-class bias and return probabilities."""
    adj = lp + b[np.newaxis, :]
    adj -= adj.max(axis=1, keepdims=True)
    e = np.exp(adj)
    return e / e.sum(axis=1, keepdims=True)


# ── F1-optimised thresholds ──────────────────────────────────────────────────

def fit_f1_thresholds(oof_lp: np.ndarray, oof_labels: np.ndarray,
                      regularise: float = 0.05,
                      verbose: bool = True) -> np.ndarray:
    """Find per-class biases b* that maximise OOF Macro F1.

    Uses Differential Evolution (global) followed by Nelder-Mead (local polish).

    Args:
        oof_lp     : (N_train, C) log-probs from OOF
        oof_labels : (N_train,)   true 0-indexed labels
        regularise : L2 penalty on b to prevent extreme solutions
    Returns:
        b* : (C,) optimal per-class biases
    """
    NAMES = ["neoplasms", "digestive", "nervous", "cardiovascular", "general"]
    C = oof_lp.shape[1]
    f1_baseline = macro_f1(oof_labels, oof_lp.argmax(1))

    def objective(b: np.ndarray) -> float:
        pred = (oof_lp + b[np.newaxis, :]).argmax(1)
        f1 = macro_f1(oof_labels, pred)
        return -f1 + regularise * np.sum(b ** 2)

    # Phase 1: Global search with Differential Evolution
    bounds = [(-3.0, 3.0)] * C
    de_result = differential_evolution(
        objective, bounds,
        maxiter=800,
        tol=1e-5,
        seed=42,
        workers=1,
        mutation=(0.5, 1.5),
        recombination=0.9,
        popsize=15,
    )

    # Phase 2: Local polish with Nelder-Mead
    nm_result = minimize(objective, de_result.x, method="Nelder-Mead",
                         options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-6})

    b_opt = nm_result.x
    f1_opt = macro_f1(oof_labels, (oof_lp + b_opt[np.newaxis, :]).argmax(1))

    if verbose:
        print(f"\nF1-threshold optimisation:")
        print(f"  OOF Macro F1: {f1_baseline:.4f} → {f1_opt:.4f}  "
              f"Δ={f1_opt - f1_baseline:+.4f}")
        print(f"  {'Class':<25} {'bias':>8}  {'effect':>12}")
        for i, name in enumerate(NAMES):
            effect = "↑↑ more" if b_opt[i] > 0.5 else (
                     "↑ more"  if b_opt[i] > 0.1 else (
                     "↓ less"  if b_opt[i] < -0.1 else "  ~same"))
            print(f"  {name:<25} {b_opt[i]:>8.4f}  {effect}")

    return b_opt


# ── Vector scaling (for stacking) ────────────────────────────────────────────

def fit_vector_nll(oof_lp: np.ndarray, oof_labels: np.ndarray) -> np.ndarray:
    """Fit per-class bias to minimise NLL (for use as first stage)."""
    C, n = oof_lp.shape[1], len(oof_labels)

    def loss(b: np.ndarray) -> float:
        probs = apply_bias(oof_lp, b)
        nll = -np.mean(np.log(np.clip(probs[np.arange(n), oof_labels], EPS, 1.0)))
        return nll + 0.01 * np.sum(b ** 2)

    res = minimize(loss, np.zeros(C), method="L-BFGS-B", options={"maxiter": 500})
    return res.x


# ── Prior adjustment ──────────────────────────────────────────────────────────

def get_target_prior() -> np.ndarray | None:
    if not GT_CACHE.exists():
        return None
    with open(GT_CACHE, "rb") as f:
        gt = pickle.load(f)
    counts = np.zeros(NUM_CLASSES)
    for lbl in gt:
        counts[lbl - 1] += 1
    return counts / counts.sum()


def prior_adj_log(probs: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return log_p(probs) + np.log(target + EPS) - np.log(source + EPS)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bert-runs", nargs="*", default=[])
    p.add_argument("--tag", default="f1opt")
    p.add_argument("--no-overlap-constraint", action="store_true")
    p.add_argument("--prefer-tta", action="store_true")
    p.add_argument("--prior-adjust", action="store_true")
    p.add_argument("--method", choices=["f1", "vec+f1", "vec"], default="f1",
                   help="f1=direct F1 optimisation; "
                        "vec+f1=vector(NLL) first then F1 on top; "
                        "vec=vector scaling only (NLL, same as calibrate.py)")
    p.add_argument("--regularise", type=float, default=0.05,
                   help="L2 regularisation on biases (default 0.05; increase to be more conservative)")
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

    def _tp(d: Path) -> np.ndarray:
        tta = d / "test_probs_tta.npy"
        return np.load(tta) if args.prefer_tta and tta.exists() else np.load(d / "test_probs.npy")

    fold_val_probs: dict[int, np.ndarray] = {}
    fold_test_probs: list[np.ndarray] = []
    fold_val_idx: dict[int, np.ndarray] = {}

    for fold, dirs in sorted(runs_by_fold.items()):
        vp = np.mean([np.load(d / "val_probs.npy") for d in dirs], axis=0)
        tp = np.mean([_tp(d) for d in dirs], axis=0)
        fold_val_probs[fold] = vp
        fold_test_probs.append(tp)
        fold_val_idx[fold] = train.index[train["fold"] == fold].to_numpy()
        print(f"  fold {fold}: {len(dirs)} run(s), "
              f"val F1={macro_f1(train.loc[fold_val_idx[fold], 'label_idx'], vp.argmax(1)):.4f}")

    # Build OOF
    oof = np.zeros((len(train), NUM_CLASSES))
    for fold, vp in fold_val_probs.items():
        oof[fold_val_idx[fold]] = vp
    oof_labels = train["label_idx"].values
    f1_raw = macro_f1(oof_labels, oof.argmax(1))
    print(f"\nOOF Macro F1 (raw): {f1_raw:.4f}")
    print(report(oof_labels, oof.argmax(1)))

    test_probs = np.stack(fold_test_probs).mean(0)
    oof_lp    = log_p(oof)
    test_lp   = log_p(test_probs)

    # Overlap constraint
    overlap: dict[str, set[int]] = {}
    for h, lbl in zip(train["h"], train["label_idx"]):
        overlap.setdefault(h, set()).add(int(lbl))

    def constrain(lp_or_probs: np.ndarray) -> np.ndarray:
        pred = lp_or_probs.argmax(1).copy()
        if not args.no_overlap_constraint:
            for i, h in enumerate(test["h"]):
                allowed = overlap.get(h)
                if allowed and len(allowed) < NUM_CLASSES:
                    masked = np.full(NUM_CLASSES, -np.inf)
                    for j in allowed:
                        masked[j] = lp_or_probs[i, j]
                    pred[i] = int(np.argmax(masked))
        return pred

    def dist(pred: np.ndarray) -> dict:
        return pd.Series(idx_to_submission_label(pred)).value_counts().sort_index().to_dict()

    rows = []

    def add_result(tag: str, test_pred: np.ndarray, oof_f1: float) -> None:
        out = write_submission(test_pred, tag=tag)
        d = dist(test_pred)
        rows.append({"tag": tag, "oof_f1": oof_f1, "dist": d})
        print(f"[{tag:<28}] oof={oof_f1:.4f}  dist={d}  → {out.name}")

    # ── Baseline ──────────────────────────────────────────────────────────
    add_result(f"{args.tag}_raw", constrain(test_probs), f1_raw)

    # ── Stage 1: vector scaling (NLL) if needed ───────────────────────────
    if args.method in ("vec", "vec+f1"):
        print("\n─── Stage 1: Vector scaling (NLL) ───")
        b_vec = fit_vector_nll(oof_lp, oof_labels)
        oof_vec  = apply_bias(oof_lp,  b_vec)
        test_vec = apply_bias(test_lp, b_vec)
        f1_vec = macro_f1(oof_labels, oof_vec.argmax(1))
        print(f"OOF F1 after vec: {f1_vec:.4f}  Δ={f1_vec-f1_raw:+.4f}")
        add_result(f"{args.tag}_vec", constrain(test_vec), f1_vec)

        # Use vec-calibrated probs as input to F1 optimisation
        oof_lp_for_f1  = log_p(oof_vec)
        test_lp_for_f1 = log_p(test_vec)
    else:
        oof_lp_for_f1  = oof_lp
        test_lp_for_f1 = test_lp

    # ── Stage 2: F1-threshold optimisation ───────────────────────────────
    if args.method in ("f1", "vec+f1"):
        stage_name = "Stage 2" if args.method == "vec+f1" else "Stage 1"
        print(f"\n─── {stage_name}: F1-threshold optimisation (regularise={args.regularise}) ───")
        b_f1 = fit_f1_thresholds(oof_lp_for_f1, oof_labels,
                                  regularise=args.regularise)
        oof_f1opt  = apply_bias(oof_lp_for_f1,  b_f1)
        test_f1opt = apply_bias(test_lp_for_f1, b_f1)
        f1_f1opt = macro_f1(oof_labels, oof_f1opt.argmax(1))
        print(f"OOF F1 after F1-opt: {f1_f1opt:.4f}  Δ={f1_f1opt-f1_raw:+.4f}")
        print(report(oof_labels, oof_f1opt.argmax(1)))
        add_result(f"{args.tag}_f1opt", constrain(test_f1opt), f1_f1opt)

    # ── Prior adjustment ──────────────────────────────────────────────────
    if args.prior_adjust:
        target = get_target_prior()
        if target is None:
            print("\n⚠  hf_gt_cache.pkl not found — skipping prior adjustment")
        else:
            print("\n─── Prior adjustment ───")
            NAMES = ["neoplasms", "digestive", "nervous", "cardiovascular", "general"]

            if args.method in ("f1", "vec+f1"):
                # Prior on top of F1-optimised probs
                source = test_f1opt.mean(axis=0)
                adj_log = prior_adj_log(test_f1opt, source, target)
                test_f1opt_prior = apply_bias(log_p(test_f1opt),
                                              np.log(target + EPS) - np.log(source + EPS))
                pred_fp = constrain(adj_log)

                oof_source = oof_f1opt.mean(axis=0)
                adj_oof = prior_adj_log(oof_f1opt, oof_source, target)
                f1_fp = macro_f1(oof_labels, adj_oof.argmax(1))

                print(f"{'Class':<25} {'source':>8} {'target':>8} {'factor':>8}")
                for i, n in enumerate(NAMES):
                    print(f"  {n:<23} {source[i]:>8.4f} {target[i]:>8.4f} "
                          f"{target[i]/(source[i]+EPS):>8.3f}")
                add_result(f"{args.tag}_f1opt_prior", pred_fp, f1_fp)

            if args.method in ("vec", "vec+f1"):
                # Prior on top of vec-calibrated probs
                source_v = test_vec.mean(axis=0)
                adj_v = prior_adj_log(test_vec, source_v, target)
                pred_vp = constrain(adj_v)
                oof_sv = oof_vec.mean(axis=0)
                f1_vp = macro_f1(oof_labels, prior_adj_log(oof_vec, oof_sv, target).argmax(1))
                add_result(f"{args.tag}_vec_prior", pred_vp, f1_vp)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"{'Tag':<32} {'OOF F1':>8}  "
          + "  ".join(f"cls{k}" for k in range(1, 6)))
    print("-" * 75)
    for r in rows:
        d = r["dist"]
        dstr = "  ".join(f"{d.get(k,0):>5}" for k in range(1, 6))
        print(f"{r['tag']:<32} {r['oof_f1']:>8.4f}  {dstr}")
    print("=" * 75)
    print(f"Target class 5: 464 (32.1%)")
    best = max(rows, key=lambda x: x["oof_f1"])
    print(f"Best OOF: {best['tag']} = {best['oof_f1']:.4f}")


if __name__ == "__main__":
    main()
