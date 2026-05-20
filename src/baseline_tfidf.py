"""TF-IDF + Logistic Regression baseline with 5-fold OOF.

Also produces a first submission (`submission_tfidf_baseline.csv`) using:
- Model prediction on test
- Overlap lookup: if a test text appears in train, restrict prediction to the
  set of labels observed for that text in train (legal: uses only train labels).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils import (
    LABEL2IDX,
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
    set_seed,
    write_submission,
)


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def build_overlap_label_sets(train: pd.DataFrame) -> dict[str, set[int]]:
    """text_hash -> set of label indices observed for that text in train."""
    out: dict[str, set[int]] = {}
    for h, lbl in zip(train["h"], train["label_idx"]):
        out.setdefault(h, set()).add(int(lbl))
    return out


def constrained_argmax(probs: np.ndarray, allowed: set[int]) -> int:
    mask = np.full(probs.shape, -np.inf, dtype=np.float64)
    for i in allowed:
        mask[i] = probs[i]
    return int(np.argmax(mask))


def main() -> None:
    set_seed(SEED)
    t0 = time.time()

    train = load_train()
    test = load_test()
    train["h"] = train["condition"].map(md5)
    test["h"] = test["condition"].map(md5)

    train = make_folds(train)
    overlap = build_overlap_label_sets(train)
    n_test_overlap = test["h"].isin(overlap.keys()).sum()
    print(f"Test rows overlapping train (by exact text): {n_test_overlap}/{len(test)}")

    # TF-IDF — fit on ALL train (per-fold refit gives same vocab basically; we'll
    # refit per fold to be strict and avoid val leakage into vocab statistics).
    oof = np.zeros((len(train), NUM_CLASSES), dtype=np.float64)
    test_probs_folds: list[np.ndarray] = []

    for fold in range(int(train["fold"].max()) + 1):
        tr_mask = train["fold"] != fold
        va_mask = train["fold"] == fold
        tr = train.loc[tr_mask]
        va = train.loc[va_mask]

        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            sublinear_tf=True,
            max_features=200_000,
            strip_accents="unicode",
            lowercase=True,
        )
        Xtr = vec.fit_transform(tr["condition"])
        Xva = vec.transform(va["condition"])
        Xte = vec.transform(test["condition"])

        clf = LogisticRegression(
            C=4.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=SEED,
        )
        # Multiclass via one-vs-rest with liblinear; for stable per-fold timing.
        from sklearn.multiclass import OneVsRestClassifier

        clf = OneVsRestClassifier(clf, n_jobs=-1)
        clf.fit(Xtr, tr["label_idx"])

        # decision_function isn't probability; use predict_proba via calibrated estimator
        # OvR on LR already exposes predict_proba.
        va_probs = clf.predict_proba(Xva)
        te_probs = clf.predict_proba(Xte)

        oof[va_mask.values] = va_probs
        test_probs_folds.append(te_probs)

        fold_pred = va_probs.argmax(1)
        f1 = macro_f1(va["label_idx"], fold_pred)
        print(f"[fold {fold}] val Macro F1 = {f1:.4f}  (n_train={tr.shape[0]}, vocab={Xtr.shape[1]})")

    oof_pred = oof.argmax(1)
    overall_f1 = macro_f1(train["label_idx"], oof_pred)
    print(f"\nOOF Macro F1: {overall_f1:.4f}")
    print("\nPer-class report (OOF):")
    print(report(train["label_idx"], oof_pred))

    # Average fold probabilities for test predictions.
    test_probs = np.mean(test_probs_folds, axis=0)

    # Unconstrained test prediction.
    test_pred_uncon = test_probs.argmax(1)

    # Constrained: for test rows whose text appears in train, restrict to that label set.
    test_pred_con = test_pred_uncon.copy()
    n_changed = 0
    for i, h in enumerate(test["h"]):
        allowed = overlap.get(h)
        if allowed and len(allowed) < NUM_CLASSES:
            new = constrained_argmax(test_probs[i], allowed)
            if new != test_pred_uncon[i]:
                n_changed += 1
            test_pred_con[i] = new
    print(f"Overlap-constraint changed {n_changed} test predictions.")

    # Save OOF probs and submissions.
    np.save(OUTPUTS_DIR / "oof_tfidf_logreg.npy", oof)
    np.save(OUTPUTS_DIR / "test_tfidf_logreg.npy", test_probs)

    p1 = write_submission(test_pred_uncon, tag="tfidf_baseline_unconstrained")
    p2 = write_submission(test_pred_con, tag="tfidf_baseline")
    print(f"\nWrote: {p1.name}")
    print(f"Wrote: {p2.name}")

    # Save metrics.
    with open(OUTPUTS_DIR / "baseline_metrics.json", "w") as f:
        json.dump(
            {
                "oof_macro_f1": overall_f1,
                "n_overlap": int(n_test_overlap),
                "n_overlap_constraint_changes": int(n_changed),
                "elapsed_sec": time.time() - t0,
                "labels_internal_order": LABEL_LIST,
                "label2idx": LABEL2IDX,
            },
            f,
            indent=2,
        )
    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
