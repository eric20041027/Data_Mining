"""Generate pseudo-labels from an existing ensemble's test predictions.

Reads test_probs.npy from a set of BERT runs (default: final_d's 4 models),
averages within-fold then across folds (matching ensemble_predict.py), takes
argmax + max_prob, filters by confidence, writes a pseudo-labelled CSV that
can be concatenated with kaggle_trainset.csv for retraining.

Output format: `label, condition` (same as kaggle_trainset.csv).

Compliance with Rule.md: pseudo-labels are derived ONLY from our own model's
predictions on test text features — never from any external label source or
from the test set's true labels.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from utils import LABEL_LIST, OUTPUTS_DIR, load_test  # noqa: E402


def aggregate_test_probs(patterns: list[str], project_root: Path) -> np.ndarray:
    """Mirror ensemble_predict.py logic: within each fold average over runs,
    then average across folds."""
    all_dirs: list[Path] = []
    for pattern in patterns:
        # Patterns are relative to project root.
        matches = sorted(Path(p) for p in glob.glob(str(project_root / pattern)))
        all_dirs.extend(matches)
    all_dirs = sorted(set(all_dirs))
    if not all_dirs:
        raise SystemExit(f"No runs found for patterns: {patterns}")

    by_fold: dict[int, list[Path]] = {}
    for d in all_dirs:
        cfg = json.loads((d / "args.json").read_text())
        by_fold.setdefault(cfg["fold"], []).append(d)

    print(f"Found {len(all_dirs)} runs across {len(by_fold)} folds")
    fold_test_means = []
    for fold, dirs in sorted(by_fold.items()):
        fold_test = np.mean([np.load(d / "test_probs.npy") for d in dirs], axis=0)
        fold_test_means.append(fold_test)
        print(f"  fold {fold}: {len(dirs)} run(s)")

    return np.mean(fold_test_means, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bert-runs",
        nargs="*",
        default=[
            "outputs/bert_runs/pubmedbert_noweight_seed42_fold*",
            "outputs/bert_runs/pubmedbert_noweight_seed2024_fold*",
            "outputs/bert_runs/biobert_noweight_seed*_fold*",
            "outputs/bert_runs/pubmedbertlarge_noweight_seed*_fold*",
        ],
        help="Glob patterns for source ensemble runs (default = final_d's 4 models)",
    )
    p.add_argument(
        "--threshold", type=float, default=0.80,
        help="Minimum max-prob to keep as pseudo-label (default 0.80)",
    )
    p.add_argument(
        "--top-n", type=int, default=None,
        help="Alternative: keep top N most confident rows (overrides --threshold)",
    )
    p.add_argument(
        "--top-n-per-class", type=int, default=None,
        help="Alternative: keep top N most confident rows PER predicted class "
             "(ensures class balance; overrides --threshold and --top-n)",
    )
    p.add_argument(
        "--out", default="outputs/pseudo_labels.csv",
        help="Output CSV path (relative to project root)",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent

    test_probs = aggregate_test_probs(args.bert_runs, root)
    print(f"\nAggregated test probs shape: {test_probs.shape}")

    max_probs = test_probs.max(axis=1)
    pred_idx = test_probs.argmax(axis=1)

    if args.top_n_per_class is not None:
        # Per-class top N: ensure each class is represented
        keep_mask = np.zeros(len(test_probs), dtype=bool)
        for cls_idx in range(len(LABEL_LIST)):
            cls_rows = np.where(pred_idx == cls_idx)[0]
            if len(cls_rows) == 0:
                print(f"  ⚠️ Class {cls_idx} ({LABEL_LIST[cls_idx]}): 0 predicted rows")
                continue
            # Take top N most confident in this class
            cls_confidence = max_probs[cls_rows]
            top_in_class = cls_rows[np.argsort(-cls_confidence)[: args.top_n_per_class]]
            keep_mask[top_in_class] = True
            avg_conf = max_probs[top_in_class].mean()
            print(f"  Class {cls_idx} ({LABEL_LIST[cls_idx]}): "
                  f"kept {len(top_in_class)}/{len(cls_rows)} (avg conf {avg_conf:.3f}, "
                  f"min {max_probs[top_in_class].min():.3f})")
        print(f"\nPer-class top-{args.top_n_per_class} selection")
    elif args.top_n is not None:
        top = np.argsort(-max_probs)[: args.top_n]
        keep_mask = np.zeros(len(test_probs), dtype=bool)
        keep_mask[top] = True
        print(f"\nKeeping top {args.top_n} most confident rows")
    else:
        keep_mask = max_probs >= args.threshold
        print(f"\nKeeping rows with max_prob >= {args.threshold}")

    n_keep = int(keep_mask.sum())
    print(f"Selected {n_keep}/{len(test_probs)} test rows ({n_keep / len(test_probs):.1%})")

    test = load_test()
    pseudo = pd.DataFrame({
        "label": [LABEL_LIST[i] for i in pred_idx[keep_mask]],
        "condition": test.loc[keep_mask, "condition"].values,
    })

    print("\n=== Pseudo-label distribution ===")
    print(pseudo["label"].value_counts())

    print("\n=== Confidence distribution of kept rows ===")
    print(pd.Series(max_probs[keep_mask]).describe().round(4))

    out_path = root / args.out
    out_path.parent.mkdir(exist_ok=True, parents=True)
    pseudo.to_csv(out_path, index=False)
    print(f"\nWrote {n_keep} pseudo-labelled rows to: {out_path}")


if __name__ == "__main__":
    main()
