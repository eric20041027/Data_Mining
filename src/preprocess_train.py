"""Target-aware deduplication + regex cleaning for the training set.

Implements the cleaning strategy from docs/improvePlan.md:

Rule 0 (regex cleaning):
  Remove structural headers like "RESULTS--", "OBJECTIVE:", "BACKGROUND--" from
  abstract text so attention focuses on medical content, not paper formatting.

Rule 1 (drop general in multi-label):
  When the same exact text appears with both "general pathological conditions"
  (class 5) AND a specific organ class (1-4), DROP the general rows. The
  specific class is more informative and matches the dataset's labeling
  preference for test rows (we confirmed test class 5 ratio ~20% << train 33%).

Rule 2 (drop 2-specific conflicts):
  After Rule 1, if a text still has ≥2 different SPECIFIC labels (e.g., both
  cardiovascular AND neoplasms), drop the entire group. These rows are
  inherently ambiguous and only add noise.

Compliance: pure train-side data cleaning. No test labels involved. The
"insight" that test prefers specific over general was derived from our own
model predictions and train label distributions only.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from utils import LABEL_LIST, OUTPUTS_DIR, TRAIN_CSV  # noqa: E402

GENERAL_LABEL = "general pathological conditions"

# Match all-caps section headers followed by "--" or ":"
# Examples: "RESULTS--", "OBJECTIVE:", "BACKGROUND--", "METHODS:"
HEADER_PATTERN = re.compile(r"\b[A-Z]{2,}(?:--|:)\s*")


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    """Strip structural headers, collapse whitespace."""
    cleaned = HEADER_PATTERN.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def preprocess(
    train_df: pd.DataFrame,
    apply_regex: bool = True,
    apply_rule1: bool = True,
    apply_rule2: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return a cleaned training DataFrame."""
    df = train_df.copy()
    df["label"] = df["label"].str.strip().str.lower()
    n_orig = len(df)

    if apply_regex:
        before_lens = df["condition"].str.len()
        df["condition"] = df["condition"].map(clean_text)
        after_lens = df["condition"].str.len()
        chars_removed = (before_lens - after_lens).sum()
        if verbose:
            print(f"Regex: removed {chars_removed:,} chars across {n_orig} rows")

    # Hash AFTER cleaning so same-content (post-cleanup) groups together.
    df["h"] = df["condition"].map(md5)

    groups = df.groupby("h")["label"].agg(set).to_dict()
    multi_label_groups = {h: labels for h, labels in groups.items() if len(labels) > 1}
    if verbose:
        print(f"\nMulti-label unique texts: {len(multi_label_groups)}")

    keep_mask = pd.Series(True, index=df.index)

    n_general_dropped = 0
    n_conflict_dropped = 0

    for h, labels in multi_label_groups.items():
        rows = df.index[df["h"] == h]

        # Rule 1: drop "general" rows when specific labels coexist
        labels_after_r1 = set(labels)
        if apply_rule1 and GENERAL_LABEL in labels:
            general_rows = rows[df.loc[rows, "label"] == GENERAL_LABEL]
            keep_mask[general_rows] = False
            n_general_dropped += len(general_rows)
            labels_after_r1 = labels - {GENERAL_LABEL}

        # Rule 2: if still ≥2 distinct specific labels, drop entire group
        if apply_rule2 and len(labels_after_r1) >= 2:
            # Only the rows we haven't already dropped (e.g., general kept under no-Rule1 case)
            still_kept = rows[keep_mask.loc[rows]]
            keep_mask[still_kept] = False
            n_conflict_dropped += len(still_kept)

    cleaned = df[keep_mask].drop(columns=["h"]).reset_index(drop=True)

    if verbose:
        print(f"\nRule 1 (drop 'general' in multi-label): {n_general_dropped} rows dropped")
        print(f"Rule 2 (drop ≥2 specific conflicts):     {n_conflict_dropped} rows dropped")
        print(f"\nOriginal: {n_orig} rows")
        print(f"Cleaned : {len(cleaned)} rows  ({100*len(cleaned)/n_orig:.1f}%)")

        print(f"\nLabel distribution:")
        dist = cleaned["label"].value_counts().reindex(LABEL_LIST)
        ratios = dist / dist.sum()
        print(pd.DataFrame({"count": dist, "ratio": ratios.round(4)}))

        # Sanity check: any duplicates remaining?
        cleaned["h"] = cleaned["condition"].map(md5)
        remaining_groups = cleaned.groupby("h")["label"].agg(set)
        still_multi = (remaining_groups.map(len) > 1).sum()
        print(f"\nMulti-label groups remaining: {still_multi} (should be 0)")
        cleaned = cleaned.drop(columns=["h"])

    return cleaned


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--no-regex", action="store_true", help="Skip regex header removal")
    p.add_argument("--no-rule1", action="store_true", help="Keep 'general' in multi-label groups")
    p.add_argument("--no-rule2", action="store_true", help="Keep ≥2-specific conflicts")
    p.add_argument(
        "--out",
        default="outputs/kaggle_trainset_cleaned.csv",
        help="Output CSV path (relative to project root)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(TRAIN_CSV)
    cleaned = preprocess(
        train,
        apply_regex=not args.no_regex,
        apply_rule1=not args.no_rule1,
        apply_rule2=not args.no_rule2,
    )
    out_path = Path(__file__).resolve().parent.parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned[["label", "condition"]].to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
