"""Quick EDA: distribution, lengths, duplicates between train and test."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from utils import LABEL_LIST, load_test, load_train


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def main() -> None:
    train = load_train()
    test = load_test()

    print("=" * 60)
    print("Shapes")
    print("=" * 60)
    print("Train:", train.shape)
    print("Test :", test.shape)

    print("\n" + "=" * 60)
    print("Class distribution (train)")
    print("=" * 60)
    counts = train["label"].value_counts().reindex(LABEL_LIST)
    ratios = counts / counts.sum()
    dist = pd.DataFrame({"count": counts, "ratio": ratios.round(4)})
    print(dist)

    print("\n" + "=" * 60)
    print("Character length stats")
    print("=" * 60)
    print("Train condition:")
    print(train["condition"].str.len().describe().round(1))
    print("\nTest condition:")
    print(test["condition"].str.len().describe().round(1))

    print("\n" + "=" * 60)
    print("Word count stats (whitespace split)")
    print("=" * 60)
    train_wc = train["condition"].str.split().str.len()
    test_wc = test["condition"].str.split().str.len()
    print("Train words:")
    print(train_wc.describe().round(1))
    print("\nTest words:")
    print(test_wc.describe().round(1))

    print("\n" + "=" * 60)
    print("Duplicates")
    print("=" * 60)
    train_hash = train["condition"].map(md5)
    test_hash = test["condition"].map(md5)
    n_train_dup = train_hash.duplicated().sum()
    n_test_dup = test_hash.duplicated().sum()
    overlap = set(train_hash) & set(test_hash)
    print(f"Duplicate rows within train: {n_train_dup}")
    print(f"Duplicate rows within test : {n_test_dup}")
    print(f"Train/test overlap (exact)  : {len(overlap)}")

    print("\n" + "=" * 60)
    print("Possible empty / very short rows")
    print("=" * 60)
    short_train = (train["condition"].str.len() < 50).sum()
    short_test = (test["condition"].str.len() < 50).sum()
    print(f"Train rows with <50 chars : {short_train}")
    print(f"Test rows  with <50 chars : {short_test}")

    print("\n" + "=" * 60)
    print("Per-class length (chars, mean)")
    print("=" * 60)
    print(train.groupby("label")["condition"].apply(lambda s: s.str.len().mean()).round(1))

    print("\n" + "=" * 60)
    print("Token-length estimate via PubMedBERT tokenizer")
    print("=" * 60)
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        )
        sample = train["condition"].sample(n=min(500, len(train)), random_state=42)
        lens = sample.map(lambda x: len(tok(x, add_special_tokens=True)["input_ids"]))
        print(f"Sampled n={len(sample)} train rows")
        print(lens.describe().round(1))
        print(f">512 tokens : {(lens > 512).mean():.2%}")
    except Exception as exc:  # noqa: BLE001
        print(f"(skipped, transformers/tokenizer unavailable: {exc})")


if __name__ == "__main__":
    main()
