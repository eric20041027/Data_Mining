"""Test-time augmentation for trained BERT runs.

For each provided run directory (containing a saved trainer checkpoint),
re-runs inference on the test set under K text augmentations, averages
all the predictions, and writes `test_probs_tta.npy` alongside the
original `test_probs.npy`.

To use TTA-augmented probs in ensemble_predict.py, pass `--prefer-tta`.

Available augmentations (`--augmentations`):
  original        : no change (anchor)
  drop_sentence   : randomly drop ~10% of sentences
  shuffle_body    : shuffle non-title sentences
  truncate_head   : keep first 85% of characters
  truncate_tail   : keep last 85% of characters

Important: TTA can only be applied to runs whose model checkpoint is still
on disk (under `trainer/checkpoint-*/`). The lightweight Drive backups we
make (val_probs + test_probs + json) do NOT include model weights, so TTA
is only practical for runs trained in the current Colab session.
"""
from __future__ import annotations

import argparse
import glob
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from utils import load_test  # noqa: E402


# ============ Augmentations ============
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    sents = SENT_SPLIT.split(text)
    return [s for s in sents if s.strip()]


def aug_original(text: str, rng: random.Random) -> str:
    return text


def aug_drop_sentence(text: str, rng: random.Random, drop_frac: float = 0.10) -> str:
    sents = split_sentences(text)
    if len(sents) <= 2:
        return text
    keep = [s for s in sents if rng.random() >= drop_frac]
    if len(keep) < 2:
        return text
    return " ".join(keep)


def aug_shuffle_body(text: str, rng: random.Random) -> str:
    sents = split_sentences(text)
    if len(sents) <= 2:
        return text
    title, body = sents[0], sents[1:]
    rng.shuffle(body)
    return " ".join([title] + body)


def aug_truncate_head(text: str, rng: random.Random, frac: float = 0.85) -> str:
    n = max(1, int(len(text) * frac))
    return text[:n]


def aug_truncate_tail(text: str, rng: random.Random, frac: float = 0.85) -> str:
    n = max(1, int(len(text) * frac))
    return text[len(text) - n :]


AUGMENTATIONS = {
    "original": aug_original,
    "drop_sentence": aug_drop_sentence,
    "shuffle_body": aug_shuffle_body,
    "truncate_head": aug_truncate_head,
    "truncate_tail": aug_truncate_tail,
}


# ============ Inference helpers ============
def find_checkpoint(run_dir: Path) -> Path:
    trainer_dir = run_dir / "trainer"
    if not trainer_dir.is_dir():
        raise FileNotFoundError(f"No trainer/ dir under {run_dir}")
    ckpts = list(trainer_dir.glob("checkpoint-*"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint under {trainer_dir}")
    return max(ckpts, key=lambda p: int(p.name.split("-")[1]))


def predict_batched(
    model, tokenizer, texts: list[str], max_length: int, batch_size: int, device: str
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits = model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            out.append(probs)
    return np.vstack(out)


# ============ Main ============
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Glob patterns for run directories (e.g., 'outputs/bert_runs/deberta_v3_large_*')",
    )
    p.add_argument(
        "--augmentations",
        nargs="+",
        default=["original", "drop_sentence", "shuffle_body", "truncate_head", "truncate_tail"],
        choices=list(AUGMENTATIONS),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    # Resolve run dirs
    dirs: list[Path] = []
    for pattern in args.run_dirs:
        matches = sorted(Path(p) for p in glob.glob(str(root / pattern)))
        dirs.extend(matches)
    dirs = sorted(set(dirs))

    if not dirs:
        raise SystemExit(f"No run dirs match: {args.run_dirs}")

    print(f"Will TTA {len(dirs)} run dirs with {len(args.augmentations)} augmentations:")
    for d in dirs:
        print(f"  {d.name}")
    print(f"Augmentations: {args.augmentations}\n")

    # Pre-generate augmented test texts once (shared across runs)
    test = load_test()
    test_texts = test["condition"].tolist()
    aug_text_sets: dict[str, list[str]] = {}
    for aug_name in args.augmentations:
        rng_aug = random.Random(args.seed)  # deterministic per aug
        fn = AUGMENTATIONS[aug_name]
        aug_text_sets[aug_name] = [fn(t, rng_aug) for t in test_texts]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for run_dir in dirs:
        print(f"\n=== TTA on {run_dir.name} ===")
        try:
            ckpt = find_checkpoint(run_dir)
        except FileNotFoundError as exc:
            print(f"  ⚠️ skip: {exc}")
            continue
        print(f"  Checkpoint: {ckpt.name}")

        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)

        per_aug_probs = []
        for aug_name in args.augmentations:
            t1 = time.time()
            probs = predict_batched(
                model, tokenizer, aug_text_sets[aug_name],
                args.max_length, args.batch_size, device
            )
            per_aug_probs.append(probs)
            print(f"  [{aug_name:15s}] {probs.shape} in {time.time()-t1:.0f}s "
                  f"(mean max-prob {probs.max(axis=1).mean():.3f})")

        avg = np.mean(per_aug_probs, axis=0)
        out_path = run_dir / "test_probs_tta.npy"
        np.save(out_path, avg)
        print(f"  Wrote: {out_path.name} ({time.time()-t0:.0f}s total)")

        # Free GPU
        del model
        torch.cuda.empty_cache()

    print("\n=== TTA complete ===")
    print("Use in ensemble: python src/ensemble_predict.py --bert-runs '...' --prefer-tta --tag <tag>")


if __name__ == "__main__":
    main()
