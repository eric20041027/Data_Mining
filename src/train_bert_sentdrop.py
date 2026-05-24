"""Train BERT with in-batch sentence dropout augmentation.

Each training sample is dynamically modified per epoch: randomly drop a fraction
of sentences (default 15%) before tokenizing. Validation and test use original
text. This forces the model to learn from whole-abstract structure rather than
memorizing specific sentence patterns.

Differs from TTA (test-time augmentation) which only perturbs inference:
this perturbs training, changing what the model learns.

Compliance: pure regularization technique on train text. No external labels
or test labels involved.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed as hf_set_seed,
)

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from utils import (  # noqa: E402
    LABEL_LIST,
    NUM_CLASSES,
    OUTPUTS_DIR,
    SEED,
    load_test,
    load_train,
    macro_f1,
    make_folds,
    set_seed,
)


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    sents = SENT_SPLIT.split(text)
    return [s for s in sents if s.strip()]


def sentence_dropout(text: str, drop_frac: float, rng: random.Random) -> str:
    """Randomly drop a fraction of sentences. Always keep ≥2 sentences."""
    sents = split_sentences(text)
    if len(sents) <= 2:
        return text
    keep = [s for s in sents if rng.random() >= drop_frac]
    if len(keep) < 2:
        return text
    return " ".join(keep)


class SentDropTrainDataset(Dataset):
    """Train dataset: applies dynamic sentence dropout per __getitem__ call."""
    def __init__(self, texts, labels, tokenizer, max_length=512, drop_frac=0.15, seed=42):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tok = tokenizer
        self.max_length = max_length
        self.drop_frac = drop_frac
        # Use a single RNG seeded once; each call gets fresh randomness
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):
        text = sentence_dropout(self.texts[idx], self.drop_frac, self.rng)
        enc = self.tok(text, truncation=True, max_length=self.max_length,
                       return_attention_mask=True)
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


class StaticDataset(Dataset):
    """Val/test dataset: no augmentation."""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], truncation=True, max_length=self.max_length,
                       return_attention_mask=True)
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "accuracy": float((preds == labels).mean()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--drop-frac", type=float, default=0.15,
                   help="Fraction of sentences to drop per training sample (0-1)")
    p.add_argument("--output-dir", default=str(OUTPUTS_DIR / "bert_runs"))
    p.add_argument("--tag", default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    hf_set_seed(args.seed)

    train = make_folds(load_train(), seed=SEED)
    test = load_test()

    tr = train[train["fold"] != args.fold].reset_index(drop=True)
    va = train[train["fold"] == args.fold].reset_index(drop=True)

    if args.smoke:
        tr = tr.head(64)
        va = va.head(32)
        args.epochs = 1

    tag = args.tag or f"{args.model.split('/')[-1]}_sentdrop{int(args.drop_frac*100):02d}_seed{args.seed}_fold{args.fold}"
    run_dir = Path(args.output_dir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=NUM_CLASSES,
        id2label={i: n for i, n in enumerate(LABEL_LIST)},
        label2id={n: i for i, n in enumerate(LABEL_LIST)},
    )

    train_ds = SentDropTrainDataset(tr["condition"], tr["label_idx"], tokenizer,
                                     args.max_length, args.drop_frac, args.seed)
    val_ds = StaticDataset(va["condition"], va["label_idx"], tokenizer, args.max_length)
    test_ds = StaticDataset(test["condition"], None, tokenizer, args.max_length)

    has_cuda = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        bf16=bool(args.bf16 and has_cuda),
        report_to=[],
        logging_steps=50,
        dataloader_num_workers=2,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_fn,
    )

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    val_out = trainer.predict(val_ds)
    val_probs = torch.softmax(torch.tensor(val_out.predictions), dim=-1).numpy()
    val_pred = val_probs.argmax(1)
    val_f1 = macro_f1(va["label_idx"], val_pred)
    print(f"[fold {args.fold}] SENTDROP val Macro F1 = {val_f1:.4f}  ({train_secs:.0f}s)")

    test_out = trainer.predict(test_ds)
    test_probs = torch.softmax(torch.tensor(test_out.predictions), dim=-1).numpy()

    np.save(run_dir / "val_probs.npy", val_probs)
    np.save(run_dir / "test_probs.npy", test_probs)
    va[["label_idx"]].to_csv(run_dir / "val_index.csv", index_label="row_id")

    metrics = {
        "fold": args.fold, "seed": args.seed, "model": args.model,
        "val_macro_f1": val_f1, "train_secs": train_secs,
        "variant": "sentence-dropout", "drop_frac": args.drop_frac,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
