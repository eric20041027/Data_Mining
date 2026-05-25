"""Train BERT classifier using only the first sentence (title) of each abstract.

The medical abstracts have a "Title. Body..." structure. The title alone carries
weaker but DIFFERENT signal than the full text — useful as ensemble diversity.

Identical to train_bert.py except texts are pre-truncated to the first sentence.
"""
from __future__ import annotations

import argparse
import json
import os
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


def extract_title(text: str) -> str:
    """First sentence, splitting on '. '. Fallback to full text if no split."""
    parts = text.split(". ", 1)
    if len(parts) == 1:
        return text
    return parts[0] + "."


class TitleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        # 把 condition 換成 title only
        self.texts = [extract_title(t) for t in texts]
        self.labels = None if labels is None else list(labels)
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None if self.class_weights is None else self.class_weights.to(logits.device)
        loss = F.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "accuracy": float((preds == labels).mean()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    )
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-length", type=int, default=128, help="Title is short, 128 is plenty.")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none")
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

    tag = args.tag or f"{args.model.split('/')[-1]}_title_seed{args.seed}_fold{args.fold}"
    run_dir = Path(args.output_dir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=NUM_CLASSES,
        id2label={i: name for i, name in enumerate(LABEL_LIST)},
        label2id={name: i for i, name in enumerate(LABEL_LIST)},
    )

    train_ds = TitleDataset(tr["condition"], tr["label_idx"], tokenizer, args.max_length)
    val_ds = TitleDataset(va["condition"], va["label_idx"], tokenizer, args.max_length)
    test_ds = TitleDataset(test["condition"], None, tokenizer, args.max_length)

    class_weights = None
    if args.class_weight == "balanced":
        counts = tr["label_idx"].value_counts().sort_index().values
        weights = len(tr) / (NUM_CLASSES * counts)
        class_weights = torch.tensor(weights, dtype=torch.float32)

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

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_fn,
        class_weights=class_weights,
    )

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    val_out = trainer.predict(val_ds)
    val_logits = val_out.predictions
    val_probs = torch.softmax(torch.tensor(val_logits), dim=-1).numpy()
    val_pred = val_probs.argmax(1)
    val_f1 = macro_f1(va["label_idx"], val_pred)
    print(f"[fold {args.fold}] TITLE val Macro F1 = {val_f1:.4f}  ({train_secs:.0f}s)")

    test_out = trainer.predict(test_ds)
    test_probs = torch.softmax(torch.tensor(test_out.predictions), dim=-1).numpy()

    np.save(run_dir / "val_probs.npy", val_probs)
    np.save(run_dir / "test_probs.npy", test_probs)
    va[["label_idx"]].to_csv(run_dir / "val_index.csv", index_label="row_id")

    metrics = {
        "fold": args.fold,
        "seed": args.seed,
        "model": args.model,
        "val_macro_f1": val_f1,
        "train_secs": train_secs,
        "variant": "title-only",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
