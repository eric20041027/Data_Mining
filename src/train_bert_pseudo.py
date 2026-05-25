"""Train BERT with pseudo-labels appended to train.

Key design decision to avoid evaluation-time leakage:
  - Original 12,994 train rows are split via the standard StratifiedKFold
    (same fold assignment as other CE models, so OOF probs are compatible
    in the same ensemble).
  - Pseudo-labelled test rows (typically 500-1000 high-confidence rows) are
    ALWAYS in the training set, regardless of which fold is val. They never
    enter val.

Why this is sound:
  - Val is measured on real train labels only — pseudo doesn't inflate OOF.
  - Pseudo rows boost training signal (especially for the 850 OOD test
    texts that the original 4-model ensemble was confident about).
  - At inference, the model has been fine-tuned to handle those exact test
    texts as well, which is the whole point of pseudo-labelling.

Compliance:
  - Pseudo labels come from our own model's predictions (`make_pseudo_labels.py`).
  - No external label source, no test ground truth involved.
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
    LABEL2IDX,
    NUM_CLASSES,
    OUTPUTS_DIR,
    SEED,
    load_test,
    load_train,
    macro_f1,
    make_folds,
    set_seed,
)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = list(texts)
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
    p.add_argument(
        "--pseudo-csv",
        default="outputs/pseudo_labels.csv",
        help="Path to pseudo-labelled CSV (columns: label, condition)",
    )
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
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    p.add_argument("--output-dir", default=str(OUTPUTS_DIR / "bert_runs"))
    p.add_argument("--tag", default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    hf_set_seed(args.seed)

    # === Original train with StratifiedKFold (same as other CE models) ===
    train = make_folds(load_train(), seed=SEED)
    test = load_test()

    # === Pseudo-labels ===
    pseudo_path = Path(args.pseudo_csv)
    if not pseudo_path.is_absolute():
        pseudo_path = Path(__file__).resolve().parent.parent / args.pseudo_csv
    pseudo = pd.read_csv(pseudo_path)
    pseudo["label"] = pseudo["label"].str.strip().str.lower()
    unknown = set(pseudo["label"]) - set(LABEL2IDX)
    if unknown:
        raise ValueError(f"Unknown labels in pseudo CSV: {unknown}")
    pseudo["label_idx"] = pseudo["label"].map(LABEL2IDX).astype(int)

    print(f"Loaded {len(pseudo)} pseudo-labelled rows from {pseudo_path}")
    print(f"  pseudo label dist: {pseudo['label_idx'].value_counts().sort_index().to_dict()}")

    # === Build train/val for this fold ===
    tr_orig = train[train["fold"] != args.fold].reset_index(drop=True)
    va_orig = train[train["fold"] == args.fold].reset_index(drop=True)

    # All pseudo rows go into train regardless of fold.
    tr_texts = pd.concat([tr_orig["condition"], pseudo["condition"]], ignore_index=True)
    tr_labels = pd.concat([tr_orig["label_idx"], pseudo["label_idx"]], ignore_index=True)
    va_texts = va_orig["condition"]
    va_labels = va_orig["label_idx"]

    print(f"Train: {len(tr_orig)} original + {len(pseudo)} pseudo = {len(tr_texts)} total rows")
    print(f"Val:   {len(va_orig)} (original train fold {args.fold} only)")

    if args.smoke:
        tr_texts = tr_texts.head(64)
        tr_labels = tr_labels.head(64)
        va_texts = va_texts.head(32)
        va_labels = va_labels.head(32)
        args.epochs = 1

    tag = (
        args.tag
        or f"{args.model.split('/')[-1]}_pseudo_seed{args.seed}_fold{args.fold}"
    )
    run_dir = Path(args.output_dir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    # === Model + datasets ===
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=NUM_CLASSES,
        id2label={i: name for i, name in enumerate(LABEL_LIST)},
        label2id={name: i for i, name in enumerate(LABEL_LIST)},
    )

    train_ds = TextDataset(tr_texts.values, tr_labels.values, tokenizer, args.max_length)
    val_ds = TextDataset(va_texts.values, va_labels.values, tokenizer, args.max_length)
    test_ds = TextDataset(test["condition"].values, None, tokenizer, args.max_length)

    class_weights = None
    if args.class_weight == "balanced":
        counts = pd.Series(tr_labels.values).value_counts().sort_index().values
        weights = len(tr_labels) / (NUM_CLASSES * counts)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"Class weights: {weights.round(3).tolist()}")

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
    val_probs = torch.softmax(torch.tensor(val_out.predictions), dim=-1).numpy()
    val_pred = val_probs.argmax(1)
    val_f1 = macro_f1(va_labels.values, val_pred)
    print(f"[fold {args.fold}] PSEUDO val Macro F1 = {val_f1:.4f}  ({train_secs:.0f}s)")

    test_out = trainer.predict(test_ds)
    test_probs = torch.softmax(torch.tensor(test_out.predictions), dim=-1).numpy()

    np.save(run_dir / "val_probs.npy", val_probs)
    np.save(run_dir / "test_probs.npy", test_probs)
    va_orig[["label_idx"]].to_csv(run_dir / "val_index.csv", index_label="row_id")

    metrics = {
        "fold": args.fold,
        "seed": args.seed,
        "model": args.model,
        "val_macro_f1": val_f1,
        "train_secs": train_secs,
        "variant": "pseudo-label",
        "n_pseudo": len(pseudo),
        "n_train_orig": len(tr_orig),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
