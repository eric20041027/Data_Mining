"""Multi-label BCE training variant of train_bert.py.

For each train text, the label is a 5-dim binary vector marking all labels
observed for that exact text in the train set (so multi-label texts get
multiple positives). Loss = BCEWithLogitsLoss.

At inference, we apply sigmoid and pick argmax for single-label output.
"""
from __future__ import annotations

import argparse
import hashlib
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


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def build_multilabel_targets(df: pd.DataFrame) -> np.ndarray:
    """For each row, return a 5-dim binary vector indicating ALL labels
    observed for that exact text in the train set.
    """
    df = df.copy()
    df["h"] = df["condition"].map(md5)
    # For each unique hash, collect set of labels observed
    text_labels: dict[str, set[int]] = {}
    for h, lbl in zip(df["h"], df["label_idx"]):
        text_labels.setdefault(h, set()).add(int(lbl))

    targets = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
    for i, h in enumerate(df["h"]):
        for lbl in text_labels[h]:
            targets[i, lbl] = 1.0
    return targets


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = list(texts)
        self.labels = labels  # array of shape (N, 5) or None
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
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


class BCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics_fn_factory(val_singlelabel_idx: np.ndarray):
    """Multi-label trained but evaluate using sigmoid → argmax → macro F1 vs
    the single-label ground truth of the val fold.
    """
    def compute_metrics(eval_pred):
        logits, _ = eval_pred
        # sigmoid → predicted top class
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = np.argmax(probs, axis=-1)
        return {
            "macro_f1": float(f1_score(val_singlelabel_idx, preds, average="macro")),
            "accuracy": float((preds == val_singlelabel_idx).mean()),
        }
    return compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
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

    # 多標籤 targets：對 train 同文本的所有觀察標籤都標為 1
    # IMPORTANT: 要從整個 train 建構，這樣 val 的同文本如果在 tr 出現過也會自動繼承
    full_targets = build_multilabel_targets(train.reset_index(drop=True))
    # 重新拿到對應 tr/va 的 targets
    tr_full = train[train["fold"] != args.fold]
    va_full = train[train["fold"] == args.fold]
    tr_targets = full_targets[tr_full.index.values]
    va_targets = full_targets[va_full.index.values]
    va_singlelabel = va["label_idx"].to_numpy()

    if args.smoke:
        tr = tr.head(64)
        tr_targets = tr_targets[:64]
        va = va.head(32)
        va_targets = va_targets[:32]
        va_singlelabel = va_singlelabel[:32]
        args.epochs = 1

    tag = args.tag or f"{args.model.split('/')[-1]}_bce_seed{args.seed}_fold{args.fold}"
    run_dir = Path(args.output_dir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # problem_type=multi_label_classification 讓 head 直接適合 BCE
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=NUM_CLASSES,
        problem_type="multi_label_classification",
        id2label={i: name for i, name in enumerate(LABEL_LIST)},
        label2id={name: i for i, name in enumerate(LABEL_LIST)},
    )

    train_ds = TextDataset(tr["condition"], tr_targets, tokenizer, args.max_length)
    val_ds = TextDataset(va["condition"], va_targets, tokenizer, args.max_length)
    test_ds = TextDataset(test["condition"], None, tokenizer, args.max_length)

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

    trainer = BCETrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_fn_factory(va_singlelabel),
    )

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    val_out = trainer.predict(val_ds)
    val_logits = val_out.predictions
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))  # sigmoid
    # 為了 ensemble 一致，這裡 normalize 成跟 softmax 一樣的「機率分布」（和為 1）
    # 不影響 argmax 但讓平均 ensemble 行為合理
    val_probs_normalized = val_probs / val_probs.sum(axis=1, keepdims=True)
    val_pred = val_probs_normalized.argmax(1)
    val_f1 = macro_f1(va_singlelabel, val_pred)
    print(f"[fold {args.fold}] BCE val Macro F1 = {val_f1:.4f}  ({train_secs:.0f}s)")

    test_out = trainer.predict(test_ds)
    test_probs_raw = 1.0 / (1.0 + np.exp(-test_out.predictions))
    test_probs = test_probs_raw / test_probs_raw.sum(axis=1, keepdims=True)

    np.save(run_dir / "val_probs.npy", val_probs_normalized)
    np.save(run_dir / "test_probs.npy", test_probs)
    # 也存原始 sigmoid 機率以備不時之需
    np.save(run_dir / "val_probs_sigmoid.npy", val_probs)
    np.save(run_dir / "test_probs_sigmoid.npy", test_probs_raw)
    va[["label_idx"]].to_csv(run_dir / "val_index.csv", index_label="row_id")

    metrics = {
        "fold": args.fold,
        "seed": args.seed,
        "model": args.model,
        "val_macro_f1": val_f1,
        "train_secs": train_secs,
        "loss": "BCE multi-label",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
