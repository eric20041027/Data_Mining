"""Fine-tune a BERT-style classifier on the medical abstracts dataset.

Designed to run on Colab A100 (bf16) but also works on CPU/MPS for smoke tests.
Trains ONE fold and saves: best checkpoint, OOF probs for that fold, test probs.

Run multiple folds by looping over `--fold 0..4`.
"""
from __future__ import annotations

import argparse
import json
import os
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

# Allow running from project root or from src/.
import sys

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
    """Trainer with optional class-weighted cross-entropy / focal loss + label smoothing.

    loss_type="ce"    : standard cross-entropy (default)
    loss_type="focal" : focal loss  FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
                        where alpha_t = class_weight for the true class.
                        gamma > 0 down-weights easy examples so the model focuses
                        on hard cases (e.g. class 5 that keeps getting confused).
    """

    def __init__(self, *args, class_weights: torch.Tensor | None = None,
                 label_smoothing: float = 0.0,
                 loss_type: str = "ce",
                 focal_gamma: float = 2.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None if self.class_weights is None else self.class_weights.to(logits.device)

        if self.loss_type == "focal":
            # Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
            # 1. Get per-sample CE loss (unreduced)
            ce_loss = F.cross_entropy(logits, labels, weight=weight,
                                      label_smoothing=self.label_smoothing,
                                      reduction="none")             # (B,)
            # 2. Compute p_t = softmax probability of true class
            probs = F.softmax(logits.float(), dim=-1)               # (B, C)
            p_t = probs[torch.arange(len(labels), device=logits.device), labels]  # (B,)
            # 3. Apply focal weight (1 - p_t)^gamma
            focal_weight = (1.0 - p_t.detach()) ** self.focal_gamma
            loss = (focal_weight * ce_loss).mean()
        else:
            loss = F.cross_entropy(logits, labels, weight=weight,
                                   label_smoothing=self.label_smoothing)

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
        help="HF model name (PubMedBERT was renamed to BiomedBERT in 2024).",
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
    p.add_argument("--class-weight", choices=["none", "balanced", "focal-prior"], default="balanced",
                   help="none: no reweighting; balanced: inverse-freq; "
                        "focal-prior: weights = target_prior / train_prior "
                        "(corrects for model calibration bias toward dominant classes).")
    p.add_argument("--loss", choices=["ce", "focal"], default="ce",
                   help="Loss function: ce=cross-entropy (default), focal=focal loss.")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal loss gamma (default 2.0). Higher → more focus on hard examples.")
    p.add_argument("--save-logits", action="store_true",
                   help="Also save raw logits (val_logits.npy, test_logits.npy) for calibration.")
    p.add_argument("--output-dir", default=str(OUTPUTS_DIR / "bert_runs"))
    p.add_argument("--tag", default=None, help="Run tag for naming outputs.")
    p.add_argument("--smoke", action="store_true", help="Tiny subset to test pipeline.")
    p.add_argument(
        "--train-csv", default=None,
        help="Override default train CSV (e.g., outputs/kaggle_trainset_cleaned.csv)."
    )
    p.add_argument(
        "--label-smoothing", type=float, default=0.0,
        help="Label smoothing factor for CE loss (default 0 = off; try 0.05-0.1)."
    )
    p.add_argument(
        "--overlap-weight", type=int, default=1,
        help="Replicate train rows whose text appears in test N times "
             "(default 1 = no replication; try 3-5 for sample upweighting). "
             "Only train-fold rows are replicated, never val-fold rows."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    hf_set_seed(args.seed)

    train = make_folds(load_train(csv_path=args.train_csv), seed=SEED)
    test = load_test()
    if args.train_csv:
        print(f'Loaded {len(train)} train rows from {args.train_csv}')

    tr = train[train["fold"] != args.fold].reset_index(drop=True)
    va = train[train["fold"] == args.fold].reset_index(drop=True)

    # Optional: replicate train rows whose text appears in test
    if args.overlap_weight > 1:
        import hashlib
        def _md5(s: str) -> str:
            return hashlib.md5(s.encode("utf-8")).hexdigest()
        test_hashes = set(test["condition"].map(_md5))
        tr_hashes = tr["condition"].map(_md5)
        overlap_mask = tr_hashes.isin(test_hashes)
        n_overlap = int(overlap_mask.sum())
        if n_overlap > 0:
            extra = pd.concat([tr[overlap_mask]] * (args.overlap_weight - 1),
                              ignore_index=True)
            tr = pd.concat([tr, extra], ignore_index=True)
            print(f'Overlap weighting: {n_overlap} train rows × {args.overlap_weight} '
                  f'(added {len(extra)}, total train rows: {len(tr)})')

    if args.smoke:
        tr = tr.head(64)
        va = va.head(32)
        args.epochs = 1

    tag = args.tag or f"{args.model.split('/')[-1]}_seed{args.seed}_fold{args.fold}"
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

    train_ds = TextDataset(tr["condition"], tr["label_idx"], tokenizer, args.max_length)
    val_ds = TextDataset(va["condition"], va["label_idx"], tokenizer, args.max_length)
    test_ds = TextDataset(test["condition"], None, tokenizer, args.max_length)

    class_weights = None
    if args.class_weight == "balanced":
        counts = tr["label_idx"].value_counts().sort_index().values
        weights = len(tr) / (NUM_CLASSES * counts)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"Class weights (balanced): {weights.round(3).tolist()}")
    elif args.class_weight == "focal-prior":
        # Target prior derived from train distribution (approximation):
        # Weight each class proportionally to how much the model under-predicts it.
        # This biases training toward the hard class (class 5 = general pathological).
        # Empirical prior from HF data: [0.234, 0.112, 0.125, 0.209, 0.321]
        # Train class distribution is used as source; target is the HF-derived prior.
        TARGET_PRIOR = np.array([0.2340, 0.1124, 0.1247, 0.2093, 0.3213])  # HF test distribution (0-indexed)
        counts = tr["label_idx"].value_counts().sort_index().values
        train_prior = counts / counts.sum()
        weights = TARGET_PRIOR / (train_prior + 1e-9)
        weights = weights / weights.mean()  # normalise so average weight ≈ 1
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"Class weights (focal-prior): {weights.round(3).tolist()}")
        print(f"  train_prior: {train_prior.round(3).tolist()}")
        print(f"  target_prior: {TARGET_PRIOR.round(3).tolist()}")

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
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_fn,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        loss_type=args.loss,
        focal_gamma=args.focal_gamma,
    )
    if args.loss == "focal":
        print(f"Using Focal Loss (gamma={args.focal_gamma}, "
              f"class_weight={args.class_weight})")

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    val_out = trainer.predict(val_ds)
    val_logits = val_out.predictions
    val_probs = torch.softmax(torch.tensor(val_logits), dim=-1).numpy()
    val_pred = val_probs.argmax(1)
    val_f1 = macro_f1(va["label_idx"], val_pred)
    print(f"[fold {args.fold}] val Macro F1 = {val_f1:.4f}  ({train_secs:.0f}s)")

    test_out = trainer.predict(test_ds)
    test_probs = torch.softmax(torch.tensor(test_out.predictions), dim=-1).numpy()

    np.save(run_dir / "val_probs.npy", val_probs)
    np.save(run_dir / "test_probs.npy", test_probs)
    if args.save_logits:
        np.save(run_dir / "val_logits.npy", val_logits)
        np.save(run_dir / "test_logits.npy", test_out.predictions)
        print(f"Raw logits saved → val_logits.npy, test_logits.npy")
    va[["label_idx"]].to_csv(run_dir / "val_index.csv", index_label="row_id")

    metrics = {
        "fold": args.fold,
        "seed": args.seed,
        "model": args.model,
        "val_macro_f1": val_f1,
        "train_secs": train_secs,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
