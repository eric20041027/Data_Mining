"""Task-adaptive MLM pretraining on train + test texts.

Continues PubMedBERT's masked language modeling objective on our specific
corpus (train.csv abstracts + test.csv abstracts, ~14,438 texts). This
narrows the model toward the vocabulary and writing style of THIS dataset
before classification fine-tuning.

Compliance: pure self-supervised text modeling. No labels of any kind
are used (MLM doesn't need them — only token reconstruction). Using test
TEXT (not labels) for self-supervision is standard "task-adaptive
pretraining" widely documented in academic literature (Gururangan et al.
2020 'Don't Stop Pretraining').
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed as hf_set_seed,
)

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from utils import load_test, load_train, set_seed, SEED  # noqa: E402


class TextOnlyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = list(texts)
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--mlm-prob", type=float, default=0.15)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument(
        "--output-dir",
        default="outputs/mlm_adapted",
        help="Where to save the adapted model.",
    )
    p.add_argument(
        "--no-test-text",
        action="store_true",
        help="Use only train texts for MLM (skip test). Default: include test.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    hf_set_seed(SEED)

    train_df = load_train()
    texts = train_df["condition"].tolist()
    if not args.no_test_text:
        test_df = load_test()
        texts = texts + test_df["condition"].tolist()
    print(f"Total texts for MLM: {len(texts)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    ds = TextOnlyDataset(texts, tokenizer, args.max_length)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=args.mlm_prob
    )

    root = Path(__file__).resolve().parent.parent
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    has_cuda = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_strategy="no",
        bf16=has_cuda,
        logging_steps=50,
        report_to=[],
        seed=SEED,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    model_dir = out_dir / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"\n✅ Adapted model saved to: {model_dir}")
    print(f"\nUse as `--model {model_dir}` in train_bert.py to fine-tune.")


if __name__ == "__main__":
    main()
