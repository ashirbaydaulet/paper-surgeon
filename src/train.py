"""Fine-tune SciBERT or BERT-base for PubMed-RCT rhetorical role classification.

Usage:
    python -m src.train --model scibert --epochs 3 --batch-size 32
    python -m src.train --model bert-base --subset 1000 --no-wandb  # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime
from typing import Dict

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data_loader import load_pubmed_rct

LABEL_NAMES = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
LABEL2ID: Dict[str, int] = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL: Dict[int, str] = {i: name for i, name in enumerate(LABEL_NAMES)}
NUM_LABELS = len(LABEL_NAMES)

MODEL_MAP = {
    "scibert": "allenai/scibert_scivocab_uncased",
    "bert-base": "bert-base-uncased",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(MODEL_MAP), default="scibert")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset", type=int, default=None,
                   help="train on first N examples only (train split only)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="defaults to models/run_<timestamp>_<config_hash>")
    p.add_argument("--wandb-project", type=str, default="paper-surgeon")
    p.add_argument("--wandb-run-name", type=str, default=None,
                   help="defaults to <model>-lr<lr>-seed<seed>")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def tokenize_split(df, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
    ds = ds.map(
        lambda batch: tokenizer(
            batch["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        ),
        batched=True,
    )
    ds = ds.map(
        lambda batch: {"labels": [LABEL2ID[lab] for lab in batch["label"]]},
        batched=True,
    )
    ds = ds.remove_columns(["text", "label"])
    return ds


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    per_class = f1_score(
        labels, preds, average=None, labels=list(range(NUM_LABELS)), zero_division=0
    )
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }
    for i, name in enumerate(LABEL_NAMES):
        metrics[f"f1_{name}"] = float(per_class[i])
    return metrics


def main() -> None:
    args = parse_args()

    config: Dict = {
        "model": args.model,
        "model_checkpoint": MODEL_MAP[args.model],
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "subset": args.subset,
    }

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg_hash = hashlib.md5(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:8]
        args.output_dir = f"models/run_{timestamp}_{cfg_hash}"
    config["output_dir"] = args.output_dir

    if args.wandb_run_name is None:
        args.wandb_run_name = f"{args.model}-lr{args.learning_rate}-seed{args.seed}"

    use_cuda = torch.cuda.is_available()
    set_all_seeds(args.seed)

    if not args.no_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, config=config
        )

    splits = load_pubmed_rct()
    train_df, dev_df, test_df = splits["train"], splits["dev"], splits["test"]
    if args.subset is not None:
        train_df = train_df.iloc[: args.subset].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[args.model])
    train_ds = tokenize_split(train_df, tokenizer, args.max_length)
    dev_ds = tokenize_split(dev_df, tokenizer, args.max_length)
    test_ds = tokenize_split(test_df, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_MAP[args.model],
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        fp16=use_cuda,
        seed=args.seed,
        report_to="none" if args.no_wandb else "wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    trainer.train()
    train_time = time.time() - start

    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print("\n=== Test metrics ===")
    for key, value in test_metrics.items():
        print(f"  {key}: {value}")

    if not args.no_wandb:
        import wandb

        wandb.log(
            {
                f"test/{k[len('test_'):]}": v
                for k, v in test_metrics.items()
                if k.startswith("test_")
            }
        )

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "config": config,
        "test_metrics": test_metrics,
        "train_time_seconds": train_time,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated() if use_cuda else None
        ),
    }
    with open(os.path.join(args.output_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
