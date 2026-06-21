"""Load and inspect the PubMed-RCT 20k dataset.

PubMed-RCT 20k is a sentence-classification corpus drawn from the abstracts of
randomized controlled trials. Every sentence carries one of five rhetorical
role labels: BACKGROUND, OBJECTIVE, METHODS, RESULTS, CONCLUSIONS.

This module loads the data from the HuggingFace Hub and, when run directly,
prints a quick exploratory summary so the data can be eyeballed before any
modeling work.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

import pandas as pd
from datasets import DatasetDict, load_dataset

# The canonical five rhetorical-role labels in PubMed-RCT.
LABELS = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]

# Candidate Hub dataset ids to try, in order. The first is the id requested by
# the task; the rest are known community mirrors of the same 20k corpus. If the
# first id has been renamed/removed, we fall back to the next that loads.
_CANDIDATE_DATASET_IDS = [
    "armanc/pubmed-rct20k",
    "armanc/pubmed_rct20k",
    "pietrolesci/pubmed-200k-rct",
    "TrishanuDas/pubmed_rct20k",
    "ml4pubmed/pubmed-rct-20k",
]

# Splits can be named differently across mirrors; map everything to dev.
_SPLIT_ALIASES = {
    "train": "train",
    "validation": "dev",
    "valid": "dev",
    "dev": "dev",
    "test": "test",
}

# Common names the sentence/label columns go by across mirrors.
_TEXT_COLUMN_CANDIDATES = ["text", "sentence", "abstract_text", "Sentence"]
_LABEL_COLUMN_CANDIDATES = ["label", "labels", "target", "Label", "class"]


def _find_column(columns, candidates) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"None of {candidates} found in columns {list(columns)}")


def _normalize_labels(series: pd.Series) -> pd.Series:
    """Coerce a label column to upper-case strings from the canonical set.

    Mirrors store labels either as strings ("METHODS") or as integer ids. When
    integers, we map them positionally onto LABELS.
    """
    if pd.api.types.is_integer_dtype(series):
        return series.map(dict(enumerate(LABELS)))
    return series.astype(str).str.strip().str.upper()


def load_pubmed_rct() -> Dict[str, pd.DataFrame]:
    """Load PubMed-RCT 20k and return its train/dev/test splits.

    Returns a dict with keys ``"train"``, ``"dev"`` and ``"test"``; each value
    is a pandas DataFrame with two columns: ``text`` (the sentence) and
    ``label`` (one of LABELS).
    """
    dataset: Optional[DatasetDict] = None
    last_error: Optional[Exception] = None
    for dataset_id in _CANDIDATE_DATASET_IDS:
        try:
            dataset = load_dataset(dataset_id)
            print(f"Loaded dataset from: {dataset_id}")
            break
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            print(f"  could not load {dataset_id!r}: {exc}")
            last_error = exc

    if dataset is None:
        raise RuntimeError(
            "Failed to load PubMed-RCT 20k from any known mirror. "
            f"Last error: {last_error}"
        )

    splits: Dict[str, pd.DataFrame] = {}
    for raw_split, hf_split in dataset.items():
        split_name = _SPLIT_ALIASES.get(raw_split.lower())
        if split_name is None:
            print(f"  skipping unrecognized split {raw_split!r}")
            continue

        df = hf_split.to_pandas()
        text_col = _find_column(df.columns, _TEXT_COLUMN_CANDIDATES)
        label_col = _find_column(df.columns, _LABEL_COLUMN_CANDIDATES)

        df = df[[text_col, label_col]].rename(
            columns={text_col: "text", label_col: "label"}
        )
        df["text"] = df["text"].astype(str)
        df["label"] = _normalize_labels(df["label"])
        splits[split_name] = df.reset_index(drop=True)

    missing = {"train", "dev", "test"} - set(splits)
    if missing:
        print(f"Warning: dataset is missing expected splits: {sorted(missing)}")

    return splits


def _length_stats(values: pd.Series) -> Dict[str, float]:
    return {
        "mean": values.mean(),
        "median": values.median(),
        "max": values.max(),
        "p95": values.quantile(0.95),
        "p99": values.quantile(0.99),
    }


def _print_class_distribution(train: pd.DataFrame) -> None:
    print("\n=== Class distribution (train) ===")
    counts = train["label"].value_counts()
    total = len(train)
    # Iterate over the canonical labels so order is stable and 0-count labels show.
    for label in LABELS:
        n = int(counts.get(label, 0))
        pct = 100.0 * n / total if total else 0.0
        print(f"  {label:<12} {n:>8,}  ({pct:5.2f}%)")
    # Surface any labels present in the data but outside the canonical set.
    extra = [l for l in counts.index if l not in LABELS]
    for label in extra:
        n = int(counts[label])
        pct = 100.0 * n / total if total else 0.0
        print(f"  {label:<12} {n:>8,}  ({pct:5.2f}%)  [unexpected label]")
    print(f"  {'TOTAL':<12} {total:>8,}")


def _print_length_stats(train: pd.DataFrame) -> None:
    char_len = train["text"].str.len()
    word_len = train["text"].str.split().map(len)

    print("\n=== Sentence length stats (train) ===")
    print(
        f"  {'metric':<10}{'mean':>10}{'median':>10}{'max':>10}{'p95':>10}{'p99':>10}"
    )
    for name, series in (("chars", char_len), ("words", word_len)):
        s = _length_stats(series)
        print(
            f"  {name:<10}{s['mean']:>10.1f}{s['median']:>10.1f}"
            f"{s['max']:>10.0f}{s['p95']:>10.1f}{s['p99']:>10.1f}"
        )


def _print_samples(train: pd.DataFrame, per_class: int = 5, seed: int = 0) -> None:
    print(f"\n=== {per_class} random sentences per class (train) ===")
    rng = random.Random(seed)
    for label in LABELS:
        subset = train[train["label"] == label]
        print(f"\n[{label}]  ({len(subset):,} examples)")
        if subset.empty:
            print("  (no examples)")
            continue
        idxs = list(subset.index)
        chosen = rng.sample(idxs, min(per_class, len(idxs)))
        for i in chosen:
            print(f"  - {train.at[i, 'text']}")


if __name__ == "__main__":
    splits = load_pubmed_rct()

    print("\n=== Split sizes ===")
    for name in ("train", "dev", "test"):
        n = len(splits[name]) if name in splits else 0
        print(f"  {name:<6} {n:>8,}")

    train = splits["train"]
    _print_class_distribution(train)
    _print_length_stats(train)
    _print_samples(train)
