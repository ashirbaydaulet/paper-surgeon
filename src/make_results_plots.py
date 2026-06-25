"""Generate final results plots for the Paper Surgeon report.

Produces (both saved under results/):
  - results/confusion_matrix.png : row-normalized 5x5 confusion matrix on the
    PubMed-RCT test set, using the fine-tuned model in models/best/.
  - results/sweep_summary.png    : SciBERT vs BERT-base macro-F1 (mean +/- std
    across seeds) at LR=2e-5, read from results/sweep_results.csv.

No CLI; running the module produces both plots.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_loader import load_pubmed_rct

LABEL_NAMES = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}

MODEL_DIR = "models/best"
RESULTS_DIR = "results"
SWEEP_CSV = os.path.join(RESULTS_DIR, "sweep_results.csv")
MAX_LENGTH = 128
BATCH_SIZE = 64
BEST_LR = 2e-5


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def _predict(texts, tokenizer, model, device) -> list[int]:
    preds: list[int] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        enc = tokenizer(
            batch,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        preds.extend(model(**enc).logits.argmax(dim=-1).tolist())
    return preds


def plot_confusion_matrix() -> None:
    test = load_pubmed_rct()["test"]
    y_true = [LABEL2ID[label] for label in test["label"]]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    device = _device()
    model.to(device)
    print(f"Predicting on {len(test):,} test sentences (device={device})...")

    y_pred = _predict(test["text"].tolist(), tokenizer, model, device)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_NAMES))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        square=True,
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        cbar_kws={"label": "Fraction of true class"},
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("PubMed-RCT Test Confusion Matrix (row-normalized)")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("Wrote", out_path)


def plot_sweep_summary() -> None:
    df = pd.read_csv(SWEEP_CSV)
    sub = df[np.isclose(df["learning_rate"], BEST_LR)]
    stats = (
        sub.groupby("model")["test_macro_f1"]
        .agg(["mean", "std", "count"])
        .reindex(["scibert", "bert-base"])
    )

    display_name = {"scibert": "SciBERT", "bert-base": "BERT-base"}
    colors = {"scibert": "#2ca02c", "bert-base": "#1f77b4"}
    models = list(stats.index)
    means = stats["mean"].to_numpy()
    stds = np.nan_to_num(stats["std"].to_numpy())  # std is NaN if a single seed
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar(
        x, means, yerr=stds, capsize=8, width=0.5,
        color=[colors[m] for m in models],
    )

    for xi, mean, std, n in zip(x, means, stds, stats["count"]):
        ax.text(
            xi, mean + std + 0.0015, f"{mean:.4f}\n±{std:.4f}  (n={int(n)})",
            ha="center", va="bottom", fontsize=9,
        )

    # Horizontal gap annotation between the two means.
    gap = means[0] - means[1]
    x_mid = 0.5
    ax.hlines(means[0], x[0], x_mid, colors="gray", linestyles="dashed", linewidth=0.9)
    ax.hlines(means[1], x_mid, x[1], colors="gray", linestyles="dashed", linewidth=0.9)
    ax.annotate(
        "", xy=(x_mid, means[0]), xytext=(x_mid, means[1]),
        arrowprops=dict(arrowstyle="<->", color="black"),
    )
    ax.text(
        x_mid + 0.05, (means[0] + means[1]) / 2,
        f"gap = {gap:+.4f}", va="center", fontsize=10, fontweight="bold",
    )

    span = means.max() + stds.max() - (means.min() - stds.max())
    ax.set_ylim(means.min() - stds.max() - 0.4 * span, means.max() + stds.max() + 0.4 * span)
    ax.set_xticks(x)
    ax.set_xticklabels([display_name[m] for m in models])
    ax.set_ylabel("Test macro-F1")
    ax.set_title("SciBERT vs BERT-base at LR=2e-5")
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "sweep_summary.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("Wrote", out_path)


if __name__ == "__main__":
    plot_confusion_matrix()
    plot_sweep_summary()
