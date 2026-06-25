"""Streamlit demo for Paper Surgeon: rhetorical role classification of abstract sentences.

Run locally with:
    streamlit run src/app.py
"""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "models/best"
MAX_LENGTH = 128

# Label -> display color. Keys match the fine-tuned model's labels.
LABEL_COLORS = {
    "BACKGROUND": "#9e9e9e",   # grey
    "OBJECTIVE": "#1f77b4",    # blue
    "METHODS": "#2ca02c",      # green
    "RESULTS": "#ff7f0e",      # orange
    "CONCLUSIONS": "#9467bd",  # purple
}


@st.cache_resource
def load_model(model_dir: str = MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)
    return tokenizer, model, device


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # Split on sentence-final punctuation followed by whitespace and a capital letter.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


@torch.no_grad()
def classify(sentences: list[str], tokenizer, model, device):
    enc = tokenizer(
        sentences,
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    probs = torch.softmax(model(**enc).logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    id2label = model.config.id2label
    labels = [id2label[int(p)] for p in predictions]
    return labels, [float(c) for c in confidences]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def render_colored(sentences, labels) -> str:
    spans = []
    for sent, label in zip(sentences, labels):
        color = LABEL_COLORS.get(label, "#000000")
        bg = _hex_to_rgba(color, 0.25)
        spans.append(
            f'<span style="background-color:{bg}; border-bottom:2px solid {color}; '
            f'padding:1px 3px; border-radius:3px;" title="{label}">{sent}</span>'
        )
    return " ".join(spans)


def render_legend() -> str:
    chips = []
    for label, color in LABEL_COLORS.items():
        chips.append(
            f'<span style="background-color:{_hex_to_rgba(color, 0.25)}; '
            f'border-bottom:2px solid {color}; padding:1px 6px; margin-right:6px; '
            f'border-radius:3px;">{label}</span>'
        )
    return " ".join(chips)


def main() -> None:
    st.set_page_config(page_title="Paper Surgeon", layout="centered")
    st.title("Paper Surgeon — Rhetorical Role Classifier")
    st.write(
        "Paste an abstract below and click **Analyze** to label each sentence "
        "with its rhetorical role."
    )

    abstract = st.text_area("Abstract", height=240, placeholder="Paste an abstract here...")
    analyze = st.button("Analyze", type="primary")

    if not analyze:
        return

    if not abstract.strip():
        st.warning("Please paste an abstract first.")
        return

    try:
        tokenizer, model, device = load_model()
    except Exception as exc:
        st.error(
            f"Could not load model from `{MODEL_DIR}/`. After the sweep, copy the "
            f"winning run there (needs config.json, model.safetensors, tokenizer files).\n\n"
            f"Details: {exc}"
        )
        return

    sentences = split_sentences(abstract)
    if not sentences:
        st.warning("No sentences detected.")
        return

    labels, confidences = classify(sentences, tokenizer, model, device)

    st.subheader("Annotated abstract")
    st.markdown(render_legend(), unsafe_allow_html=True)
    st.markdown(
        f"<div style='line-height:2.0; margin-top:12px;'>{render_colored(sentences, labels)}</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Per-sentence predictions")
    df = pd.DataFrame(
        {
            "Sentence #": range(1, len(sentences) + 1),
            "Predicted label": labels,
            "Max prob": [round(c, 3) for c in confidences],
        }
    )
    st.table(df.set_index("Sentence #"))


if __name__ == "__main__":
    main()
