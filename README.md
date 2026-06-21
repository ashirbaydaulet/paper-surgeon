# paper-surgeon

A tool that takes scientific abstracts and structures them by labeling each sentence with its rhetorical role (background, objective, method, result, conclusion). The trained model is the core deliverable. A simple Streamlit demo lets a user paste an abstract and see it color-coded by role.

## Scope — exactly three levels

- **Level 0 — Metadata and ingestion.** Take a scientific abstract as plain text input. Optionally take a PDF and extract the abstract from it.
- **Level 1 — Sentence segmentation.** Split the abstract into sentences, each with an ID and position.
- **Level 2 — Sentence role classification.** A fine-tuned transformer assigns each sentence one of five labels: BACKGROUND, OBJECTIVE, METHODS, RESULTS, CONCLUSIONS.

Anything beyond this (paragraph-level roles, claim extraction, knowledge graphs, cross-paper analysis, full-paper parsing) is out of scope.

## Dataset

PubMed-RCT 20k from HuggingFace (`armanc/pubmed-rct20k` or equivalent). Pre-split into train/dev/test. ~180k labeled sentences across ~20k abstracts.

## Model

Fine-tune SciBERT (`allenai/scibert_scivocab_uncased`) for sentence-level classification. Standard setup: tokenize sentence, take `[CLS]` embedding, linear head to 5 classes, cross-entropy loss.

Baseline for comparison: vanilla `bert-base-uncased` fine-tuned with identical protocol. One ablation, that's it.

## Storage layer

SQLite, designed so future extension is possible without rewriting. One `units` table with: `id`, `type` (paper / sentence), `parent_id`, `text`, `position`, and a JSON `annotations` column. Future-proofing only — no extra features now.

## Deliverables

- Trained SciBERT model with logged training process.
- Evaluation: per-class F1, confusion matrix, in-domain test results, qualitative out-of-domain test on ~10 arXiv abstracts.
- Streamlit demo: paste abstract, see sentence-by-sentence color-coded predictions.
- Report (written separately).

## Stack

Python 3.11, PyTorch, HuggingFace transformers + datasets, scikit-learn, matplotlib, pandas, streamlit, pymupdf, wandb, sqlite3 (stdlib).

## Compute

Training on Google Colab with a T4 GPU. Everything else (data loading, EDA, evaluation, demo) runs on the laptop without a GPU.

## Repository structure

```
paper-surgeon/
├── src/
│   ├── data_loader.py    # PubMed-RCT loading + sentence segmentation
│   ├── train.py          # SciBERT fine-tuning loop
│   ├── evaluate.py       # Metrics, confusion matrix, qualitative tests
│   ├── predict.py        # Inference on new abstracts
│   ├── db.py             # SQLite storage layer
│   └── app.py            # Streamlit demo
├── notebooks/eda.ipynb
├── data/                 # gitignored
├── models/               # gitignored
├── results/              # plots and tables
├── requirements.txt
└── README.md
```
