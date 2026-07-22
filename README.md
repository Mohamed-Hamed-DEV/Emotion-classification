# Emotion Classification with Attention

A multi-class emotion classifier that predicts one of 6 core emotions — **joy, sadness, anger, fear, surprise, disgust** — from free text, built and compared across four architectures, then deployed as an interactive web app.

## Overview

This project explores emotion classification using both traditional sequence models with attention mechanisms and a fine-tuned transformer, comparing their performance and interpretability on the [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) dataset.

## Architectures compared

| Model | Description |
|---|---|
| LSTM | Baseline recurrent model with GloVe embeddings |
| GRU | Lighter recurrent alternative to LSTM |
| BiLSTM + Attention | Bidirectional LSTM with a custom additive attention layer for interpretability |
| DistilBERT (fine-tuned) | Transformer fine-tuned with HuggingFace `Trainer` |

## Pipeline

1. **Dataset preparation** — Load GoEmotions, map its 27 fine-grained labels down to 6 core emotions, drop ambiguous/multi-label rows
2. **Text preprocessing** — Lowercasing, URL/HTML stripping, contraction expansion, tokenization, padding
3. **Word embeddings** — GloVe (100d), chosen for its coverage on short, informal text
4. **Model building** — LSTM, GRU, BiLSTM+Attention (custom Keras layer), and fine-tuned DistilBERT
5. **Evaluation** — Accuracy, macro F1, per-class precision/recall, confusion matrices across all 4 models
6. **Attention visualization** — Heatmaps showing which words drove each prediction
7. **Deployment** — Interactive Gradio app for live predictions with confidence scores and attention heatmaps

## Results

| Model | Accuracy | Macro F1 |
| DistilBERT | 0.846566000974184 | 0.754629274489025 |
| LSTM | 0.752557233317096 | 0.667396601450703 |
| BiLSTM + Attention | 0.669264490988796 | 0.551058863585737 |
| GRU | 0.337554797856794 | 0.273861195828552 |



## Repository structure

```
emotion-classification/
├── notebook/
│   └── emotion-classification.ipynb   # full training pipeline (run on Kaggle, GPU recommended)
├── deployment/
│   ├── app.py                          # Gradio app
│   ├── requirements.txt
│   └── README.md                       # deployment setup instructions
└── README.md
```

