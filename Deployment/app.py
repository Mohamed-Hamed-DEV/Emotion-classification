"""
Emotion Classification with Attention — Gradio Deployment App
Run locally in VS Code with:  python app.py

Expects a folder called `deployment_artifacts/` (downloaded from the Kaggle
notebook's Output panel) placed next to this file, containing:
    - distilbert_emotion/            (HuggingFace model + tokenizer)
    - bilstm_attention_model.keras   (custom Keras model)
    - keras_tokenizer.json
    - config.json
    - model_comparison.csv
"""

import os
import json
import re

import numpy as np
import pandas as pd
import gradio as gr

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
ART_DIR = os.path.join(os.path.dirname(__file__), "deployment_artifacts")

EMOTION_EMOJI = {
    "joy": "😄", "sadness": "😢", "anger": "😠",
    "fear": "😱", "surprise": "😲", "disgust": "🤢",
}

# ----------------------------------------------------------------------
# TEXT CLEANING (must match preprocessing used in the notebook)
# ----------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


# ----------------------------------------------------------------------
# LOAD ARTIFACTS (loaded once at startup)
# ----------------------------------------------------------------------
if not os.path.isdir(ART_DIR):
    raise FileNotFoundError(
        f"Couldn't find '{ART_DIR}'. Download the `deployment_artifacts` folder "
        "from your Kaggle notebook's Output panel and place it next to app.py."
    )

with open(os.path.join(ART_DIR, "config.json")) as f:
    CFG = json.load(f)
CFG["id2label"] = {int(k): v for k, v in CFG["id2label"].items()}
LABELS = [CFG["id2label"][i] for i in range(len(CFG["id2label"]))]

print("Loading DistilBERT...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

_bert_path = os.path.join(ART_DIR, "distilbert_emotion")
BERT_TOKENIZER = AutoTokenizer.from_pretrained(_bert_path)
BERT_MODEL = AutoModelForSequenceClassification.from_pretrained(_bert_path)
BERT_MODEL.eval()

BILSTM_TOKENIZER = None
BILSTM_MODEL = None
BILSTM_PROBE = None


def _lazy_load_bilstm():
    """Loads the custom BiLSTM+Attention model on first use (keeps startup fast)."""
    global BILSTM_TOKENIZER, BILSTM_MODEL, BILSTM_PROBE
    if BILSTM_MODEL is not None:
        return

    print("Loading BiLSTM + Attention model...")
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    from tensorflow.keras.layers import Layer
    from tensorflow.keras import backend as K

    class AttentionLayer(Layer):
        """Must match the training-time definition exactly."""

        def __init__(self, units=64, **kwargs):
            super().__init__(**kwargs)
            self.units = units

        def build(self, input_shape):
            self.W = self.add_weight(name="att_W", shape=(input_shape[-1], self.units),
                                      initializer="glorot_uniform", trainable=True)
            self.b = self.add_weight(name="att_b", shape=(self.units,),
                                      initializer="zeros", trainable=True)
            self.u = self.add_weight(name="att_u", shape=(self.units, 1),
                                      initializer="glorot_uniform", trainable=True)
            super().build(input_shape)

        def call(self, hidden_states, mask=None):
            score = K.tanh(K.dot(hidden_states, self.W) + self.b)
            score = K.dot(score, self.u)
            score = K.squeeze(score, axis=-1)
            if mask is not None:
                score = tf.where(mask, score, -1e9 * tf.ones_like(score))
            weights = tf.nn.softmax(score, axis=1)
            weights_exp = tf.expand_dims(weights, axis=-1)
            context = tf.reduce_sum(hidden_states * weights_exp, axis=1)
            return context, weights

        def compute_mask(self, inputs, mask=None):
            return None

    model = tf.keras.models.load_model(
        os.path.join(ART_DIR, "bilstm_attention_model.keras"),
        custom_objects={"AttentionLayer": AttentionLayer},
    )
    with open(os.path.join(ART_DIR, "keras_tokenizer.json")) as f:
        tok = tokenizer_from_json(f.read())

    attn_layer = next(l for l in model.layers if l.__class__.__name__ == "AttentionLayer")
    context, weights = attn_layer.output
    probe = tf.keras.Model(inputs=model.input, outputs=[model.output, weights])

    BILSTM_TOKENIZER, BILSTM_MODEL, BILSTM_PROBE = tok, model, probe


# ----------------------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------------------
def predict_distilbert(text):
    cleaned = clean_text(text)
    inputs = BERT_TOKENIZER(cleaned, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = BERT_MODEL(**inputs).logits
        probs = torch.softmax(logits, dim=-1).numpy()[0]
    return probs, None


def predict_bilstm_attention(text):
    _lazy_load_bilstm()
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    cleaned = clean_text(text)
    seq = BILSTM_TOKENIZER.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=CFG["max_len"], padding="post", truncating="post")

    probs, att_weights = BILSTM_PROBE.predict(padded, verbose=0)
    probs, att_weights = probs[0], att_weights[0]

    tokens = cleaned.split()
    n = len(tokens)
    w = att_weights[:n]
    w = w / (w.sum() + 1e-9)
    token_weights = list(zip(tokens, w.tolist()))

    return probs, token_weights


def run_prediction(text, model_choice):
    if not text or not text.strip():
        return "Please enter some text first.", {}, None

    if model_choice == "DistilBERT (fine-tuned)":
        probs, token_weights = predict_distilbert(text)
    else:
        probs, token_weights = predict_bilstm_attention(text)

    pred_id = int(np.argmax(probs))
    pred_label = LABELS[pred_id]
    confidence = float(probs[pred_id])

    headline = f"{EMOTION_EMOJI.get(pred_label, '')} Predicted emotion: **{pred_label.capitalize()}** ({confidence:.1%} confidence)"
    label_scores = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

    highlighted = None
    if token_weights is not None:
        max_w = max(w for _, w in token_weights) or 1e-9
        highlighted = [(tok, w / max_w) for tok, w in token_weights]

    return headline, label_scores, highlighted


# ----------------------------------------------------------------------
# GRADIO UI
# ----------------------------------------------------------------------
comp_path = os.path.join(ART_DIR, "model_comparison.csv")
comparison_df = pd.read_csv(comp_path) if os.path.exists(comp_path) else None

with gr.Blocks(title="Emotion Classifier") as demo:
    gr.Markdown("# 🎭 Emotion Classifier")
    gr.Markdown(
        "Type a sentence and the model will predict the dominant emotion "
        "(joy, sadness, anger, fear, surprise, disgust)."
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Enter text",
                placeholder="I can't believe this happened, I'm so excited!",
                lines=3,
            )
            model_choice = gr.Radio(
                choices=["DistilBERT (fine-tuned)", "BiLSTM + Attention"],
                value="DistilBERT (fine-tuned)",
                label="Model",
            )
            predict_btn = gr.Button("Predict emotion", variant="primary")

        with gr.Column(scale=1):
            if comparison_df is not None:
                gr.Markdown("**Model comparison**")
                gr.Dataframe(comparison_df, interactive=False)

    headline_out = gr.Markdown()
    scores_out = gr.Label(label="Confidence across all 6 classes", num_top_classes=6)
    attention_out = gr.HighlightedText(
        label="Attention heatmap (BiLSTM + Attention only)",
        combine_adjacent=False,
    )

    predict_btn.click(
        fn=run_prediction,
        inputs=[text_input, model_choice],
        outputs=[headline_out, scores_out, attention_out],
    )
    text_input.submit(
        fn=run_prediction,
        inputs=[text_input, model_choice],
        outputs=[headline_out, scores_out, attention_out],
    )

if __name__ == "__main__":
    demo.launch()
