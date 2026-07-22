# Emotion Classifier — VS Code Deployment (Gradio)

This is the Phase 2/3 deployment package that goes with the Kaggle training notebook
(`emotion_classification_kaggle.ipynb`). It's a Gradio web app that loads your
trained model and predicts one of 6 emotions (joy, sadness, anger, fear, surprise,
disgust) from free text, with a confidence chart and an optional attention heatmap.

## 1. Train on Kaggle
Run `emotion_classification_kaggle.ipynb` on Kaggle with GPU enabled. At the end it
saves everything the app needs into `/kaggle/working/deployment_artifacts/`.

## 2. Download the artifacts
On Kaggle: **Output** tab (right sidebar) → download the `deployment_artifacts`
folder. It should contain:

```
deployment_artifacts/
├── distilbert_emotion/            # fine-tuned HuggingFace model + tokenizer
├── bilstm_attention_model.keras   # custom BiLSTM + Attention model
├── keras_tokenizer.json
├── config.json
└── model_comparison.csv
```

## 3. Set up the project in VS Code
Place the downloaded `deployment_artifacts` folder next to `app.py`, so your
project looks like this:

```
deployment/
├── app.py
├── requirements.txt
├── README.md
└── deployment_artifacts/
    ├── distilbert_emotion/
    ├── bilstm_attention_model.keras
    ├── keras_tokenizer.json
    ├── config.json
    └── model_comparison.csv
```

Open this folder in VS Code, then in the integrated terminal:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## 4. Run the app

```bash
python app.py
```

Gradio will print a local URL (usually `http://127.0.0.1:7860`) — open it in your
browser. To also get a temporary public shareable link, change the last line of
`app.py` to `demo.launch(share=True)`.

## Choosing which model to use
Both models are available live in the app via a radio button — no code changes
needed:
- **DistilBERT (fine-tuned)**: loaded at startup, typically the highest accuracy.
- **BiLSTM + Attention**: loaded lazily on first use, slightly lower accuracy but
  shows a live attention heatmap highlighting which words drove the prediction —
  useful for the "explainability" part of the project.

## Notes
- The text cleaning in `app.py` (`clean_text`) intentionally mirrors the
  preprocessing used during training in the notebook — don't change one without
  the other, or predictions will be inconsistent.
- If you only trained/exported one of the two models, just select that one in the
  app — the other option will only fail if you actually click it.
- First DistilBERT load can take a few seconds; the BiLSTM+Attention model loads
  lazily the first time you select it, so its first prediction is a bit slower too.
