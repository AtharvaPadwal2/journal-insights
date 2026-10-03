from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_PATH = Path(__file__).resolve().parent / "models" / "sentiment_roberta"
LABELS = {"negative", "neutral", "positive"}


@lru_cache(maxsize=1)
def load_sentiment_analyzer():
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"Sentiment model not found: {MODEL_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
    model.to("cpu")
    model.eval()

    model_labels = {
        str(label).lower()
        for label in model.config.id2label.values()
    }
    if model_labels != LABELS:
        raise ValueError("Unexpected sentiment model labels.")

    return tokenizer, model


def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter some text.")

    tokenizer, model = load_sentiment_analyzer()

    # Match the model card's username and URL preprocessing.
    words = []
    for word in text.strip().split():
        if word.startswith("@") and len(word) > 1:
            word = "@user"
        elif word.startswith("http"):
            word = "http"
        words.append(word)

    inputs = tokenizer(
        " ".join(words),
        return_tensors="pt",
        truncation=False,
    )

    if inputs["input_ids"].shape[1] > 512:
        raise ValueError(
            "Sentiment analysis currently supports up to 512 tokens. "
            "Please enter a shorter passage."
        )

    with torch.inference_mode():
        probabilities = model(**inputs).logits.softmax(dim=-1)[0]

    scores = {
        str(model.config.id2label[index]).lower(): float(score.item())
        for index, score in enumerate(probabilities)
    }

    return {
        "label": max(scores, key=scores.get),
        "scores": scores,
        "method": "CardiffNLP Twitter-RoBERTa sentiment",
        "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "notes": [
            "Scores are uncalibrated model predictions.",
            "Journal reliability still requires broader evaluation.",
        ],
    }