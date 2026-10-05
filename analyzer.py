from pathlib import Path
from functools import lru_cache
from time import perf_counter
import re

import joblib

from sentiment import analyze_sentiment
from explain import get_model_clues
from themes import analyze_themes
from prompts import get_reflection_question, build_interpretation

ROOT = Path(__file__).resolve().parent
MAX_CHARACTERS = 5000


@lru_cache(maxsize=1)
def load_model():
    path = ROOT / "models" / "emotion_logreg.joblib"

    if not path.exists():
        raise FileNotFoundError(
            "LogReg model missing. Run train_model.py first."
        )

    return joblib.load(path)


@lru_cache(maxsize=1)
def load_hartmann():
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )

    path = ROOT / "models" / "emotion_hartmann"

    if not path.exists():
        raise FileNotFoundError(
            "Hartmann model missing from models/emotion_hartmann."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        path,
        local_files_only=True,
    )
    model.eval()

    return tokenizer, model


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())

    return [
        part.strip()
        for part in parts
        if part.strip() and any(char.isalnum() for char in part)
    ]


def format_prediction(class_scores):
    return {
        "emotion": max(class_scores, key=class_scores.get),
        "scores": class_scores,
    }


def predict_logreg(texts):
    saved = load_model()
    model = saved["pipeline"]
    labels = saved["labels"]

    predictions = []

    for scores in model.predict_proba(texts):
        class_scores = {
            labels[int(class_id)]: float(score)
            for class_id, score in zip(model.classes_, scores)
        }
        predictions.append(format_prediction(class_scores))

    return predictions, saved["metadata"]


def predict_hartmann(texts):
    import torch

    tokenizer, model = load_hartmann()

    # RoBERTa reserves positions for padding and special positioning.
    position_limit = (
        model.config.max_position_embeddings
        - model.config.pad_token_id
        - 1
    )
    token_limit = min(
        int(tokenizer.model_max_length),
        position_limit,
    )

    predictions = []

    for index, text in enumerate(texts):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
        )

        if inputs["input_ids"].shape[1] > token_limit:
            section = (
                "Your full entry"
                if index == 0
                else f"Sentence {index}"
            )
            raise ValueError(
                f"{section} exceeds Hartmann's "
                f"{token_limit}-token limit. Please shorten it."
            )

        with torch.inference_mode():
            scores = model(**inputs).logits.softmax(dim=-1)[0]

        class_scores = {
            model.config.id2label[class_id].lower(): float(score.item())
            for class_id, score in enumerate(scores)
        }
        predictions.append(format_prediction(class_scores))

    metadata = {
        "model_name": "j-hartmann/emotion-english-distilroberta-base",
        "version": "local-checkpoint",
        "source": "Pretrained Hugging Face checkpoint",
    }

    return predictions, metadata


def analyze_text(text, emotion_model="logreg"):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter some text.")

    if len(text) > MAX_CHARACTERS:
        raise ValueError(
            f"Please keep your entry within "
            f"{MAX_CHARACTERS:,} characters."
        )

    if emotion_model not in {"logreg", "hartmann"}:
        raise ValueError("Unknown emotion model.")

    text = text.strip()
    sentences = split_sentences(text)

    if not sentences:
        raise ValueError("Please include words in your entry.")

    # Total analysis time includes any first-use model loading.
    start = perf_counter()

    texts = [text] + sentences

    if emotion_model == "logreg":
        predictions, metadata = predict_logreg(texts)
        clues = get_model_clues(text, load_model()["pipeline"])
    else:
        predictions, metadata = predict_hartmann(texts)
        clues = []

    overall = predictions[0]

    sentence_results = [
        {
            "sentence": sentence,
            **prediction,
        }
        for sentence, prediction in zip(
            sentences, predictions[1:]
        )
    ]

    prediction_changes = sum(
        previous["emotion"] != current["emotion"]
        for previous, current in zip(
            sentence_results, sentence_results[1:]
        )
    )

    sentiment = analyze_sentiment(text)
    topics = analyze_themes(text)

    if emotion_model == "logreg":
        reflection = get_reflection_question(
            overall["emotion"],
            topics["themes"],
        )
        interpretation = build_interpretation(
            overall,
            sentiment,
            sentence_results,
            topics["themes"],
        )
    else:
        # Existing prompt templates were written for the six-class model.
        reflection = {
            "question": (
                "Which part of this entry feels most important "
                "to you right now?"
            )
        }
        interpretation = (
            f"The emotion model predicts "
            f"{overall['emotion']} for the whole entry. "
            f"The separate sentiment model predicts "
            f"{sentiment['label']} sentiment. "
            "These predictions may miss context or mixed feelings."
        )

    notes = [
        "English is supported; Hindi/Hinglish reliability is unverified.",
        "Scores are uncalibrated model predictions, not emotional percentages.",
        "Each emotion model predicts one label per text.",
        "Negation and changes from past to present feelings may be misread.",
    ]

    if emotion_model == "hartmann":
        notes.append(
            "Hartmann supports neutral and disgust, "
            "but has no love class."
        )
    else:
        notes.append(
            "LogReg has no neutral class."
        )

    if len(text.split()) < 3:
        notes.append(
            "Very short text may give unreliable predictions."
        )

    for note in sentiment.get("notes", []):
        if note not in notes:
            notes.append(note)

    return {
        "overall": overall,
        "sentiment": sentiment,
        "model_clues": clues,
        "model_clues_available": emotion_model == "logreg",
        "topics": topics,
        "reflection": reflection,
        "interpretation": interpretation,
        "sentences": sentence_results,
        "prediction_changes": prediction_changes,
        "latency_ms": (perf_counter() - start) * 1000,
        "model": metadata,
        "emotion_model": emotion_model,
        "notes": notes,
    }