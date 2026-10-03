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
MODEL_PATH = ROOT / "models" / "emotion_logreg.joblib"
MAX_CHARACTERS = 5000


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model missing. Run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


def split_sentences(text):
    # Lightweight splitter: punctuation followed by whitespace, or newlines.
    # Abbreviations such as "Dr." may create extra segments.
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())

    return [
        part.strip()
        for part in parts
        if part.strip() and any(char.isalnum() for char in part)
    ]


def analyze_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter some text.")

    if len(text) > MAX_CHARACTERS:
        raise ValueError(
            f"Please keep your entry within {MAX_CHARACTERS:,} characters."
        )

    text = text.strip()
    sentences = split_sentences(text)

    if not sentences:
        raise ValueError("Please include words in your entry.")

    saved_model = load_model()
    model = saved_model["pipeline"]
    labels = saved_model["labels"]

    # Model loading is excluded from inference latency.
    start = perf_counter()

    # Analyze the whole entry and all sentences in one batch.
    probabilities = model.predict_proba([text] + sentences)

    def format_prediction(scores):
        class_scores = {
            labels[int(class_id)]: float(score)
            for class_id, score in zip(model.classes_, scores)
        }

        return {
            "emotion": max(class_scores, key=class_scores.get),
            "scores": class_scores,
        }

    overall = format_prediction(probabilities[0])

    sentence_results = [
        {
            "sentence": sentence,
            **format_prediction(scores),
        }
        for sentence, scores in zip(sentences, probabilities[1:])
    ]

    prediction_changes = sum(
        previous["emotion"] != current["emotion"]
        for previous, current in zip(
            sentence_results, sentence_results[1:]
        )
    )

    notes = [
        "English is supported; Hindi/Hinglish reliability is unverified.",
        "Scores are uncalibrated model predictions, not emotional percentages.",
    ]

    if len(text.split()) < 3:
        notes.append("Very short text may give unreliable predictions.")

    sentiment = analyze_sentiment(text)
    topics = analyze_themes(text)

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
    return {
        "overall": overall,
        "sentiment": sentiment,
        "model_clues": get_model_clues(text, model),
        "topics": topics,
        "reflection": reflection,
        "interpretation": interpretation,
        "sentences": sentence_results,
        "prediction_changes": prediction_changes,
        "latency_ms": (perf_counter() - start) * 1000,
        "model": saved_model["metadata"],
        "notes": notes,
    }