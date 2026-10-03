from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from sentiment import analyze_sentiment

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "evaluation" / "journal_dev.csv"

LABELS = ["negative", "neutral", "positive"]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Evaluation file missing: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, keep_default_na=False)

    required = {"id", "text", "sentiment"}
    if not required.issubset(data.columns):
        raise ValueError("CSV needs id, text and sentiment columns.")

    if data.empty:
        raise ValueError("Evaluation file is empty.")

    data["sentiment"] = data["sentiment"].str.strip().str.lower()

    if not data["sentiment"].isin(LABELS).all():
        raise ValueError("Labels must be negative, neutral or positive.")

    if data["text"].str.strip().eq("").any():
        raise ValueError("Some examples have empty text.")

    predictions = [
        analyze_sentiment(text)["label"]
        for text in data["text"]
    ]

    accuracy = accuracy_score(data["sentiment"], predictions)
    macro_f1 = f1_score(
        data["sentiment"],
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    print("VADER — small development set")
    print("Labels are proposed; human review is still required.")
    print(f"Examples: {len(data)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}\n")

    print(classification_report(
        data["sentiment"],
        predictions,
        labels=LABELS,
        zero_division=0,
    ))

    print("MISMATCHES")
    mistakes = 0

    for row, prediction in zip(
        data.itertuples(index=False), predictions
    ):
        if row.sentiment != prediction:
            mistakes += 1
            print(f"\n{row.id}: {row.text}")
            print(f"Expected: {row.sentiment} | Predicted: {prediction}")

    if not mistakes:
        print("No mismatches on these examples.")

    print("\nThese development results do not establish journal-wide accuracy.")


if __name__ == "__main__":
    main()