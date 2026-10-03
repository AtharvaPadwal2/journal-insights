from pathlib import Path
from time import perf_counter
import json

import joblib
import sklearn
from datasets import load_from_disk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "emotion"
MODEL_PATH = ROOT / "models" / "emotion_logreg.joblib"
REPORT_PATH = ROOT / "reports" / "logreg_validation.json"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Clean dataset missing. Run prepare_dataset.py first."
        )

    dataset = load_from_disk(str(DATA_PATH))
    train = dataset["train"]
    validation = dataset["validation"]

    labels = train.features["label"].names
    label_ids = list(range(len(labels)))

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=50000,
                sublinear_tf=True,
                stop_words=None,  # Preserve words like "not" and "never".
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    print(f"Training on {len(train)} examples...")
    start = perf_counter()
    model.fit(train["text"], train["label"])
    training_seconds = perf_counter() - start

    print("Evaluating on validation data...")
    predictions = model.predict(validation["text"])

    accuracy = accuracy_score(validation["label"], predictions)
    macro_f1 = f1_score(
        validation["label"],
        predictions,
        labels=label_ids,
        average="macro",
        zero_division=0,
    )

    report = {
        "model": "TF-IDF + Logistic Regression",
        "dataset": "DAIR-AI Emotion — cleaned split configuration",
        "evaluation_split": "validation",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "labels": labels,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "training_seconds": training_seconds,
        "sklearn_version": sklearn.__version__,
        "classification_report": classification_report(
            validation["label"],
            predictions,
            labels=label_ids,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            validation["label"],
            predictions,
            labels=label_ids,
        ).tolist(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save the vectorizer, classifier and label mapping together.
    joblib.dump(
        {
            "pipeline": model,
            "labels": labels,
            "metadata": {
                "model_name": "tfidf_logreg",
                "version": "baseline-1",
                "sklearn_version": sklearn.__version__,
                "dataset": report["dataset"],
            },
        },
        MODEL_PATH,
    )

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nValidation accuracy: {accuracy:.4f}")
    print(f"Validation Macro F1: {macro_f1:.4f}")
    print(f"Training time: {training_seconds:.2f} seconds")

    print("\nPer-class validation results:")
    print(classification_report(
        validation["label"],
        predictions,
        labels=label_ids,
        target_names=labels,
        zero_division=0,
    ))

    print(f"Model saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()