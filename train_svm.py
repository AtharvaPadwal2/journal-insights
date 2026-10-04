from pathlib import Path
from time import perf_counter
import json

import joblib
import sklearn
from datasets import load_from_disk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "emotion"
MODEL_PATH = ROOT / "models" / "emotion_svm.joblib"
REPORT_PATH = ROOT / "reports" / "svm_validation.json"
BASELINE_PATH = ROOT / "models" / "emotion_logreg.joblib"


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
                stop_words=None,
            ),
        ),
        (
            "classifier",
            LinearSVC(
                C=1.0,
                dual="auto",
                max_iter=5000,
                random_state=42,
            ),
        ),
    ])

    print(f"Training on {len(train)} examples...")
    start = perf_counter()
    model.fit(train["text"], train["label"])
    training_seconds = perf_counter() - start

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
        "model": "TF-IDF + LinearSVC",
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

    # Compare the saved baseline on exactly the same validation rows.
    if BASELINE_PATH.exists():
        baseline = joblib.load(BASELINE_PATH)
        if baseline["labels"] != labels:
            raise ValueError("Baseline label mapping does not match.")

        baseline_predictions = baseline["pipeline"].predict(
            validation["text"]
        )
        baseline_accuracy = accuracy_score(
            validation["label"], baseline_predictions
        )
        baseline_f1 = f1_score(
            validation["label"],
            baseline_predictions,
            labels=label_ids,
            average="macro",
            zero_division=0,
        )

        report["baseline_comparison"] = {
            "model": "Saved TF-IDF + Logistic Regression",
            "accuracy": float(baseline_accuracy),
            "macro_f1": float(baseline_f1),
            "macro_f1_difference": float(macro_f1 - baseline_f1),
        }

        print("\nSame-validation comparison:")
        print(
            f"LogReg    accuracy={baseline_accuracy:.4f}"
            f"  Macro F1={baseline_f1:.4f}"
        )
        print(
            f"LinearSVC accuracy={accuracy:.4f}"
            f"  Macro F1={macro_f1:.4f}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": model,
            "labels": labels,
            "metadata": {
                "model_name": "tfidf_linear_svc",
                "version": "baseline-svm-1",
                "sklearn_version": sklearn.__version__,
                "dataset": report["dataset"],
                "score_type": "uncalibrated decision margins",
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
    print(classification_report(
        validation["label"],
        predictions,
        labels=label_ids,
        target_names=labels,
        zero_division=0,
    ))

    print(f"Model saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")
    print("Test split was not evaluated.")
    print("LinearSVC margins are not probabilities.")


if __name__ == "__main__":
    main()