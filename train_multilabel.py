import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    accuracy_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/processed/semeval2018/clean"
MODEL_PATH = ROOT / "models/emotion_multilabel.joblib"
REPORT_PATH = ROOT / "reports/multilabel_validation.json"

LABELS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "love", "optimism", "pessimism",
    "sadness", "surprise", "trust",
]


def read_split(split):
    with (DATA_DIR / f"{split}.jsonl").open(
        encoding="utf-8"
    ) as file:
        rows = [json.loads(line) for line in file]

    if not rows:
        raise ValueError(f"{split} is empty.")

    for row in rows:
        if not row["text"].strip():
            raise ValueError(f"{split}: empty text.")
        if set(row["emotion_labels"]) - set(LABELS):
            raise ValueError(f"{split}: unknown emotion label.")

    return rows


def main():
    train = read_split("train")
    validation = read_split("validation")

    encoder = MultiLabelBinarizer(classes=LABELS)

    y_train = encoder.fit_transform(
        [row["emotion_labels"] for row in train]
    )
    y_validation = encoder.transform(
        [row["emotion_labels"] for row in validation]
    )

    if np.any(y_train.sum(axis=0) == 0):
        raise ValueError("A label has no positive training examples.")

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
            OneVsRestClassifier(
                LogisticRegression(
                    C=1.0,
                    class_weight=None,
                    max_iter=2000,
                    random_state=42,
                ),
                n_jobs=2,
            ),
        ),
    ])

    print(f"Training on {len(train)} entries...")
    start = perf_counter()

    model.fit(
        [row["text"] for row in train],
        y_train,
    )

    training_seconds = perf_counter() - start

    scores = model.predict_proba(
        [row["text"] for row in validation]
    )

    print("Choosing thresholds using training-only CV...")

    # Each training row is scored by a model that did not train on it.
    # The entire pipeline is fitted separately within each fold.
    oof_scores = cross_val_predict(
        model,
        [row["text"] for row in train],
        y_train,
        cv=KFold(
            n_splits=3,
            shuffle=True,
            random_state=42,
        ),
        method="predict_proba",
        n_jobs=1,
    )

    thresholds = np.full(len(LABELS), 0.5)
    threshold_cv_f1 = {}

    # Start at 0.5; on equal F1, prefer thresholds nearer 0.5.
    candidates = sorted(
        np.linspace(0.1, 0.9, 17),
        key=lambda value: abs(value - 0.5),
    )

    for index, label in enumerate(LABELS):
        best_f1 = -1.0

        for threshold in candidates:
            predicted = (
                oof_scores[:, index] >= threshold
            ).astype(int)

            candidate_f1 = f1_score(
                y_train[:, index],
                predicted,
                zero_division=0,
            )

            if candidate_f1 > best_f1:
                best_f1 = candidate_f1
                thresholds[index] = threshold

        threshold_cv_f1[label] = float(best_f1)

        print(
            f"{label}: threshold={thresholds[index]:.2f}, "
            f"OOF selection F1={best_f1:.4f}"
        )

    # Evaluate the final training-fitted model on validation.
    predictions = (scores >= thresholds).astype(int)

    per_label = classification_report(
        y_validation,
        predictions,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    actual_neutral = y_validation.sum(axis=1) == 0
    predicted_empty = predictions.sum(axis=1) == 0

    report = {
        "model": "TF-IDF + OneVsRest Logistic Regression",
        "dataset": "SemEval-2018 English E-c",
        "evaluation_split": "validation",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "labels": LABELS,
        "thresholds": thresholds.tolist(),
        "threshold_selection": "training-only 3-fold OOF",
        "threshold_selection_f1": threshold_cv_f1,
        "macro_f1": float(f1_score(
            y_validation, predictions,
            average="macro", zero_division=0,
        )),
        "micro_f1": float(f1_score(
            y_validation, predictions,
            average="micro", zero_division=0,
        )),
        "exact_match_accuracy": float(accuracy_score(
            y_validation, predictions,
        )),
        "hamming_loss": float(hamming_loss(
            y_validation, predictions,
        )),
        "neutral_detection_f1": float(f1_score(
            actual_neutral, predicted_empty,
            zero_division=0,
        )),
        "actual_neutral_rows": int(actual_neutral.sum()),
        "predicted_empty_rows": int(predicted_empty.sum()),
        "classification_report": per_label,
        "training_seconds": training_seconds,
        "sklearn_version": sklearn.__version__,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": model,
            "labels": LABELS,
            "thresholds": thresholds.tolist(),
            "metadata": {
                "model_name": "tfidf_ovr_logreg",
                "version": "multilabel-unweighted-1",
                "dataset": report["dataset"],
                "sklearn_version": sklearn.__version__,
                "score_type": "uncalibrated per-label probabilities",
            },
        },
        MODEL_PATH,
    )

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nMacro F1: {report['macro_f1']:.4f}")
    print(f"Micro F1: {report['micro_f1']:.4f}")
    print(
        f"Exact-match accuracy: "
        f"{report['exact_match_accuracy']:.4f}"
    )
    print(f"Hamming loss: {report['hamming_loss']:.4f}")
    print(
        f"Neutral detection F1: "
        f"{report['neutral_detection_f1']:.4f}"
    )

    print(classification_report(
        y_validation,
        predictions,
        target_names=LABELS,
        zero_division=0,
    ))

    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")
    print("Test split was not loaded.")
    print("Manual/synthetic pilot entries were not used.")
    print("An empty prediction can mean missed emotions, not just neutral.")


if __name__ == "__main__":
    main()