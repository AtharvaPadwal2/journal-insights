import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/processed/semeval2018/clean"
CACHE_DIR = ROOT / "data/processed/semeval2018/embeddings_minilm"
MODEL_PATH = ROOT / "models/emotion_embedding_logreg.joblib"
REPORT_PATH = ROOT / "reports/embedding_validation.json"


def evaluate(targets, predictions, labels):
    return {
        "macro_f1": float(f1_score(
            targets, predictions, average="macro", zero_division=0,
        )),
        "micro_f1": float(f1_score(
            targets, predictions, average="micro", zero_division=0,
        )),
        "exact_match_accuracy": float(accuracy_score(
            targets, predictions,
        )),
        "hamming_loss": float(hamming_loss(targets, predictions)),
        "neutral_detection_f1": float(f1_score(
            targets.sum(axis=1) == 0,
            predictions.sum(axis=1) == 0,
            zero_division=0,
        )),
        "classification_report": classification_report(
            targets, predictions,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    metadata = json.loads(
        (CACHE_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    labels = metadata["labels"]
    arrays = {}

    for split in ("train", "validation"):
        source_hash = hashlib.sha256(
            (DATA_DIR / f"{split}.jsonl").read_bytes()
        ).hexdigest()

        if source_hash != metadata["splits"][split]["source_sha256"]:
            raise ValueError(f"{split}: cache is stale. Rebuild it.")

        with np.load(
            CACHE_DIR / f"{split}.npz",
            allow_pickle=False,
        ) as cached:
            arrays[split] = (
                cached["embeddings"].copy(),
                cached["targets"].copy(),
            )

    x_train, y_train = arrays["train"]
    x_validation, y_validation = arrays["validation"]

    model = OneVsRestClassifier(
        LogisticRegression(
            C=1.0,
            class_weight=None,
            max_iter=2000,
            random_state=42,
        ),
        n_jobs=1,
    )

    start = perf_counter()
    print("Selecting thresholds using training-only CV...")

    oof_scores = cross_val_predict(
        model,
        x_train,
        y_train,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        method="predict_proba",
        n_jobs=1,
    )

    thresholds = np.full(len(labels), 0.5)
    candidates = sorted(
        np.linspace(0.05, 0.95, 19),
        key=lambda value: abs(value - 0.5),
    )

    for index, label in enumerate(labels):
        best_f1 = -1.0

        for threshold in candidates:
            score = f1_score(
                y_train[:, index],
                oof_scores[:, index] >= threshold,
                zero_division=0,
            )
            if score > best_f1:
                best_f1 = score
                thresholds[index] = threshold

        print(f"{label}: threshold={thresholds[index]:.2f}")

    print("\nFitting final classifier on all training rows...")
    model.fit(x_train, y_train)
    training_seconds = perf_counter() - start

    scores = model.predict_proba(x_validation)

    report = {
        "encoder": metadata["encoder"],
        "classifier": "OneVsRest Logistic Regression",
        "classifier_parameters": {"C": 1.0, "class_weight": None},
        "labels": labels,
        "train_rows": len(x_train),
        "validation_rows": len(x_validation),
        "evaluation_split": "validation",
        "threshold_selection": "training-only 3-fold OOF",
        "thresholds": thresholds.tolist(),
        "training_and_cv_seconds": training_seconds,
        "sklearn_version": sklearn.__version__,
        "cache_metadata": metadata,
        "fixed_0_5": evaluate(
            y_validation, (scores >= 0.5).astype(int), labels,
        ),
        "oof_tuned": evaluate(
            y_validation, (scores >= thresholds).astype(int), labels,
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "classifier": model,
        "labels": labels,
        "thresholds": thresholds.tolist(),
        "metadata": {
            "version": "frozen-minilm-logreg-1",
            "encoder": metadata["encoder"],
            "normalize_embeddings": True,
            "max_seq_length": metadata["max_seq_length"],
            "sklearn_version": sklearn.__version__,
            "score_type": "uncalibrated per-label probabilities",
        },
    }, MODEL_PATH)

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    for name in ("fixed_0_5", "oof_tuned"):
        print(f"\n{name}")
        for metric in (
            "macro_f1", "micro_f1", "exact_match_accuracy",
            "hamming_loss", "neutral_detection_f1",
        ):
            print(f"{metric}: {report[name][metric]:.4f}")

    print("\nOOF-tuned per-label results:")
    print(classification_report(
        y_validation,
        (scores >= thresholds).astype(int),
        target_names=labels,
        zero_division=0,
    ))

    print(f"Training + CV time: {training_seconds:.2f}s")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")
    print("Encoder was not fine-tuned.")
    print("Test split and annotation pilot were not used.")


if __name__ == "__main__":
    main()