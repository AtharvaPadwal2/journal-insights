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
from sklearn.multiclass import OneVsRestClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/processed/goemotions/clean_minilm256"
CACHE_DIR = ROOT / "data/processed/goemotions/embeddings_minilm"
MODEL_PATH = ROOT / "models/emotion_goemotions_embedding.joblib"
REPORT_PATH = ROOT / "reports/goemotions_embedding_validation.json"


def main():
    if MODEL_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError(
            "Model/report already exists; refusing to overwrite."
        )

    metadata = json.loads(
        (CACHE_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    labels = metadata["labels"]

    if len(labels) != 28 or len(set(labels)) != 28:
        raise ValueError("Expected 28 unique GoEmotions labels.")

    datasets = {}

    for split in ("train", "validation"):
        path = DATA_DIR / f"{split}.jsonl"
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        if source_hash != metadata["splits"][split]["source_sha256"]:
            raise ValueError(f"{split}: stale embedding cache.")

        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        with np.load(
            CACHE_DIR / f"{split}.npz",
            allow_pickle=False,
        ) as cached:
            embeddings = cached["embeddings"].copy()
            targets = cached["targets"].copy()
            ids = cached["ids"].tolist()

        expected_targets = np.array([
            [
                int(label in row["emotion_labels"])
                for label in labels
            ]
            for row in records
        ], dtype=np.int8)

        if ids != [row["id"] for row in records]:
            raise ValueError(f"{split}: cached ID/order mismatch.")

        if not np.array_equal(targets, expected_targets):
            raise ValueError(f"{split}: cached targets mismatch.")

        if (
            embeddings.shape != (len(records), 384)
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError(f"{split}: invalid embeddings.")

        datasets[split] = (embeddings, targets)

    x_train, y_train = datasets["train"]
    x_validation, y_validation = datasets["validation"]

    if any(
        len(np.unique(y_train[:, index])) != 2
        for index in range(len(labels))
    ):
        raise ValueError("A training label lacks positive/negative examples.")

    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=1.0,
            class_weight=None,
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        ),
        n_jobs=1,
    )

    print(f"Training on {len(x_train)} entries...")
    print(f"Labels: {len(labels)}")
    start = perf_counter()
    classifier.fit(x_train, y_train)
    training_seconds = perf_counter() - start

    scores = np.asarray(classifier.predict_proba(x_validation))

    if scores.shape != y_validation.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid validation scores.")

    thresholds = np.full(len(labels), 0.5)
    predictions = (scores >= thresholds).astype(np.int8)

    details = classification_report(
        y_validation,
        predictions,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
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
    }

    report = {
        "model": "Frozen MiniLM + OneVsRest Logistic Regression",
        "evaluation_split": "filtered GoEmotions validation",
        "labels": labels,
        "train_rows": len(x_train),
        "validation_rows": len(x_validation),
        "threshold_selection": "fixed 0.5; no tuning",
        "thresholds": thresholds.tolist(),
        "metrics": metrics,
        "classification_report": details,
        "empty_prediction_rows": int(
            (predictions.sum(axis=1) == 0).sum()
        ),
        "training_seconds": training_seconds,
        "sklearn_version": sklearn.__version__,
        "cache_metadata": metadata,
        "test_evaluated": False,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "classifier": classifier,
        "labels": labels,
        "thresholds": thresholds.tolist(),
        "metadata": {
            "version": "goemotions-embedding-baseline-1",
            "encoder": metadata["encoder"],
            "max_seq_length": metadata["max_seq_length"],
            "normalize_embeddings": metadata["normalize_embeddings"],
            "sklearn_version": sklearn.__version__,
            "score_type": "uncalibrated per-label probabilities",
        },
    }, MODEL_PATH)

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nPer-label results:")
    print(classification_report(
        y_validation,
        predictions,
        target_names=labels,
        zero_division=0,
    ))

    print("Empty prediction rows:", report["empty_prediction_rows"])
    print(f"Training time: {training_seconds:.2f}s")
    print("Model saved:", MODEL_PATH)
    print("Report saved:", REPORT_PATH)
    print("Neutral is an explicit label; empty predictions are not neutral.")
    print("Encoder unchanged. Test not evaluated.")
    print("Existing 11-label model unchanged.")


if __name__ == "__main__":
    main()