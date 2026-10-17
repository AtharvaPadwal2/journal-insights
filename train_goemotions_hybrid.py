import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import sklearn
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/processed/goemotions/clean_minilm256"
CACHE_DIR = ROOT / "data/processed/goemotions/embeddings_minilm"
MODEL_PATH = ROOT / "models/emotion_goemotions_hybrid.joblib"
REPORT_PATH = ROOT / "reports/goemotions_hybrid_validation.json"


class HybridFeatures(BaseEstimator, TransformerMixin):
    def fit(self, rows, y=None):
        self.vectorizer_ = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.vectorizer_.fit([row["text"] for row in rows])
        return self

    def transform(self, rows):
        return hstack(
            [
                self.vectorizer_.transform(
                    [row["text"] for row in rows]
                ),
                csr_matrix(np.stack([
                    row["embedding"] for row in rows
                ]).astype(np.float32)),
            ],
            format="csr",
        )


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
        "empty_prediction_rows": int(
            (predictions.sum(axis=1) == 0).sum()
        ),
        "classification_report": classification_report(
            targets,
            predictions,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    if MODEL_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError("Hybrid model/report already exists.")

    metadata = json.loads(
        (CACHE_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    labels = metadata["labels"]
    datasets = {}

    for split in ("train", "validation"):
        path = DATA_DIR / f"{split}.jsonl"

        if hashlib.sha256(path.read_bytes()).hexdigest() != (
            metadata["splits"][split]["source_sha256"]
        ):
            raise ValueError(f"{split}: stale cache.")

        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        with np.load(
            CACHE_DIR / f"{split}.npz", allow_pickle=False
        ) as cache:
            embeddings = cache["embeddings"].copy()
            targets = cache["targets"].copy()
            ids = cache["ids"].tolist()

        expected = np.array([
            [int(label in row["emotion_labels"]) for label in labels]
            for row in records
        ])

        if (
            ids != [row["id"] for row in records]
            or not np.array_equal(targets, expected)
            or embeddings.shape != (len(records), 384)
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError(f"{split}: invalid cache.")

        if split == "train":
            texts = [
                " ".join(row["text"].casefold().split())
                for row in records
            ]
            if len(set(texts)) != len(texts):
                raise ValueError("Duplicate training texts found.")

        rows = [
            {"text": row["text"], "embedding": embedding}
            for row, embedding in zip(records, embeddings)
        ]
        datasets[split] = (rows, targets)

    x_train, y_train = datasets["train"]
    x_validation, y_validation = datasets["validation"]

    model = Pipeline([
        ("features", HybridFeatures()),
        (
            "classifier",
            OneVsRestClassifier(
                LogisticRegression(
                    C=1.0,
                    class_weight=None,
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=42,
                ),
                n_jobs=1,
            ),
        ),
    ])

    start = perf_counter()
    print(f"Training entries: {len(x_train)}")
    print("Generating training-only OOF scores...")
    print("TF-IDF is fitted independently inside each fold.")

    oof = cross_val_predict(
        model,
        x_train,
        y_train,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        method="predict_proba",
        n_jobs=1,
    )

    if oof.shape != y_train.shape or not np.isfinite(oof).all():
        raise ValueError("Invalid OOF scores.")

    thresholds = np.full(len(labels), 0.5)
    candidates = sorted(
        np.arange(1, 96) / 100,
        key=lambda value: (abs(value - 0.5), value),
    )

    for index, label in enumerate(labels):
        best_score = -1.0

        for threshold in candidates:
            score = f1_score(
                y_train[:, index],
                oof[:, index] >= threshold,
                zero_division=0,
            )
            if score > best_score:
                best_score = score
                thresholds[index] = threshold

        print(f"{label}: threshold={thresholds[index]:.2f}")

    print("\nFitting final hybrid classifier...")
    model.fit(x_train, y_train)
    training_seconds = perf_counter() - start

    scores = model.predict_proba(x_validation)
    if scores.shape != y_validation.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid validation scores.")

    report = {
        "model": "TF-IDF + frozen MiniLM + OneVsRest Logistic Regression",
        "labels": labels,
        "train_rows": len(x_train),
        "validation_rows": len(x_validation),
        "evaluation_split": "filtered GoEmotions validation",
        "threshold_selection": "training-only 3-fold OOF per-label F1",
        "thresholds": thresholds.tolist(),
        "training_and_cv_seconds": training_seconds,
        "cache_metadata": metadata,
        "sklearn_version": sklearn.__version__,
        "models": {},
        "test_evaluated": False,
    }

    for name, cutoffs in (
        ("fixed_0_5", np.full(len(labels), 0.5)),
        ("oof_tuned", thresholds),
    ):
        metrics = evaluate(
            y_validation, (scores >= cutoffs).astype(int), labels
        )
        report["models"][name] = metrics

        print(f"\n{name}")
        for key in (
            "macro_f1", "micro_f1", "exact_match_accuracy",
            "hamming_loss", "empty_prediction_rows",
        ):
            print(f"{key}: {metrics[key]}")

    print("\nTuned per-label results:")
    print(classification_report(
        y_validation,
        (scores >= thresholds).astype(int),
        target_names=labels,
        zero_division=0,
    ))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save standard sklearn objects, not the custom transformer.
    joblib.dump({
        "vectorizer": model.named_steps["features"].vectorizer_,
        "classifier": model.named_steps["classifier"],
        "embedding_weight": 1.0,
        "labels": labels,
        "thresholds": thresholds.tolist(),
        "metadata": {
            "version": "goemotions-hybrid-oof-f1-1",
            "encoder": metadata["encoder"],
            "max_seq_length": metadata["max_seq_length"],
            "normalize_embeddings": metadata["normalize_embeddings"],
            "sklearn_version": sklearn.__version__,
            "score_type": "uncalibrated per-label probabilities",
            "threshold_selection": report["threshold_selection"],
        },
    }, MODEL_PATH)

    REPORT_PATH.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"\nTraining + CV time: {training_seconds:.2f}s")
    print("Model saved:", MODEL_PATH)
    print("Report saved:", REPORT_PATH)
    print("Encoder unchanged. Test not evaluated.")
    print("Existing models preserved.")
    print("Empty predictions do not establish neutrality.")


if __name__ == "__main__":
    main()