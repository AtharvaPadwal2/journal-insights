import hashlib
import json

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
)
from sklearn.model_selection import KFold, cross_val_predict

from train_goemotions import ROOT, DATA_DIR, CACHE_DIR, MODEL_PATH

OUTPUT_MODEL = ROOT / "models/emotion_goemotions_embedding_tuned.joblib"
OUTPUT_REPORT = ROOT / "reports/goemotions_threshold_validation.json"


def main():
    if OUTPUT_MODEL.exists() or OUTPUT_REPORT.exists():
        raise FileExistsError("Tuned model/report already exists.")

    saved = joblib.load(MODEL_PATH)
    metadata = json.loads(
        (CACHE_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    labels = saved["labels"]

    if labels != metadata["labels"]:
        raise ValueError("Model/cache label order mismatch.")

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
            x = cache["embeddings"].copy()
            y = cache["targets"].copy()
            ids = cache["ids"].tolist()

        expected = np.array([
            [int(label in row["emotion_labels"]) for label in labels]
            for row in records
        ])

        if (
            ids != [row["id"] for row in records]
            or not np.array_equal(y, expected)
            or x.shape != (len(records), 384)
            or not np.isfinite(x).all()
        ):
            raise ValueError(f"{split}: invalid cache.")

        if split == "train":
            normalized = [
                " ".join(row["text"].casefold().split())
                for row in records
            ]
            if len(set(normalized)) != len(normalized):
                raise ValueError("Duplicate training texts; grouped CV needed.")

        datasets[split] = (x, y)

    x_train, y_train = datasets["train"]
    x_validation, y_validation = datasets["validation"]

    print("Generating training-only out-of-fold scores...")
    oof = cross_val_predict(
        clone(saved["classifier"]),
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

    # Reuse baseline classifier already fitted on all training rows.
    scores = saved["classifier"].predict_proba(x_validation)
    report = {
        "labels": labels,
        "threshold_selection": "training-only 3-fold OOF per-label F1",
        "thresholds": thresholds.tolist(),
        "evaluation_split": "filtered GoEmotions validation",
        "train_rows": len(x_train),
        "validation_rows": len(x_validation),
        "models": {},
        "test_evaluated": False,
    }

    for name, cutoffs in (
        ("fixed_0_5", np.full(len(labels), 0.5)),
        ("oof_tuned", thresholds),
    ):
        predictions = (scores >= cutoffs).astype(int)
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
            "empty_prediction_rows": int(
                (predictions.sum(axis=1) == 0).sum()
            ),
        }

        report["models"][name] = {
            "metrics": metrics,
            "classification_report": classification_report(
                y_validation, predictions,
                target_names=labels,
                output_dict=True,
                zero_division=0,
            ),
        }

        print(f"\n{name}")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    print("\nTuned per-label results:")
    print(classification_report(
        y_validation,
        (scores >= thresholds).astype(int),
        target_names=labels,
        zero_division=0,
    ))

    saved["thresholds"] = thresholds.tolist()
    saved["metadata"] = {
        **saved["metadata"],
        "version": "goemotions-embedding-oof-f1-1",
        "threshold_selection": report["threshold_selection"],
    }

    joblib.dump(saved, OUTPUT_MODEL)
    OUTPUT_REPORT.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("Model saved:", OUTPUT_MODEL)
    print("Report saved:", OUTPUT_REPORT)
    print("Baseline preserved. Encoder unchanged. Test not evaluated.")
    print("Empty predictions do not establish neutrality.")


if __name__ == "__main__":
    main()