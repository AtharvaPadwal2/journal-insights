import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import sklearn
import torch
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    fbeta_score,
    hamming_loss,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/processed/semeval2018/clean"
CACHE_DIR = ROOT / "data/processed/semeval2018/embeddings_minilm"

TRAINING_CSV = ROOT / "data/annotation/journal_training.csv"
DIAGNOSTIC_CSV = ROOT / "data/annotation/journal_labels.csv"

MODEL_PATH = ROOT / "models/emotion_hybrid_journal_pilot.joblib"
REPORT_PATH = ROOT / "reports/hybrid_journal_pilot_validation.json"
BASELINE_REPORT = ROOT / "reports/hybrid_precision_validation.json"


def normalize_text(text):
    return " ".join(text.casefold().split())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HybridFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_weight=1.0):
        self.embedding_weight = embedding_weight

    def fit(self, rows, y=None):
        self.vectorizer_ = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
            stop_words=None,
            dtype=np.float32,
        )
        self.vectorizer_.fit([row["text"] for row in rows])
        return self

    def transform(self, rows):
        word_features = self.vectorizer_.transform(
            [row["text"] for row in rows]
        )
        embeddings = np.stack(
            [row["embedding"] for row in rows]
        ).astype(np.float32)

        return hstack(
            [
                word_features,
                csr_matrix(embeddings * self.embedding_weight),
            ],
            format="csr",
        )


def evaluate(targets, predictions, labels):
    return {
        "macro_f1": float(f1_score(
            targets, predictions,
            average="macro", zero_division=0,
        )),
        "micro_f1": float(f1_score(
            targets, predictions,
            average="micro", zero_division=0,
        )),
        "exact_match_accuracy": float(
            accuracy_score(targets, predictions)
        ),
        "hamming_loss": float(
            hamming_loss(targets, predictions)
        ),
        "neutral_detection_f1": float(f1_score(
            targets.sum(axis=1) == 0,
            predictions.sum(axis=1) == 0,
            zero_division=0,
        )),
        "classification_report": classification_report(
            targets,
            predictions,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "id", "text", "emotion_labels", "annotation_state",
            "review_status", "source", "group_id",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name}: required columns missing.")

        rows = list(reader)

    for line, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path.name}, line {line}: malformed CSV.")

    return rows


def load_cached_split(split, metadata, labels):
    path = DATA_DIR / f"{split}.jsonl"

    if sha256(path) != metadata["splits"][split]["source_sha256"]:
        raise ValueError(f"{split}: embedding cache is stale.")

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

    if ids != [row["id"] for row in records]:
        raise ValueError(f"{split}: cached row order mismatch.")

    for row in records:
        if set(row["emotion_labels"]) - set(labels):
            raise ValueError(f"{split}: unknown labels.")

    expected = np.array([
        [int(label in row["emotion_labels"]) for label in labels]
        for row in records
    ])

    if not np.array_equal(targets, expected):
        raise ValueError(f"{split}: cached labels mismatch.")

    if (
        embeddings.ndim != 2
        or len(embeddings) != len(records)
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError(f"{split}: invalid embeddings.")

    rows = [
        {"text": record["text"], "embedding": embedding}
        for record, embedding in zip(records, embeddings)
    ]
    return records, rows, targets


def load_journal_candidates(labels):
    candidates = []
    skipped = {"pending": 0, "unclear": 0}
    seen_ids = set()
    seen_texts = set()

    for row in read_csv(TRAINING_CSV):
        entry_id = row["id"].strip()
        text = row["text"].strip()
        state = row["annotation_state"].strip()
        status = row["review_status"].strip()
        source = row["source"].strip()
        group_id = row["group_id"].strip()

        proposed = row["emotion_labels"].strip()
        emotions = proposed.split("|") if proposed else []

        if not entry_id or not text or not source or not group_id:
            raise ValueError("Journal training row has empty required fields.")

        if entry_id in seen_ids or normalize_text(text) in seen_texts:
            raise ValueError(f"{entry_id}: duplicate ID or text.")

        seen_ids.add(entry_id)
        seen_texts.add(normalize_text(text))

        if status not in {"pending", "reviewed"}:
            raise ValueError(f"{entry_id}: invalid review status.")

        if state not in {"emotional", "neutral", "unclear"}:
            raise ValueError(f"{entry_id}: invalid annotation state.")

        if (
            set(emotions) - set(labels)
            or len(emotions) != len(set(emotions))
        ):
            raise ValueError(f"{entry_id}: invalid emotion labels.")

        if state == "emotional" and not emotions:
            raise ValueError(f"{entry_id}: emotional row needs labels.")

        if state in {"neutral", "unclear"} and emotions:
            raise ValueError(f"{entry_id}: {state} row must have no labels.")

        if status == "pending":
            skipped["pending"] += 1
            continue

        if state == "unclear":
            skipped["unclear"] += 1
            continue

        candidates.append({
            "id": entry_id,
            "text": text,
            "emotion_labels": emotions,
            "source": source,
            "group_id": group_id,
        })

    if not candidates:
        raise ValueError("No reviewed, usable journal training entries.")

    return candidates, skipped


def main():
    metadata = json.loads(
        (CACHE_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    labels = metadata["labels"]

    train_records, x_train, y_train = load_cached_split(
        "train", metadata, labels
    )
    validation_records, x_validation, y_validation = load_cached_split(
        "validation", metadata, labels
    )

    journal_records, skipped = load_journal_candidates(labels)
    diagnostic_records = read_csv(DIAGNOSTIC_CSV)

    # Exact normalized-text checks do not detect paraphrases.
    blocked_texts = {
        normalize_text(row["text"])
        for row in train_records + validation_records + diagnostic_records
    }
    diagnostic_ids = {row["id"] for row in diagnostic_records}
    diagnostic_groups = {
        row["group_id"] for row in diagnostic_records
    }

    for row in journal_records:
        if normalize_text(row["text"]) in blocked_texts:
            raise ValueError(
                f"{row['id']}: training text overlaps existing data."
            )
        if (
            row["id"] in diagnostic_ids
            or row["group_id"] in diagnostic_groups
        ):
            raise ValueError(
                f"{row['id']}: overlaps diagnostic ID/group."
            )

    torch.set_num_threads(2)
    encoder = SentenceTransformer(
        metadata["encoder"],
        device="cpu",
        local_files_only=True,
    )
    encoder.max_seq_length = metadata["max_seq_length"]
    # Existing MiniLM cache was created with normalized embeddings.
    normalize_embeddings = metadata.get("normalize_embeddings", True)

    for row in journal_records:
        token_ids = encoder.tokenizer(
            row["text"],
            truncation=False,
            add_special_tokens=True,
        )["input_ids"]

        if len(token_ids) > encoder.max_seq_length:
            raise ValueError(
                f"{row['id']}: exceeds encoder token limit."
            )

    encoding_start = perf_counter()
    journal_embeddings = encoder.encode(
        [row["text"] for row in journal_records],
        batch_size=4,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    encoding_seconds = perf_counter() - encoding_start

    if (
        journal_embeddings.shape[1] != x_train[0]["embedding"].shape[0]
        or not np.isfinite(journal_embeddings).all()
    ):
        raise ValueError("Journal embeddings do not match cached features.")

    base_train_count = len(x_train)

    # Shared normalized tweet texts remain together during CV.
    groups = [
        "semeval:" + normalize_text(row["text"])
        for row in train_records
    ]

    x_train.extend([
        {"text": row["text"], "embedding": embedding}
        for row, embedding in zip(journal_records, journal_embeddings)
    ])
    groups.extend([
        "journal:" + row["group_id"]
        for row in journal_records
    ])

    journal_targets = np.array([
        [int(label in row["emotion_labels"]) for label in labels]
        for row in journal_records
    ])
    y_train = np.vstack([y_train, journal_targets])

    model = Pipeline([
        ("features", HybridFeatures(embedding_weight=1.0)),
        (
            "classifier",
            OneVsRestClassifier(
                LogisticRegression(
                    C=1.0,
                    class_weight=None,
                    max_iter=2000,
                    random_state=42,
                ),
                n_jobs=1,
            ),
        ),
    ])

    print(f"\nSemEval training rows: {base_train_count}")
    print(f"Reviewed journal training rows: {len(journal_records)}")
    print(f"Skipped journal rows: {skipped}")
    print(f"Total training rows: {len(x_train)}")

    # Materialized grouped folds; TF-IDF fits within each fold.
    folds = list(
        GroupKFold(n_splits=3).split(x_train, y_train, groups)
    )

    for fold_number, (fit_indices, _) in enumerate(folds, start=1):
        positives = y_train[fit_indices].sum(axis=0)
        if np.any(positives == 0) or np.any(
            positives == len(fit_indices)
        ):
            raise ValueError(
                f"Fold {fold_number}: a label lacks both target classes."
            )

    start = perf_counter()
    print("\nSelecting thresholds using training-only grouped CV...")

    oof_scores = cross_val_predict(
        model,
        x_train,
        y_train,
        cv=folds,
        method="predict_proba",
        n_jobs=1,
    )

    thresholds = np.full(len(labels), 0.5)
    candidates = sorted(
        np.linspace(0.05, 0.95, 19),
        key=lambda value: (abs(value - 0.5), value),
    )

    for index, label in enumerate(labels):
        best_score = -1.0

        for threshold in candidates:
            score = fbeta_score(
                y_train[:, index],
                oof_scores[:, index] >= threshold,
                beta=0.5,
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
    predictions = (scores >= thresholds).astype(int)

    report = {
        "encoder": metadata["encoder"],
        "model": "TF-IDF + MiniLM + OneVsRest Logistic Regression",
        "embedding_weight": 1.0,
        "C": 1.0,
        "class_weight": None,
        "labels": labels,
        "semeval_train_rows": base_train_count,
        "journal_train_rows": len(journal_records),
        "train_rows": len(x_train),
        "validation_rows": len(x_validation),
        "evaluation_split": "SemEval validation",
        "threshold_selection": "training-only 3-fold grouped OOF F0.5",
        "thresholds": thresholds.tolist(),
        "journal_encoding_seconds": encoding_seconds,
        "training_and_cv_seconds": training_seconds,
        "sklearn_version": sklearn.__version__,
        "cache_metadata": metadata,
        "journal_training_sha256": sha256(TRAINING_CSV),
        "diagnostic_csv_sha256": sha256(DIAGNOSTIC_CSV),
        "journal_source_counts": {
            source: sum(
                row["source"] == source for row in journal_records
            )
            for source in sorted({
                row["source"] for row in journal_records
            })
        },
        "skipped_journal_rows": skipped,
        "overlap_checks": {
            "normalized_text": "train, validation, diagnostic",
            "diagnostic_ids_and_groups": "checked",
            "paraphrases": "not checked",
            "test": "not loaded or checked",
        },
        "fixed_0_5": evaluate(
            y_validation, (scores >= 0.5).astype(int), labels
        ),
        "oof_tuned": evaluate(y_validation, predictions, labels),
    }

    # Historical comparison: CV changed as well as training data.
    if BASELINE_REPORT.exists():
        baseline = json.loads(
            BASELINE_REPORT.read_text(encoding="utf-8")
        )
        baseline_hash = (
            baseline.get("cache_metadata", {})
            .get("splits", {})
            .get("validation", {})
            .get("source_sha256")
        )

        if (
            baseline.get("labels") == labels
            and baseline_hash
            == metadata["splits"]["validation"]["source_sha256"]
        ):
            metrics = (
                "macro_f1", "micro_f1", "exact_match_accuracy",
                "hamming_loss", "neutral_detection_f1",
            )
            report["historical_baseline_comparison"] = {
                metric: {
                    "baseline": baseline["oof_tuned"][metric],
                    "pilot": report["oof_tuned"][metric],
                    "difference": (
                        report["oof_tuned"][metric]
                        - baseline["oof_tuned"][metric]
                    ),
                }
                for metric in metrics
            }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "vectorizer": model.named_steps["features"].vectorizer_,
        "classifier": model.named_steps["classifier"],
        "embedding_weight": 1.0,
        "labels": labels,
        "thresholds": thresholds.tolist(),
        "metadata": {
            "version": "hybrid-journal-pilot-1",
            "encoder": metadata["encoder"],
            "normalize_embeddings": normalize_embeddings,
            "max_seq_length": metadata["max_seq_length"],
            "sklearn_version": sklearn.__version__,
            "score_type": "uncalibrated per-label probabilities",
            "journal_training_sha256": sha256(TRAINING_CSV),
            "journal_train_rows": len(journal_records),
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
        predictions,
        target_names=labels,
        zero_division=0,
    ))

    comparison = report.get("historical_baseline_comparison")
    if comparison:
        print("\nHistorical baseline -> journal pilot:")
        for metric, values in comparison.items():
            print(
                f"{metric}: {values['baseline']:.4f}"
                f" -> {values['pilot']:.4f}"
            )
        print("CV grouping also changed; this is not a controlled ablation.")

    print(f"\nTraining + CV time: {training_seconds:.2f}s")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")
    print("Encoder unchanged. Test split not loaded.")
    print("Diagnostic entries were not used for training.")
    print("Paraphrase overlap requires manual checking.")
    print("SemEval metrics do not establish journal-wide accuracy.")


if __name__ == "__main__":
    main()