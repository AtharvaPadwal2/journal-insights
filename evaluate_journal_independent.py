import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data/evaluation/journal_independent_test.csv"
MODEL_PATH = ROOT / "models/emotion_hybrid_journal_targeted.joblib"
REPORT_PATH = ROOT / "reports/journal_independent_test.json"


def main():
    if REPORT_PATH.exists():
        raise FileExistsError(
            "Report already exists; refusing to overwrite."
        )

    saved = joblib.load(MODEL_PATH)
    metadata = saved["metadata"]
    labels = saved["labels"]
    thresholds = np.asarray(saved["thresholds"], dtype=float)

    with CSV_PATH.open(encoding="utf-8-sig", newline="") as file:
        records = list(csv.DictReader(file))

    usable = []
    seen_ids = set()
    seen_texts = set()

    for row in records:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed CSV record.")

        entry_id = row["id"].strip()
        text = " ".join(row["text"].casefold().split())

        if (
            not entry_id
            or not text
            or entry_id in seen_ids
            or text in seen_texts
        ):
            raise ValueError("Empty or duplicate ID/text.")

        seen_ids.add(entry_id)
        seen_texts.add(text)

        if row["review_status"] not in {"reviewed", "pending"}:
            raise ValueError(f"{entry_id}: invalid review status.")

        state = row["annotation_state"]
        emotions = (
            row["emotion_labels"].split("|")
            if row["emotion_labels"] else []
        )

        if (
            state not in {"emotional", "neutral", "unclear"}
            or set(emotions) - set(labels)
            or len(emotions) != len(set(emotions))
            or (state == "emotional" and not emotions)
            or (state in {"neutral", "unclear"} and emotions)
        ):
            raise ValueError(f"{entry_id}: invalid labels/state.")

        if row["review_status"] == "reviewed" and state != "unclear":
            usable.append(row)

    if not usable:
        raise ValueError("No reviewed, usable entries.")

    targets = np.array([
        [
            int(label in row["emotion_labels"].split("|"))
            for label in labels
        ]
        for row in usable
    ], dtype=np.int8)

    torch.set_num_threads(2)
    encoder = SentenceTransformer(
        metadata["encoder"],
        device="cpu",
        local_files_only=True,
    )
    encoder.max_seq_length = metadata["max_seq_length"]

    for row in usable:
        tokens = encoder.tokenizer(
            row["text"],
            truncation=False,
            add_special_tokens=True,
            verbose=False,
        )["input_ids"]

        if len(tokens) > encoder.max_seq_length:
            raise ValueError(f"{row['id']}: token limit exceeded.")

    embeddings = encoder.encode(
        [row["text"] for row in usable],
        batch_size=4,
        normalize_embeddings=metadata["normalize_embeddings"],
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid embeddings.")

    features = hstack([
        saved["vectorizer"].transform(
            [row["text"] for row in usable]
        ),
        csr_matrix(embeddings * saved["embedding_weight"]),
    ], format="csr")

    scores = np.asarray(saved["classifier"].predict_proba(features))

    if (
        scores.shape != targets.shape
        or thresholds.shape != (len(labels),)
        or not np.isfinite(scores).all()
        or not np.isfinite(thresholds).all()
    ):
        raise ValueError("Invalid scores or thresholds.")

    predictions = (scores >= thresholds).astype(np.int8)
    actual_neutral = targets.sum(axis=1) == 0
    empty_selection = predictions.sum(axis=1) == 0

    metrics = {
        "macro_f1": float(f1_score(
            targets, predictions, average="macro", zero_division=0,
        )),
        "micro_f1": float(f1_score(
            targets, predictions, average="micro", zero_division=0,
        )),
        "micro_precision": float(precision_score(
            targets, predictions, average="micro", zero_division=0,
        )),
        "micro_recall": float(recall_score(
            targets, predictions, average="micro", zero_division=0,
        )),
        "exact_match_accuracy": float(accuracy_score(
            targets, predictions,
        )),
        "hamming_loss": float(hamming_loss(targets, predictions)),
        "empty_selection_vs_neutral_f1": float(f1_score(
            actual_neutral, empty_selection, zero_division=0,
        )),
        "neutral_rows": int(actual_neutral.sum()),
        "neutral_rows_with_empty_selection": int(
            (actual_neutral & empty_selection).sum()
        ),
        "emotional_rows_with_empty_selection": int(
            (~actual_neutral & empty_selection).sum()
        ),
        "missed_labels": int(
            ((targets == 1) & (predictions == 0)).sum()
        ),
        "extra_labels": int(
            ((targets == 0) & (predictions == 1)).sum()
        ),
    }

    details = classification_report(
        targets,
        predictions,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    report = {
        "evaluated_rows": len(usable),
        "skipped_rows": len(records) - len(usable),
        "labels": labels,
        "thresholds": thresholds.tolist(),
        "source_counts": dict(Counter(
            row["source"] for row in usable
        )),
        "model_sha256": hashlib.sha256(
            MODEL_PATH.read_bytes()
        ).hexdigest(),
        "evaluation_sha256": hashlib.sha256(
            CSV_PATH.read_bytes()
        ).hexdigest(),
        "metrics": metrics,
        "classification_report": details,
        "entry_results": [
            {
                "id": row["id"],
                "expected": [
                    label for label, value in zip(labels, actual)
                    if value
                ],
                "predicted": [
                    label for label, value in zip(labels, predicted)
                    if value
                ],
            }
            for row, actual, predicted in zip(
                usable, targets, predictions
            )
        ],
        "training_performed": False,
        "thresholds_changed": False,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nEvaluated rows: {len(usable)}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nPer-emotion results:")
    print(classification_report(
        targets, predictions,
        target_names=labels,
        zero_division=0,
    ))

    print("Report saved:", REPORT_PATH)
    print("No training or threshold changes.")
    print("No raw journal text saved in report.")
    print("Empty selection does not establish neutrality.")
    print("Results apply to this sample; provenance and independence")
    print("depend on how the entries were authored and reviewed.")


if __name__ == "__main__":
    main()