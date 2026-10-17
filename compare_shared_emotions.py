import csv
import hashlib
import json
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
)

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data/evaluation/journal_fresh_validation.csv"
REPORT_PATH = ROOT / "reports/shared_emotion_comparison.json"

MODEL_PATHS = {
    "journal_targeted": ROOT / "models/emotion_hybrid_journal_targeted.joblib",
    "goemotions_hybrid": ROOT / "models/emotion_goemotions_hybrid.joblib",
}

SHARED = [
    "anger", "disgust", "fear", "joy",
    "love", "optimism", "sadness", "surprise",
]


def normalize(text):
    return " ".join(text.casefold().split())


def main():
    models = {
        name: joblib.load(path)
        for name, path in MODEL_PATHS.items()
    }
    reference = models["journal_targeted"]["metadata"]

    for name, saved in models.items():
        if not set(SHARED).issubset(saved["labels"]):
            raise ValueError(f"{name}: shared labels missing.")

        for key in (
            "encoder", "normalize_embeddings", "max_seq_length"
        ):
            if saved["metadata"][key] != reference[key]:
                raise ValueError(f"{name}: encoder settings mismatch.")

    with CSV_PATH.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "id", "text", "emotion_labels",
            "annotation_state", "review_status",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Validation CSV columns missing.")
        records = list(reader)

    blocked = set()

    # Check text overlap against both models' training sources.
    for path in (
        ROOT / "data/annotation/journal_training_targeted.csv",
    ):
        with path.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                blocked.add(normalize(row["text"]))

    for path in (
        ROOT / "data/processed/semeval2018/clean/train.jsonl",
        ROOT / "data/processed/goemotions/clean_minilm256/train.jsonl",
    ):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                blocked.add(normalize(json.loads(line)["text"]))

    usable = []
    seen_ids = set()
    seen_texts = set()
    valid_labels = set(models["journal_targeted"]["labels"])

    for row in records:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed evaluation CSV row.")

        entry_id = row["id"].strip()
        text = normalize(row["text"])

        if not entry_id or not text:
            raise ValueError("Empty evaluation ID/text.")

        if entry_id in seen_ids or text in seen_texts:
            raise ValueError("Duplicate evaluation ID/text.")

        seen_ids.add(entry_id)
        seen_texts.add(text)

        if text in blocked:
            raise ValueError(f"{entry_id}: training-text overlap.")

        if row["review_status"] != "reviewed":
            continue
        if row["annotation_state"] == "unclear":
            continue

        annotated = (
            row["emotion_labels"].split("|")
            if row["emotion_labels"] else []
        )

        if (
            set(annotated) - valid_labels
            or len(annotated) != len(set(annotated))
        ):
            raise ValueError(f"{entry_id}: invalid annotations.")

        state = row["annotation_state"]
        if (
            state not in {"emotional", "neutral"}
            or (state == "emotional" and not annotated)
            or (state == "neutral" and annotated)
        ):
            raise ValueError(f"{entry_id}: labels/state mismatch.")

        usable.append(row)

    if not usable:
        raise ValueError("No usable reviewed entries.")

    texts = [row["text"] for row in usable]
    targets = np.array([
        [
            int(label in row["emotion_labels"].split("|"))
            for label in SHARED
        ]
        for row in usable
    ], dtype=np.int8)

    torch.set_num_threads(2)
    encoder = SentenceTransformer(
        reference["encoder"],
        device="cpu",
        local_files_only=True,
    )
    encoder.max_seq_length = reference["max_seq_length"]

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
        texts,
        batch_size=4,
        normalize_embeddings=reference["normalize_embeddings"],
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid embeddings.")

    report = {
        "evaluated_rows": len(usable),
        "skipped_rows": len(records) - len(usable),
        "shared_labels": SHARED,
        "validation_sha256": hashlib.sha256(
            CSV_PATH.read_bytes()
        ).hexdigest(),
        "comparison_scope": "Whole-entry predictions; shared labels only",
        "models": {},
    }

    print(f"\nEvaluated rows: {len(usable)}")
    print("Shared labels:", ", ".join(SHARED))

    for name, saved in models.items():
        features = hstack([
            saved["vectorizer"].transform(texts),
            csr_matrix(embeddings * saved["embedding_weight"]),
        ], format="csr")

        scores = np.asarray(
            saved["classifier"].predict_proba(features)
        )
        thresholds = np.asarray(saved["thresholds"], dtype=float)

        if (
            scores.shape != (len(usable), len(saved["labels"]))
            or thresholds.shape != (len(saved["labels"]),)
            or not np.isfinite(scores).all()
            or not np.isfinite(thresholds).all()
        ):
            raise ValueError(f"{name}: invalid scores/thresholds.")

        indices = [saved["labels"].index(label) for label in SHARED]
        predictions = (
            scores[:, indices] >= thresholds[indices]
        ).astype(np.int8)

        metrics = {
            "macro_f1": float(f1_score(
                targets, predictions,
                average="macro", zero_division=0,
            )),
            "micro_f1": float(f1_score(
                targets, predictions,
                average="micro", zero_division=0,
            )),
            "shared_label_exact_match": float(accuracy_score(
                targets, predictions,
            )),
            "hamming_loss": float(hamming_loss(targets, predictions)),
            "missed_labels": int(
                ((targets == 1) & (predictions == 0)).sum()
            ),
            "extra_labels": int(
                ((targets == 0) & (predictions == 1)).sum()
            ),
        }

        report["models"][name] = {
            "metrics": metrics,
            "classification_report": classification_report(
                targets,
                predictions,
                target_names=SHARED,
                output_dict=True,
                zero_division=0,
            ),
            "model_sha256": hashlib.sha256(
                MODEL_PATHS[name].read_bytes()
            ).hexdigest(),
            "shared_thresholds": thresholds[indices].tolist(),
        }

        print(f"\n{name}")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        print(classification_report(
            targets,
            predictions,
            target_names=SHARED,
            zero_division=0,
        ))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Report saved:", REPORT_PATH)
    print("Non-shared labels excluded from targets and predictions.")
    print("Empty shared-label predictions do not establish neutrality.")
    print("Development comparison only; paraphrases not detected.")
    print("No training or threshold changes. No journal text saved.")


if __name__ == "__main__":
    main()