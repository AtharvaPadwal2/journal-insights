import hashlib
import json

import joblib
import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer

from train_hybrid_classifier import (
    DATA_DIR,
    DIAGNOSTIC_CSV,
    ROOT,
    TRAINING_CSV,
    evaluate,
    normalize_text,
    read_csv,
)
MODEL_PATHS = {
    "journal_targeted": ROOT / "models/emotion_hybrid_journal_targeted.joblib",
    "journal_targeted_f1": ROOT / "models/emotion_hybrid_journal_targeted_f1.joblib",
}
CSV_PATH = ROOT / "data/evaluation/journal_fresh_validation.csv"
REPORT_PATH = ROOT / "reports/journal_targeted_f1_comparison.json"
OLD_VALIDATION_CSV = (
    ROOT / "data/evaluation/journal_human_validation.csv"
)


def main():
    records = read_csv(CSV_PATH)
    models = {
        name: joblib.load(path)
        for name, path in MODEL_PATHS.items()
    }

    reference = models["journal_targeted"]
    labels = reference["labels"]
    metadata = reference["metadata"]

    for name, saved in models.items():
        if saved["labels"] != labels:
            raise ValueError(f"{name}: label order mismatch.")
        for key in (
            "encoder", "normalize_embeddings", "max_seq_length"
        ):
            if saved["metadata"][key] != metadata[key]:
                raise ValueError(f"{name}: encoder settings mismatch.")

    # Check exact normalized-text overlap with known development data.
    blocked = set()
    for path in (
        TRAINING_CSV,
        ROOT / "data/annotation/journal_training_expanded.csv",
        DIAGNOSTIC_CSV,
        OLD_VALIDATION_CSV,
    ):
        blocked.update(
            normalize_text(row["text"])
            for row in read_csv(path)
        )

    for split in ("train", "validation"):
        path = DATA_DIR / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                blocked.add(normalize_text(json.loads(line)["text"]))

    usable = []
    seen_ids = set()
    seen_texts = set()

    for row in records:
        if (
            row["id"] in seen_ids
            or normalize_text(row["text"]) in seen_texts
        ):
            raise ValueError("Duplicate validation ID or text.")

        seen_ids.add(row["id"])
        seen_texts.add(normalize_text(row["text"]))

        if normalize_text(row["text"]) in blocked:
            raise ValueError(f"{row['id']}: development-data overlap.")

        if row["review_status"] != "reviewed":
            continue
        if row["annotation_state"] == "unclear":
            continue

        emotions = (
            row["emotion_labels"].split("|")
            if row["emotion_labels"] else []
        )
        if set(emotions) - set(labels):
            raise ValueError(f"{row['id']}: unsupported labels.")
        if row["annotation_state"] not in {"emotional", "neutral"}:
            raise ValueError(f"{row['id']}: invalid state.")
        if (
            row["annotation_state"] == "neutral" and emotions
            or row["annotation_state"] == "emotional" and not emotions
        ):
            raise ValueError(f"{row['id']}: labels/state mismatch.")

        usable.append(row)

    if not usable:
        raise ValueError("No reviewed, usable validation entries.")

    texts = [row["text"] for row in usable]
    targets = np.array([
        [
            int(label in row["emotion_labels"].split("|"))
            for label in labels
        ]
        for row in usable
    ])

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
        )["input_ids"]
        if len(tokens) > encoder.max_seq_length:
            raise ValueError(f"{row['id']}: encoder token limit exceeded.")

    embeddings = encoder.encode(
        texts,
        batch_size=4,
        normalize_embeddings=metadata["normalize_embeddings"],
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    report = {
        "validation_sha256": hashlib.sha256(
            CSV_PATH.read_bytes()
        ).hexdigest(),
        "evaluated_rows": len(usable),
        "skipped_rows": len(records) - len(usable),
        "labels": labels,
        "source_counts": {
            source: sum(row["source"] == source for row in usable)
            for source in sorted({row["source"] for row in usable})
        },
        "label_support": dict(zip(
            labels, targets.sum(axis=0).tolist()
        )),
        "models": {},
    }

    all_predictions = {}

    for name, saved in models.items():
        features = hstack([
            saved["vectorizer"].transform(texts),
            csr_matrix(embeddings * saved["embedding_weight"]),
        ], format="csr")

        scores = np.asarray(
            saved["classifier"].predict_proba(features)
        )
        thresholds = np.asarray(saved["thresholds"])

        if (
            scores.shape != targets.shape
            or thresholds.shape != (len(labels),)
            or not np.isfinite(scores).all()
        ):
            raise ValueError(f"{name}: invalid scores or thresholds.")

        predictions = (scores >= thresholds).astype(int)
        all_predictions[name] = predictions
        metrics = evaluate(targets, predictions, labels)

        # Exclude labels absent from this tiny sample for this extra metric.
        supported = targets.sum(axis=0) > 0
        metrics["supported_labels_macro_f1"] = evaluate(
            targets[:, supported],
            predictions[:, supported],
            [
                label for label, present in zip(labels, supported)
                if present
            ],
        )["macro_f1"]

        report["models"][name] = {
            "metrics": metrics,
            "thresholds": thresholds.tolist(),
            "model_sha256": hashlib.sha256(
                MODEL_PATHS[name].read_bytes()
            ).hexdigest(),
        }

        print(f"\n{name}")
        for metric in (
            "macro_f1",
            "supported_labels_macro_f1",
            "micro_f1",
            "exact_match_accuracy",
            "hamming_loss",
            "neutral_detection_f1",
        ):
            print(f"{metric}: {metrics[metric]:.4f}")
        # Categories follow the original evaluation-generation ID ranges.
    # These are intended categories, not independently verified tags.
    category_ranges = {
        "single_emotion": (1, 88),
        "mixed_emotions": (89, 128),
        "emotional_transitions": (129, 160),
        "negation_and_contrast": (161, 180),
        "neutral_routine": (181, 200),
    }

    report["category_results"] = {}

    print("\nCATEGORY BREAKDOWN")

    for category, (first_id, last_id) in category_ranges.items():
        indices = []

        for index, row in enumerate(usable):
            entry_id = row["id"]

            if (
                entry_id.startswith("jfv")
                and entry_id[3:].isdigit()
                and first_id <= int(entry_id[3:]) <= last_id
            ):
                indices.append(index)

        if not indices:
            continue

        category_targets = targets[indices]
        category_result = {
            "evaluated_rows": len(indices),
            "models": {},
        }

        print(f"\n{category}: {len(indices)} entries")

        for name in ("journal_targeted", "journal_targeted_f1"):
            category_predictions = all_predictions[name][indices]

            true_positives = int(np.sum(
                (category_targets == 1)
                & (category_predictions == 1)
            ))
            missed_labels = int(np.sum(
                (category_targets == 1)
                & (category_predictions == 0)
            ))
            extra_labels = int(np.sum(
                (category_targets == 0)
                & (category_predictions == 1)
            ))
            exact_matches = int(np.sum(
                np.all(
                    category_targets == category_predictions,
                    axis=1,
                )
            ))

            denominator = (
                2 * true_positives + missed_labels + extra_labels
            )
            micro_f1 = (
                2 * true_positives / denominator
                if denominator else None
            )

            category_result["models"][name] = {
                "micro_f1": micro_f1,
                "exact_matches": exact_matches,
                "missed_labels": missed_labels,
                "extra_labels": extra_labels,
            }

            f1_text = (
                f"{micro_f1:.4f}"
                if micro_f1 is not None
                else "N/A"
            )

            print(
                f"{name}: micro_F1={f1_text}"
                f" | exact={exact_matches}/{len(indices)}"
                f" | missed={missed_labels}"
                f" | extra={extra_labels}"
            )

        report["category_results"][category] = category_result
    print("\nENTRY COMPARISON")
    for index, row in enumerate(usable):
        print(f"\n{row['id']}")
        print(
            "Expected:",
            row["emotion_labels"] or row["annotation_state"],
        )
        for name, predictions in all_predictions.items():
            predicted = [
                label
                for label, present in zip(labels, predictions[index])
                if present
            ]
            print(
                f"{name}:",
                " | ".join(predicted) or "No labels above threshold",
            )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"\nReport saved: {REPORT_PATH}")
    print("No training or threshold changes performed.")
    print("Test split not loaded; paraphrases not automatically checked.")
    print("Empty predictions do not establish neutrality.")
    print(
        f"{len(usable)} reviewed entries evaluated. "
        "Results apply to this sample; journal-wide accuracy "
        "is not established."
    )


if __name__ == "__main__":
    main()