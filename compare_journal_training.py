import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from train_hybrid_classifier import (
    CACHE_DIR,
    DIAGNOSTIC_CSV,
    ROOT,
    HybridFeatures,
    evaluate,
    load_cached_split,
    load_journal_candidates,
    normalize_text,
    read_csv,
)

REPORT_PATH = ROOT / "reports/journal_training_ablation.json"


def build_model():
    return Pipeline([
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


def select_thresholds(targets, scores):
    thresholds = np.full(targets.shape[1], 0.5)
    candidates = sorted(
        np.linspace(0.05, 0.95, 19),
        key=lambda value: (abs(value - 0.5), value),
    )

    for index in range(targets.shape[1]):
        best = -1.0
        for threshold in candidates:
            score = fbeta_score(
                targets[:, index],
                scores[:, index] >= threshold,
                beta=0.5,
                zero_division=0,
            )
            if score > best:
                best = score
                thresholds[index] = threshold

    return thresholds


def run_experiment(rows, targets, folds, validation, labels):
    model = build_model()

    for fit_indices, _ in folds:
        positives = targets[fit_indices].sum(axis=0)
        if np.any(positives == 0) or np.any(
            positives == len(fit_indices)
        ):
            raise ValueError("A CV fold lacks both classes for a label.")

    oof_scores = cross_val_predict(
        model,
        rows,
        targets,
        cv=folds,
        method="predict_proba",
        n_jobs=1,
    )
    thresholds = select_thresholds(targets, oof_scores)

    model.fit(rows, targets)
    validation_rows, validation_targets = validation
    scores = model.predict_proba(validation_rows)

    return {
        "train_rows": len(rows),
        "thresholds": thresholds.tolist(),
        "fixed_0_5": evaluate(
            validation_targets,
            (scores >= 0.5).astype(int),
            labels,
        ),
        "oof_tuned": evaluate(
            validation_targets,
            (scores >= thresholds).astype(int),
            labels,
        ),
    }


def main():
    metadata = json.loads(
        (CACHE_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    labels = metadata["labels"]

    train_records, base_rows, base_targets = load_cached_split(
        "train", metadata, labels
    )
    validation_records, validation_rows, validation_targets = (
        load_cached_split("validation", metadata, labels)
    )
    journals, _ = load_journal_candidates(labels)
    diagnostics = read_csv(DIAGNOSTIC_CSV)

    blocked = {
        normalize_text(row["text"])
        for row in train_records + validation_records + diagnostics
    }
    diagnostic_ids = {row["id"] for row in diagnostics}
    diagnostic_groups = {row["group_id"] for row in diagnostics}

    for row in journals:
        if (
            normalize_text(row["text"]) in blocked
            or row["id"] in diagnostic_ids
            or row["group_id"] in diagnostic_groups
        ):
            raise ValueError(f"{row['id']}: training overlap detected.")

    torch.set_num_threads(2)
    encoder = SentenceTransformer(
        metadata["encoder"],
        device="cpu",
        local_files_only=True,
    )
    encoder.max_seq_length = metadata["max_seq_length"]

    for row in journals:
        tokens = encoder.tokenizer(
            row["text"],
            truncation=False,
            add_special_tokens=True,
        )["input_ids"]
        if len(tokens) > encoder.max_seq_length:
            raise ValueError(f"{row['id']}: encoder token limit exceeded.")

    embeddings = encoder.encode(
        [row["text"] for row in journals],
        batch_size=4,
        normalize_embeddings=metadata.get("normalize_embeddings", True),
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    if (
        not np.isfinite(embeddings).all()
        or embeddings.shape[1] != base_rows[0]["embedding"].shape[0]
    ):
        raise ValueError("Invalid journal embeddings.")

    combined_rows = base_rows + [
        {"text": row["text"], "embedding": embedding}
        for row, embedding in zip(journals, embeddings)
    ]
    combined_targets = np.vstack([
        base_targets,
        np.array([
            [int(label in row["emotion_labels"]) for label in labels]
            for row in journals
        ]),
    ])

    groups = [
        "semeval:" + normalize_text(row["text"])
        for row in train_records
    ] + [
        "journal:" + row["group_id"]
        for row in journals
    ]

    # Construct once, then keep the same SemEval fold assignments
    # in both experiments.
    combined_folds = list(
        GroupKFold(n_splits=3).split(
            combined_rows, combined_targets, groups
        )
    )
    base_count = len(base_rows)
    base_folds = [
        (
            fit_indices[fit_indices < base_count],
            held_indices[held_indices < base_count],
        )
        for fit_indices, held_indices in combined_folds
    ]

    validation = (validation_rows, validation_targets)
    report = {
        "labels": labels,
        "evaluation_split": "SemEval validation",
        "journal_rows": len(journals),
        "threshold_selection": "training-only grouped OOF F0.5",
        "semeval_fold_assignments_identical": True,
    }

    for name, rows, targets, folds in [
        ("without_journals", base_rows, base_targets, base_folds),
        (
            "with_journals",
            combined_rows,
            combined_targets,
            combined_folds,
        ),
    ]:
        print(f"\nTraining: {name} ({len(rows)} rows)")
        report[name] = run_experiment(
            rows, targets, folds, validation, labels
        )

    metrics = [
        "macro_f1",
        "micro_f1",
        "exact_match_accuracy",
        "hamming_loss",
        "neutral_detection_f1",
    ]

    for mode in ("fixed_0_5", "oof_tuned"):
        print(f"\n{mode}: WITHOUT -> WITH journals")
        for metric in metrics:
            before = report["without_journals"][mode][metric]
            after = report["with_journals"][mode][metric]
            print(f"{metric}: {before:.4f} -> {after:.4f}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved: {REPORT_PATH}")
    print("No model files saved or overwritten.")
    print("Encoder unchanged. Test split not loaded.")
    print("Diagnostic entries not used for training.")
    print("Paraphrase overlap is not automatically detected.")
    print("These metrics measure SemEval validation performance.")


if __name__ == "__main__":
    main()