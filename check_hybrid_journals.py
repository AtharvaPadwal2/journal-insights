import csv
import re
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models/emotion_hybrid_journal_pilot.joblib"
CSV_PATH = ROOT / "data/annotation/journal_labels.csv"


def split_sentences(text):
    # Simple diagnostic splitter; abbreviations may split incorrectly.
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())

    return [
        part.strip()
        for part in parts
        if part.strip() and any(char.isalnum() for char in part)
    ]


def print_prediction(probabilities, labels, thresholds):
    predicted = [
        label
        for label, score, threshold in zip(
            labels, probabilities, thresholds
        )
        if score >= threshold
    ]

    top_indices = np.argsort(probabilities)[::-1][:3]
    top_scores = ", ".join(
        f"{labels[index]}={probabilities[index]:.3f}"
        for index in top_indices
    )

    print(
        "Predicted:",
        " | ".join(predicted)
        if predicted else "No labels above threshold",
    )
    print("Top model scores:", top_scores)


def main():
    saved = joblib.load(MODEL_PATH)
    metadata = saved["metadata"]
    labels = saved["labels"]
    thresholds = np.asarray(saved["thresholds"], dtype=float)

    if thresholds.shape != (len(labels),):
        raise ValueError("Threshold count does not match labels.")

    torch.set_num_threads(2)

    encoder = SentenceTransformer(
        metadata["encoder"],
        device="cpu",
        local_files_only=True,
    )
    encoder.max_seq_length = metadata["max_seq_length"]

    with CSV_PATH.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Annotation CSV has no entries.")

    # Encode full entries and sentences together, preserving their order.
    texts = []
    entry_ranges = []

    for row in rows:
        full_text = row["text"].strip()
        sentences = split_sentences(full_text)

        if not sentences:
            raise ValueError(f"{row['id']}: no usable text.")

        start = len(texts)
        texts.append(full_text)

        # Avoid repeating an entry that already contains one sentence.
        separate_sentences = (
            sentences if len(sentences) > 1 else []
        )
        texts.extend(separate_sentences)

        entry_ranges.append((start, separate_sentences))

        for section, text in [
            ("whole entry", full_text),
            *[
                (f"sentence {index}", sentence)
                for index, sentence in enumerate(
                    separate_sentences, start=1
                )
            ],
        ]:
            tokens = encoder.tokenizer(
                text,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]

            if len(tokens) > encoder.max_seq_length:
                raise ValueError(
                    f"{row['id']} {section}: exceeds "
                    f"{encoder.max_seq_length}-token limit."
                )

    embeddings = encoder.encode(
        texts,
        batch_size=4,
        normalize_embeddings=metadata["normalize_embeddings"],
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    features = hstack(
        [
            saved["vectorizer"].transform(texts),
            csr_matrix(
                embeddings * saved["embedding_weight"]
            ),
        ],
        format="csr",
    )

    scores = np.asarray(
        saved["classifier"].predict_proba(features)
    )

    if scores.shape != (len(texts), len(labels)):
        raise ValueError("Unexpected classifier score shape.")

    if not np.isfinite(scores).all():
        raise ValueError("Classifier returned non-finite scores.")

    print("\nWHOLE ENTRY VS SENTENCE DIAGNOSTICS")

    for row, (start, sentences) in zip(rows, entry_ranges):
        print(f"\n{'=' * 60}")
        print(f"{row['id']} | Review: {row['review_status']}")
        print(
            "Entry annotation:",
            row["emotion_labels"] or row["annotation_state"],
        )

        print("\nWHOLE ENTRY")
        print("Text:", texts[start])
        print_prediction(scores[start], labels, thresholds)

        for index, sentence in enumerate(sentences, start=1):
            print(f"\nSENTENCE {index}")
            print("Text:", sentence)
            print_prediction(
                scores[start + index],
                labels,
                thresholds,
            )

        if not sentences:
            print("\nSingle sentence: no separate comparison.")

    print("\nScores are uncalibrated, not emotion intensities.")
    print("Entry annotations are not sentence-level ground truth.")
    print("Pending annotations are not verified ground truth.")
    print("Synthetic examples do not establish real-world accuracy.")
    print("No labels above threshold does not establish neutrality.")
    print("Sentence predictions are not combined into a final result.")
    print("No training performed; no files written.")


if __name__ == "__main__":
    main()