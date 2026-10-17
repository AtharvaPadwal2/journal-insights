import re
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent

TEXT = """It is a wonderful evening of navratri. The happiness is in the air, but along with it is the memories of shravya. The girl i am truly in love with. I cannot forget her ever. Whatever happened between us is killing me from inside. Daily basis i think abt her, i expect her to comeback. When i know somewhere all doors are closing one by one but i still hope she might comeback someday. I love you shravya"""

MODEL_PATH = ROOT / "models/emotion_goemotions_hybrid.joblib"


def main():
    saved = joblib.load(MODEL_PATH)
    metadata = saved["metadata"]
    labels = saved["labels"]
    thresholds = np.asarray(saved["thresholds"], dtype=float)

    torch.set_num_threads(2)

    encoder = SentenceTransformer(
        metadata["encoder"],
        device="cpu",
        local_files_only=True,
    )
    encoder.max_seq_length = metadata["max_seq_length"]

    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", TEXT.strip())
        if part.strip() and any(char.isalnum() for char in part)
    ]
    windows = [
        f"{sentences[index]} {sentences[index + 1]}"
        for index in range(len(sentences) - 1)
    ]

    texts = [TEXT.strip()] + sentences + windows

    titles = (
        ["WHOLE ENTRY"]
        + [
            f"SENTENCE {index + 1}"
            for index in range(len(sentences))
        ]
        + [
            f"CONTEXT WINDOW {index + 1}: "
            f"SENTENCES {index + 1}–{index + 2}"
            for index in range(len(windows))
        ]
    )

    for index, text in enumerate(texts):
        tokens = encoder.tokenizer(
            text,
            truncation=False,
            add_special_tokens=True,
            verbose=False,
        )["input_ids"]

        if len(tokens) > encoder.max_seq_length:
            section = "Whole entry" if index == 0 else f"Sentence {index}"
            raise ValueError(
                f"{section}: {len(tokens)} tokens exceeds "
                f"{encoder.max_seq_length}. No text truncated."
            )

    embeddings = encoder.encode(
        texts,
        batch_size=4,
        normalize_embeddings=metadata["normalize_embeddings"],
        convert_to_numpy=True,
    )

    features = hstack([
        saved["vectorizer"].transform(texts),
        csr_matrix(embeddings * saved["embedding_weight"]),
    ], format="csr")

    scores = np.asarray(saved["classifier"].predict_proba(features))

    if (
        scores.shape != (len(texts), len(labels))
        or thresholds.shape != (len(labels),)
        or not np.isfinite(scores).all()
        or not np.isfinite(thresholds).all()
    ):
        raise ValueError("Invalid model scores or thresholds.")

    for index, (text, probabilities) in enumerate(zip(texts, scores)):
        title = titles[index]
        selected = [
            label
            for label, score, threshold in zip(
                labels, probabilities, thresholds
            )
            if score >= threshold
        ]

        print(f"\n{title}")
        print(text)
        print("Selected:", " | ".join(selected) or "No labels above threshold")

        # All selected labels plus the five highest-scoring labels.
        shown = set(np.argsort(probabilities)[::-1][:5].tolist())
        shown.update(np.flatnonzero(probabilities >= thresholds).tolist())

        print("Label                 Score   Threshold")
        for position in sorted(
            shown, key=lambda item: probabilities[item], reverse=True
        ):
            marker = "*" if probabilities[position] >= thresholds[position] else " "
            print(
                f"{marker} {labels[position]:<19} "
                f"{probabilities[position]:.3f}   "
                f"{thresholds[position]:.3f}"
            )

    print("\n* Label crossed its saved threshold.")
    print("Scores are uncalibrated, not emotion intensities.")
    print("Sentence selections are not combined into a whole-entry prediction.")
    print("One example does not establish journal accuracy.")
    print("No training or threshold changes. No journal text saved.")


if __name__ == "__main__":
    main()