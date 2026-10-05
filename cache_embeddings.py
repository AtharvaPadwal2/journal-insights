import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/processed/semeval2018/clean"
CACHE_DIR = ROOT / "data/processed/semeval2018/embeddings_minilm"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LABELS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "love", "optimism", "pessimism",
    "sadness", "surprise", "trust",
]


def main():
    if CACHE_DIR.exists():
        raise FileExistsError(
            "Embedding cache already exists. Rename it before rerunning."
        )

    torch.set_num_threads(2)

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        local_files_only=True,
    )

    splits = {}
    metadata = {
        "encoder": MODEL_NAME,
        "max_seq_length": model.max_seq_length,
        "normalized": True,
        "labels": LABELS,
        "splits": {},
    }

    # Validate inputs before creating the cache.
    for split in ("train", "validation"):
        path = DATA_DIR / f"{split}.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

        if not rows:
            raise ValueError(f"{split}: empty dataset.")

        for row in rows:
            if not row["text"].strip():
                raise ValueError(f"{split}: empty text.")

            if set(row["emotion_labels"]) - set(LABELS):
                raise ValueError(f"{split}: unknown label.")

            token_ids = model.tokenizer(
                row["text"],
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]

            if len(token_ids) > model.max_seq_length:
                raise ValueError(
                    f"{split}, ID {row['id']}: "
                    f"{len(token_ids)} tokens exceeds "
                    f"{model.max_seq_length}. "
                    "Chunking needs a separate policy."
                )

        splits[split] = rows
        metadata["splits"][split] = {
            "rows": len(rows),
            "source_sha256": hashlib.sha256(
                path.read_bytes()
            ).hexdigest(),
        }

    CACHE_DIR.mkdir(parents=True)

    for split, rows in splits.items():
        print(f"\nEncoding {split}: {len(rows)} entries...")
        start = perf_counter()

        embeddings = model.encode(
            [row["text"] for row in rows],
            batch_size=4,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

        if not np.isfinite(embeddings).all():
            raise ValueError(f"{split}: invalid embedding values.")

        targets = np.array([
            [
                int(label in row["emotion_labels"])
                for label in LABELS
            ]
            for row in rows
        ], dtype=np.int32)

        np.savez_compressed(
            CACHE_DIR / f"{split}.npz",
            embeddings=embeddings,
            targets=targets,
            ids=np.array([row["id"] for row in rows]),
        )

        print("Shape:", embeddings.shape)
        print(f"Encoding time: {perf_counter() - start:.2f}s")

    (CACHE_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nCache saved.")
    print("Encoder weights unchanged; no classifier trained.")
    print("Test split was not loaded.")


if __name__ == "__main__":
    main()