import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/processed/goemotions/clean_minilm256"
CACHE_DIR = ROOT / "data/processed/goemotions/embeddings_minilm"

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if CACHE_DIR.exists():
        raise FileExistsError(
            f"{CACHE_DIR} already exists; refusing to overwrite."
        )

    preparation = json.loads(
        (DATA_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    labels = preparation["labels"]

    torch.set_num_threads(2)

    encoder = SentenceTransformer(
        ENCODER,
        device="cpu",
        local_files_only=True,
    )
    token_limit = encoder.max_seq_length

    datasets = {}

    # Check lengths before encoding to prevent silent truncation.
    for split in ("train", "validation"):
        path = DATA_DIR / f"{split}.jsonl"
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if not records:
            raise ValueError(f"{split}: empty dataset.")

        texts = [row["text"] for row in records]

        for start in range(0, len(texts), 256):
            tokenized = encoder.tokenizer(
                texts[start:start + 256],
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]

            for offset, tokens in enumerate(tokenized):
                if len(tokens) > token_limit:
                    entry_id = records[start + offset]["id"]
                    raise ValueError(
                        f"{split}/{entry_id}: {len(tokens)} tokens "
                        f"exceeds limit {token_limit}. "
                        "No cache written."
                    )

        datasets[split] = records
        print(f"{split}: token-length checks passed.")

    CACHE_DIR.mkdir(parents=True)

    metadata = {
        "encoder": ENCODER,
        "max_seq_length": token_limit,
        "normalize_embeddings": True,
        "labels": labels,
        "device": "cpu",
        "splits": {},
    }

    for split, records in datasets.items():
        print(f"\nEncoding {split}: {len(records)} entries...")
        start = perf_counter()

        embeddings = encoder.encode(
            [row["text"] for row in records],
            batch_size=4,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

        if (
            embeddings.shape != (
                len(records),
                encoder.get_sentence_embedding_dimension(),
            )
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError(f"{split}: invalid embeddings.")

        targets = np.array([
            [
                int(label in row["emotion_labels"])
                for label in labels
            ]
            for row in records
        ], dtype=np.int8)

        np.savez_compressed(
            CACHE_DIR / f"{split}.npz",
            embeddings=embeddings,
            targets=targets,
            ids=np.array([row["id"] for row in records]),
        )

        metadata["splits"][split] = {
            "rows": len(records),
            "source_sha256": hashlib.sha256(
                (DATA_DIR / f"{split}.jsonl").read_bytes()
            ).hexdigest(),
            "encoding_seconds": perf_counter() - start,
        }

        print("Shape:", embeddings.shape)
        print(
            "Encoding time:",
            f"{metadata['splits'][split]['encoding_seconds']:.2f}s",
        )

    (CACHE_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nCache saved:", CACHE_DIR)
    print("Encoder unchanged. No classifier trained.")
    print("Test split not encoded.")


if __name__ == "__main__":
    main()