from time import perf_counter

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(2)

start = perf_counter()

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)

print(f"Load/download time: {perf_counter() - start:.2f}s")
print("Device:", model.device)
print("Token limit:", model.max_seq_length)

texts = [
    "I feel nervous about tomorrow, even though I prepared well.",
    "I trust my friend to support me.",
    "I attended classes and came home at five.",
]

start = perf_counter()

embeddings = model.encode(
    texts,
    batch_size=4,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False,
)

print("Embedding shape:", embeddings.shape)
print("All values finite:", bool(np.isfinite(embeddings).all()))
print(f"Encoding time: {perf_counter() - start:.2f}s")
print("No classifier was trained.")