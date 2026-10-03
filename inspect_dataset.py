from pathlib import Path
from collections import Counter
from datasets import load_from_disk

data_path = Path(__file__).resolve().parent / "data" / "raw" / "emotion"
dataset = load_from_disk(str(data_path))

labels = dataset["train"].features["label"].names
text_sets = {}

for name in ("train", "validation", "test"):
    split = dataset[name]

    # Normalize only for duplicate checking.
    texts = [" ".join(text.lower().split()) for text in split["text"]]
    text_sets[name] = set(texts)

    print(f"\n{name.upper()}")
    print("Rows:", len(split))
    print("Empty texts:", sum(not text for text in texts))
    print("Duplicate texts within split:", len(texts) - len(set(texts)))

    for label_id, count in sorted(Counter(split["label"]).items()):
        print(f"  {labels[label_id]}: {count}")

print("\nSHARED TEXTS BETWEEN SPLITS")
for first, second in (
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
):
    overlap = text_sets[first] & text_sets[second]
    print(f"{first} / {second}: {len(overlap)}")