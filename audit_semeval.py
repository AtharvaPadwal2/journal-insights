import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/processed/semeval2018"


def normalize(text):
    return " ".join(text.casefold().split())


def main():
    text_groups = {}
    id_sets = {}

    for split in ("train", "validation", "test"):
        groups = defaultdict(list)
        ids = set()

        with (DATA_DIR / f"{split}.jsonl").open(
            encoding="utf-8"
        ) as file:
            for line in file:
                row = json.loads(line)
                ids.add(row["id"])
                groups[normalize(row["text"])].append(
                    frozenset(row["emotion_labels"])
                )

        text_groups[split] = groups
        id_sets[split] = ids

        duplicate_rows = sum(
            len(values) - 1 for values in groups.values()
        )
        conflicting_texts = sum(
            len(set(values)) > 1 for values in groups.values()
        )

        print(f"\n{split}")
        print(f"Duplicate text rows: {duplicate_rows}")
        print(f"Texts with different label sets: {conflicting_texts}")

    for first, second in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        shared_texts = (
            text_groups[first].keys()
            & text_groups[second].keys()
        )

        conflicting_shared = sum(
            len(set(
                text_groups[first][text]
                + text_groups[second][text]
            )) > 1
            for text in shared_texts
        )

        print(f"\n{first} / {second}")
        print(f"Shared IDs: {len(id_sets[first] & id_sets[second])}")
        print(f"Shared normalized texts: {len(shared_texts)}")
        print(f"Shared texts with different labels: {conflicting_shared}")

    print("\nAudit only. No files changed; no model trained.")
    print("Paraphrases and near-duplicates are not detected.")


if __name__ == "__main__":
    main()