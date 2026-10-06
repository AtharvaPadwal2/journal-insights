import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data/processed/semeval2018"
OUTPUT_DIR = INPUT_DIR / "clean"


def normalize(text):
    return " ".join(text.casefold().split())


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            "Clean directory already exists. Rename it before rerunning."
        )

    splits = {}

    for split in ("train", "validation", "test"):
        with (INPUT_DIR / f"{split}.jsonl").open(
            encoding="utf-8"
        ) as file:
            splits[split] = [
                json.loads(line) for line in file
            ]

    groups = defaultdict(list)

    for row in splits["train"]:
        groups[normalize(row["text"])].append(row)

    clean_train = []
    conflicting_rows = 0
    duplicate_rows = 0

    for rows in groups.values():
        label_sets = {
            frozenset(row["emotion_labels"]) for row in rows
        }

        if len(label_sets) > 1:
            conflicting_rows += len(rows)
            continue

        clean_train.append(rows[0])
        duplicate_rows += len(rows) - 1

    splits["train"] = clean_train

    # Confirm separation before writing.
    for key in ("id", "text"):
        values = {
            split: {
                normalize(row[key]) if key == "text" else row[key]
                for row in rows
            }
            for split, rows in splits.items()
        }

        for first, second in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            if values[first] & values[second]:
                raise ValueError(
                    f"{key} overlap between {first} and {second}."
                )

    OUTPUT_DIR.mkdir()

    for split, rows in splits.items():
        with (OUTPUT_DIR / f"{split}.jsonl").open(
            "w", encoding="utf-8"
        ) as file:
            for row in rows:
                file.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )

        print(f"{split}: {len(rows)} rows")

    print(f"Conflicting training rows removed: {conflicting_rows}")
    print(f"Redundant training rows removed: {duplicate_rows}")
    print("Validation and test records preserved.")
    print("Original imported files preserved.")
    print("No model was trained.")


if __name__ == "__main__":
    main()