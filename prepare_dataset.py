from pathlib import Path
from collections import defaultdict
import json

from datasets import DatasetDict, load_from_disk

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "emotion"
CLEAN_PATH = ROOT / "data" / "processed" / "emotion"
REPORT_PATH = ROOT / "reports" / "dataset_cleaning.json"


def normalize(text):
    # Only for duplicate detection; original text is preserved.
    return " ".join(text.lower().split())


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError("Raw dataset missing. Run download_dataset.py first.")

    if CLEAN_PATH.exists():
        raise FileExistsError(
            f"Clean dataset already exists at {CLEAN_PATH}. "
            "Rename it before running again."
        )

    dataset = load_from_disk(str(RAW_PATH))
    split_names = ("train", "validation", "test")

    # Find identical normalized texts with different labels.
    labels_by_text = defaultdict(set)

    for name in split_names:
        for row in dataset[name]:
            key = normalize(row["text"])
            if key:
                labels_by_text[key].add(row["label"])

    conflicts = {
        text
        for text, labels in labels_by_text.items()
        if len(labels) > 1
    }

    cleaned = DatasetDict()
    seen = set()

    report = {
        "normalization": "lowercase and collapse whitespace",
        "priority": ["test", "validation", "train"],
        "conflicting_unique_texts": len(conflicts),
        "splits": {},
    }

    # Preserve held-out examples first.
    for name in ("test", "validation", "train"):
        keep_indices = []
        local_seen = set()
        removed = {
            "empty": 0,
            "conflicting_labels": 0,
            "within_split_duplicates": 0,
            "cross_split_duplicates": 0,
        }

        for index, row in enumerate(dataset[name]):
            key = normalize(row["text"])

            if not key:
                removed["empty"] += 1
            elif key in conflicts:
                removed["conflicting_labels"] += 1
            elif key in local_seen:
                removed["within_split_duplicates"] += 1
            elif key in seen:
                removed["cross_split_duplicates"] += 1
            else:
                keep_indices.append(index)

            local_seen.add(key)

        cleaned[name] = dataset[name].select(keep_indices)

        seen.update(
            normalize(text) for text in cleaned[name]["text"]
        )

        report["splits"][name] = {
            "original_rows": len(dataset[name]),
            "clean_rows": len(cleaned[name]),
            "removed": removed,
        }

    # Verify the result before saving.
    text_sets = {
        name: {normalize(text) for text in cleaned[name]["text"]}
        for name in split_names
    }

    for name in split_names:
        if not len(cleaned[name]):
            raise ValueError(f"{name} is empty after cleaning.")

        assert len(text_sets[name]) == len(cleaned[name])
        assert "" not in text_sets[name]

    for first, second in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        assert not text_sets[first] & text_sets[second], (
            f"Overlap remains between {first} and {second}"
        )

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.save_to_disk(str(CLEAN_PATH))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    for name in split_names:
        stats = report["splits"][name]
        print(
            f"{name}: {stats['original_rows']} "
            f"-> {stats['clean_rows']} rows"
        )
        print("Removed:", stats["removed"])

    print(f"\nConflicting unique texts: {len(conflicts)}")
    print("Verified: no empty texts, duplicates or split overlap.")
    print(f"Clean dataset: {CLEAN_PATH}")
    print(f"Cleaning report: {REPORT_PATH}")


if __name__ == "__main__":
    main()