import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RAW_DIR = (
    ROOT / "data/raw/semeval2018/extracted"
    / "SemEval2018-Task1-all-data/English/E-c"
)
OUTPUT_DIR = ROOT / "data/processed/semeval2018"

LABELS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "love", "optimism", "pessimism",
    "sadness", "surprise", "trust",
]

FILES = {
    "train": "2018-E-c-En-train.txt",
    "validation": "2018-E-c-En-dev.txt",
    "test": "2018-E-c-En-test-gold.txt",
}


def read_split(path):
    records = []
    seen_ids = set()

    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(
            file,
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
        )

        expected = ["ID", "Tweet", *LABELS]
        if reader.fieldnames != expected:
            raise ValueError(
                f"{path.name}: unexpected header {reader.fieldnames}"
            )

        for line, row in enumerate(reader, start=2):
            if None in row or any(
                row.get(column) is None for column in expected
            ):
                raise ValueError(
                    f"{path.name}, line {line}: malformed row."
                )

            entry_id = row["ID"].strip()
            text = row["Tweet"]

            if not entry_id or entry_id in seen_ids:
                raise ValueError(
                    f"{path.name}, line {line}: empty or duplicate ID."
                )

            if not text.strip():
                raise ValueError(
                    f"{path.name}, line {line}: empty text."
                )

            active_labels = []

            for label in LABELS:
                value = row[label].strip()
                if value not in {"0", "1"}:
                    raise ValueError(
                        f"{path.name}, line {line}: "
                        f"invalid {label} value {value!r}."
                    )
                if value == "1":
                    active_labels.append(label)

            seen_ids.add(entry_id)

            records.append({
                "id": entry_id,
                "text": text,
                "emotion_labels": active_labels,
                "annotation_state": (
                    "emotional" if active_labels else "neutral"
                ),
                "source": "semeval2018_english_ec",
                "group_id": entry_id,
            })

    if not records:
        raise ValueError(f"{path.name}: empty dataset.")

    return records


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. "
            "Rename it before importing again."
        )

    # Validate every split before writing output.
    splits = {
        split: read_split(RAW_DIR / filename)
        for split, filename in FILES.items()
    }

    OUTPUT_DIR.mkdir(parents=True)

    for split, records in splits.items():
        output_path = OUTPUT_DIR / f"{split}.jsonl"

        with output_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        counts = Counter(
            label
            for record in records
            for label in record["emotion_labels"]
        )
        neutral_count = sum(
            record["annotation_state"] == "neutral"
            for record in records
        )

        print(f"\n{split}: {len(records)} rows")
        print(f"Neutral: {neutral_count}")
        print("Emotion counts:", {
            label: counts[label] for label in LABELS
        })

    print("\nImport complete. Original splits preserved.")
    print("All 11 labels preserved; text was not normalized.")
    print("Duplicate-text and split-overlap checks are next.")
    print("No model was trained.")
    print("Do not commit raw or processed dataset files.")


if __name__ == "__main__":
    main()