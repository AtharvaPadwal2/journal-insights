import csv
from collections import Counter
from pathlib import Path

from emotion_labels import EMOTION_LABELS, ANNOTATION_STATES

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "annotation" / "journal_labels.csv"

REQUIRED_COLUMNS = {
    "id",
    "text",
    "emotion_labels",
    "annotation_state",
    "review_status",
    "source",
    "group_id",
}


def main():
    errors = []
    seen_ids = set()
    seen_texts = set()
    label_counts = Counter()
    review_counts = Counter()

    with CSV_PATH.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        rows = list(reader)

    if not rows:
        raise ValueError("CSV is empty.")

    for line, row in enumerate(rows, start=2):
        if None in row or any(
            row.get(column) is None for column in REQUIRED_COLUMNS
        ):
            errors.append(f"Line {line}: malformed CSV row.")
            continue

        entry_id = row["id"].strip()
        text = row["text"].strip()
        state = row["annotation_state"].strip()
        review = row["review_status"].strip()
        raw_labels = row["emotion_labels"].strip()
        source = row["source"].strip()
        group_id = row["group_id"].strip()

        if not source:
            errors.append(f"Line {line}: source is required.")

        if not group_id:
            errors.append(f"Line {line}: group_id is required.")

        labels = (
            [label.strip() for label in raw_labels.split("|")]
            if raw_labels else []
        )

        if not entry_id or entry_id in seen_ids:
            errors.append(f"Line {line}: empty or duplicate ID.")
        seen_ids.add(entry_id)

        normalized_text = " ".join(text.lower().split())
        if not normalized_text or normalized_text in seen_texts:
            errors.append(f"Line {line}: empty or duplicate text.")
        seen_texts.add(normalized_text)

        if len(labels) != len(set(labels)):
            errors.append(f"Line {line}: repeated emotion label.")

        invalid = set(labels) - set(EMOTION_LABELS)
        if invalid:
            errors.append(
                f"Line {line}: invalid labels {sorted(invalid)}."
            )

        if state not in {"emotional", *ANNOTATION_STATES}:
            errors.append(f"Line {line}: invalid annotation state.")
        elif state == "emotional" and not labels:
            errors.append(f"Line {line}: emotional row needs labels.")
        elif state in ANNOTATION_STATES and labels:
            errors.append(
                f"Line {line}: {state} row must have empty labels."
            )

        if review not in {"pending", "reviewed"}:
            errors.append(f"Line {line}: invalid review status.")

        label_counts.update(labels)
        review_counts.update([review])

    if errors:
        raise ValueError(
            "Validation failed:\n" + "\n".join(errors)
        )

    print(f"Format validation passed: {len(rows)} rows")
    print("Review status:", dict(review_counts))
    print("Emotion counts:", dict(label_counts))
    print("Format validation does not verify label correctness.")
    print("No model was trained.")


if __name__ == "__main__":
    main()