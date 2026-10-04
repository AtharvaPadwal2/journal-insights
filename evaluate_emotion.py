from pathlib import Path
import json

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/evaluation/emotion_dev.csv"
REPORT_PATH = ROOT / "reports/emotion_dev_comparison.json"

MODEL_PATHS = {
    "LogReg": ROOT / "models/emotion_logreg.joblib",
    "LinearSVC": ROOT / "models/emotion_svm.joblib",
}

VALID_LABELS = {
    "sadness", "joy", "love", "anger", "fear",
    "surprise", "neutral", "mixed", "unclear",
}


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Evaluation file missing: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, keep_default_na=False)

    required = {
        "id", "text", "emotion", "label_source", "review_status",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if data.empty:
        raise ValueError("Evaluation file is empty.")

    for column in required:
        data[column] = data[column].astype(str).str.strip()

    data["emotion"] = data["emotion"].str.lower()
    data["review_status"] = data["review_status"].str.lower()

    if data["id"].eq("").any() or data["id"].duplicated().any():
        raise ValueError("IDs must be nonempty and unique.")

    if data["text"].eq("").any():
        raise ValueError("Evaluation texts cannot be empty.")

    invalid = set(data["emotion"]) - VALID_LABELS
    if invalid:
        raise ValueError(f"Unexpected labels: {sorted(invalid)}")

    models = {}
    for name, path in MODEL_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Model missing: {path}")
        models[name] = joblib.load(path)

    results = data.to_dict(orient="records")
    summaries = {}

    for name, saved in models.items():
        pipeline = saved["pipeline"]
        labels = saved["labels"]
        supported = set(labels)

        predictions = pipeline.predict(data["text"].tolist())
        proposed_matches = 0
        supported_rows = 0
        reviewed_rows = 0
        reviewed_matches = 0

        for row, predicted_id in zip(results, predictions):
            prediction = labels[int(predicted_id)]
            eligible = row["emotion"] in supported
            match = prediction == row["emotion"] if eligible else None

            row.setdefault("predictions", {})[name] = {
                "emotion": prediction,
                "expected_label_supported": eligible,
                "matches_proposed_label": match,
            }

            if eligible:
                supported_rows += 1
                proposed_matches += int(match)

                # Mark reviewed only after actual human review.
                if row["review_status"] == "reviewed":
                    reviewed_rows += 1
                    reviewed_matches += int(match)

        summaries[name] = {
            "model_metadata": saved.get("metadata", {}),
            "total_rows": len(data),
            "supported_label_rows": supported_rows,
            "unsupported_label_rows": len(data) - supported_rows,
            "matches_on_supported_proposed_labels": proposed_matches,
            "human_reviewed_supported_rows": reviewed_rows,
            "human_reviewed_matches": reviewed_matches,
            "human_reviewed_supported_accuracy": (
                reviewed_matches / reviewed_rows
                if reviewed_rows else None
            ),
        }

    for row in results:
        print(f"\n{row['id']}: {row['text']}")
        print(
            f"Proposed label: {row['emotion']} "
            f"| Review: {row['review_status']}"
        )

        for name, prediction in row["predictions"].items():
            print(f"{name}: {prediction['emotion']}")
            if not prediction["expected_label_supported"]:
                print("  Expected label is outside this model's classes.")

    print("\nSUPPORTED-LABEL DIAGNOSTIC MATCHES")
    for name, summary in summaries.items():
        print(
            f"{name}: "
            f"{summary['matches_on_supported_proposed_labels']}/"
            f"{summary['supported_label_rows']}"
        )

    notes = [
        "Development examples are not used for training by this script.",
        "Pending AI-proposed labels are not human-reviewed ground truth.",
        "Unsupported labels are reported separately, not silently discarded.",
        "These results do not establish journal-wide accuracy.",
    ]

    report = {
        "evaluation_file": str(DATA_PATH.relative_to(ROOT)),
        "summaries": summaries,
        "examples": results,
        "notes": notes,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for note in notes:
        print(note)

    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()