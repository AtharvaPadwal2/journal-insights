from pathlib import Path
import json

import joblib
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent


def main():
    entries = pd.read_csv(
        ROOT / "data/evaluation/journal_parth.csv",
        keep_default_na=False,
        dtype=str,
    )
    labels = pd.read_csv(
        ROOT / "data/evaluation/journal_parth_labels.csv",
        keep_default_na=False,
        dtype=str,
    )

    if entries["id"].duplicated().any() or labels["id"].duplicated().any():
        raise ValueError("Duplicate IDs found.")

    if set(entries["id"]) != set(labels["id"]):
        raise ValueError("Entry IDs and label IDs must match.")

    data = entries.merge(labels, on="id", validate="one_to_one")
    texts = data["text"].tolist()

    if any(not text.strip() for text in texts):
        raise ValueError("An entry has empty text.")

    allowed = {
        "sadness", "joy", "love", "anger",
        "fear", "surprise", "neutral", "unclear",
    }

    for value in data["proposed_labels"]:
        proposed = {label.strip() for label in value.split("|")}
        if not proposed or not proposed <= allowed:
            raise ValueError(f"Invalid labels: {value}")

    predictions = {}
    model_classes = {}

    for name, filename in [
        ("LogReg", "emotion_logreg.joblib"),
        ("LinearSVC", "emotion_svm.joblib"),
    ]:
        saved = joblib.load(ROOT / "models" / filename)
        predicted_ids = saved["pipeline"].predict(texts)

        predictions[name] = [
            saved["labels"][int(class_id)]
            for class_id in predicted_ids
        ]
        model_classes[name] = [
            saved["labels"][int(class_id)]
            for class_id in saved["pipeline"].classes_
        ]

    for name, folder in [
        ("DistilBERT", "emotion_distilbert"),
        ("Hartmann", "emotion_hartmann"),
    ]:
        print(f"Running {name}...")
        path = ROOT / "models" / folder

        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True
        )
        model.eval()

        id_to_label = {
            int(key): value.lower()
            for key, value in model.config.id2label.items()
        }
        model_classes[name] = list(id_to_label.values())

        # RoBERTa's usable input limit differs from its position table size.
        token_limit = min(
            int(tokenizer.model_max_length),
            int(model.config.max_position_embeddings),
        )

        predicted_labels = []

        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
            )

            if inputs["input_ids"].shape[1] > token_limit:
                raise ValueError(
                    f"An entry exceeds {name}'s {token_limit}-token limit."
                )

            with torch.inference_mode():
                class_id = model(**inputs).logits.argmax(dim=-1).item()

            predicted_labels.append(id_to_label[class_id])

        predictions[name] = predicted_labels
        del model, tokenizer

    results = []

    for index, row in enumerate(data.itertuples(index=False)):
        proposed = [
            label.strip()
            for label in row.proposed_labels.split("|")
        ]

        print(f"\n{row.id} [{row.category}]: {row.text}")
        print(f"AI-proposed labels: {' | '.join(proposed)}")

        row_predictions = {}

        for name, values in predictions.items():
            predicted = values[index]
            unsupported = sorted(
                set(proposed) - set(model_classes[name])
            )

            print(f"  {name}: {predicted}")
            if unsupported:
                print(
                    "    Unsupported proposed labels: "
                    + ", ".join(unsupported)
                )

            row_predictions[name] = {
                "prediction": predicted,
                "matches_any_proposed_label": predicted in proposed,
                "unsupported_proposed_labels": unsupported,
            }

        results.append({
            "id": row.id,
            "text": row.text,
            "category": row.category,
            "proposed_labels": proposed,
            "label_source": "ai_proposed",
            "review_status": "pending",
            "predictions": row_predictions,
        })

    report = ROOT / "reports/parth_model_comparison.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "model_classes": model_classes,
                "results": results,
                "notes": [
                    "AI-proposed labels; human review pending.",
                    "Models predict one label per entry.",
                    "Matching one proposed label does not capture every emotion.",
                    "These examples are not used for training.",
                    "This comparison does not establish journal-wide accuracy.",
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nReport saved: {report}")
    print("Diagnostic comparison only; human review pending.")


if __name__ == "__main__":
    main()