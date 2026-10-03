from pathlib import Path
from time import perf_counter

import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "emotion_logreg.joblib"


def main():
    if not MODEL_PATH.exists():
        print("Model missing. Run train_model.py first.")
        return

    # Load only our own locally trained model.
    saved_model = joblib.load(MODEL_PATH)
    model = saved_model["pipeline"]
    labels = saved_model["labels"]

    print("Journal Insights — Emotion Prediction")
    print("English text supported. Type 'exit' to quit.")
    print("Entries are not saved.")
    print("Scores are model predictions, not emotional percentages.\n")

    while True:
        try:
            text = input("Enter your text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nClosed.")
            break

        if text.lower() == "exit":
            break

        if not text:
            print("Please enter some text.\n")
            continue

        if len(text) > 5000:
            print("Please keep your entry within 5,000 characters.\n")
            continue

        if len(text.split()) < 3:
            print("Note: Very short text may give unreliable results.")

        start = perf_counter()
        probabilities = model.predict_proba([text])[0]
        elapsed_ms = (perf_counter() - start) * 1000

        # Match scores to the classifier's actual class order.
        results = [
            (labels[int(class_id)], float(score))
            for class_id, score in zip(model.classes_, probabilities)
        ]
        results.sort(key=lambda item: item[1], reverse=True)

        print(f"\nPredicted emotion: {results[0][0]}")
        print("Model scores (uncalibrated):")

        for emotion, score in results:
            print(f"  {emotion:<10} {score:.4f}")

        print(f"Inference time: {elapsed_ms:.2f} ms\n")


if __name__ == "__main__":
    main()