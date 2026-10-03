from pathlib import Path
from datasets import load_dataset

data_path = Path(__file__).resolve().parent / "data" / "raw" / "emotion"
data_path.parent.mkdir(parents=True, exist_ok=True)

print("Downloading emotion dataset...")

dataset = load_dataset("dair-ai/emotion", "split")
dataset.save_to_disk(str(data_path))

for split_name, split_data in dataset.items():
    print(f"{split_name}: {len(split_data)} examples")

print("Labels:", dataset["train"].features["label"].names)
print(f"Saved to: {data_path}")