import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "lean4_qa_complete_messages.jsonl"
TRAIN_PATH = ROOT / "lean4_qa_complete_messages_train.jsonl"
VALID_PATH = ROOT / "lean4_qa_complete_messages_valid.jsonl"
TRAIN_RATIO = 0.95
SEED = 42


def main() -> None:
    with SOURCE_PATH.open("r", encoding="utf-8") as source_file:
        rows = [json.loads(line) for line in source_file]

    random.Random(SEED).shuffle(rows)
    split_index = int(len(rows) * TRAIN_RATIO)
    train_rows = rows[:split_index]
    valid_rows = rows[split_index:]

    with TRAIN_PATH.open("w", encoding="utf-8") as train_file:
        for row in train_rows:
            train_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with VALID_PATH.open("w", encoding="utf-8") as valid_file:
        for row in valid_rows:
            valid_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"total_rows={len(rows)}")
    print(f"train_rows={len(train_rows)}")
    print(f"valid_rows={len(valid_rows)}")
    print(f"seed={SEED}")
    print(f"train_ratio={TRAIN_RATIO}")


if __name__ == "__main__":
    main()
