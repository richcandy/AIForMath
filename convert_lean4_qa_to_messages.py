import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "lean4_qa_complete.jsonl"
OUTPUT_PATH = ROOT / "lean4_qa_complete_messages.jsonl"


def main() -> None:
    written_rows = 0

    with (
        SOURCE_PATH.open("r", encoding="utf-8") as source_file,
        OUTPUT_PATH.open("w", encoding="utf-8") as output_file,
    ):
        for line in source_file:
            row = json.loads(line)
            message_row = {
                "messages": [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]},
                ]
            }
            output_file.write(json.dumps(message_row, ensure_ascii=False) + "\n")
            written_rows += 1

    print(f"written_rows={written_rows}")
    print(f"output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
