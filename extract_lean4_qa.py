import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PARQUET_PATH = ROOT / "NuminaMath-LEAN" / "data" / "train-00000-of-00001.parquet"
OUTPUT_PATH = ROOT / "lean4_qa_complete.jsonl"


BLOCK_COMMENT_RE = re.compile(r"/-(?:.|\n)*?-/", re.DOTALL)
LINE_COMMENT_RE = re.compile(r"(?m)^\s*--.*$")


def strip_comments(text: str) -> str:
    text = BLOCK_COMMENT_RE.sub("", text)
    text = LINE_COMMENT_RE.sub("", text)
    lines = [line.rstrip() for line in text.splitlines()]

    cleaned_lines = []
    previous_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        cleaned_lines.append(line)
        previous_blank = is_blank

    return "\n".join(cleaned_lines).strip()


def extract_proof_body(formal_ground_truth: str) -> str:
    marker = ":= by"
    marker_index = formal_ground_truth.find(marker)
    if marker_index == -1:
        return ""

    body = formal_ground_truth[marker_index + len(marker) :].strip()
    return body


def main() -> None:
    df = pd.read_parquet(
        PARQUET_PATH,
        columns=["formal_statement", "formal_ground_truth", "ground_truth_type"],
    )

    df = df[
        (df["ground_truth_type"] == "complete")
        & df["formal_statement"].notna()
        & df["formal_ground_truth"].notna()
    ].copy()

    seen_questions = set()
    total_rows = 0
    written_rows = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for row in df.itertuples(index=False):
            total_rows += 1

            question = strip_comments(str(row.formal_statement))
            proof_source = strip_comments(str(row.formal_ground_truth))
            answer = extract_proof_body(proof_source)

            if not question or not answer:
                continue
            if question in seen_questions:
                continue

            seen_questions.add(question)
            output_file.write(
                json.dumps(
                    {"question": question, "answer": answer},
                    ensure_ascii=False,
                )
                + "\n"
            )
            written_rows += 1

    print(f"input_complete_rows={total_rows}")
    print(f"written_rows={written_rows}")
    print(f"output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
