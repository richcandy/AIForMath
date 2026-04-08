import json
import random
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "lean4_qa_complete.jsonl"
OUTPUT_DIR = ROOT / "experiment_data" / "lean4_small_1600_200_200"
TRAIN_SIZE = 1600
VALID_SIZE = 200
TEST_SIZE = 200
TOTAL_SIZE = TRAIN_SIZE + VALID_SIZE + TEST_SIZE
SEED = 42


@dataclass
class Sample:
    sample_id: str
    question: str
    answer: str
    question_chars: int
    answer_chars: int
    total_chars: int
    question_lines: int
    answer_lines: int
    total_lines: int


def build_sample(index: int, row: dict[str, str]) -> Sample:
    question = row["question"].strip()
    answer = row["answer"].strip()
    question_lines = len(question.splitlines())
    answer_lines = len(answer.splitlines())
    return Sample(
        sample_id=f"lean4qa-{index:05d}",
        question=question,
        answer=answer,
        question_chars=len(question),
        answer_chars=len(answer),
        total_chars=len(question) + len(answer),
        question_lines=question_lines,
        answer_lines=answer_lines,
        total_lines=question_lines + answer_lines,
    )


def load_samples() -> list[Sample]:
    samples = []
    with SOURCE_PATH.open("r", encoding="utf-8", newline="") as source_file:
        for index, line in enumerate(source_file):
            row = json.loads(line)
            samples.append(build_sample(index, row))
    return samples


def write_split(path: Path, split_name: str, samples: list[Sample]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        for sample in samples:
            output_file.write(
                json.dumps(
                    {
                        "id": sample.sample_id,
                        "split": split_name,
                        "question": sample.question,
                        "answer": sample.answer,
                        "question_chars": sample.question_chars,
                        "answer_chars": sample.answer_chars,
                        "total_chars": sample.total_chars,
                        "question_lines": sample.question_lines,
                        "answer_lines": sample.answer_lines,
                        "total_lines": sample.total_lines,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def summarize(samples: list[Sample]) -> dict[str, float | int]:
    return {
        "count": len(samples),
        "avg_question_chars": round(
            sum(s.question_chars for s in samples) / len(samples), 2
        ),
        "avg_answer_chars": round(
            sum(s.answer_chars for s in samples) / len(samples), 2
        ),
        "avg_total_chars": round(sum(s.total_chars for s in samples) / len(samples), 2),
        "avg_question_lines": round(
            sum(s.question_lines for s in samples) / len(samples), 2
        ),
        "avg_answer_lines": round(
            sum(s.answer_lines for s in samples) / len(samples), 2
        ),
        "avg_total_lines": round(sum(s.total_lines for s in samples) / len(samples), 2),
        "max_total_chars": max(s.total_chars for s in samples),
        "max_total_lines": max(s.total_lines for s in samples),
    }


def main() -> None:
    all_samples = load_samples()
    ranked = sorted(
        all_samples,
        key=lambda sample: (
            sample.total_chars,
            sample.answer_lines,
            sample.question_lines,
            sample.sample_id,
        ),
    )
    selected = ranked[:TOTAL_SIZE]

    random.Random(SEED).shuffle(selected)
    train_samples = selected[:TRAIN_SIZE]
    valid_samples = selected[TRAIN_SIZE : TRAIN_SIZE + VALID_SIZE]
    test_samples = selected[TRAIN_SIZE + VALID_SIZE :]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_split(OUTPUT_DIR / "train.jsonl", "train", train_samples)
    write_split(OUTPUT_DIR / "valid.jsonl", "valid", valid_samples)
    write_split(OUTPUT_DIR / "test.jsonl", "test", test_samples)

    stats = {
        "source_path": str(SOURCE_PATH),
        "selection_strategy": "shortest_total_chars_then_shuffle",
        "seed": SEED,
        "train": summarize(train_samples),
        "valid": summarize(valid_samples),
        "test": summarize(test_samples),
    }
    (OUTPUT_DIR / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"output_dir={OUTPUT_DIR}")
    print(f"train_rows={len(train_samples)}")
    print(f"valid_rows={len(valid_samples)}")
    print(f"test_rows={len(test_samples)}")


if __name__ == "__main__":
    main()
