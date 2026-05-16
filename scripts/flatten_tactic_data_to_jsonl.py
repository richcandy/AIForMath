import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / "Deepseek_highschool_data"
DEFAULT_OUTPUT_ROOT = DEFAULT_SOURCE_ROOT / "flatten_data"
SYSTEM_PROMPT = (
    "You are a Lean 4 tactic generator. Given a goal state, "
    "output exactly ONE Lean tactic that advances or solves the goal.\n"
    "Rules:\n"
    "- Output only the tactic text; no prose or code fences.\n"
    "- Single line only; no `by` blocks.\n"
    "- Never use `sorry` or `admit`."
)
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten traced_tactics JSON splits into messages JSONL single-step records."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def strip_tags(text: str) -> str:
    return re.sub(r"</?a>", "", text).strip()


def build_message_record(state_before: str, tactic_text: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state_before},
            {"role": "assistant", "content": tactic_text},
        ]
    }


def iter_step_records(rows: list[dict[str, Any]]):
    for theorem_index, row in enumerate(rows):
        traced_tactics = row.get("traced_tactics") or []
        for step_index, step in enumerate(traced_tactics):
            raw_tactic = strip_tags(str(step.get("tactic") or ""))
            state_before = strip_tags(str(step.get("state_before") or ""))
            if not raw_tactic or not state_before:
                continue

            tactic_first_line = raw_tactic.splitlines()[0].strip()
            if not tactic_first_line or tactic_first_line in {"sorry", "admit"}:
                continue

            yield theorem_index, step_index, row, build_message_record(state_before, tactic_first_line)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def flatten_split(source_path: Path, output_path: Path) -> dict[str, int]:
    rows = load_json(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    theorem_count = len(rows)
    step_count = 0

    with temp_path.open("w", encoding="utf-8") as output_file:
        for _, _, _, record in iter_step_records(rows):
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            step_count += 1

    temp_path.replace(output_path)
    return {
        "source_theorems": theorem_count,
        "step_records": step_count,
    }


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    summary: dict[str, Any] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "created_at_unix": int(time.time()),
        "format": "messages_jsonl",
        "system_prompt": SYSTEM_PROMPT,
        "splits": {},
    }

    for split_name in SPLIT_NAMES:
        source_path = source_root / f"{split_name}.json"
        output_path = output_root / f"{split_name}.jsonl"
        stats = flatten_split(source_path, output_path)
        summary["splits"][split_name] = {
            "source_path": str(source_path),
            "output_path": str(output_path),
            **stats,
        }
        print(
            f"{split_name}: source_theorems={stats['source_theorems']} step_records={stats['step_records']}",
            flush=True,
        )

    manifest_path = output_root / "manifest.json"
    atomic_write_json(manifest_path, summary)
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
