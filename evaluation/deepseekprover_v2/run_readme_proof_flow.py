import argparse
import json
import re
import sys
import textwrap
from threading import Thread
from time import perf_counter
from pathlib import Path

import torch
from transformers import TextIteratorStreamer

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_EVAL_DIR = CURRENT_DIR.parent
if str(PARENT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_EVAL_DIR))

from evaluate_lean_proofs import build_summary, evaluate_row, validate_project_dir

from common import (
    ROOT,
    build_generation_inputs,
    build_prompt_a_proof_prompt,
    build_readme_proof_prompt,
    collect_problem_paths,
    extract_formal_statement_from_file,
    extract_question_from_file,
    load_model_and_tokenizer,
)


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "deepseekprover_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single-sample README-style DeepSeek proof generation flow and write raw/extracted/evaluated JSONL outputs."
    )
    parser.add_argument("--split", choices=["valid", "test", "both"], default="test")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "readme_flow")
    parser.add_argument("--run-name", default="minif2f_prefix")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--prompt-mode", choices=["auto", "chat", "plain"], default="auto")
    parser.add_argument("--prompt-style", choices=["deepseek_readme", "prompt_a"], default="deepseek_readme")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def normalize_model_text(text: str) -> str:
    return text.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ").strip()


def strip_wrappers(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^#{1,6}\s*", "", stripped)
    stripped = re.sub(r"^(?:Assistant|assistant|Answer|answer)\s*[:：]\s*", "", stripped)
    return stripped.strip()


def normalize_proof_body(text: str) -> str:
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    return textwrap.dedent("\n".join(lines)).rstrip()


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    blocks = []
    for match in re.finditer(r"```([A-Za-z0-9_-]*)\s*\n?(.*?)```", text, flags=re.S):
        language = (match.group(1) or "").strip().lower()
        content = match.group(2).strip()
        blocks.append((language, content))
    return blocks


def strip_prompt_echo(text: str, formal_statement: str) -> str:
    stripped = text.strip()
    prompt_prefix = build_prompt(formal_statement, "deepseek_readme").strip()
    if stripped.startswith(prompt_prefix):
        return stripped[len(prompt_prefix) :].strip()
    return stripped


def build_prompt(formal_statement: str, prompt_style: str) -> str:
    if prompt_style == "prompt_a":
        return build_prompt_a_proof_prompt(formal_statement)
    return build_readme_proof_prompt(formal_statement)


def looks_like_lean_code(text: str) -> bool:
    markers = ["theorem ", "lemma ", "example ", ":= by", " by\n", "simp", "rw ", "exact ", "have ", "calc"]
    return any(marker in text for marker in markers)


def extract_proof_body(candidate: str) -> tuple[str, str]:
    candidate = strip_wrappers(candidate)
    theorem_match = re.search(r":=\s*by\b", candidate)
    if theorem_match is not None and re.search(r"(?m)^\s*(?:import\s+|theorem\s+|lemma\s+|example\s+)", candidate):
        proof = normalize_proof_body(candidate[theorem_match.end() :])
        if proof:
            return proof, "theorem_to_body"

    candidate = re.sub(r"^```(?:lean4?|lean)?", "", candidate).strip()
    candidate = re.sub(r"```$", "", candidate).strip()
    candidate = re.sub(r"^(?:Lean\s*4\s*Proof|Detailed\s*Proof|Proof\s*Plan)\s*:??\s*", "", candidate, flags=re.I)
    if candidate.startswith("by"):
        candidate = candidate[2:].strip()
    return normalize_proof_body(candidate), "body_direct"


def strip_prompt_echo_with_style(text: str, formal_statement: str, prompt_style: str) -> str:
    stripped = text.strip()
    prompt_prefix = build_prompt(formal_statement, prompt_style).strip()
    if stripped.startswith(prompt_prefix):
        return stripped[len(prompt_prefix) :].strip()
    return stripped


def extract_from_raw_output(raw_output: str, formal_statement: str, prompt_style: str) -> dict:
    normalized = strip_prompt_echo_with_style(normalize_model_text(raw_output), formal_statement, prompt_style)
    code_blocks = extract_code_blocks(normalized)
    lean_blocks = [content for language, content in code_blocks if language in {"lean", "lean4"}]
    untyped_blocks = [content for language, content in code_blocks if not language]

    candidates = []
    for block in reversed(lean_blocks):
        candidates.append((block, "lean_code_block"))
    for block in reversed(untyped_blocks):
        if looks_like_lean_code(block):
            candidates.append((block, "generic_code_block"))
    if looks_like_lean_code(normalized):
        candidates.append((normalized, "full_output"))

    for candidate, source in candidates:
        proof, strategy = extract_proof_body(candidate)
        if proof and not re.search(r"\b(?:sorry|admit)\b", proof.lower()):
            return {
                "extracted_proof": proof,
                "extraction_status": "ok",
                "extraction_source": source,
                "extraction_strategy": strategy,
            }

    reason = "contains_sorry_or_admit" if re.search(r"\b(?:sorry|admit)\b", normalized.lower()) else "no_lean_proof_found"
    return {
        "extracted_proof": "",
        "extraction_status": reason,
        "extraction_source": None,
        "extraction_strategy": None,
    }


def generate_raw_output(model, tokenizer, formal_statement: str, args: argparse.Namespace) -> dict:
    user_prompt = build_prompt(formal_statement, args.prompt_style)
    inputs = build_generation_inputs(model, tokenizer, user_prompt, args.prompt_mode)
    prompt_length = inputs["input_ids"].shape[1]
    generation_kwargs = {
        **inputs,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    start_time = perf_counter()
    first_token_seconds = None
    if args.stream:
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        generation_kwargs["streamer"] = streamer
        result_holder = {}
        error_holder = {}

        def run_generation() -> None:
            try:
                with torch.no_grad():
                    result_holder["output"] = model.generate(**generation_kwargs)
            except BaseException as exc:  # pragma: no cover - surfaced after join
                error_holder["error"] = exc

        worker = Thread(target=run_generation, daemon=True)
        worker.start()
        for chunk in streamer:
            if first_token_seconds is None:
                first_token_seconds = perf_counter() - start_time
            sys.stdout.write(chunk)
            sys.stdout.flush()
        worker.join()
        sys.stdout.write("\n")
        sys.stdout.flush()
        if "error" in error_holder:
            raise error_holder["error"]
        output = result_holder["output"]
    else:
        with torch.no_grad():
            output = model.generate(**generation_kwargs)

    generation_seconds = perf_counter() - start_time
    continuation_ids = output[0][prompt_length:]
    full_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
    special_ids = set(tokenizer.all_special_ids)
    output_token_count = sum(1 for token_id in continuation_ids.tolist() if token_id not in special_ids)
    tokens_per_second = output_token_count / generation_seconds if generation_seconds > 0 else 0.0
    return {
        "full_output": full_text,
        "continuation_output": continuation_text,
        "prompt_token_count": int(prompt_length),
        "output_token_count": int(output_token_count),
        "generation_seconds": generation_seconds,
        "tokens_per_second": tokens_per_second,
        "time_to_first_token_seconds": first_token_seconds,
    }


def build_record_base(problem_path: Path, question: str) -> dict:
    split_name = problem_path.parent.name
    theorem_id = problem_path.stem
    return {
        "id": f"{split_name}/{theorem_id}",
        "split": split_name,
        "file_path": str(problem_path.relative_to(ROOT)),
        "question": question,
    }


def main() -> None:
    args = parse_args()
    args.model_root = args.model_root.resolve()
    args.project_dir = args.project_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    validate_project_dir(args.project_dir)
    problem_paths = collect_problem_paths(args.split)
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.start_index:
        problem_paths = problem_paths[args.start_index :]
    if args.max_samples is not None:
        problem_paths = problem_paths[: args.max_samples]

    model, tokenizer = load_model_and_tokenizer(args.model_root, args.trust_remote_code)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"{args.run_name}.raw.jsonl"
    extracted_path = args.output_dir / f"{args.run_name}.extracted.jsonl"
    evaluated_path = args.output_dir / f"{args.run_name}.evaluated.jsonl"
    summary_path = args.output_dir / f"{args.run_name}.summary.json"

    raw_rows = []
    extracted_rows = []
    total_problems = len(problem_paths)
    for problem_index, problem_path in enumerate(problem_paths, start=1):
        question = extract_question_from_file(problem_path)
        formal_statement = extract_formal_statement_from_file(problem_path)
        record = build_record_base(problem_path, question)
        if args.stream:
            print(f"\n=== [{problem_index}/{total_problems}] {record['id']} ===", flush=True)
        generated = generate_raw_output(model, tokenizer, formal_statement, args)
        print(
            "generation_stats "
            f"id={record['id']} "
            f"prompt_tokens={generated['prompt_token_count']} "
            f"output_tokens={generated['output_token_count']} "
            f"seconds={generated['generation_seconds']:.2f} "
            f"tokens_per_second={generated['tokens_per_second']:.2f} "
            f"time_to_first_token={generated['time_to_first_token_seconds'] if generated['time_to_first_token_seconds'] is not None else 'n/a'}",
            flush=True,
        )
        extraction = extract_from_raw_output(generated["continuation_output"], formal_statement, args.prompt_style)

        raw_rows.append(
            {
                **record,
                "prompt_style": args.prompt_style,
                "formal_statement": formal_statement,
                "prompt": build_prompt(formal_statement, args.prompt_style),
                "full_output": generated["full_output"],
                "continuation_output": generated["continuation_output"],
                "prompt_token_count": generated["prompt_token_count"],
                "output_token_count": generated["output_token_count"],
                "generation_seconds": generated["generation_seconds"],
                "tokens_per_second": generated["tokens_per_second"],
                "time_to_first_token_seconds": generated["time_to_first_token_seconds"],
            }
        )
        extracted_rows.append(
            {
                **record,
                "prompt_style": args.prompt_style,
                "formal_statement": formal_statement,
                "prompt": build_prompt(formal_statement, args.prompt_style),
                "full_output": generated["full_output"],
                "raw_output": generated["continuation_output"],
                "prompt_token_count": generated["prompt_token_count"],
                "output_token_count": generated["output_token_count"],
                "generation_seconds": generated["generation_seconds"],
                "tokens_per_second": generated["tokens_per_second"],
                "time_to_first_token_seconds": generated["time_to_first_token_seconds"],
                **extraction,
                "predictions": [extraction["extracted_proof"]] if extraction["extracted_proof"] else [],
            }
        )

    with raw_path.open("w", encoding="utf-8") as raw_file:
        for row in raw_rows:
            raw_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with extracted_path.open("w", encoding="utf-8") as extracted_file:
        for row in extracted_rows:
            extracted_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    eval_args = argparse.Namespace(
        input=extracted_path,
        output=evaluated_path,
        project_dir=args.project_dir,
        question_field="question",
        proof_field="predictions",
        id_field="id",
        timeout_seconds=args.timeout_seconds,
        jobs=1,
        max_samples=None,
        normalize=False,
    )
    results = [evaluate_row(index + 1, row, eval_args) for index, row in enumerate(extracted_rows)]

    with evaluated_path.open("w", encoding="utf-8") as evaluated_file:
        for row, result in zip(extracted_rows, results, strict=True):
            payload = {
                **row,
                **result,
            }
            evaluated_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    extraction_status_counts = {}
    total_prompt_tokens = 0
    total_output_tokens = 0
    total_generation_seconds = 0.0
    total_tokens_per_second = 0.0
    total_time_to_first_token = 0.0
    time_to_first_token_count = 0
    for row in extracted_rows:
        status = row["extraction_status"]
        extraction_status_counts[status] = extraction_status_counts.get(status, 0) + 1
        total_prompt_tokens += row["prompt_token_count"]
        total_output_tokens += row["output_token_count"]
        total_generation_seconds += row["generation_seconds"]
        total_tokens_per_second += row["tokens_per_second"]
        if row["time_to_first_token_seconds"] is not None:
            total_time_to_first_token += row["time_to_first_token_seconds"]
            time_to_first_token_count += 1

    sample_count = len(extracted_rows)
    average_prompt_tokens = total_prompt_tokens / sample_count if sample_count else 0.0
    average_output_tokens = total_output_tokens / sample_count if sample_count else 0.0
    average_generation_seconds = total_generation_seconds / sample_count if sample_count else 0.0
    average_tokens_per_second = total_tokens_per_second / sample_count if sample_count else 0.0
    average_time_to_first_token = (
        total_time_to_first_token / time_to_first_token_count if time_to_first_token_count else None
    )

    summary = build_summary(eval_args, results)
    summary.update(
        {
            "raw_output": str(raw_path),
            "extracted_output": str(extracted_path),
            "evaluated_output": str(evaluated_path),
            "summary_path": str(summary_path),
            "split": args.split,
            "start_index": args.start_index,
            "model_root": str(args.model_root),
            "prompt_style": args.prompt_style,
            "prompt_mode": args.prompt_mode,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "stream": args.stream,
            "extraction_status_counts": extraction_status_counts,
            "average_prompt_tokens": average_prompt_tokens,
            "average_output_tokens": average_output_tokens,
            "average_generation_seconds": average_generation_seconds,
            "average_tokens_per_second": average_tokens_per_second,
            "average_time_to_first_token_seconds": average_time_to_first_token,
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"raw_output={raw_path}")
    print(f"extracted_output={extracted_path}")
    print(f"evaluated_output={evaluated_path}")
    print(f"summary={summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
