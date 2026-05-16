import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import StoppingCriteriaList

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_EVAL_DIR = CURRENT_DIR.parent
if str(PARENT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_EVAL_DIR))

from evaluate_lean_proofs import build_summary, evaluate_row, validate_project_dir

from common import (
    ROOT,
    STOP_PROOF_MARKERS,
    TextStoppingCriteria,
    build_generation_inputs,
    build_generation_row,
    build_proof_prompt,
    cleanup_proof,
    collect_problem_paths,
    extract_question_from_file,
    load_model_and_tokenizer,
)


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "deepseekprover_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MiniF2F proof-body pass@32 with DeepSeekProverV2 and Lean verification."
    )
    parser.add_argument("--split", choices=["valid", "test", "both"], default="test")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "minif2f_proofbody_pass32.jsonl",
    )
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--samples-per-problem", type=int, default=32)
    parser.add_argument("--max-attempts-per-problem", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--prompt-mode", choices=["auto", "chat", "plain"], default="auto")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def generate_proof_candidates(model, tokenizer, question: str, args: argparse.Namespace, rng: random.Random) -> list[str]:
    user_prompt = build_proof_prompt(question)
    inputs = build_generation_inputs(model, tokenizer, user_prompt, args.prompt_mode)
    prompt_length = inputs["input_ids"].shape[1]
    stopping = StoppingCriteriaList(
        [TextStoppingCriteria(tokenizer, prompt_length, STOP_PROOF_MARKERS)]
    )

    candidates = []
    seen = set()
    max_attempts = max(args.max_attempts_per_problem, args.samples_per_problem)
    fork_devices = [model.device] if getattr(model.device, "type", None) == "cuda" else []

    for _ in range(max_attempts):
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(rng.randint(1, 2**31 - 1))
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=args.repetition_penalty,
                    stopping_criteria=stopping,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=1,
                )

        decoded = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
        proof = cleanup_proof(decoded)
        if not proof or proof in seen:
            continue
        seen.add(proof)
        candidates.append(proof)
        if len(candidates) >= args.samples_per_problem:
            break

    return candidates


def main() -> None:
    args = parse_args()
    args.model_root = args.model_root.resolve()
    args.project_dir = args.project_dir.resolve()
    args.output = args.output.resolve()
    summary_path = args.summary_path.resolve() if args.summary_path else args.output.with_suffix(".summary.json")

    validate_project_dir(args.project_dir)
    problem_paths = collect_problem_paths(args.split)
    if args.max_samples is not None:
        problem_paths = problem_paths[: args.max_samples]

    model, tokenizer = load_model_and_tokenizer(args.model_root, args.trust_remote_code)
    rng = random.Random(args.sample_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    generation_rows = []
    for problem_path in problem_paths:
        question = extract_question_from_file(problem_path)
        predictions = generate_proof_candidates(model, tokenizer, question, args, rng)
        generation_rows.append(build_generation_row(problem_path, question, predictions))

    with args.output.open("w", encoding="utf-8") as output_file:
        for row in generation_rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    eval_args = argparse.Namespace(
        input=args.output,
        output=args.output,
        project_dir=args.project_dir,
        question_field="question",
        proof_field="predictions",
        id_field="id",
        timeout_seconds=args.timeout_seconds,
        jobs=1,
        max_samples=None,
        normalize=True,
    )
    results = [evaluate_row(index + 1, row, eval_args) for index, row in enumerate(generation_rows)]

    evaluation_output = args.output.with_suffix(".evaluated.jsonl")
    with evaluation_output.open("w", encoding="utf-8") as output_file:
        for row, result in zip(generation_rows, results, strict=True):
            payload = {
                "id": row["id"],
                "split": row["split"],
                "file_path": row["file_path"],
                "question": row["question"],
                "predictions": row["predictions"],
                **result,
            }
            output_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    summary = build_summary(eval_args, results)
    summary.update(
        {
            "generation_output": str(args.output),
            "evaluation_output": str(evaluation_output),
            "summary_path": str(summary_path),
            "split": args.split,
            "model_root": str(args.model_root),
            "prompt_mode": args.prompt_mode,
            "samples_per_problem": args.samples_per_problem,
            "max_attempts_per_problem": args.max_attempts_per_problem,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"generation_output={args.output}")
    print(f"evaluation_output={evaluation_output}")
    print(f"summary={summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
