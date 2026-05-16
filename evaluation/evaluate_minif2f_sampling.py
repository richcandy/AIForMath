import argparse
import json
import random
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from evaluate_lean_proofs import build_summary, evaluate_row, validate_project_dir


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = ROOT / "Qwen2.5-Math-1.5B"
DEFAULT_PROJECT_DIR = ROOT
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "evaluation"
STOP_MARKERS = ["```", "\nimport ", "\ntheorem ", "\nlemma ", "\nexample ", "\n/--"]
PROOF_SYSTEM_PROMPT = (
    "You are a Lean 4 theorem prover. Given a Lean theorem declaration ending with ':= by', "
    "output only the proof body that should come after 'by'.\n"
    "Rules:\n"
    "- Output only Lean proof code.\n"
    "- Do not repeat the theorem statement.\n"
    "- Do not use code fences.\n"
    "- Never use `sorry` or `admit`.\n"
)


class TextStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int, stop_markers: list[str]):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.stop_markers = stop_markers

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[0][self.prompt_length :]
        if generated_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        return any(marker in text for marker in self.stop_markers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MiniF2F theorem proving by sampling multiple proof candidates per problem."
    )
    parser.add_argument("--split", choices=["valid", "test", "both"], default="test")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--adapter-root", type=Path, default=None)
    parser.add_argument("--adapter-dir", type=Path, default=ROOT / "outputs" / "qwen2_5_math_algebra_medium_lora")
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "minif2f_sampling_results.jsonl")
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--samples-per-problem", type=int, default=32)
    parser.add_argument("--max-attempts-per-problem", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--sample-seed", type=int, default=42)
    return parser.parse_args()


def resolve_model_dir(model_root: Path) -> Path:
    ref_file = model_root / "refs" / "main"
    if ref_file.exists():
        return model_root / "snapshots" / ref_file.read_text(encoding="utf-8").strip()
    return model_root


def resolve_latest_checkpoint(adapter_dir: Path) -> Path:
    candidates = []
    for path in adapter_dir.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if path.is_dir() and suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directories found under {adapter_dir}")
    return max(candidates)[1]


def load_model_and_tokenizer(model_root: Path, adapter_root: Path | None):
    resolved_model = resolve_model_dir(model_root)
    tokenizer_root = adapter_root if adapter_root and (adapter_root / "tokenizer.json").exists() else resolved_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if adapter_root is not None:
        model = PeftModel.from_pretrained(model, adapter_root)
    model.eval()
    return model, tokenizer


def extract_question_from_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    updated = re.sub(r":=\s*by\s*(?:sorry|admit)\s*$", ":= by", text, count=1, flags=re.S)
    if updated == text:
        raise ValueError(f"Could not locate trailing ':= by sorry/admit' in {path}")
    return updated.rstrip()


def collect_problem_paths(split: str) -> list[Path]:
    split_map = {"valid": "Valid", "test": "Test"}
    splits = [split_map[split]] if split != "both" else ["Valid", "Test"]
    paths = []
    for split_name in splits:
        split_dir = ROOT / "MiniF2F" / split_name
        paths.extend(sorted(split_dir.glob("*.lean")))
    return paths


def cleanup_proof(text: str) -> str:
    cleaned = text.strip()
    for marker in STOP_MARKERS:
        index = cleaned.find(marker)
        if index != -1:
            cleaned = cleaned[:index].strip()
    cleaned = cleaned.strip().strip("`")
    cleaned = re.sub(r"^(?:Assistant|assistant|Proof|proof|Answer|answer)\s*[:：]\s*", "", cleaned)
    if cleaned.startswith("by"):
        cleaned = cleaned[2:].strip()
    return cleaned.strip()


def generate_proof_candidates(model, tokenizer, question: str, args: argparse.Namespace, rng: random.Random) -> list[str]:
    messages = [
        {"role": "system", "content": PROOF_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    stopping = StoppingCriteriaList(
        [TextStoppingCriteria(tokenizer, inputs["input_ids"].shape[1], STOP_MARKERS)]
    )

    prompt_length = inputs["input_ids"].shape[1]
    candidates = []
    seen = set()
    max_attempts = max(args.max_attempts_per_problem, args.samples_per_problem)

    for _ in range(max_attempts):
        with torch.random.fork_rng(devices=[model.device]):
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


def build_generation_row(problem_path: Path, question: str, predictions: list[str]) -> dict:
    split_name = problem_path.parent.name
    theorem_id = problem_path.stem
    return {
        "id": f"{split_name}/{theorem_id}",
        "split": split_name,
        "file_path": str(problem_path.relative_to(ROOT)),
        "question": question,
        "predictions": predictions,
    }


def main() -> None:
    args = parse_args()
    args.model_root = args.model_root.resolve()
    args.project_dir = args.project_dir.resolve()
    args.output = args.output.resolve()
    args.adapter_dir = args.adapter_dir.resolve()
    if args.adapter_root is None:
        args.adapter_root = resolve_latest_checkpoint(args.adapter_dir)
    else:
        args.adapter_root = args.adapter_root.resolve()
    summary_path = args.summary_path.resolve() if args.summary_path else args.output.with_suffix(".summary.json")

    validate_project_dir(args.project_dir)
    problem_paths = collect_problem_paths(args.split)
    if args.max_samples is not None:
        problem_paths = problem_paths[: args.max_samples]

    model, tokenizer = load_model_and_tokenizer(args.model_root, args.adapter_root)
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
            "adapter_root": str(args.adapter_root),
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
