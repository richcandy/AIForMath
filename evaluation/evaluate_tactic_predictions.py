import argparse
import json
import re
import random
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = ROOT / "DeepSeek-Prover-V2"
DEFAULT_OUTPUT = ROOT / "outputs" / "evaluation" / "tactic_eval_results.jsonl"
TACTIC_SYSTEM_PROMPT = (
    "You are a Lean 4 tactic generator. Given a goal state, output exactly ONE Lean tactic that advances or solves the goal.\n"
    "Rules:\n"
    "- Output only the tactic text; no prose or code fences.\n"
    "- Single line only; no `by` blocks.\n"
    "- Never use `sorry` or `admit`.\n"
)
STOP_MARKERS = ["\n", "```", "User:", "Assistant:"]


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
        description="Evaluate single-step Lean tactic predictions on traced-tactics data."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--adapter-root", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-goal-tokens", type=int, default=2800)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--sample-seed", type=int, default=42)
    return parser.parse_args()


def resolve_model_dir(model_root: Path) -> Path:
    ref_file = model_root / "refs" / "main"
    if ref_file.exists():
        return model_root / "snapshots" / ref_file.read_text(encoding="utf-8").strip()
    return model_root


def flatten_traced_tactics(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")

    rows = []
    for theorem_index, theorem in enumerate(data):
        theorem_id = theorem.get("full_name") or theorem.get("file_path") or f"theorem-{theorem_index:05d}"
        theorem_statement = str(theorem.get("theorem_statement") or "")
        for tactic_index, tactic in enumerate(theorem.get("traced_tactics", [])):
            rows.append(
                {
                    "id": f"{theorem_id}#step-{tactic_index:03d}",
                    "theorem_id": theorem_id,
                    "file_path": theorem.get("file_path"),
                    "theorem_statement": theorem_statement,
                    "state_before": str(tactic.get("state_before") or "").strip(),
                    "gold_tactic": str(tactic.get("tactic") or "").strip(),
                    "state_after": str(tactic.get("state_after") or "").strip(),
                }
            )
    return rows


def cleanup_tactic(text: str) -> str:
    cleaned = text.strip()
    for marker in STOP_MARKERS:
        index = cleaned.find(marker)
        if index != -1:
            cleaned = cleaned[:index].strip()
    cleaned = cleaned.strip().strip("`")
    cleaned = cleaned.replace("Assistant:", "").replace("assistant:", "").strip()
    if cleaned.startswith("by"):
        cleaned = cleaned[2:].strip()
    cleaned = cleaned.splitlines()[0].strip() if cleaned.strip() else ""
    cleaned = cleaned.replace("<a>", "").replace("</a>", "")
    return re.sub(r"\s+", " ", cleaned).strip()


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


def truncate_goal_text(tokenizer, goal_text: str, max_goal_tokens: int) -> tuple[str, bool]:
    if max_goal_tokens <= 0:
        return goal_text, False

    encoded = tokenizer(goal_text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"][0]
    if input_ids.shape[0] <= max_goal_tokens:
        return goal_text, False

    truncated_ids = input_ids[-max_goal_tokens:]
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True).strip()
    return truncated_text, True


def generate_tactics(model, tokenizer, goal_text: str, args: argparse.Namespace) -> list[str]:
    goal_text, _ = truncate_goal_text(tokenizer, goal_text, args.max_goal_tokens)
    messages = [
        {"role": "system", "content": TACTIC_SYSTEM_PROMPT},
        {"role": "user", "content": goal_text},
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

    generation_kwargs = {
        **inputs,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": args.repetition_penalty,
        "stopping_criteria": stopping,
    }
    prompt_length = inputs["input_ids"].shape[1]
    predictions = []
    seen = set()

    if args.num_return_sequences <= 1:
        generation_kwargs["do_sample"] = args.temperature > 0
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p
        else:
            generation_kwargs["num_beams"] = 1

        with torch.no_grad():
            output = model.generate(**generation_kwargs)

        for sequence in output:
            decoded = tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)
            tactic = cleanup_tactic(decoded)
            if not tactic or tactic in seen:
                continue
            seen.add(tactic)
            predictions.append(tactic)
        return predictions

    sample_temperature = args.temperature if args.temperature > 0 else 0.8
    sample_top_p = args.top_p if args.temperature > 0 else 0.95
    max_attempts = max(args.num_return_sequences * 4, 8)
    rng = random.Random(args.sample_seed)

    for _ in range(max_attempts):
        with torch.random.fork_rng(devices=[model.device]):
            torch.manual_seed(rng.randint(1, 2**31 - 1))
            with torch.no_grad():
                output = model.generate(
                    **generation_kwargs,
                    do_sample=True,
                    temperature=sample_temperature,
                    top_p=sample_top_p,
                    num_return_sequences=1,
                )

        decoded = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
        tactic = cleanup_tactic(decoded)
        if not tactic or tactic in seen:
            continue
        seen.add(tactic)
        predictions.append(tactic)
        if len(predictions) >= args.num_return_sequences:
            break

    return predictions


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.model_root = args.model_root.resolve()
    if args.adapter_root is not None:
        args.adapter_root = args.adapter_root.resolve()
    summary_path = args.summary_path.resolve() if args.summary_path else args.output.with_suffix(".summary.json")

    rows = flatten_traced_tactics(args.input)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    model, tokenizer = load_model_and_tokenizer(args.model_root, args.adapter_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    top1_correct = 0
    topk_correct = 0
    nonempty_predictions = 0
    with args.output.open("w", encoding="utf-8") as output_file:
        for row in rows:
            predictions = generate_tactics(model, tokenizer, row["state_before"], args)
            gold = cleanup_tactic(row["gold_tactic"])
            if predictions:
                nonempty_predictions += 1
            top1_hit = bool(predictions and predictions[0] == gold)
            topk_hit = gold in predictions
            top1_correct += int(top1_hit)
            topk_correct += int(topk_hit)

            payload = {
                "id": row["id"],
                "theorem_id": row["theorem_id"],
                "file_path": row["file_path"],
                "state_before": row["state_before"],
                "gold_tactic": gold,
                "predictions": predictions,
                "top1_correct": top1_hit,
                "topk_correct": topk_hit,
            }
            output_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    total = len(rows)
    summary = {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "model_root": str(args.model_root),
        "adapter_root": str(args.adapter_root) if args.adapter_root else None,
        "total": total,
        "top1_accuracy": round(top1_correct / total, 4) if total else 0.0,
        "topk_accuracy": round(topk_correct / total, 4) if total else 0.0,
        "nonempty_prediction_rate": round(nonempty_predictions / total, 4) if total else 0.0,
        "num_return_sequences": args.num_return_sequences,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results={args.output}")
    print(f"summary={summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
