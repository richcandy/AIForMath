import argparse
import json
import random
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pantograph.server import Server, ServerError, TacticFailure
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from evaluate_lean_proofs import build_summary, evaluate_row, validate_project_dir


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = ROOT / "Qwen2.5-Math-1.5B"
DEFAULT_PROJECT_DIR = ROOT
DEFAULT_OUTPUT = ROOT / "outputs" / "evaluation" / "minif2f_bfs_results.jsonl"
TACTIC_SYSTEM_PROMPT = (
    "You are a Lean 4 tactic generator. Given a goal state, output exactly ONE Lean tactic that advances or solves the goal.\n"
    "Rules:\n"
    "- Output only the tactic text; no prose or code fences.\n"
    "- Single line only; no `by` blocks.\n"
    "- Never use `sorry` or `admit`.\n"
)
STOP_MARKERS = ["\n", "```", "User:", "Assistant:"]


@dataclass
class SearchNode:
    goal_state: Any
    proof_steps: list[str]
    depth: int


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
        description="Evaluate MiniF2F with BFS single-step tactic generation and Lean verification."
    )
    parser.add_argument("--split", choices=["valid", "test", "both"], default="test")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--adapter-root", type=Path, default=None)
    parser.add_argument("--adapter-dir", type=Path, default=ROOT / "outputs" / "qwen2_5_math_algebra_medium_lora")
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--attempts-per-problem", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument("--num-tactics", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-goal-tokens", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--fixed-tactic",
        action="append",
        default=[],
        help="Inject one or more fixed tactics ahead of model proposals.",
    )
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


def resolve_default_checkpoint(adapter_dir: Path) -> Path:
    latest_checkpoint = resolve_latest_checkpoint(adapter_dir)
    trainer_state_path = latest_checkpoint / "trainer_state.json"
    if not trainer_state_path.exists():
        return latest_checkpoint
    try:
        trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return latest_checkpoint

    best_model_checkpoint = trainer_state.get("best_model_checkpoint")
    if not isinstance(best_model_checkpoint, str):
        return latest_checkpoint

    best_path = Path(best_model_checkpoint)
    if not best_path.is_absolute():
        best_path = (adapter_dir / best_path).resolve()
    return best_path if best_path.exists() else latest_checkpoint


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
        paths.extend(sorted((ROOT / "MiniF2F" / split_name).glob("*.lean")))
    return paths


def build_search_target(question: str) -> str:
    lines = question.rstrip().splitlines()
    while lines and lines[0].startswith("import "):
        lines.pop(0)
    return f"{'\n'.join(lines).rstrip()}\n  sorry\n"


def build_goal_text(goal_state: Any) -> str:
    return "\n\n".join(str(goal) for goal in goal_state.goals)


def truncate_goal_text(tokenizer, goal_text: str, max_goal_tokens: int) -> str:
    if max_goal_tokens <= 0:
        return goal_text
    encoded = tokenizer(goal_text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"][0]
    if input_ids.shape[0] <= max_goal_tokens:
        return goal_text
    return tokenizer.decode(input_ids[-max_goal_tokens:], skip_special_tokens=True).strip()


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
    cleaned = cleaned.splitlines()[0].strip() if cleaned else ""
    cleaned = cleaned.replace("<a>", "").replace("</a>", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    bad_fragments = ["You are a Lean", "Given a goal state", "⊢", "User:", "Assistant:"]
    if any(fragment in cleaned for fragment in bad_fragments):
        return ""
    return cleaned


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def generate_tactics(model, tokenizer, goal_text: str, args: argparse.Namespace, rng: random.Random) -> list[str]:
    goal_text = truncate_goal_text(tokenizer, goal_text, args.max_goal_tokens)
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

    prompt_length = inputs["input_ids"].shape[1]
    predictions = []
    seen = set()
    max_attempts = max(args.num_tactics * 4, 8)
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
        tactic = cleanup_tactic(decoded)
        if not tactic or tactic in seen:
            continue
        seen.add(tactic)
        predictions.append(tactic)
        if len(predictions) >= args.num_tactics:
            break
    return predictions


def serialize_goals(goal_state: Any) -> tuple[str, ...]:
    return tuple(str(goal) for goal in goal_state.goals)


def bfs_prove(server: Server, model, tokenizer, question: str, args: argparse.Namespace, rng: random.Random) -> dict[str, Any]:
    targets = server.load_sorry(build_search_target(question))
    if not targets:
        return {
            "solved": False,
            "proof": "",
            "expanded_nodes": 0,
            "visited_states": 0,
            "failure": "no_targets",
            "explored_tactics": [],
        }

    root_state = targets[0].goal_state
    queue = deque([SearchNode(root_state, [], 0)])
    visited = {serialize_goals(root_state)}
    expanded_nodes = 0
    explored_tactics = []

    while queue and expanded_nodes < args.max_nodes:
        node = queue.popleft()
        if len(node.goal_state.goals) == 0:
            return {
                "solved": True,
                "proof": "\n".join(node.proof_steps),
                "expanded_nodes": expanded_nodes,
                "visited_states": len(visited),
                "failure": None,
                "explored_tactics": explored_tactics,
            }
        if node.depth >= args.max_depth:
            continue

        expanded_nodes += 1
        goal_text = build_goal_text(node.goal_state)
        model_tactics = generate_tactics(model, tokenizer, goal_text, args, rng)
        candidate_tactics = dedupe_keep_order(args.fixed_tactic + model_tactics)
        explored_tactics.append(
            {
                "depth": node.depth,
                "goal": goal_text,
                "tactics": candidate_tactics,
            }
        )

        for tactic in candidate_tactics:
            try:
                next_state = server.goal_tactic(node.goal_state, tactic)
            except (TacticFailure, ServerError):
                continue

            next_signature = serialize_goals(next_state)
            if next_signature in visited:
                continue
            visited.add(next_signature)

            next_steps = node.proof_steps + [tactic]
            if len(next_state.goals) == 0:
                return {
                    "solved": True,
                    "proof": "\n".join(next_steps),
                    "expanded_nodes": expanded_nodes,
                    "visited_states": len(visited),
                    "failure": None,
                    "explored_tactics": explored_tactics,
                }
            queue.append(SearchNode(next_state, next_steps, node.depth + 1))

    return {
        "solved": False,
        "proof": "",
        "expanded_nodes": expanded_nodes,
        "visited_states": len(visited),
        "failure": "search_exhausted",
        "explored_tactics": explored_tactics,
    }


def build_generation_row(problem_path: Path, question: str, predictions: list[str], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    split_name = problem_path.parent.name
    theorem_id = problem_path.stem
    return {
        "id": f"{split_name}/{theorem_id}",
        "split": split_name,
        "file_path": str(problem_path.relative_to(ROOT)),
        "question": question,
        "predictions": predictions,
        "search_attempts": attempts,
    }


def main() -> None:
    args = parse_args()
    args.model_root = args.model_root.resolve()
    args.project_dir = args.project_dir.resolve()
    args.output = args.output.resolve()
    args.adapter_dir = args.adapter_dir.resolve()
    if args.adapter_root is None:
        args.adapter_root = resolve_default_checkpoint(args.adapter_dir)
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
    results = []
    evaluation_output = args.output.with_suffix(".evaluated.jsonl")
    with Server(imports=["Mathlib"], project_path=str(args.project_dir)) as server:
        with args.output.open("w", encoding="utf-8") as output_file, evaluation_output.open(
            "w", encoding="utf-8"
        ) as evaluated_file:
            total_problems = len(problem_paths)
            for problem_index, problem_path in enumerate(problem_paths, start=1):
                question = extract_question_from_file(problem_path)
                predictions = []
                attempts = []
                for attempt_index in range(args.attempts_per_problem):
                    attempt_rng = random.Random(rng.randint(1, 2**31 - 1) + attempt_index)
                    result = bfs_prove(server, model, tokenizer, question, args, attempt_rng)
                    predictions.append(result["proof"])
                    attempts.append(
                        {
                            "attempt_index": attempt_index,
                            "solved": result["solved"],
                            "expanded_nodes": result["expanded_nodes"],
                            "visited_states": result["visited_states"],
                            "failure": result["failure"],
                            "proof": result["proof"],
                            "explored_tactics": result["explored_tactics"],
                        }
                    )

                row = build_generation_row(problem_path, question, predictions, attempts)
                generation_rows.append(row)
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_file.flush()

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
                evaluation_result = evaluate_row(problem_index, row, eval_args)
                results.append(evaluation_result)
                payload = {
                    "id": row["id"],
                    "split": row["split"],
                    "file_path": row["file_path"],
                    "question": row["question"],
                    "predictions": row["predictions"],
                    "search_attempts": row["search_attempts"],
                    **evaluation_result,
                }
                evaluated_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                evaluated_file.flush()
                print(
                    f"completed={problem_index}/{total_problems} id={row['id']} "
                    f"pass_at_1={int(evaluation_result['pass_at_1'])} pass_at_k={int(evaluation_result['pass_at_k'])}",
                    flush=True,
                )

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
    summary = build_summary(eval_args, results)
    summary.update(
        {
            "generation_output": str(args.output),
            "evaluation_output": str(evaluation_output),
            "summary_path": str(summary_path),
            "split": args.split,
            "model_root": str(args.model_root),
            "adapter_root": str(args.adapter_root),
            "attempts_per_problem": args.attempts_per_problem,
            "max_depth": args.max_depth,
            "max_nodes": args.max_nodes,
            "num_tactics": args.num_tactics,
            "max_new_tokens": args.max_new_tokens,
            "max_goal_tokens": args.max_goal_tokens,
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
