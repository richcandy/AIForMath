import argparse
import json
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pantograph.server import Server, ServerError, TacticFailure
from transformers import StoppingCriteriaList

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_EVAL_DIR = CURRENT_DIR.parent
if str(PARENT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_EVAL_DIR))

from evaluate_lean_proofs import build_summary, evaluate_row, validate_project_dir

from common import (
    ROOT,
    STOP_TACTIC_MARKERS,
    TextStoppingCriteria,
    build_generation_inputs,
    build_tactic_prompt,
    cleanup_tactic,
    collect_problem_paths,
    extract_question_from_file,
    load_model_and_tokenizer,
    truncate_goal_text,
)


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "deepseekprover_v2"


@dataclass
class SearchNode:
    goal_state: Any
    proof_steps: list[str]
    depth: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MiniF2F tactic+BFS pass@32 with DeepSeekProverV2 and Lean verification."
    )
    parser.add_argument("--split", choices=["valid", "test", "both"], default="test")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "minif2f_bfs_pass32.jsonl",
    )
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--attempts-per-problem", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument("--num-tactics", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-goal-tokens", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--prompt-mode", choices=["auto", "chat", "plain"], default="auto")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--fixed-tactic",
        action="append",
        default=[],
        help="Inject one or more fixed tactics ahead of model proposals.",
    )
    return parser.parse_args()


def build_search_target(question: str) -> str:
    lines = question.rstrip().splitlines()
    while lines and lines[0].startswith("import "):
        lines.pop(0)
    theorem_block = "\n".join(lines).rstrip()
    return f"{theorem_block}\n  sorry\n"


def build_goal_text(goal_state: Any) -> str:
    return "\n\n".join(str(goal) for goal in goal_state.goals)


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
    user_prompt = build_tactic_prompt(goal_text)
    inputs = build_generation_inputs(model, tokenizer, user_prompt, args.prompt_mode)
    prompt_length = inputs["input_ids"].shape[1]
    stopping = StoppingCriteriaList(
        [TextStoppingCriteria(tokenizer, prompt_length, STOP_TACTIC_MARKERS)]
    )

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
    summary_path = args.summary_path.resolve() if args.summary_path else args.output.with_suffix(".summary.json")

    validate_project_dir(args.project_dir)
    problem_paths = collect_problem_paths(args.split)
    if args.max_samples is not None:
        problem_paths = problem_paths[: args.max_samples]

    model, tokenizer = load_model_and_tokenizer(args.model_root, args.trust_remote_code)
    rng = random.Random(args.sample_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

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
            "prompt_mode": args.prompt_mode,
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
