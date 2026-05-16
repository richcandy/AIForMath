import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pantograph.server import Server, ServerError, TacticFailure
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = ROOT / "Qwen2.5-Math-1.5B"
DEFAULT_OUTPUT = ROOT / "outputs" / "generation" / "bfs_tactic_predictions.jsonl"
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
        description="Solve Lean theorem goals with single-step tactic generation and BFS search."
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--tokenizer-root", type=Path, default=None)
    parser.add_argument("--project-dir", type=Path, default=ROOT)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--num-tactics", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--fixed-tactic",
        action="append",
        default=[],
        help="Inject one or more fixed tactics ahead of model proposals. Useful for smoke tests.",
    )
    return parser.parse_args()


def resolve_model_dir(model_root: Path) -> Path:
    ref_file = model_root / "refs" / "main"
    if ref_file.exists():
        return model_root / "snapshots" / ref_file.read_text(encoding="utf-8").strip()
    return model_root


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as input_file:
        for index, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("id", f"sample-{index:05d}")
            rows.append(row)
    return rows


def build_goal_text(goal_state: Any) -> str:
    return "\n\n".join(str(goal) for goal in goal_state.goals)


def build_search_target(question: str) -> str:
    lines = question.rstrip().splitlines()
    while lines and lines[0].startswith("import "):
        lines.pop(0)
    return f"{'\n'.join(lines).rstrip()}\n  sorry\n"


def cleanup_tactic(text: str) -> str:
    text = text.strip()
    for marker in STOP_MARKERS:
        index = text.find(marker)
        if index != -1:
            text = text[:index].strip()
    text = text.strip().strip("`")
    text = text.replace("Assistant:", "").replace("assistant:", "").strip()
    if text.startswith("by"):
        text = text[2:].strip()
    text = text.splitlines()[0].strip() if text.strip() else ""
    if not text:
        return ""
    bad_fragments = ["You are a Lean", "Given a goal state", "⊢", "User:", "Assistant:"]
    if any(fragment in text for fragment in bad_fragments):
        return ""
    return text


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def generate_tactics(
    model,
    tokenizer,
    goal_text: str,
    args: argparse.Namespace,
) -> list[str]:
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
        "num_return_sequences": args.num_tactics,
        "repetition_penalty": args.repetition_penalty,
        "stopping_criteria": stopping,
    }
    if args.temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    else:
        generation_kwargs["do_sample"] = False
        generation_kwargs["num_beams"] = args.num_tactics

    with torch.no_grad():
        output = model.generate(**generation_kwargs)

    tactics = []
    prompt_length = inputs["input_ids"].shape[1]
    for sequence in output:
        text = tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)
        tactic = cleanup_tactic(text)
        if tactic:
            tactics.append(tactic)
    return dedupe_keep_order(tactics)


def serialize_goals(goal_state: Any) -> tuple[str, ...]:
    return tuple(str(goal) for goal in goal_state.goals)


def bfs_prove(server: Server, model, tokenizer, question: str, args: argparse.Namespace) -> dict[str, Any]:
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
        model_tactics = generate_tactics(model, tokenizer, goal_text, args)
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


def main() -> None:
    args = parse_args()
    args.model_root = args.model_root.resolve()
    args.project_dir = args.project_dir.resolve()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    tokenizer_root = resolve_model_dir((args.tokenizer_root or args.model_root).resolve())
    model_root = resolve_model_dir(args.model_root)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_root, dtype=torch.float16).cuda().eval()

    rows = iter_jsonl(args.input)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Server(imports=["Mathlib"], project_path=str(args.project_dir)) as server:
        with args.output.open("w", encoding="utf-8") as output_file:
            for row in rows:
                result = bfs_prove(server, model, tokenizer, row["question"], args)
                payload = {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": row.get("answer"),
                    "raw_output": result["proof"],
                    "predictions": [result["proof"]],
                    "search": {
                        "solved": result["solved"],
                        "expanded_nodes": result["expanded_nodes"],
                        "visited_states": result["visited_states"],
                        "failure": result["failure"],
                        "explored_tactics": result["explored_tactics"],
                    },
                }
                output_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"output={args.output}")
    print(f"samples={len(rows)}")


if __name__ == "__main__":
    main()
