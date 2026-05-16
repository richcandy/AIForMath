import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import openai
from openai import OpenAI
from pantograph.server import Server, ServerError, TacticFailure


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "Deepseek_highschool_data" / "test.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "evaluation" / "deepseek_vllm_tactic_execution_16.jsonl"
DEFAULT_API_BASE = "http://127.0.0.1:11451/v1"
DEFAULT_MODEL_NAME = "my-tactic-lora"
TACTIC_SYSTEM_PROMPT = (
    "You are a Lean 4 tactic generator. Given a goal state, output exactly ONE Lean tactic that advances or solves the goal.\n"
    "Rules:\n"
    "- Output only the tactic text; no prose or code fences.\n"
    "- Single line only; no `by` blocks.\n"
    "- Never use `sorry` or `admit`.\n"
)
STOP_MARKERS = ["\n", "```", "User:", "Assistant:"]
BAD_TACTIC_FRAGMENTS = ["You are a Lean", "Given a goal state", "User:", "Assistant:", "⊢"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether generated single-step tactics execute and advance proof states in Lean."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--project-dir", type=Path, default=ROOT)
    parser.add_argument("--source-root", type=Path, default=ROOT / ".lake" / "packages" / "mathlib")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-return-sequences", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def flatten_traced_tactics(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")

    rows = []
    for theorem_index, theorem in enumerate(data):
        theorem_id = theorem.get("full_name") or theorem.get("file_path") or f"theorem-{theorem_index:05d}"
        for tactic_index, tactic in enumerate(theorem.get("traced_tactics", [])):
            rows.append(
                {
                    "id": f"{theorem_id}#step-{tactic_index:03d}",
                    "theorem_id": theorem_id,
                    "step_index": tactic_index,
                    "file_path": theorem.get("file_path"),
                    "start": theorem.get("start"),
                    "state_before": str(tactic.get("state_before") or "").strip(),
                    "gold_tactic": str(tactic.get("tactic") or "").strip(),
                    "state_after": str(tactic.get("state_after") or "").strip(),
                }
            )
    return rows


def cleanup_tactic(text: str) -> str:
    cleaned = text.replace("Ġ", " ").replace("Ċ", "\n").strip()
    for marker in STOP_MARKERS:
        index = cleaned.find(marker)
        if index != -1:
            cleaned = cleaned[:index].strip()
    cleaned = cleaned.strip().strip("`")
    cleaned = cleaned.replace("Assistant:", "").replace("assistant:", "").strip()
    if cleaned.startswith("by"):
        cleaned = cleaned[2:].strip()
    cleaned = cleaned.replace("<a>", "").replace("</a>", "")
    cleaned = cleaned.splitlines()[0].strip() if cleaned.strip() else ""
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if any(fragment in cleaned for fragment in BAD_TACTIC_FRAGMENTS):
        return ""
    return cleaned


def normalize_lean_notation(text: str) -> str:
    normalized = text
    normalized = re.sub(r"\b([A-Za-z0-9_'.✝]+)\[X\]", r"Polynomial \1", normalized)
    normalized = re.sub(r"\b([A-Za-z0-9_'.✝]+)\[Y\]", r"Polynomial \1", normalized)
    return normalized


def normalize_goal_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "no goals"
    return re.sub(r"\s+", " ", stripped)


def extract_file_prefix_context(file_path: Path, theorem_start_line: int) -> list[str]:
    lines = file_path.read_text(encoding="utf-8").splitlines()
    context = []
    in_continuation = False
    for line in lines[: theorem_start_line - 1]:
        stripped = line.lstrip()
        if in_continuation:
            if not stripped:
                context.append(line)
                in_continuation = False
                continue
            if line.startswith(" ") or line.startswith("\t"):
                context.append(line)
                continue
            in_continuation = False

        keep_line = False
        if stripped.startswith((
            "open ",
            "open scoped ",
            "scoped ",
            "set_option ",
            "attribute ",
            "local notation",
            "notation ",
            "infix",
            "prefix",
            "postfix",
            "variable ",
            "variables ",
        )):
            keep_line = True

        if keep_line:
            context.append(line)
            if stripped.startswith(("variable ", "variables ")) and not stripped.rstrip().endswith("]"):
                in_continuation = True
    return context


def state_to_text(goal_state: Any) -> str:
    if len(goal_state.goals) == 0:
        return "no goals"
    return normalize_goal_text("\n\n".join(str(goal) for goal in goal_state.goals))


def state_to_sorry_code(state_before: str, file_path: Path | None = None, theorem_start_line: int | None = None) -> str:
    state_before = normalize_lean_notation(state_before)
    context_lines = []
    goal_lines = []
    seen_goal = False
    universe_names = []
    for line in state_before.splitlines():
        if line.startswith("⊢"):
            seen_goal = True
            goal_lines.append(line[1:].strip())
            continue
        if seen_goal:
            goal_lines.append(line.rstrip())
        else:
            context_lines.append(line.rstrip())
            match = re.search(r"Type\s+([A-Za-z0-9_]+)", line)
            if match:
                universe_names.append(match.group(1))

    goal = "\n".join(goal_lines).strip()
    if not goal:
        raise ValueError("Missing goal in state_before")

    scoped_opens = []
    full_text = "\n".join(context_lines + [goal])
    if "[X]" in full_text or ".coeff" in full_text or ".natDegree" in full_text:
        scoped_opens.append("open scoped Polynomial")
    if "∑" in full_text or "∏" in full_text:
        scoped_opens.append("open scoped BigOperators")

    binders = []
    for line in context_lines:
        if not line.strip():
            continue
        name, _, typ = line.partition(":")
        name = name.strip()
        typ = typ.strip()
        if not name or not typ:
            continue
        if name.startswith("inst"):
            binders.append(f"[{typ}]")
        else:
            binders.append(f"({name} : {typ})")

    header_parts = []
    if universe_names:
        header_parts.append(f"universe {' '.join(dict.fromkeys(universe_names))}")
    if file_path is not None and theorem_start_line is not None:
        header_parts.extend(extract_file_prefix_context(file_path, theorem_start_line))
    header_parts.extend(scoped_opens)
    binder_text = " ".join(binders)
    header_parts.append(f"theorem synthetic_goal {binder_text} : {goal} := by\n  sorry")
    return "\n".join(header_parts) + "\n"


def build_client(args: argparse.Namespace) -> OpenAI:
    return OpenAI(api_key=args.api_key, base_url=args.api_base, timeout=args.request_timeout)


def resolve_model_name(client: OpenAI, requested_model_name: str | None) -> str:
    if requested_model_name:
        return requested_model_name
    models = client.models.list().data
    if not models:
        raise ValueError("No served models returned by /v1/models")
    return models[0].id


def generate_tactics(client: OpenAI, model_name: str, goal_text: str, args: argparse.Namespace) -> list[str]:
    messages = [
        {"role": "system", "content": TACTIC_SYSTEM_PROMPT},
        {"role": "user", "content": goal_text},
    ]
    extra_body: dict[str, Any] = {"repetition_penalty": args.repetition_penalty}
    request_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": args.max_new_tokens,
        "n": args.num_return_sequences,
        "stop": STOP_MARKERS,
        "temperature": args.temperature,
        "extra_body": extra_body,
    }
    if args.temperature > 0:
        request_kwargs["top_p"] = args.top_p
    elif args.num_return_sequences > 1:
        extra_body["use_beam_search"] = True

    response = client.chat.completions.create(**request_kwargs)

    predictions = []
    seen = set()
    for choice in response.choices:
        tactic = cleanup_tactic(choice.message.content or "")
        if not tactic or tactic in seen:
            continue
        seen.add(tactic)
        predictions.append(tactic)
    return predictions


def build_empty_result(row: dict[str, Any], gold_tactic: str, gold_state_after: str, prior_states: set[str], error: str) -> dict[str, Any]:
    return {
        **row,
        "gold_tactic": gold_tactic,
        "gold_state_after": gold_state_after,
        "prior_theorem_state_count": len(prior_states),
        "state_before_matches_trace": False,
        "predictions": [],
        "top1_exact": False,
        "topk_exact": False,
        "top1_executable": False,
        "topk_executable": False,
        "top1_state_match": False,
        "topk_state_match": False,
        "top1_closed": False,
        "topk_closed": False,
        "top1_same_as_input_state": False,
        "topk_same_as_input_state": False,
        "top1_revisited_prior_theorem_state": False,
        "topk_revisited_prior_theorem_state": False,
        "top1_progressed": False,
        "topk_progressed": False,
        "compiles_state": False,
        "compile_error": error,
        "compile_attempt_errors": [error],
        "generation_error": None,
        "execution_records": [],
    }


def load_goal_targets(
    server: Server,
    state_before: str,
    source_file: Path | None,
    theorem_start_line: int | None,
):
    compile_errors = []
    code_variants = [state_to_sorry_code(state_before, source_file, theorem_start_line)]
    if source_file is not None and theorem_start_line is not None:
        code_variants.append(state_to_sorry_code(state_before))

    last_error = None
    for code in code_variants:
        try:
            targets = server.load_sorry(code)
            return targets, compile_errors
        except (ServerError, RuntimeError, ValueError) as exc:
            last_error = exc
            compile_errors.append(str(exc))

    if last_error is None:
        raise ValueError("No synthetic theorem variants generated")
    raise last_error


def evaluate_step(
    server: Server,
    client: OpenAI,
    model_name: str,
    row: dict[str, Any],
    prior_states: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    state_before = str(row["state_before"])
    gold_tactic = cleanup_tactic(str(row["gold_tactic"]))
    gold_state_after = normalize_goal_text(str(row["state_after"]))
    source_file = args.source_root / str(row["file_path"]) if row.get("file_path") else None
    theorem_start_line = row.get("start", [None, None])[0]
    try:
        targets, compile_errors = load_goal_targets(
            server, state_before, source_file, theorem_start_line
        )
    except (ServerError, RuntimeError, ValueError) as exc:
        return build_empty_result(row, gold_tactic, gold_state_after, prior_states, str(exc))
    if not targets:
        return build_empty_result(row, gold_tactic, gold_state_after, prior_states, "no_targets")

    goal_state = targets[0].goal_state
    current_state_text = state_to_text(goal_state)

    generation_error = None
    try:
        predictions = generate_tactics(client, model_name, current_state_text, args)
    except (openai.APIError, openai.APIConnectionError, openai.APITimeoutError) as exc:
        generation_error = str(exc)
        predictions = []

    execution_records = []
    top1_exact = bool(predictions and predictions[0] == gold_tactic)
    topk_exact = gold_tactic in predictions
    top1_executable = False
    topk_executable = False
    top1_state_match = False
    topk_state_match = False
    top1_closed = False
    topk_closed = False
    top1_same_as_input_state = False
    topk_same_as_input_state = False
    top1_revisited_prior_theorem_state = False
    topk_revisited_prior_theorem_state = False
    top1_progressed = False
    topk_progressed = False

    for pred_index, tactic in enumerate(predictions):
        try:
            next_state = server.goal_tactic(goal_state, tactic)
            after_text = state_to_text(next_state)
            executable = True
            state_match = after_text == gold_state_after
            closed = len(next_state.goals) == 0
            same_as_input_state = after_text == current_state_text
            revisited_prior_theorem_state = after_text in prior_states
            progressed = not same_as_input_state and not revisited_prior_theorem_state
        except (TacticFailure, ServerError) as exc:
            executable = False
            state_match = False
            closed = False
            same_as_input_state = False
            revisited_prior_theorem_state = False
            progressed = False
            after_text = str(exc)

        execution_records.append(
            {
                "rank": pred_index + 1,
                "tactic": tactic,
                "executable": executable,
                "state_match": state_match,
                "closed_goal": closed,
                "same_as_input_state": same_as_input_state,
                "revisited_prior_theorem_state": revisited_prior_theorem_state,
                "progressed": progressed,
                "result": after_text,
            }
        )
        if pred_index == 0:
            top1_executable = executable
            top1_state_match = state_match
            top1_closed = closed
            top1_same_as_input_state = same_as_input_state
            top1_revisited_prior_theorem_state = revisited_prior_theorem_state
            top1_progressed = progressed
        topk_executable = topk_executable or executable
        topk_state_match = topk_state_match or state_match
        topk_closed = topk_closed or closed
        topk_same_as_input_state = topk_same_as_input_state or same_as_input_state
        topk_revisited_prior_theorem_state = (
            topk_revisited_prior_theorem_state or revisited_prior_theorem_state
        )
        topk_progressed = topk_progressed or progressed

    return {
        **row,
        "gold_tactic": gold_tactic,
        "gold_state_after": gold_state_after,
        "prior_theorem_state_count": len(prior_states),
        "compiles_state": True,
        "compile_error": None,
        "compile_attempt_errors": compile_errors,
        "generation_error": generation_error,
        "state_before_matches_trace": current_state_text == normalize_goal_text(state_before),
        "predictions": predictions,
        "top1_exact": top1_exact,
        "topk_exact": topk_exact,
        "top1_executable": top1_executable,
        "topk_executable": topk_executable,
        "top1_state_match": top1_state_match,
        "topk_state_match": topk_state_match,
        "top1_closed": top1_closed,
        "topk_closed": topk_closed,
        "top1_same_as_input_state": top1_same_as_input_state,
        "topk_same_as_input_state": topk_same_as_input_state,
        "top1_revisited_prior_theorem_state": top1_revisited_prior_theorem_state,
        "topk_revisited_prior_theorem_state": topk_revisited_prior_theorem_state,
        "top1_progressed": top1_progressed,
        "topk_progressed": topk_progressed,
        "execution_records": execution_records,
    }


def update_prior_states(seen_states_by_theorem: dict[str, set[str]], row: dict[str, Any]) -> None:
    theorem_id = str(row["theorem_id"])
    theorem_states = seen_states_by_theorem.setdefault(theorem_id, set())
    theorem_states.add(normalize_goal_text(str(row["state_before"])))
    theorem_states.add(normalize_goal_text(str(row["state_after"])))


def metric_rate(results: list[dict[str, Any]], key: str) -> float:
    total = len(results)
    if not total:
        return 0.0
    return round(sum(bool(row[key]) for row in results) / total, 4)


def print_progress(processed: int, total: int, results: list[dict[str, Any]], start_time: float) -> None:
    elapsed = max(time.monotonic() - start_time, 1e-9)
    compile_rate = sum(bool(row["compiles_state"]) for row in results) / processed
    gen_rate = sum(row["generation_error"] is None for row in results) / processed
    exec_rate = sum(bool(row["top1_executable"]) for row in results) / processed
    progress_rate = sum(bool(row["top1_progressed"]) for row in results) / processed
    speed = processed / elapsed
    print(
        "processed="
        f"{processed}/{total} "
        f"compile={compile_rate:.4f} "
        f"gen={gen_rate:.4f} "
        f"top1_exec={exec_rate:.4f} "
        f"top1_progress={progress_rate:.4f} "
        f"steps_per_sec={speed:.2f}"
    )


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.project_dir = args.project_dir.resolve()
    args.source_root = args.source_root.resolve()
    summary_path = args.summary_path.resolve() if args.summary_path else args.output.with_suffix(".summary.json")

    rows = flatten_traced_tactics(args.input)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    client = build_client(args)
    model_name = resolve_model_name(client, args.model_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    seen_states_by_theorem: dict[str, set[str]] = {}
    total = len(rows)
    start_time = time.monotonic()
    print(
        f"starting total_steps={total} model={model_name} num_return_sequences={args.num_return_sequences}"
    )
    with args.output.open("w", encoding="utf-8") as output_file:
        with Server(imports=["Mathlib"], project_path=str(args.project_dir)) as server:
            for index, row in enumerate(rows, start=1):
                prior_states = set(seen_states_by_theorem.get(str(row["theorem_id"]), set()))
                result = evaluate_step(server, client, model_name, row, prior_states, args)
                results.append(result)
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                update_prior_states(seen_states_by_theorem, row)

                if args.progress_every > 0 and (index % args.progress_every == 0 or index == total):
                    output_file.flush()
                    print_progress(index, total, results, start_time)

    total = len(results)
    summary = {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "api_base": args.api_base,
        "model_name": model_name,
        "total_steps": total,
        "generation_success_rate": round(
            sum(row["generation_error"] is None for row in results) / total, 4
        )
        if total
        else 0.0,
        "compiles_state_rate": metric_rate(results, "compiles_state"),
        "top1_exact_accuracy": metric_rate(results, "top1_exact"),
        "topk_exact_accuracy": metric_rate(results, "topk_exact"),
        "top1_executable_rate": metric_rate(results, "top1_executable"),
        "topk_executable_rate": metric_rate(results, "topk_executable"),
        "top1_state_match_rate": metric_rate(results, "top1_state_match"),
        "topk_state_match_rate": metric_rate(results, "topk_state_match"),
        "top1_same_as_input_state_rate": metric_rate(results, "top1_same_as_input_state"),
        "topk_same_as_input_state_rate": metric_rate(results, "topk_same_as_input_state"),
        "top1_revisited_prior_theorem_state_rate": metric_rate(
            results, "top1_revisited_prior_theorem_state"
        ),
        "topk_revisited_prior_theorem_state_rate": metric_rate(
            results, "topk_revisited_prior_theorem_state"
        ),
        "top1_progress_rate": metric_rate(results, "top1_progressed"),
        "topk_progress_rate": metric_rate(results, "topk_progressed"),
        "top1_closed_rate": metric_rate(results, "top1_closed"),
        "topk_closed_rate": metric_rate(results, "topk_closed"),
        "state_alignment_rate": metric_rate(results, "state_before_matches_trace"),
        "num_return_sequences": args.num_return_sequences,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results={args.output}")
    print(f"summary={summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
