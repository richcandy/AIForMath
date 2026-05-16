import argparse
import importlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from pantograph.server import Server, ServerError, TacticFailure
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "Deepseek_highschool_data" / "test.json"
DEFAULT_MODEL_ROOT = ROOT / "Deepseek-Prover-V2"
DEFAULT_ADAPTER_ROOT = ROOT / "outputs" / "deepseekprover_v2_highschool_tactic" / "checkpoint-384"
DEFAULT_OUTPUT = ROOT / "outputs" / "evaluation" / "deepseek_highschool_test_pass32_ckpt384.jsonl"
DEFAULT_SOURCE_ROOT = ROOT / ".lake" / "packages" / "mathlib"
TACTIC_SYSTEM_PROMPT = (
    "You are a Lean 4 tactic generator. Given a goal state, output exactly ONE Lean tactic that advances or solves the goal.\n"
    "Rules:\n"
    "- Output only the tactic text; no prose or code fences.\n"
    "- Single line only; no `by` blocks.\n"
    "- Never use `sorry` or `admit`.\n"
)
STOP_MARKERS = ["\n", "```", "User:", "Assistant:"]
BAD_TACTIC_FRAGMENTS = ["You are a Lean", "Given a goal state", "User:", "Assistant:"]


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
        description="Evaluate single-step Lean tactic pass@k using batched generation and Pantograph execution."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--project-dir", type=Path, default=ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-return-sequences", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-goal-tokens", type=int, default=2800)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--oversample-factor", type=int, default=2)
    parser.add_argument("--max-generation-rounds", type=int, default=3)
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        default="auto",
    )
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
                    "full_name": theorem.get("full_name"),
                    "commit": theorem.get("commit"),
                    "file_path": theorem.get("file_path"),
                    "start": theorem.get("start"),
                    "theorem_statement": theorem_statement,
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
    cleaned = cleaned.splitlines()[0].strip() if cleaned else ""
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
        if stripped.startswith(
            (
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
            )
        ):
            keep_line = True

        if keep_line:
            context.append(line)
            if stripped.startswith(("variable ", "variables ")) and not stripped.rstrip().endswith("]"
            ):
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


def resolve_attention_implementation(requested: str, use_cuda: bool) -> str:
    if not use_cuda:
        return "eager"
    if requested != "auto":
        return requested
    try:
        importlib.import_module("flash_attn")
    except Exception:
        return "sdpa"
    return "flash_attention_2"


def load_model_and_tokenizer(model_root: Path, adapter_root: Path | None, args: argparse.Namespace):
    resolved_model = resolve_model_dir(model_root)
    config = AutoConfig.from_pretrained(resolved_model)
    tokenizer_root = adapter_root if adapter_root and (adapter_root / "tokenizer.json").exists() else resolved_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    torch_dtype_name = str(getattr(config, "torch_dtype", "float16"))
    use_cuda = torch.cuda.is_available()
    if use_cuda and ("bfloat16" in torch_dtype_name or torch.cuda.is_bf16_supported()):
        torch_dtype = torch.bfloat16
    elif use_cuda:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "attn_implementation": resolve_attention_implementation(args.attn_implementation, use_cuda),
    }
    if use_cuda:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(resolved_model, **model_kwargs)
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


def decode_unique_predictions(tokenizer, output, prompt_length: int, seen: set[str]) -> list[str]:
    predictions = []
    for sequence in output:
        decoded = tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)
        tactic = cleanup_tactic(decoded)
        if not tactic or tactic in seen:
            continue
        seen.add(tactic)
        predictions.append(tactic)
    return predictions


def generate_tactics(model, tokenizer, goal_text: str, args: argparse.Namespace, seed: int) -> list[str]:
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
    seen: set[str] = set()
    predictions: list[str] = []
    fork_devices = [model.device] if getattr(model.device, "type", None) == "cuda" else []

    if args.num_return_sequences <= 1:
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed)
            with torch.no_grad():
                if args.temperature > 0:
                    output = model.generate(
                        **generation_kwargs,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=1,
                        num_return_sequences=1,
                    )
                else:
                    output = model.generate(
                        **generation_kwargs,
                        do_sample=False,
                        num_beams=1,
                        num_return_sequences=1,
                    )
        return decode_unique_predictions(tokenizer, output, prompt_length, seen)[:1]

    for round_index in range(args.max_generation_rounds):
        needed = args.num_return_sequences - len(predictions)
        if needed <= 0:
            break
        batch_size = max(needed * args.oversample_factor, needed)

        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed + round_index)
            with torch.no_grad():
                if args.temperature > 0:
                    output = model.generate(
                        **generation_kwargs,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_return_sequences=batch_size,
                    )
                else:
                    output = model.generate(
                        **generation_kwargs,
                        do_sample=False,
                        num_beams=batch_size,
                        num_return_sequences=batch_size,
                    )

        predictions.extend(decode_unique_predictions(tokenizer, output, prompt_length, seen))

    return predictions[: args.num_return_sequences]


def evaluate_step(server: Server, model, tokenizer, row: dict[str, Any], args: argparse.Namespace, step_seed: int) -> dict[str, Any]:
    state_before = str(row["state_before"])
    source_file = args.source_root / str(row["file_path"]) if row.get("file_path") else None
    theorem_start_line = row.get("start", [None, None])[0]
    try:
        targets = server.load_sorry(state_to_sorry_code(state_before, source_file, theorem_start_line))
    except (ServerError, RuntimeError, ValueError) as exc:
        return {
            **row,
            "goal_text": normalize_goal_text(state_before),
            "predictions": [],
            "compile_error": str(exc),
            "compiles_state": False,
            "state_before_matches_trace": False,
            "top1_pass": False,
            "pass_at_k": False,
            "top1_finished": False,
            "pass_at_k_finished": False,
            "top1_executable": False,
            "pass_at_k_executable": False,
            "unique_predictions": 0,
            "execution_records": [],
        }
    if not targets:
        return {
            **row,
            "goal_text": normalize_goal_text(state_before),
            "predictions": [],
            "compile_error": "no_targets",
            "compiles_state": False,
            "state_before_matches_trace": False,
            "top1_pass": False,
            "pass_at_k": False,
            "top1_finished": False,
            "pass_at_k_finished": False,
            "top1_executable": False,
            "pass_at_k_executable": False,
            "unique_predictions": 0,
            "execution_records": [],
        }

    goal_state = targets[0].goal_state
    current_state_text = state_to_text(goal_state)
    predictions = generate_tactics(model, tokenizer, current_state_text, args, step_seed)

    execution_records = []
    top1_pass = False
    pass_at_k = False
    top1_finished = False
    pass_at_k_finished = False
    top1_executable = False
    pass_at_k_executable = False

    for pred_index, tactic in enumerate(predictions):
        try:
            next_state = server.goal_tactic(goal_state, tactic)
            after_text = state_to_text(next_state)
            executable = True
            closed = len(next_state.goals) == 0
            progressed = closed or after_text != current_state_text
        except (TacticFailure, ServerError) as exc:
            executable = False
            closed = False
            progressed = False
            after_text = str(exc)

        execution_records.append(
            {
                "rank": pred_index + 1,
                "tactic": tactic,
                "executable": executable,
                "progressed": progressed,
                "closed_goal": closed,
                "result": after_text,
            }
        )

        if pred_index == 0:
            top1_executable = executable
            top1_pass = executable and progressed
            top1_finished = executable and closed

        pass_at_k_executable = pass_at_k_executable or executable
        pass_at_k = pass_at_k or (executable and progressed)
        pass_at_k_finished = pass_at_k_finished or (executable and closed)

    return {
        **row,
        "goal_text": current_state_text,
        "predictions": predictions,
        "compile_error": None,
        "compiles_state": True,
        "state_before_matches_trace": current_state_text == normalize_goal_text(state_before),
        "top1_pass": top1_pass,
        "pass_at_k": pass_at_k,
        "top1_finished": top1_finished,
        "pass_at_k_finished": pass_at_k_finished,
        "top1_executable": top1_executable,
        "pass_at_k_executable": pass_at_k_executable,
        "unique_predictions": len(predictions),
        "execution_records": execution_records,
    }


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.model_root = args.model_root.resolve()
    args.project_dir = args.project_dir.resolve()
    args.source_root = args.source_root.resolve()
    if args.adapter_root is not None:
        args.adapter_root = args.adapter_root.resolve()
    summary_path = args.summary_path.resolve() if args.summary_path else args.output.with_suffix(".summary.json")

    rows = flatten_traced_tactics(args.input)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    model, tokenizer = load_model_and_tokenizer(args.model_root, args.adapter_root, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with Server(imports=["Mathlib"], project_path=str(args.project_dir)) as server:
        for index, row in enumerate(rows):
            results.append(evaluate_step(server, model, tokenizer, row, args, args.sample_seed + index))

    with args.output.open("w", encoding="utf-8") as output_file:
        for row in results:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(results)
    summary = {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "model_root": str(args.model_root),
        "adapter_root": str(args.adapter_root) if args.adapter_root else None,
        "total_steps": total,
        "compiles_state_rate": round(sum(r["compiles_state"] for r in results) / total, 4) if total else 0.0,
        "top1_pass_rate": round(sum(r["top1_pass"] for r in results) / total, 4) if total else 0.0,
        "pass_at_k": round(sum(r["pass_at_k"] for r in results) / total, 4) if total else 0.0,
        "top1_finish_rate": round(sum(r["top1_finished"] for r in results) / total, 4) if total else 0.0,
        "pass_at_k_finish_rate": round(sum(r["pass_at_k_finished"] for r in results) / total, 4) if total else 0.0,
        "top1_executable_rate": round(sum(r["top1_executable"] for r in results) / total, 4) if total else 0.0,
        "pass_at_k_executable_rate": round(sum(r["pass_at_k_executable"] for r in results) / total, 4) if total else 0.0,
        "state_alignment_rate": round(sum(r["state_before_matches_trace"] for r in results) / total, 4) if total else 0.0,
        "nonempty_prediction_rate": round(sum(r["unique_predictions"] > 0 for r in results) / total, 4) if total else 0.0,
        "avg_unique_predictions": round(sum(r["unique_predictions"] for r in results) / total, 2) if total else 0.0,
        "num_return_sequences": args.num_return_sequences,
        "max_new_tokens": args.max_new_tokens,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results={args.output}")
    print(f"summary={summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
