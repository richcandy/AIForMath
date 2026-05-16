import asyncio
import argparse
import json
import multiprocessing as mp
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import openai
from openai import AsyncOpenAI, OpenAI
from pantograph.server import Server, ServerError, TacticFailure, get_lean_path_async


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT / "MiniF2F_Highschool"
DEFAULT_OUTPUT = ROOT / "formal_results_storage" / "minif2f_highschool_beam_vllm_new_para2.jsonl"
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
KILL_ONLY_HAMMERS = {"linarith", "nlinarith", "ring", "aesop"}
RETRIABLE_SERVER_EXCEPTIONS = (ServerError, RuntimeError, BrokenPipeError, OSError, EOFError)
REWRITE_PREFIXES = ("rw", "erw", "rwa", "simp_rw", "nth_rewrite")

WORKER_SERVER: Server | None = None
WORKER_SERVER_CONFIG: dict[str, Any] = {}


SearchEventCallback = Callable[[dict[str, Any]], None]


@dataclass
class SearchNode:
    proof_steps: list[str]
    goal_text: str
    state_signature: tuple[str, ...]
    depth: int
    votes: int = 0
    best_rank: int = 0


@dataclass
class CandidateState:
    proof_steps: list[str]
    goal_text: str
    state_signature: tuple[str, ...]
    depth: int
    votes: int
    best_rank: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MiniF2F_Highschool with vLLM-guided beam search and parallel Lean verification."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--split", choices=["valid", "test", "both"], default="test")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--project-dir", type=Path, default=ROOT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--problem",
        action="append",
        default=[],
        help="Run only the specified problem stem(s) or filenames. Can be repeated.",
    )
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=21)
    parser.add_argument("--num-tactics", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-goal-chars", type=int, default=12000)
    parser.add_argument("--lean-workers", type=int, default=6)
    parser.add_argument("--time-budget-seconds", type=int, default=480)
    parser.add_argument("--max-expanded-nodes", type=int, default=128)
    parser.add_argument("--max-verified-tactics", type=int, default=1280)
    parser.add_argument("--server-timeout", type=int, default=60)
    parser.add_argument("--max-hammers-per-node", type=int, default=3)
    return parser.parse_args()


def build_client(args: argparse.Namespace) -> OpenAI:
    return OpenAI(api_key=args.api_key, base_url=args.api_base, timeout=args.request_timeout)


def resolve_model_name(client: OpenAI, requested_model_name: str | None) -> str:
    if requested_model_name:
        return requested_model_name
    models = client.models.list().data
    if not models:
        raise ValueError("No served models returned by /v1/models")
    return models[0].id


def normalize_goal_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "no goals"
    return re.sub(r"\s+", " ", stripped)


def cleanup_tactic(text: str) -> str:
    cleaned = text.replace("Ġ", " ").replace("Ċ", "\n").strip()
    for marker in STOP_MARKERS:
        index = cleaned.find(marker)
        if index != -1:
            cleaned = cleaned[:index].strip()
    cleaned = re.sub(r"--.*$", "", cleaned, flags=re.M).strip()
    cleaned = cleaned.strip().strip("`")
    cleaned = cleaned.replace("Assistant:", "").replace("assistant:", "").strip()
    if cleaned.startswith("by"):
        cleaned = cleaned[2:].strip()
    cleaned = cleaned.replace("<a>", "").replace("</a>", "")
    cleaned = cleaned.splitlines()[0].strip() if cleaned.strip() else ""
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if any(fragment in cleaned for fragment in BAD_TACTIC_FRAGMENTS):
        return ""
    if not tactic_looks_safe(cleaned):
        return ""
    return cleaned


def delimiters_balanced(text: str) -> bool:
    opening = {"(": ")", "[": "]", "{": "}", "⟨": "⟩"}
    closing = {value: key for key, value in opening.items()}
    stack: list[str] = []
    for char in text:
        if char in opening:
            stack.append(char)
        elif char in closing:
            if not stack or stack[-1] != closing[char]:
                return False
            stack.pop()
    return not stack


def tactic_looks_safe(tactic: str) -> bool:
    stripped = tactic.strip()
    if not stripped:
        return False
    if not delimiters_balanced(stripped):
        return False
    if stripped.endswith(("[", "(", "{", ",", ":", "⟨")):
        return False
    if stripped.endswith(" at"):
        return False

    lowered = stripped.lower()
    for prefix in REWRITE_PREFIXES:
        if not lowered.startswith(prefix):
            continue
        remainder = stripped[len(prefix) :].strip()
        if not remainder:
            return False
        if remainder in {"[", "(", "{"}:
            return False
        if remainder.endswith(("[", "(", "{", ",", ":")):
            return False
        break
    return True


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def dedupe_ranked_tactics(
    ranked_tactics: list[tuple[int, str, str]]
) -> list[tuple[int, str, str]]:
    seen = set()
    ordered = []
    for rank, tactic, source in ranked_tactics:
        if not tactic or tactic in seen:
            continue
        seen.add(tactic)
        ordered.append((rank, tactic, source))
    return ordered


def truncate_goal_text(goal_text: str, max_goal_chars: int) -> str:
    if max_goal_chars <= 0 or len(goal_text) <= max_goal_chars:
        return goal_text
    return goal_text[-max_goal_chars:].strip()


def get_smart_hammers(goal_text: str) -> list[str]:
    hammers = ["norm_num"]
    if any(op in goal_text for op in ["<", ">", "≤", "≥"]):
        hammers.extend(["linarith", "nlinarith"])
    if "*" in goal_text or "^" in goal_text:
        hammers.append("ring")
    hammers.append("aesop")
    return dedupe_keep_order(hammers)


def extract_question_from_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    updated = re.sub(r":=\s*by\s*(?:sorry|admit)\s*$", ":= by", text, count=1, flags=re.S)
    if updated == text:
        raise ValueError(f"Could not locate trailing ':= by sorry/admit' in {path}")
    return updated.rstrip()


def collect_problem_paths(input_root: Path, split: str) -> list[Path]:
    split_map = {"valid": "Valid", "test": "Test"}
    splits = [split_map[split]] if split != "both" else ["Valid", "Test"]
    paths = []
    for split_name in splits:
        paths.extend(sorted((input_root / split_name).glob("*.lean")))
    return paths


def filter_problem_paths(problem_paths: list[Path], requested: list[str]) -> list[Path]:
    if not requested:
        return problem_paths

    requested_names = {item.strip() for item in requested if item.strip()}
    requested_stems = {Path(item).stem for item in requested_names}
    filtered = [
        path
        for path in problem_paths
        if path.name in requested_names or path.stem in requested_stems
    ]
    missing = sorted(
        name
        for name in requested_names
        if not any(path.name == name or path.stem == Path(name).stem for path in filtered)
    )
    if missing:
        raise ValueError(f"Requested problems not found: {', '.join(missing)}")
    return filtered


def build_search_target(question: str) -> str:
    lines = question.rstrip().splitlines()
    while lines and lines[0].startswith("import "):
        lines.pop(0)
    return f"{'\n'.join(lines).rstrip()}\n  sorry\n"


def goal_state_signature(goal_state: Any) -> tuple[str, ...]:
    if len(goal_state.goals) == 0:
        return ("no goals",)
    return tuple(str(goal) for goal in goal_state.goals)


def goal_signature_to_text(state_signature: tuple[str, ...]) -> str:
    if not state_signature or state_signature == ("no goals",):
        return "no goals"
    return normalize_goal_text("\n\n".join(state_signature))


def build_problem_id(problem_path: Path, input_root: Path) -> str:
    split_name = problem_path.parent.name.lower()
    return f"{split_name}/{problem_path.stem}"


def build_generation_request(goal_text: str, args: argparse.Namespace, model_name: str) -> dict[str, Any]:
    truncated_goal = truncate_goal_text(goal_text, args.max_goal_chars)
    extra_body: dict[str, Any] = {"repetition_penalty": args.repetition_penalty}
    request_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": TACTIC_SYSTEM_PROMPT},
            {"role": "user", "content": truncated_goal},
        ],
        "max_tokens": args.max_new_tokens,
        "n": args.num_tactics,
        "stop": STOP_MARKERS,
        "temperature": args.temperature,
        "extra_body": extra_body,
    }
    if args.temperature > 0:
        request_kwargs["top_p"] = args.top_p
    elif args.num_tactics > 1:
        extra_body["use_beam_search"] = True
    return request_kwargs


async def generate_tactics(
    client: AsyncOpenAI, goal_text: str, args: argparse.Namespace, model_name: str
) -> list[str]:
    response = await client.chat.completions.create(
        **build_generation_request(goal_text, args, model_name)
    )
    predictions = []
    seen = set()
    for choice in response.choices:
        tactic = cleanup_tactic(choice.message.content or "")
        if not tactic or tactic in seen:
            continue
        seen.add(tactic)
        predictions.append(tactic)
    return predictions


def init_lean_worker(project_dir: str, lean_path: str, timeout: int) -> None:
    global WORKER_SERVER_CONFIG
    WORKER_SERVER_CONFIG = {
        "project_dir": project_dir,
        "lean_path": lean_path,
        "timeout": timeout,
    }
    restart_worker_server(force_recreate=True)


def worker_server_alive() -> bool:
    if WORKER_SERVER is None:
        return False
    proc = getattr(WORKER_SERVER, "proc", None)
    return proc is not None and getattr(proc, "returncode", None) is None


def restart_worker_server(force_recreate: bool = False) -> None:
    global WORKER_SERVER
    if not WORKER_SERVER_CONFIG:
        raise RuntimeError("Worker server config is not initialized")

    if WORKER_SERVER is not None:
        try:
            if not force_recreate:
                WORKER_SERVER.restart()
                return
        except Exception:
            pass
        try:
            WORKER_SERVER._close()
        except Exception:
            pass

    WORKER_SERVER = Server(
        imports=["Mathlib"],
        project_path=WORKER_SERVER_CONFIG["project_dir"],
        lean_path=WORKER_SERVER_CONFIG["lean_path"],
        timeout=WORKER_SERVER_CONFIG["timeout"],
    )


def ensure_worker_server() -> None:
    if not worker_server_alive():
        restart_worker_server(force_recreate=True)


def verify_tactic_batch(
    question: str,
    proof_steps: list[str],
    ranked_tactics: list[tuple[int, str, str]],
) -> dict[str, Any]:
    global WORKER_SERVER
    assert WORKER_SERVER_CONFIG, "Lean worker was not initialized"

    last_error = None
    for attempt in range(2):
        try:
            ensure_worker_server()
            targets = WORKER_SERVER.load_sorry(build_search_target(question))
            if not targets:
                return {"replay_error": "no_targets", "results": []}
            current_state = targets[0].goal_state
            for step in proof_steps:
                current_state = WORKER_SERVER.goal_tactic(current_state, step)

            results = []
            for rank, tactic, source in ranked_tactics:
                try:
                    next_state = WORKER_SERVER.goal_tactic(current_state, tactic)
                except (TacticFailure, ServerError) as exc:
                    results.append(
                        {
                            "rank": rank,
                            "tactic": tactic,
                            "source": source,
                            "ok": False,
                            "closed": False,
                            "error": str(exc),
                        }
                    )
                    continue

                next_signature = goal_state_signature(next_state)
                results.append(
                    {
                        "rank": rank,
                        "tactic": tactic,
                        "source": source,
                        "ok": True,
                        "closed": next_signature == ("no goals",),
                        "next_signature": next_signature,
                        "next_goal_text": goal_signature_to_text(next_signature),
                        "error": None,
                    }
                )

            return {"replay_error": None, "results": results}
        except RETRIABLE_SERVER_EXCEPTIONS as exc:
            last_error = str(exc)
            restart_worker_server(force_recreate=True)
        except (TacticFailure, ValueError, AssertionError) as exc:
            return {"replay_error": str(exc), "results": []}
        finally:
            try:
                if worker_server_alive():
                    WORKER_SERVER.gc()
            except Exception:
                pass

        if attempt == 0:
            continue

    return {
        "replay_error": f"worker_server_restart_failed: {last_error or 'unknown'}",
        "results": [],
    }


async def generate_layer_tactics(
    beam: list[SearchNode],
    client: AsyncOpenAI,
    args: argparse.Namespace,
    model_name: str,
) -> list[tuple[SearchNode, list[str], str | None]]:
    async def generate_for_node(node: SearchNode) -> tuple[SearchNode, list[str], str | None]:
        try:
            tactics = await generate_tactics(client, node.goal_text, args, model_name)
            return node, tactics, None
        except (openai.APIError, openai.APIConnectionError, openai.APITimeoutError) as exc:
            return node, [], str(exc)

    results = list(await asyncio.gather(*(generate_for_node(node) for node in beam)))
    results.sort(key=lambda item: (item[0].depth, len(item[0].proof_steps), item[0].goal_text))
    return results


def choose_better_candidate(candidate: CandidateState, challenger: CandidateState) -> CandidateState:
    if challenger.best_rank < candidate.best_rank:
        return challenger
    if challenger.best_rank == candidate.best_rank and len(challenger.proof_steps) < len(candidate.proof_steps):
        return challenger
    return candidate


def build_candidate_tactics(
    model_tactics: list[str], goal_text: str, args: argparse.Namespace
) -> tuple[list[tuple[int, str, str]], int, int]:
    llm_tactics = dedupe_keep_order(model_tactics)[: args.num_tactics]
    smart_hammers = get_smart_hammers(goal_text)[: args.max_hammers_per_node]
    ranked = [(rank, tactic, "llm") for rank, tactic in enumerate(llm_tactics, start=1)]
    ranked.extend(
        (args.num_tactics + offset, tactic, "hammer")
        for offset, tactic in enumerate(smart_hammers, start=1)
    )
    combined = dedupe_ranked_tactics(ranked)
    llm_count = sum(source == "llm" for _, _, source in combined)
    hammer_count = sum(source == "hammer" for _, _, source in combined)
    return combined, llm_count, hammer_count


async def search_theorem(
    question: str,
    process_pool: ProcessPoolExecutor,
    args: argparse.Namespace,
    model_name: str,
    async_client: AsyncOpenAI,
    on_event: SearchEventCallback | None = None,
) -> dict[str, Any]:
    start_time = time.monotonic()
    deadline = start_time + args.time_budget_seconds
    layer_summaries = []
    generation_errors = []
    verification_errors = []
    generation_wall_time = 0.0
    verification_wall_time = 0.0
    verified_llm_tactics = 0
    verified_hammer_tactics = 0

    def emit(event_type: str, **payload: Any) -> None:
        if on_event is None:
            return
        on_event({"type": event_type, **payload})

    server = await Server.create(
        imports=["Mathlib"],
        project_path=str(args.project_dir),
        lean_path=args.lean_path,
        timeout=args.server_timeout,
    )
    try:
        try:
            targets = None
            last_root_error = None
            for attempt in range(2):
                try:
                    targets = await server.load_sorry_async(build_search_target(question))
                    break
                except RETRIABLE_SERVER_EXCEPTIONS as exc:
                    last_root_error = exc
                    await server.restart_async()
                except ValueError as exc:
                    last_root_error = exc
                    break
            if targets is None:
                raise last_root_error or RuntimeError("load_sorry failed")
        except (ServerError, RuntimeError, ValueError) as exc:
            return {
                "solved": False,
                "proof": "",
                "proof_steps": [],
                "closed_at_depth": None,
                "expanded_nodes": 0,
                "verified_tactics": 0,
                "visited_states": 0,
                "failure": f"root_compile_error: {exc}",
                "elapsed_seconds": round(time.monotonic() - start_time, 2),
                "generation_wall_time": 0.0,
                "verification_wall_time": 0.0,
                "verified_llm_tactics": 0,
                "verified_hammer_tactics": 0,
                "layer_summaries": [],
                "generation_errors": [],
                "verification_errors": [],
            }

        if not targets:
            return {
                "solved": False,
                "proof": "",
                "proof_steps": [],
                "closed_at_depth": None,
                "expanded_nodes": 0,
                "verified_tactics": 0,
                "visited_states": 0,
                "failure": "no_targets",
                "elapsed_seconds": round(time.monotonic() - start_time, 2),
                "generation_wall_time": 0.0,
                "verification_wall_time": 0.0,
                "verified_llm_tactics": 0,
                "verified_hammer_tactics": 0,
                "layer_summaries": [],
                "generation_errors": [],
                "verification_errors": [],
            }

        root_state = targets[0].goal_state
        root_signature = goal_state_signature(root_state)
        emit(
            "root_state_loaded",
            goal_text=goal_signature_to_text(root_signature),
            state_signature=root_signature,
        )
        if root_signature == ("no goals",):
            return {
                "solved": True,
                "proof": "",
                "proof_steps": [],
                "closed_at_depth": 0,
                "expanded_nodes": 0,
                "verified_tactics": 0,
                "visited_states": 1,
                "failure": None,
                "elapsed_seconds": round(time.monotonic() - start_time, 2),
                "generation_wall_time": 0.0,
                "verification_wall_time": 0.0,
                "verified_llm_tactics": 0,
                "verified_hammer_tactics": 0,
                "layer_summaries": [],
                "generation_errors": [],
                "verification_errors": [],
            }
    finally:
        server._close()

    beam = [SearchNode([], goal_signature_to_text(root_signature), root_signature, 0)]
    visited_states = {root_signature}
    expanded_nodes = 0
    verified_tactics = 0
    failure = "beam_exhausted"

    while beam:
        if time.monotonic() >= deadline:
            failure = "time_budget_exhausted"
            break
        if expanded_nodes >= args.max_expanded_nodes:
            failure = "expanded_budget_exhausted"
            break
        if verified_llm_tactics >= args.max_verified_tactics:
            failure = "verified_budget_exhausted"
            break

        expandable = [node for node in beam if node.depth < args.max_depth]
        if not expandable:
            failure = "depth_exhausted"
            break

        layer_depth = expandable[0].depth
        emit(
            "layer_started",
            depth=layer_depth,
            beam_size=len(beam),
            expandable_nodes=len(expandable),
            expanded_nodes=expanded_nodes,
            verified_tactics=verified_tactics,
        )
        generation_start = time.monotonic()
        generated_entries = await generate_layer_tactics(expandable, async_client, args, model_name)
        generation_elapsed = time.monotonic() - generation_start
        generation_wall_time += generation_elapsed
        node_batches = []
        layer_expanded = 0
        layer_generated = 0
        layer_generated_llm = 0
        layer_generated_hammer = 0

        for node, tactics, error in generated_entries:
            if expanded_nodes >= args.max_expanded_nodes:
                failure = "expanded_budget_exhausted"
                break
            if time.monotonic() >= deadline:
                failure = "time_budget_exhausted"
                break

            expanded_nodes += 1
            layer_expanded += 1
            if error is not None:
                generation_errors.append(error)
                continue

            combined_tactics, llm_count, hammer_count = build_candidate_tactics(tactics, node.goal_text, args)
            layer_generated += len(combined_tactics)
            layer_generated_llm += llm_count
            layer_generated_hammer += hammer_count
            if not combined_tactics:
                continue
            node_batches.append((node, combined_tactics))

        emit(
            "layer_generation_completed",
            depth=layer_depth,
            expanded_nodes=layer_expanded,
            generated_tactics=layer_generated,
            generated_llm_tactics=layer_generated_llm,
            generated_hammer_tactics=layer_generated_hammer,
            generation_seconds=round(generation_elapsed, 2),
            node_batches=[
                {
                    "proof_steps": node.proof_steps,
                    "goal_text": node.goal_text,
                    "depth": node.depth,
                    "candidates": [
                        {"rank": rank, "tactic": tactic, "source": source}
                        for rank, tactic, source in ranked_tactics
                    ],
                }
                for node, ranked_tactics in node_batches
            ],
            generation_errors=list(generation_errors),
        )

        if failure in {"expanded_budget_exhausted", "time_budget_exhausted"}:
            break

        future_map: dict[Any, SearchNode] = {}
        remaining_budget = args.max_verified_tactics - verified_llm_tactics
        layer_verified = 0
        layer_verified_llm = 0
        layer_verified_hammer = 0
        for node, ranked_tactics in node_batches:
            llm_candidates = [candidate for candidate in ranked_tactics if candidate[2] == "llm"]
            hammer_candidates = [candidate for candidate in ranked_tactics if candidate[2] == "hammer"]
            if remaining_budget <= 0 and not hammer_candidates:
                failure = "verified_budget_exhausted"
                break
            allowed_llm = llm_candidates[: max(0, remaining_budget)]
            allowed = dedupe_ranked_tactics(allowed_llm + hammer_candidates)
            if not allowed:
                continue
            layer_verified += len(allowed)
            layer_verified_llm += len(allowed_llm)
            layer_verified_hammer += sum(candidate[2] == "hammer" for candidate in allowed)
            remaining_budget -= len(allowed_llm)
            future = process_pool.submit(verify_tactic_batch, question, node.proof_steps, allowed)
            future_map[future] = node

        verified_tactics += layer_verified
        verified_llm_tactics += layer_verified_llm
        verified_hammer_tactics += layer_verified_hammer
        next_candidates: dict[tuple[str, ...], CandidateState] = {}
        if not future_map:
            if failure != "verified_budget_exhausted":
                failure = "beam_exhausted"
            break

        solved_steps: list[str] | None = None
        verification_start = time.monotonic()
        for future in as_completed(future_map):
            node = future_map[future]
            try:
                payload = future.result()
            except Exception as exc:
                verification_errors.append(f"worker_future_error: {exc}")
                emit(
                    "node_verification_completed",
                    depth=node.depth,
                    proof_steps=node.proof_steps,
                    goal_text=node.goal_text,
                    replay_error=f"worker_future_error: {exc}",
                    results=[],
                )
                continue
            replay_error = payload.get("replay_error")
            if replay_error:
                verification_errors.append(replay_error)
                emit(
                    "node_verification_completed",
                    depth=node.depth,
                    proof_steps=node.proof_steps,
                    goal_text=node.goal_text,
                    replay_error=replay_error,
                    results=[],
                )
                continue

            emit(
                "node_verification_completed",
                depth=node.depth,
                proof_steps=node.proof_steps,
                goal_text=node.goal_text,
                replay_error=None,
                results=payload["results"],
            )

            for result in payload["results"]:
                if not result["ok"]:
                    continue

                next_steps = node.proof_steps + [result["tactic"]]
                if result["closed"]:
                    solved_steps = next_steps
                    break

                if result["source"] == "hammer" and result["tactic"] in KILL_ONLY_HAMMERS:
                    continue

                next_signature = tuple(result["next_signature"])
                if next_signature in visited_states:
                    continue

                challenger = CandidateState(
                    proof_steps=next_steps,
                    goal_text=result["next_goal_text"],
                    state_signature=next_signature,
                    depth=node.depth + 1,
                    votes=1,
                    best_rank=result["rank"],
                )
                existing = next_candidates.get(next_signature)
                if existing is None:
                    next_candidates[next_signature] = challenger
                else:
                    existing.votes += 1
                    existing.best_rank = min(existing.best_rank, challenger.best_rank)
                    better = choose_better_candidate(existing, challenger)
                    better.votes = existing.votes
                    better.best_rank = existing.best_rank
                    next_candidates[next_signature] = better

        verification_elapsed = time.monotonic() - verification_start
        verification_wall_time += verification_elapsed
        if time.monotonic() >= deadline and solved_steps is None:
            failure = "time_budget_exhausted"
            break

        if solved_steps is not None:
            emit(
                "solution_found",
                proof_steps=solved_steps,
                closed_at_depth=len(solved_steps),
                expanded_nodes=expanded_nodes,
                verified_tactics=verified_tactics,
            )
            return {
                "solved": True,
                "proof": "\n".join(solved_steps),
                "proof_steps": solved_steps,
                "closed_at_depth": len(solved_steps),
                "expanded_nodes": expanded_nodes,
                "verified_tactics": verified_tactics,
                "visited_states": len(visited_states) + len(next_candidates),
                "failure": None,
                "elapsed_seconds": round(time.monotonic() - start_time, 2),
                "generation_wall_time": round(generation_wall_time, 2),
                "verification_wall_time": round(verification_wall_time, 2),
                "verified_llm_tactics": verified_llm_tactics,
                "verified_hammer_tactics": verified_hammer_tactics,
                "layer_summaries": layer_summaries,
                "generation_errors": generation_errors,
                "verification_errors": verification_errors,
            }

        visited_states.update(next_candidates.keys())
        ranked_candidates = sorted(
            next_candidates.values(),
            key=lambda candidate: (-candidate.votes, candidate.best_rank, len(candidate.proof_steps)),
        )
        beam = [
            SearchNode(
                proof_steps=candidate.proof_steps,
                goal_text=candidate.goal_text,
                state_signature=candidate.state_signature,
                depth=candidate.depth,
                votes=candidate.votes,
                best_rank=candidate.best_rank,
            )
            for candidate in ranked_candidates[: args.beam_width]
        ]
        layer_summaries.append(
            {
                "depth": layer_depth,
                "expanded_nodes": layer_expanded,
                "generated_tactics": layer_generated,
                "generated_llm_tactics": layer_generated_llm,
                "generated_hammer_tactics": layer_generated_hammer,
                "verified_tactics": layer_verified,
                "verified_llm_tactics": layer_verified_llm,
                "verified_hammer_tactics": layer_verified_hammer,
                "unique_next_states": len(next_candidates),
                "beam_out": len(beam),
                "generation_seconds": round(generation_elapsed, 2),
                "verification_seconds": round(verification_elapsed, 2),
            }
        )
        emit("layer_completed", **layer_summaries[-1])

        if not beam:
            failure = "beam_exhausted"
            break

    emit(
        "search_finished",
        solved=False,
        failure=failure,
        expanded_nodes=expanded_nodes,
        verified_tactics=verified_tactics,
        visited_states=len(visited_states),
    )
    return {
        "solved": False,
        "proof": "",
        "proof_steps": [],
        "closed_at_depth": None,
        "expanded_nodes": expanded_nodes,
        "verified_tactics": verified_tactics,
        "visited_states": len(visited_states),
        "failure": failure,
        "elapsed_seconds": round(time.monotonic() - start_time, 2),
        "generation_wall_time": round(generation_wall_time, 2),
        "verification_wall_time": round(verification_wall_time, 2),
        "verified_llm_tactics": verified_llm_tactics,
        "verified_hammer_tactics": verified_hammer_tactics,
        "layer_summaries": layer_summaries,
        "generation_errors": generation_errors,
        "verification_errors": verification_errors,
    }


def build_summary(args: argparse.Namespace, model_name: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    solved = sum(bool(row["solved"]) for row in results)
    avg_expanded = round(sum(row["expanded_nodes"] for row in results) / total, 2) if total else 0.0
    avg_verified = round(sum(row["verified_tactics"] for row in results) / total, 2) if total else 0.0
    avg_verified_llm = (
        round(sum(row["verified_llm_tactics"] for row in results) / total, 2) if total else 0.0
    )
    avg_verified_hammer = (
        round(sum(row["verified_hammer_tactics"] for row in results) / total, 2) if total else 0.0
    )
    avg_generation = (
        round(sum(row["generation_wall_time"] for row in results) / total, 2) if total else 0.0
    )
    avg_verification = (
        round(sum(row["verification_wall_time"] for row in results) / total, 2) if total else 0.0
    )
    solved_depths = [row["closed_at_depth"] for row in results if row["closed_at_depth"] is not None]
    return {
        "input_root": str(args.input_root),
        "split": args.split,
        "output_path": str(args.output),
        "api_base": args.api_base,
        "model_name": model_name,
        "total_theorems": total,
        "success_rate": round(solved / total, 4) if total else 0.0,
        "avg_expanded_nodes": avg_expanded,
        "avg_verified_tactics": avg_verified,
        "avg_verified_llm_tactics": avg_verified_llm,
        "avg_verified_hammer_tactics": avg_verified_hammer,
        "avg_generation_wall_time": avg_generation,
        "avg_verification_wall_time": avg_verification,
        "avg_solution_depth": round(sum(solved_depths) / len(solved_depths), 2) if solved_depths else None,
        "time_budget_seconds": args.time_budget_seconds,
        "max_expanded_nodes": args.max_expanded_nodes,
        "max_verified_tactics": args.max_verified_tactics,
        "beam_width": args.beam_width,
        "max_depth": args.max_depth,
        "num_tactics": args.num_tactics,
        "max_hammers_per_node": args.max_hammers_per_node,
        "lean_workers": args.lean_workers,
        "failure_counts": {
            key: sum(row["failure"] == key for row in results)
            for key in sorted({row["failure"] for row in results if row["failure"] is not None})
        },
    }


async def async_main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output = args.output.resolve()
    args.project_dir = args.project_dir.resolve()
    summary_path = args.summary_path.resolve() if args.summary_path else args.output.with_suffix(".summary.json")

    problem_paths = collect_problem_paths(args.input_root, args.split)
    problem_paths = filter_problem_paths(problem_paths, args.problem)
    if args.max_samples is not None:
        problem_paths = problem_paths[: args.max_samples]

    sync_client = build_client(args)
    model_name = resolve_model_name(sync_client, args.model_name)
    async_client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_base, timeout=args.request_timeout)
    args.lean_path = await get_lean_path_async(str(args.project_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    mp_context = mp.get_context("spawn")
    results = []
    total = len(problem_paths)
    solved_count = 0

    with ProcessPoolExecutor(
        max_workers=args.lean_workers,
        mp_context=mp_context,
        initializer=init_lean_worker,
        initargs=(str(args.project_dir), args.lean_path, args.server_timeout),
    ) as process_pool:
        with args.output.open("w", encoding="utf-8") as output_file:
            for index, problem_path in enumerate(problem_paths, start=1):
                question = extract_question_from_file(problem_path)
                result = await search_theorem(question, process_pool, args, model_name, async_client)
                solved_count += int(result["solved"])
                payload = {
                    "id": build_problem_id(problem_path, args.input_root),
                    "split": problem_path.parent.name.lower(),
                    "file_path": str(problem_path.relative_to(ROOT)),
                    "question": question,
                    "theorem_statement": question,
                    "proof": result["proof"],
                    "predictions": [result["proof"]] if result["solved"] else [],
                    **result,
                }
                results.append(payload)
                output_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                output_file.flush()
                print(
                    f"completed={index}/{total} solved={solved_count} id={payload['id']} "
                    f"success={int(payload['solved'])} expanded={payload['expanded_nodes']} "
                    f"verified={payload['verified_tactics']} llm={payload['verified_llm_tactics']} "
                    f"hammer={payload['verified_hammer_tactics']} gen={payload['generation_wall_time']} "
                    f"verify={payload['verification_wall_time']} elapsed={payload['elapsed_seconds']} "
                    f"failure={payload['failure']}",
                    flush=True,
                )

    summary = build_summary(args, model_name, results)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results={args.output}")
    print(f"summary={summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
