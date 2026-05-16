import argparse
import json
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "evaluation" / "lean_eval_results.jsonl"
DEFAULT_PROJECT_DIR = ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Lean 4 proof candidates by compiling them in a local Lake project."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--proof-field", default="predictions")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply deterministic proof-output cleanup before compile.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def validate_project_dir(project_dir: Path) -> None:
    if not (project_dir / "lean-toolchain").exists():
        raise FileNotFoundError(f"Missing lean-toolchain under {project_dir}")
    if not ((project_dir / "lakefile.lean").exists() or (project_dir / "lakefile.toml").exists()):
        raise FileNotFoundError(f"Missing lakefile.lean or lakefile.toml under {project_dir}")


def extract_question(row: dict[str, Any], question_field: str) -> str:
    for field_name in [question_field, "formal_statement", "prompt"]:
        value = row.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.rstrip()

    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.rstrip()

    raise ValueError(f"Missing question field '{question_field}'")


def extract_candidates(row: dict[str, Any], proof_field: str) -> list[str]:
    candidate_keys = [proof_field, "predictions", "prediction", "raw_output", "answer", "proof"]
    for key in candidate_keys:
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, list):
            return [ensure_text(candidate) for candidate in value]
        return [ensure_text(value)]
    raise ValueError(f"Missing proof field '{proof_field}'")


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```(?:lean4?|lean)?\s*", "", stripped, count=1)
    stripped = re.sub(r"\s*```$", "", stripped, count=1)
    return stripped.strip()


def rebalance_proof_indentation(proof: str) -> str:
    lines = proof.splitlines()
    if len(lines) <= 1:
        return proof.rstrip()

    trailing_indents = [
        len(line) - len(line.lstrip(" ")) for line in lines[1:] if line.strip()
    ]
    if not trailing_indents:
        return proof.rstrip()

    common_indent = min(trailing_indents)
    if common_indent <= 0:
        return proof.rstrip()

    rebuilt = [lines[0]]
    for line in lines[1:]:
        rebuilt.append("" if not line.strip() else line[common_indent:])
    return "\n".join(rebuilt).rstrip()


def normalize_proof(raw_output: str, question: str) -> tuple[str, str]:
    proof = strip_code_fences(raw_output).strip()
    proof = re.sub(
        r"^(?:Assistant|assistant|Proof|proof|Answer|answer|答案)\s*[:：]\s*",
        "",
        proof,
    )

    if question and question in proof:
        proof = proof.split(question, 1)[1].strip()

    theorem_match = re.search(r":=\s*by\b", proof)
    if theorem_match is not None and re.search(r"(?m)^\s*(?:import\s+|theorem\s+|lemma\s+|example\s+)", proof):
        proof = proof[theorem_match.end() :].strip()

    if re.match(r"^by(?:\s|$)", proof):
        proof = re.sub(r"^by\s*", "", proof, count=1)

    proof = rebalance_proof_indentation(proof)
    if not proof.strip():
        return "", "empty_proof"
    return proof.rstrip(), "ok"


def indent_proof_block(proof: str) -> str:
    return "\n".join(f"  {line}" if line.strip() else "" for line in proof.splitlines())


def classify_failure(stdout: str, stderr: str) -> str:
    combined = f"{stdout}\n{stderr}".lower()
    if "timeout" in combined:
        return "timeout"
    if "unknown identifier" in combined or "unknown constant" in combined:
        return "unknown_identifier"
    if "unsolved goals" in combined:
        return "unsolved_goals"
    if "type mismatch" in combined:
        return "type_mismatch"
    if "expected token" in combined or "unexpected token" in combined or "syntax" in combined or "parse" in combined:
        return "syntax_error"
    if "tactic" in combined:
        return "tactic_failure"
    if "error" in combined:
        return "compile_error"
    return "compile_error"


def build_failure_result(raw_output: str, normalized_proof: str, status: str) -> dict[str, Any]:
    return {
        "raw_output": raw_output,
        "normalized_proof": normalized_proof,
        "passed": False,
        "status": status,
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }


def compile_candidate(
    project_dir: Path,
    question: str,
    raw_output: str,
    normalized_proof: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    lowered = normalized_proof.lower()
    if re.search(r"\bsorry\b", lowered):
        return build_failure_result(raw_output, normalized_proof, "contains_sorry")
    if re.search(r"\badmit\b", lowered):
        return build_failure_result(raw_output, normalized_proof, "contains_admit")
    if not normalized_proof.strip():
        return build_failure_result(raw_output, normalized_proof, "empty_proof")

    full_code = f"{question.rstrip()}\n{indent_proof_block(normalized_proof)}\n"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".lean",
            prefix="EvalProof_",
            dir=project_dir,
            delete=False,
        ) as temp_file:
            temp_file.write(full_code)
            temp_path = Path(temp_file.name)

        result = subprocess.run(
            ["lake", "env", "lean", temp_path.name],
            cwd=project_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
            check=False,
        )
        stdout = ensure_text(result.stdout)
        stderr = ensure_text(result.stderr)
        passed = result.returncode == 0
        return {
            "raw_output": raw_output,
            "normalized_proof": normalized_proof,
            "passed": passed,
            "status": "pass" if passed else classify_failure(stdout, stderr),
            "returncode": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "raw_output": raw_output,
            "normalized_proof": normalized_proof,
            "passed": False,
            "status": "timeout",
            "returncode": None,
            "stdout": ensure_text(exc.stdout),
            "stderr": ensure_text(exc.stderr),
        }
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def evaluate_row(index: int, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    record_id = str(row.get(args.id_field, f"sample-{index:05d}"))
    question = extract_question(row, args.question_field)
    candidates = extract_candidates(row, args.proof_field)
    candidate_results = []

    for raw_output in candidates:
        if args.normalize:
            normalized_proof, normalization_status = normalize_proof(raw_output, question)
        else:
            normalized_proof = raw_output.strip()
            normalization_status = "ok" if normalized_proof else "empty_proof"

        if normalization_status != "ok":
            candidate_results.append(build_failure_result(raw_output, normalized_proof, normalization_status))
            continue

        candidate_results.append(
            compile_candidate(
                project_dir=args.project_dir,
                question=question,
                raw_output=raw_output,
                normalized_proof=normalized_proof,
                timeout_seconds=args.timeout_seconds,
            )
        )

    return {
        "id": record_id,
        "pass_at_1": bool(candidate_results and candidate_results[0]["passed"]),
        "pass_at_k": any(candidate["passed"] for candidate in candidate_results),
        "candidate_results": candidate_results,
    }


def build_summary(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    pass_at_1 = sum(1 for result in results if result["pass_at_1"])
    pass_at_k = sum(1 for result in results if result["pass_at_k"])
    failure_counts: dict[str, int] = {}
    for result in results:
        for candidate in result["candidate_results"]:
            if candidate["passed"]:
                continue
            status = candidate["status"]
            failure_counts[status] = failure_counts.get(status, 0) + 1
    return {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "project_dir": str(args.project_dir),
        "question_field": args.question_field,
        "proof_field": args.proof_field,
        "total": total,
        "pass_at_1": round(pass_at_1 / total, 4) if total else 0.0,
        "pass_at_k": round(pass_at_k / total, 4) if total else 0.0,
        "failure_counts": failure_counts,
    }


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.project_dir = args.project_dir.resolve()
    summary_path = args.summary_path.resolve() if args.summary_path else args.output.with_suffix(".summary.json")

    validate_project_dir(args.project_dir)
    rows = list(iter_jsonl(args.input))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            results = list(executor.map(lambda item: evaluate_row(item[0], item[1], args), rows))
    else:
        results = [evaluate_row(index, row, args) for index, row in rows]

    with args.output.open("w", encoding="utf-8") as output_file:
        for result in results:
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = build_summary(args, results)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"results={args.output}")
    print(f"summary={summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
