import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent


def classify_failure(stderr: str, stdout: str, proof: str) -> str:
    combined = f"{stdout}\n{stderr}".lower()
    if "sorry" in proof:
        return "contains_sorry"
    if "timeout" in combined:
        return "timeout"
    if "unknown constant" in combined or "unknown identifier" in combined:
        return "unknown_identifier"
    if "type mismatch" in combined:
        return "type_mismatch"
    if "tactic" in combined:
        return "tactic_failure"
    if "error" in combined:
        return "compile_error"
    return "other"


def compile_candidate(
    project_dir: Path, question: str, proof: str, timeout_seconds: int
) -> dict[str, object]:
    full_code = f"{question}\n{proof}\n"
    with TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        source_path = temp_dir / "EvalProof.lean"
        source_path.write_text(full_code, encoding="utf-8")

        try:
            result = subprocess.run(
                ["lake", "env", "lean", str(source_path)],
                cwd=project_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout_seconds,
                check=False,
            )
            passed = result.returncode == 0 and "sorry" not in proof
            status = (
                "pass"
                if passed
                else classify_failure(result.stderr, result.stdout, proof)
            )
            return {
                "passed": passed,
                "status": status,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "full_code": full_code,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "passed": False,
                "status": "timeout",
                "returncode": None,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "full_code": full_code,
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    args = parser.parse_args()

    predictions_path = args.predictions.resolve()
    output_path = args.output.resolve()
    project_dir = args.project_dir.resolve()

    total = 0
    pass_at_1 = 0
    pass_at_k = 0
    failure_counts: dict[str, int] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        predictions_path.open("r", encoding="utf-8", newline="") as input_file,
        output_path.open("w", encoding="utf-8", newline="") as output_file,
    ):
        for line in input_file:
            row = json.loads(line)
            total += 1
            question = str(row["question"])
            predictions = list(row["predictions"])
            candidate_results = []

            for proof in predictions:
                result = compile_candidate(
                    project_dir=project_dir,
                    question=question,
                    proof=str(proof),
                    timeout_seconds=args.timeout_seconds,
                )
                candidate_results.append(
                    {
                        "proof": proof,
                        "passed": result["passed"],
                        "status": result["status"],
                        "returncode": result["returncode"],
                        "stdout": result["stdout"],
                        "stderr": result["stderr"],
                    }
                )

            first_passed = bool(candidate_results and candidate_results[0]["passed"])
            any_passed = any(candidate["passed"] for candidate in candidate_results)
            if first_passed:
                pass_at_1 += 1
            if any_passed:
                pass_at_k += 1
            for candidate in candidate_results:
                if not candidate["passed"]:
                    failure_counts[candidate["status"]] = (
                        failure_counts.get(candidate["status"], 0) + 1
                    )

            output_file.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "pass_at_1": first_passed,
                        "pass_at_k": any_passed,
                        "candidate_results": candidate_results,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "predictions_path": str(predictions_path),
        "project_dir": str(project_dir),
        "total": total,
        "pass_at_1": round(pass_at_1 / total, 4) if total else 0.0,
        "pass_at_k": round(pass_at_k / total, 4) if total else 0.0,
        "failure_counts": failure_counts,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"results={output_path}")
    print(f"summary={summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
