import argparse
import json
import random
from pathlib import Path

from evaluate_tactic_predictions import (
    DEFAULT_MODEL_ROOT,
    cleanup_tactic,
    flatten_traced_tactics,
    generate_tactics,
    load_model_and_tokenizer,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "training_data" / "highschool_algebra_mathlib_medium" / "val.json"
DEFAULT_ADAPTER_DIR = ROOT / "outputs" / "qwen2_5_math_algebra_medium_lora"
DEFAULT_OUTPUT = ROOT / "outputs" / "evaluation" / "medium_tactic_smoke.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded single-step tactic accuracy evaluation on a random sample."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--adapter-root",
        type=Path,
        default=None,
        help="Specific adapter checkpoint. If omitted, use the latest checkpoint under --adapter-dir.",
    )
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-goal-tokens", type=int, default=2800)
    parser.add_argument("--num-return-sequences", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    return parser.parse_args()


def resolve_latest_checkpoint(adapter_dir: Path) -> Path:
    candidates = []
    for path in adapter_dir.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if path.is_dir() and suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directories found under {adapter_dir}")
    return max(candidates)[1]


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.model_root = args.model_root.resolve()
    args.adapter_dir = args.adapter_dir.resolve()
    args.output = args.output.resolve()
    if args.adapter_root is None:
        args.adapter_root = resolve_latest_checkpoint(args.adapter_dir)
    else:
        args.adapter_root = args.adapter_root.resolve()

    summary_path = args.summary_path.resolve() if args.summary_path else args.output.with_suffix(".summary.json")

    rows = flatten_traced_tactics(args.input)
    if not rows:
        raise ValueError(f"No traced tactics found in {args.input}")

    sample_size = min(args.max_samples, len(rows))
    rng = random.Random(args.sample_seed)
    sampled_rows = rng.sample(rows, sample_size)

    model, tokenizer = load_model_and_tokenizer(args.model_root, args.adapter_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    top1_correct = 0
    topk_correct = 0
    nonempty_predictions = 0

    with args.output.open("w", encoding="utf-8") as output_file:
        for row in sampled_rows:
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

    total = len(sampled_rows)
    summary = {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "summary_path": str(summary_path),
        "model_root": str(args.model_root),
        "adapter_root": str(args.adapter_root),
        "sample_seed": args.sample_seed,
        "total": total,
        "top1_accuracy": round(top1_correct / total, 4) if total else 0.0,
        "topk_accuracy": round(topk_correct / total, 4) if total else 0.0,
        "nonempty_prediction_rate": round(nonempty_predictions / total, 4) if total else 0.0,
        "num_return_sequences": args.num_return_sequences,
        "max_goal_tokens": args.max_goal_tokens,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results={args.output}")
    print(f"summary={summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
