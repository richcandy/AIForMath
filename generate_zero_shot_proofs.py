import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "Qwen2.5-Math-1.5B"


def resolve_snapshot_dir(model_root: Path) -> Path:
    ref_file = model_root / "refs" / "main"
    if ref_file.exists():
        return model_root / "snapshots" / ref_file.read_text(encoding="utf-8").strip()
    return model_root


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return [json.loads(line) for line in input_file]


def build_prompt(question: str) -> str:
    return (
        "Complete the Lean 4 proof body only. Do not restate the theorem. "
        "Do not use sorry.\n\n"
        f"{question}\n"
    )


def trim_completion(text: str) -> str:
    return text.strip()


def print_progress(current: int, total: int, start_time: float) -> None:
    if total <= 0:
        return
    elapsed = time.time() - start_time
    percent = current / total * 100
    print(
        f"\rgenerated_rows={current}/{total} ({percent:5.1f}%) elapsed={elapsed:.1f}s",
        end="",
        flush=True,
    )
    if current == total:
        print(flush=True)


def generate_candidates(
    model,
    tokenizer,
    prompt: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            num_return_sequences=k,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    candidates = []
    for output in outputs:
        completion = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
        candidates.append(trim_completion(completion))
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    output_path = args.output.resolve()
    model_dir = resolve_snapshot_dir(args.model_dir.resolve())

    rows = load_rows(dataset_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto")
    model.to(device)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        for index, row in enumerate(rows, start=1):
            prompt = build_prompt(str(row["question"]))
            predictions = generate_candidates(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                k=args.k,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            output_file.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "answer": row["answer"],
                        "prompt": prompt,
                        "predictions": predictions,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            output_file.flush()
            print_progress(index, len(rows), start_time)

    print(f"dataset={dataset_path}")
    print(f"output={output_path}")
    print(f"rows={len(rows)}")
    print(f"k={args.k}")


if __name__ == "__main__":
    main()
