import argparse
import importlib
import json
import math
import os
import signal
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, IterableDataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = ROOT / "Deepseek-Prover-V2"
DEFAULT_TRAIN_DATA_PATH = ROOT / "Deepseek_highschool_data" / "flatten_data" / "train.jsonl"
DEFAULT_EVAL_DATA_PATH = ROOT / "Deepseek_highschool_data" / "flatten_data" / "val.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "deepseekprover_v2_lean_dojo"
PREVIEW_RECORDS = 3


def get_distributed_context() -> tuple[int, int, bool]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    return local_rank, world_size, is_distributed


def ignore_sighup() -> None:
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DeepSeekProverV2 on LeanDojo-style single-step tactic messages JSONL."
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--train-data-path", type=Path, default=DEFAULT_TRAIN_DATA_PATH)
    parser.add_argument("--eval-data-path", type=Path, default=DEFAULT_EVAL_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Training steps. If unset for streaming data, infer from manifest or line count.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer-size", type=int, default=1024)
    parser.add_argument("--preview-records", type=int, default=PREVIEW_RECORDS)
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        help="Trainer optimizer name.",
    )
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        default="auto",
        help="Attention backend. 'auto' prefers Flash Attention 2 and falls back to SDPA.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU mode. Useful only for import/config checks.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit loading on CUDA.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load datasets and model, then exit before training.",
    )
    return parser.parse_args()


def resolve_snapshot_dir(model_root: Path) -> Path:
    ref_file = model_root / "refs" / "main"
    if ref_file.exists():
        return model_root / "snapshots" / ref_file.read_text(encoding="utf-8").strip()
    return model_root


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def validate_messages_record(record: dict[str, Any], source: str) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Record from {source} is missing a non-empty messages list")
    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise ValueError(f"Message {index} in {source} is not a JSON object")
        if not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            raise ValueError(f"Message {index} in {source} must contain string role/content fields")


def preview_jsonl_records(data_path: Path, limit: int) -> list[dict[str, Any]]:
    preview = []
    with data_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a JSON object on line {line_number} in {data_path}")
            validate_messages_record(payload, f"{data_path}:{line_number}")
            preview.append(payload)
            if len(preview) >= limit:
                break
    if not preview:
        raise ValueError(f"Dataset is empty: {data_path}")
    return preview


def load_streaming_train_dataset(data_path: Path, shuffle_buffer_size: int, seed: int) -> IterableDataset:
    dataset = load_dataset("json", data_files=str(data_path), split="train", streaming=True)
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    return dataset


def load_eval_dataset(data_path: Path) -> Dataset:
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty: {data_path}")
    for index in range(min(len(dataset), PREVIEW_RECORDS)):
        validate_messages_record(dict(dataset[index]), f"{data_path}:{index + 1}")
    return dataset


def count_jsonl_records(data_path: Path) -> int:
    count = 0
    with data_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                count += 1
    return count


def load_manifest_step_count(data_path: Path) -> int | None:
    manifest_path = data_path.parent / "manifest.json"
    if not manifest_path.exists():
        return None

    payload = load_json(manifest_path)
    if not isinstance(payload, dict):
        return None
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        return None

    resolved_data_path = data_path.resolve()
    stem = data_path.stem
    for split_name, split_payload in splits.items():
        if not isinstance(split_payload, dict):
            continue
        output_path = split_payload.get("output_path")
        if isinstance(output_path, str) and Path(output_path).resolve() == resolved_data_path:
            step_records = split_payload.get("step_records")
            if isinstance(step_records, int) and step_records > 0:
                return step_records
        if split_name == stem:
            step_records = split_payload.get("step_records")
            if isinstance(step_records, int) and step_records > 0:
                return step_records
    return None


def resolve_train_step_count(data_path: Path) -> int:
    manifest_count = load_manifest_step_count(data_path)
    if manifest_count is not None:
        return manifest_count

    line_count = count_jsonl_records(data_path)
    if line_count <= 0:
        raise ValueError(f"Dataset contains no records: {data_path}")
    return line_count


def compute_max_steps(sample_count: int, args: argparse.Namespace, world_size: int) -> int:
    effective_batch_size = args.batch_size * args.grad_accum * max(world_size, 1)
    if effective_batch_size <= 0:
        raise ValueError("Effective batch size must be positive")
    steps_per_epoch = math.ceil(sample_count / effective_batch_size)
    return max(1, math.ceil(steps_per_epoch * args.epochs))


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def resolve_attention_implementation(args: argparse.Namespace, use_cuda: bool) -> str:
    if not use_cuda:
        return "eager"
    if args.attn_implementation != "auto":
        return args.attn_implementation

    try:
        importlib.import_module("flash_attn")
    except Exception as exc:
        print(
            f"Flash Attention 2 unavailable ({exc.__class__.__name__}: {exc}). Falling back to sdpa.",
            flush=True,
        )
        return "sdpa"
    return "flash_attention_2"


def load_model_and_tokenizer(
    model_dir: Path,
    args: argparse.Namespace,
    lora_config: LoraConfig,
):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    use_cuda = torch.cuda.is_available() and not args.use_cpu
    use_4bit = use_cuda and not args.no_4bit
    local_rank, world_size, is_distributed = get_distributed_context()
    attn_implementation = resolve_attention_implementation(args, use_cuda)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "attn_implementation": attn_implementation,
    }

    if use_cuda:
        if is_distributed:
            torch.cuda.set_device(local_rank)
        else:
            model_kwargs["device_map"] = "auto"

    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif use_cuda:
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    if is_distributed:
        model_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    if local_rank == 0:
        model.print_trainable_parameters()

    model.config.use_cache = False
    return (
        model,
        tokenizer,
        config,
        use_cuda,
        use_4bit,
        is_distributed,
        world_size,
        local_rank,
        attn_implementation,
    )


def build_trainer(
    model,
    tokenizer,
    train_dataset: IterableDataset,
    eval_dataset: Dataset | None,
    output_dir: Path,
    args: argparse.Namespace,
    is_distributed: bool,
    resolved_max_steps: int,
) -> SFTTrainer:
    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    train_args = SFTConfig(
        output_dir=str(output_dir),
        logging_dir=str(output_dir / "logs"),
        num_train_epochs=args.epochs,
        max_steps=resolved_max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps if has_eval else None,
        save_steps=args.save_steps,
        eval_strategy="steps" if has_eval else "no",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        report_to="tensorboard",
        max_length=args.max_length,
        packing=False,
        completion_only_loss=True,
        assistant_only_loss=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=torch.cuda.is_available() and not args.use_cpu,
        fp16=False,
        seed=args.seed,
        ddp_find_unused_parameters=False if is_distributed else None,
        optim=args.optim,
    )

    return SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )


def main() -> None:
    ignore_sighup()
    args = parse_args()
    args.model_root = args.model_root.resolve()
    args.train_data_path = args.train_data_path.resolve()
    args.eval_data_path = args.eval_data_path.resolve() if args.eval_data_path is not None else None
    args.output_dir = args.output_dir.resolve()

    model_dir = resolve_snapshot_dir(args.model_root)
    train_preview = preview_jsonl_records(args.train_data_path, args.preview_records)
    train_dataset = load_streaming_train_dataset(args.train_data_path, args.shuffle_buffer_size, args.seed)
    eval_dataset = load_eval_dataset(args.eval_data_path) if args.eval_data_path is not None else None
    lora_config = build_lora_config(args)
    local_rank, world_size, is_distributed = get_distributed_context()

    train_sample_count = resolve_train_step_count(args.train_data_path)
    resolved_max_steps = args.max_steps
    if resolved_max_steps <= 0:
        resolved_max_steps = compute_max_steps(train_sample_count, args, world_size)

    effective_batch_size = args.batch_size * args.grad_accum * max(world_size, 1)

    if local_rank == 0:
        print(f"Model root: {args.model_root}")
        print(f"Resolved model dir: {model_dir}")
        print(f"Train data path: {args.train_data_path}")
        print(f"Eval data path: {args.eval_data_path}")
        print(f"Train dataset is streaming: True")
        print(f"Shuffle buffer size: {args.shuffle_buffer_size}")
        print(f"Train sample count: {train_sample_count}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Resolved max steps: {resolved_max_steps}")
        print(json.dumps(train_preview[0], ensure_ascii=False, indent=2))
        if eval_dataset is not None:
            print(f"Loaded {len(eval_dataset)} validation samples from {args.eval_data_path}")
            print(json.dumps(dict(eval_dataset[0]), ensure_ascii=False, indent=2))

    model, tokenizer, config, use_cuda, use_4bit, is_distributed, world_size, local_rank, attn_implementation = load_model_and_tokenizer(
        model_dir, args, lora_config
    )
    if local_rank == 0:
        print(f"Model config type: {config.model_type}")
        print(f"CUDA enabled: {use_cuda}")
        print(f"4-bit loading: {use_4bit}")
        print(f"Attention implementation: {attn_implementation}")
        print(f"Distributed training: {is_distributed}")
        print(f"World size: {world_size}")
        print(f"Optimizer: {args.optim}")
        print(f"Max length: {args.max_length}")
        print(f"Tokenizer pad token: {tokenizer.pad_token}")
        print(f"Tokenizer padding side: {tokenizer.padding_side}")
        print(f"Tokenizer truncation side: {tokenizer.truncation_side}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        args.output_dir,
        args,
        is_distributed,
        resolved_max_steps,
    )
    if args.dry_run:
        if local_rank == 0:
            print("Dry run complete. Datasets, model, and trainer initialized successfully.")
        return

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    if local_rank == 0:
        tokenizer.save_pretrained(args.output_dir)
        print(f"Training finished. Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
