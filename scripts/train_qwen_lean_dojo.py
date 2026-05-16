import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


os.environ.setdefault("GITHUB_ACCESS_TOKEN", "local-smoke-test-token")

try:
    from lean_dojo_v2.trainer.sft_trainer import SFTDataset as LeanDojoSFTDataset
except Exception:
    LeanDojoSFTDataset = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = ROOT / "Qwen2.5-Math-1.5B"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "qwen2_5_math_lean_dojo"
TACTIC_SYSTEM_PROMPT = (
    "You are a Lean 4 tactic generator. Given a goal state, "
    "output exactly ONE Lean tactic that advances or solves the goal.\n"
    "Rules:\n"
    "- Output only the tactic text; no prose or code fences.\n"
    "- Single line only; no `by` blocks.\n"
    "- Never use `sorry` or `admit`.\n"
)


def remove_marks(text: str) -> str:
    return text.replace("<a>", "").replace("</a>", "")


def get_distributed_context() -> tuple[int, int, bool]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    return local_rank, world_size, is_distributed


class LocalTacticDataset:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        with data_path.open(encoding="utf-8") as data_file:
            self.json_data = json.load(data_file)
        self.data = self._process_data(self.json_data)

    def _process_data(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        processed = []
        for item in data:
            for tactic in item.get("traced_tactics", []):
                tactic_text = remove_marks(str(tactic.get("tactic", ""))).strip()
                if not tactic_text or tactic_text == "sorry":
                    continue
                processed.append(
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": TACTIC_SYSTEM_PROMPT,
                            },
                            {
                                "role": "user",
                                "content": remove_marks(
                                    str(tactic.get("state_before", ""))
                                ).strip(),
                            },
                            {
                                "role": "assistant",
                                "content": tactic_text.splitlines()[0].strip(),
                            },
                        ]
                    }
                )
        return processed

    def to_hf(self) -> Dataset:
        return Dataset.from_list(self.data)


def resolve_snapshot_dir(model_root: Path) -> Path:
    ref_file = model_root / "refs" / "main"
    if ref_file.exists():
        return model_root / "snapshots" / ref_file.read_text(encoding="utf-8").strip()
    return model_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Qwen2.5-Math-1.5B on LeanDojo-style tactic or message data."
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help="Local HuggingFace model directory or snapshot root.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Training data path. Supports traced_tactics JSON or messages JSON/JSONL.",
    )
    parser.add_argument(
        "--data-format",
        choices=["auto", "traced_tactics", "messages_json", "messages_jsonl"],
        default="auto",
        help="Explicitly set the dataset format or auto-detect it.",
    )
    parser.add_argument(
        "--eval-data-path",
        type=Path,
        default=None,
        help="Optional validation dataset path.",
    )
    parser.add_argument(
        "--eval-data-format",
        choices=["auto", "traced_tactics", "messages_json", "messages_jsonl"],
        default="auto",
        help="Explicitly set the validation dataset format or auto-detect it.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to save checkpoints, adapters, and tokenizer files.",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--optim",
        type=str,
        default=None,
        help="Trainer optimizer name. Defaults to adafactor for full finetune, adamw_torch otherwise.",
    )
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="Disable LoRA and train the base model directly.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank. Ignored with --full-finetune.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha. Ignored with --full-finetune.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout. Ignored with --full-finetune.",
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
        help="Resume trainer state from a saved checkpoint directory.",
    )
    parser.add_argument(
        "--deepspeed-stage",
        type=int,
        choices=[2, 3],
        default=None,
        help="Enable DeepSpeed ZeRO stage 2 or 3.",
    )
    parser.add_argument(
        "--deepspeed-offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="CPU offload for DeepSpeed ZeRO. Defaults to true for full finetune when DeepSpeed is enabled.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load dataset and model config, then exit before training.",
    )
    return parser.parse_args()


def load_json_records(data_path: Path) -> list[dict[str, Any]]:
    with data_path.open(encoding="utf-8") as data_file:
        data = json.load(data_file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {data_path}")
    return data


def load_jsonl_records(data_path: Path) -> list[dict[str, Any]]:
    records = []
    with data_path.open(encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} in {data_path}"
                )
            records.append(payload)
    return records


def detect_data_format(data_path: Path, requested_format: str) -> str:
    if requested_format != "auto":
        return requested_format

    if data_path.suffix == ".jsonl":
        records = load_jsonl_records(data_path)
        if records and isinstance(records[0].get("messages"), list):
            return "messages_jsonl"
    elif data_path.suffix == ".json":
        records = load_json_records(data_path)
        if not records:
            raise ValueError(f"Dataset is empty: {data_path}")
        first = records[0]
        if isinstance(first.get("traced_tactics"), list):
            return "traced_tactics"
        if isinstance(first.get("messages"), list):
            return "messages_json"

    raise ValueError(
        "Could not auto-detect dataset format. Use --data-format explicitly."
    )


def validate_messages(records: list[dict[str, Any]], data_path: Path) -> Dataset:
    normalized = []
    for index, record in enumerate(records, start=1):
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                f"Record {index} in {data_path} is missing a non-empty messages list"
            )
        normalized.append({"messages": messages})
    return Dataset.from_list(normalized)


def load_sft_dataset(data_path: Path, data_format: str) -> tuple[Dataset, str]:
    resolved_format = detect_data_format(data_path, data_format)

    if resolved_format == "traced_tactics":
        if LeanDojoSFTDataset is not None:
            print("Using lean_dojo_v2.trainer.sft_trainer.SFTDataset")
            dataset = LeanDojoSFTDataset(str(data_path)).to_hf()
        else:
            print("Falling back to local LeanDojo-compatible SFTDataset")
            dataset = LocalTacticDataset(data_path).to_hf()
        return dataset, resolved_format

    if resolved_format == "messages_json":
        return validate_messages(load_json_records(data_path), data_path), resolved_format

    if resolved_format == "messages_jsonl":
        return validate_messages(load_jsonl_records(data_path), data_path), resolved_format

    raise ValueError(f"Unsupported dataset format: {resolved_format}")


def build_lora_config(args: argparse.Namespace) -> LoraConfig | None:
    if args.full_finetune:
        return None
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


def resolve_optimizer(args: argparse.Namespace) -> str:
    if args.optim is not None:
        return args.optim
    return "adafactor" if args.full_finetune else "adamw_torch"


def should_use_deepspeed_offload(args: argparse.Namespace) -> bool:
    if args.deepspeed_offload is not None:
        return args.deepspeed_offload
    return args.full_finetune and args.deepspeed_stage is not None


def write_deepspeed_config(output_dir: Path, args: argparse.Namespace) -> str | None:
    if args.deepspeed_stage is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    use_offload = should_use_deepspeed_offload(args)
    zero_optimization: dict[str, Any] = {
        "stage": args.deepspeed_stage,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
    }
    if args.deepspeed_stage == 3:
        zero_optimization["stage3_prefetch_bucket_size"] = "auto"
        zero_optimization["stage3_param_persistence_threshold"] = "auto"

    if use_offload:
        zero_optimization["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }
        if args.deepspeed_stage == 3:
            zero_optimization["offload_param"] = {
                "device": "cpu",
                "pin_memory": True,
            }

    config = {
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "zero_optimization": zero_optimization,
        "fp16": {
            "enabled": bool(torch.cuda.is_available() and not args.use_cpu and args.no_4bit)
        },
        "bf16": {"enabled": False},
    }
    config_path = output_dir / "deepspeed_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return str(config_path)


def load_model_and_tokenizer(
    model_dir: Path,
    args: argparse.Namespace,
    lora_config: LoraConfig | None,
):
    if args.full_finetune and not args.no_4bit and torch.cuda.is_available() and not args.use_cpu:
        raise ValueError("--full-finetune requires --no-4bit on CUDA")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_cuda = torch.cuda.is_available() and not args.use_cpu
    use_4bit = use_cuda and not args.no_4bit and lora_config is not None
    local_rank, world_size, is_distributed = get_distributed_context()
    model_kwargs: dict[str, Any] = {}

    if use_cuda:
        if is_distributed:
            torch.cuda.set_device(local_rank)
        elif lora_config is not None:
            model_kwargs["device_map"] = "auto"

    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif use_cuda:
        model_kwargs["dtype"] = torch.float16
    else:
        model_kwargs["dtype"] = torch.float32

    if is_distributed:
        model_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)

    if lora_config is not None:
        if use_4bit:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        if local_rank == 0:
            model.print_trainable_parameters()

    model.config.use_cache = False
    return model, tokenizer, use_cuda, use_4bit, is_distributed, world_size, local_rank


def build_trainer(
    model,
    tokenizer,
    dataset: Dataset,
    eval_dataset: Dataset | None,
    output_dir: Path,
    args: argparse.Namespace,
    is_distributed: bool,
) -> SFTTrainer:
    deepspeed_config_path = write_deepspeed_config(output_dir, args)
    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    train_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
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
        report_to="none",
        max_length=args.max_length,
        packing=False,
        completion_only_loss=True,
        assistant_only_loss=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=False,
        fp16=torch.cuda.is_available() and not args.use_cpu and args.no_4bit,
        seed=args.seed,
        ddp_find_unused_parameters=False if is_distributed else None,
        optim=resolve_optimizer(args),
        deepspeed=deepspeed_config_path,
    )

    return SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )


def main() -> None:
    args = parse_args()
    model_dir = resolve_snapshot_dir(args.model_root)
    dataset, resolved_format = load_sft_dataset(args.data_path, args.data_format)
    eval_dataset = None
    eval_format = None
    if args.eval_data_path is not None:
        eval_dataset, eval_format = load_sft_dataset(args.eval_data_path, args.eval_data_format)
    lora_config = build_lora_config(args)
    local_rank, world_size, is_distributed = get_distributed_context()

    if len(dataset) == 0:
        raise ValueError(f"Dataset contains no training samples: {args.data_path}")

    if local_rank == 0:
        print(f"Loaded {len(dataset)} samples from {args.data_path}")
        print(f"Resolved dataset format: {resolved_format}")
        print(json.dumps(dataset[0], ensure_ascii=False, indent=2))
        if eval_dataset is not None:
            print(f"Loaded {len(eval_dataset)} validation samples from {args.eval_data_path}")
            print(f"Resolved validation format: {eval_format}")

    model, tokenizer, use_cuda, use_4bit, is_distributed, world_size, local_rank = load_model_and_tokenizer(
        model_dir, args, lora_config
    )
    if local_rank == 0:
        print(f"Model loaded from {model_dir}")
        print(f"CUDA enabled: {use_cuda}")
        print(f"4-bit loading: {use_4bit}")
        print(f"LoRA enabled: {lora_config is not None}")
        print(f"Distributed training: {is_distributed}")
        print(f"World size: {world_size}")
        print(f"Optimizer: {resolve_optimizer(args)}")
        print(f"DeepSpeed stage: {args.deepspeed_stage}")
        print(f"DeepSpeed offload: {should_use_deepspeed_offload(args)}")

    if args.dry_run:
        if local_rank == 0:
            print("Dry run complete. Dataset and model loaded successfully.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(
        model,
        tokenizer,
        dataset,
        eval_dataset,
        args.output_dir,
        args,
        is_distributed,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    if local_rank == 0:
        tokenizer.save_pretrained(args.output_dir)
        print(f"Training finished. Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
