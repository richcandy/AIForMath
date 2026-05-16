import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL_ROOT = ROOT / "Qwen2.5-Math-1.5B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a DeepSpeed ZeRO checkpoint into a normal Hugging Face model directory."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Training output directory that contains zero_to_fp32.py and global_step*.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for merged fp32 weights and model files.",
    )
    parser.add_argument(
        "--base-model-root",
        type=Path,
        default=DEFAULT_BASE_MODEL_ROOT,
        help="Directory to copy tokenizer/config files from.",
    )
    parser.add_argument(
        "--safe-serialization",
        action="store_true",
        help="Use safetensors when zero_to_fp32 supports it.",
    )
    return parser.parse_args()


def copy_model_metadata(base_model_root: Path, output_dir: Path) -> None:
    files_to_copy = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]
    for file_name in files_to_copy:
        source_path = base_model_root / file_name
        if source_path.exists():
            shutil.copy2(source_path, output_dir / file_name)


def main() -> None:
    args = parse_args()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.base_model_root = args.base_model_root.resolve()

    zero_script = args.checkpoint_dir / "zero_to_fp32.py"
    if not zero_script.exists():
        raise FileNotFoundError(f"Missing zero_to_fp32.py under {args.checkpoint_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        str(zero_script),
        str(args.checkpoint_dir),
        str(args.output_dir),
    ]
    if args.safe_serialization:
        command.append("--safe_serialization")

    subprocess.run(command, check=True)
    copy_model_metadata(args.base_model_root, args.output_dir)

    print(f"checkpoint_dir={args.checkpoint_dir}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
