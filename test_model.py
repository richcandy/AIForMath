import json
from threading import Thread
from pathlib import Path

from tokenizers import Tokenizer
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    TextIteratorStreamer,
)


MODEL_ROOT = Path("./Qwen2.5-Math-1.5B")


def resolve_snapshot_dir(model_root: Path) -> Path:
    ref_file = model_root / "refs" / "main"
    if ref_file.exists():
        return model_root / "snapshots" / ref_file.read_text(encoding="utf-8").strip()
    return model_root


def load_tokenizer(model_dir: Path) -> PreTrainedTokenizerFast:
    tokenizer_config = json.loads(
        (model_dir / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file(str(model_dir / "tokenizer.json")),
        eos_token=tokenizer_config.get("eos_token"),
        pad_token=tokenizer_config.get("pad_token"),
        bos_token=tokenizer_config.get("bos_token"),
        unk_token=tokenizer_config.get("unk_token"),
        additional_special_tokens=tokenizer_config.get("additional_special_tokens"),
        clean_up_tokenization_spaces=tokenizer_config.get(
            "clean_up_tokenization_spaces", False
        ),
    )
    tokenizer.chat_template = tokenizer_config["chat_template"]
    return tokenizer


def generate_reply(model, tokenizer, messages, max_new_tokens: int = 512) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "streamer": streamer,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    chunks = []
    print("Assistant: ", end="", flush=True)
    for text in streamer:
        chunks.append(text)
        print(text, end="", flush=True)
    print()

    thread.join()
    return "".join(chunks)


def read_user_input() -> str:
    lines = []
    prompt = "\nUser: "
    while True:
        line = input(prompt)
        if not lines:
            stripped = line.strip()
            if stripped in {"/exit", "/quit", "/clear"}:
                return stripped
        if line.strip() == "/send":
            return "\n".join(lines).strip()
        lines.append(line)
        prompt = "... "


def main() -> None:
    model_dir = resolve_snapshot_dir(MODEL_ROOT)
    tokenizer = load_tokenizer(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).cuda()
    messages = []

    print("开始对话，输入 /clear 清空上下文，输入 /exit 或 /quit 退出。")
    print("支持多行输入：连续输入多行，输入 /send 发送。")

    while True:
        user_input = read_user_input()
        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            print("已退出。")
            break
        if user_input == "/clear":
            messages.clear()
            print("上下文已清空。")
            continue

        messages.append({"role": "user", "content": user_input})
        reply = generate_reply(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
