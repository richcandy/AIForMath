import re
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria


ROOT = Path(__file__).resolve().parents[2]
STOP_PROOF_MARKERS = ["```", "\nimport ", "\ntheorem ", "\nlemma ", "\nexample ", "\n/--"]
STOP_TACTIC_MARKERS = ["\n", "```", "User:", "Assistant:"]

PROOF_PROMPT_TEMPLATE = (
    "Complete the following Lean 4 code by generating only the proof body that should come after `by`.\n\n"
    "Rules:\n"
    "- Output only Lean proof code.\n"
    "- Do not repeat the theorem statement.\n"
    "- Do not include explanations, markdown, or code fences.\n"
    "- Never use `sorry` or `admit`.\n\n"
    "Lean 4 theorem declaration:\n{question}"
)

TACTIC_PROMPT_TEMPLATE = (
    "Given the following Lean 4 goal state, output exactly one Lean tactic that advances or solves the goal.\n\n"
    "Rules:\n"
    "- Output only the tactic text.\n"
    "- Single line only.\n"
    "- Do not use `by`, markdown, explanations, `sorry`, or `admit`.\n\n"
    "Goal state:\n{goal_text}"
)

README_PROOF_PROMPT_TEMPLATE = (
    "Complete the following Lean 4 code:\n\n"
    "```lean4\n{question}\n```\n\n"
    "Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.\n"
    "The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."
)

PROMPT_A_PROOF_TEMPLATE = (
    "Complete the following Lean 4 code:\n\n"
    "```lean4\n{question}\n```\n\n"
    "Replace the trailing `sorry` with a complete Lean 4 proof.\n"
    "Output only the final Lean 4 code."
)


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


def resolve_model_dir(model_root: Path) -> Path:
    ref_file = model_root / "refs" / "main"
    if ref_file.exists():
        return model_root / "snapshots" / ref_file.read_text(encoding="utf-8").strip()
    return model_root


def load_model_and_tokenizer(model_root: Path, trust_remote_code: bool):
    resolved_model = resolve_model_dir(model_root)
    config = AutoConfig.from_pretrained(resolved_model, trust_remote_code=trust_remote_code, attn_implementation="flash_attnention_2")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype_name = str(getattr(config, "torch_dtype", "float16"))
    torch_dtype = torch.bfloat16 if "bfloat16" in torch_dtype_name else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def build_proof_prompt(question: str) -> str:
    return PROOF_PROMPT_TEMPLATE.format(question=question.strip())


def build_tactic_prompt(goal_text: str) -> str:
    return TACTIC_PROMPT_TEMPLATE.format(goal_text=goal_text.strip())


def build_readme_proof_prompt(question: str) -> str:
    return README_PROOF_PROMPT_TEMPLATE.format(question=question.strip())


def build_prompt_a_proof_prompt(question: str) -> str:
    return PROMPT_A_PROOF_TEMPLATE.format(question=question.strip())


def build_plain_prompt(user_prompt: str) -> str:
    return f"User:\n{user_prompt.strip()}\n\nAssistant:\n"


def build_generation_inputs(model, tokenizer, user_prompt: str, prompt_mode: str):
    messages = [{"role": "user", "content": user_prompt}]

    use_chat_template = prompt_mode == "chat" or (
        prompt_mode == "auto" and bool(getattr(tokenizer, "chat_template", None))
    )
    if use_chat_template:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

    plain_prompt = build_plain_prompt(user_prompt)
    return tokenizer(plain_prompt, return_tensors="pt").to(model.device)


def extract_question_from_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    updated = re.sub(r":=\s*by\s*(?:sorry|admit)\s*$", ":= by", text, count=1, flags=re.S)
    if updated == text:
        raise ValueError(f"Could not locate trailing ':= by sorry/admit' in {path}")
    return updated.rstrip()


def extract_formal_statement_from_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not re.search(r":=\s*by\s*(?:sorry|admit)\s*$", text, flags=re.S):
        raise ValueError(f"Could not locate trailing ':= by sorry/admit' in {path}")
    return text.rstrip()


def collect_problem_paths(split: str) -> list[Path]:
    split_map = {"valid": "Valid", "test": "Test"}
    splits = [split_map[split]] if split != "both" else ["Valid", "Test"]
    paths = []
    for split_name in splits:
        split_dir = ROOT / "MiniF2F" / split_name
        paths.extend(sorted(split_dir.glob("*.lean")))
    return paths


def cleanup_proof(text: str) -> str:
    cleaned = text.replace("Ġ", " ").replace("Ċ", "\n").strip()
    cleaned = re.sub(r"^```(?:lean4?|lean)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    for marker in STOP_PROOF_MARKERS:
        index = cleaned.find(marker)
        if index != -1:
            cleaned = cleaned[:index].strip()
    cleaned = cleaned.strip().strip("`")
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"^(?:Assistant|assistant|Proof|proof|Answer|answer)\s*[:：]\s*", "", cleaned)
    cleaned = re.sub(r"^(?:Lean\s*4\s*Proof|Proof\s*Plan)\s*:??\s*", "", cleaned, flags=re.I)
    theorem_match = re.search(r":=\s*by\b", cleaned)
    if theorem_match is not None and "theorem" in cleaned:
        cleaned = cleaned[theorem_match.end() :].strip()
    if cleaned.startswith("by"):
        cleaned = cleaned[2:].strip()
    return cleaned.strip()


def truncate_goal_text(tokenizer, goal_text: str, max_goal_tokens: int) -> str:
    if max_goal_tokens <= 0:
        return goal_text
    encoded = tokenizer(goal_text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"][0]
    if input_ids.shape[0] <= max_goal_tokens:
        return goal_text
    return tokenizer.decode(input_ids[-max_goal_tokens:], skip_special_tokens=True).strip()


def cleanup_tactic(text: str) -> str:
    cleaned = text.replace("Ġ", " ").replace("Ċ", "\n").strip()
    for marker in STOP_TACTIC_MARKERS:
        index = cleaned.find(marker)
        if index != -1:
            cleaned = cleaned[:index].strip()
    cleaned = cleaned.strip().strip("`")
    cleaned = cleaned.replace("Assistant:", "").replace("assistant:", "").strip()
    if cleaned.startswith("by"):
        cleaned = cleaned[2:].strip()
    cleaned = cleaned.splitlines()[0].strip() if cleaned else ""
    cleaned = cleaned.replace("<a>", "").replace("</a>", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    bad_fragments = ["You are a Lean", "Given a goal state", "⊢", "User:", "Assistant:"]
    if any(fragment in cleaned for fragment in bad_fragments):
        return ""
    return cleaned


def build_generation_row(problem_path: Path, question: str, predictions: list[str]) -> dict:
    split_name = problem_path.parent.name
    theorem_id = problem_path.stem
    return {
        "id": f"{split_name}/{theorem_id}",
        "split": split_name,
        "file_path": str(problem_path.relative_to(ROOT)),
        "question": question,
        "predictions": predictions,
    }
