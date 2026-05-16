from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunConfig:
    api_base: str = "http://127.0.0.1:11451/v1"
    api_key: str = "EMPTY"
    model_name: str = "my-tactic-lora"
    request_timeout: float = 120.0
    project_dir: Path = Path(".")
    input_root: Path = Path("MiniF2F_Highschool")
    split: str = "test"
    beam_width: int = 5
    max_depth: int = 21
    num_tactics: int = 12
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    max_goal_chars: int = 12000
    lean_workers: int = 6
    time_budget_seconds: int = 480
    max_expanded_nodes: int = 128
    max_verified_tactics: int = 1280
    server_timeout: int = 60
    max_hammers_per_node: int = 3


@dataclass
class ProblemOption:
    problem_id: str
    label: str
    path: Path
    question: str


@dataclass
class VerificationRecord:
    rank: int
    tactic: str
    source: str
    ok: bool
    closed: bool
    error: str | None = None
    next_goal_text: str | None = None


@dataclass
class CandidateBatch:
    depth: int
    proof_steps: list[str]
    goal_text: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    replay_error: str | None = None
    verification_results: list[VerificationRecord] = field(default_factory=list)


@dataclass
class RunViewState:
    status: str = "idle"
    problem_id: str | None = None
    question: str = ""
    root_goal_text: str = ""
    current_depth: int = 0
    expanded_nodes: int = 0
    verified_tactics: int = 0
    visited_states: int = 0
    solved: bool = False
    failure: str | None = None
    proof_steps: list[str] = field(default_factory=list)
    layer_summaries: list[dict[str, Any]] = field(default_factory=list)
    candidate_batches: list[CandidateBatch] = field(default_factory=list)
    generation_errors: list[str] = field(default_factory=list)
    verification_errors: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    result_payload: dict[str, Any] | None = None
