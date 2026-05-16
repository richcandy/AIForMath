from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import openai
from openai import AsyncOpenAI
from pantograph.server import get_lean_path_async

from evaluation.evaluate_minif2f_highschool_beam_vllm import (
    ROOT,
    build_client,
    build_problem_id,
    collect_problem_paths,
    extract_question_from_file,
    filter_problem_paths,
    init_lean_worker,
    resolve_model_name,
    search_theorem,
)
from tui.services.event_bus import SearchEventHandler, coerce_search_event
from tui.state import ProblemOption, RunConfig


class SearchService:
    def __init__(self, config: RunConfig):
        self.config = config

    def list_problems(self) -> list[ProblemOption]:
        problem_paths = collect_problem_paths(self.config.input_root.resolve(), self.config.split)
        selected = filter_problem_paths(problem_paths, [])
        options = []
        for path in selected:
            question = extract_question_from_file(path)
            problem_id = build_problem_id(path, self.config.input_root.resolve())
            options.append(
                ProblemOption(
                    problem_id=problem_id,
                    label=problem_id,
                    path=path,
                    question=question,
                )
            )
        return options

    async def run_problem(
        self,
        question: str,
        on_event: SearchEventHandler | None = None,
    ) -> dict[str, Any]:
        def emit(event_type: str, **payload: Any) -> None:
            if on_event is not None:
                on_event(coerce_search_event({"type": event_type, **payload}))

        args = self._build_args_namespace()
        emit("startup_stage", message=f"Preparing search with api_base={args.api_base}")
        try:
            sync_client = build_client(args)
            emit("startup_stage", message="Checking served models")
            served_models = sync_client.models.list().data
            if not served_models:
                raise RuntimeError("No served models returned by /v1/models")

            model_name = resolve_model_name(sync_client, args.model_name)
            served_model_ids = {model.id for model in served_models}
            if model_name not in served_model_ids:
                available = ", ".join(sorted(served_model_ids))
                raise RuntimeError(
                    f"Configured model '{model_name}' is not served. Available models: {available}"
                )

            async_client = AsyncOpenAI(
                api_key=args.api_key,
                base_url=args.api_base,
                timeout=args.request_timeout,
            )
            emit("startup_stage", message="Resolving Lean path")
            args.lean_path = await get_lean_path_async(str(args.project_dir))
        except (openai.APIError, openai.APIConnectionError, openai.APITimeoutError) as exc:
            raise RuntimeError(f"vLLM/OpenAI API check failed: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Search startup failed: {exc}") from exc

        start_method = "fork" if os.name == "posix" else "spawn"
        mp_context = mp.get_context(start_method)
        emit(
            "startup_stage",
            message=f"Starting {args.lean_workers} Lean worker processes with {start_method}",
        )
        try:
            with ProcessPoolExecutor(
                max_workers=args.lean_workers,
                mp_context=mp_context,
                initializer=init_lean_worker,
                initargs=(str(args.project_dir), args.lean_path, args.server_timeout),
            ) as process_pool:
                result = await search_theorem(
                    question,
                    process_pool,
                    args,
                    model_name,
                    async_client,
                    on_event=(lambda raw: on_event(coerce_search_event(raw))) if on_event else None,
                )
        except Exception as exc:
            raise RuntimeError(f"Beam search execution failed: {exc}") from exc

        emit("search_result", result=result)
        return result

    def _build_args_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(
            input_root=self.config.input_root.resolve(),
            split=self.config.split,
            output=(ROOT / "outputs" / "evaluation" / "tui_session.jsonl").resolve(),
            summary_path=None,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
            model_name=self.config.model_name,
            request_timeout=self.config.request_timeout,
            project_dir=self.config.project_dir.resolve(),
            max_samples=1,
            problem=[],
            beam_width=self.config.beam_width,
            max_depth=self.config.max_depth,
            num_tactics=self.config.num_tactics,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            max_goal_chars=self.config.max_goal_chars,
            lean_workers=self.config.lean_workers,
            time_budget_seconds=self.config.time_budget_seconds,
            max_expanded_nodes=self.config.max_expanded_nodes,
            max_verified_tactics=self.config.max_verified_tactics,
            server_timeout=self.config.server_timeout,
            max_hammers_per_node=self.config.max_hammers_per_node,
        )
