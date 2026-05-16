from __future__ import annotations

import asyncio
from collections.abc import Callable

from tui.services.event_bus import SearchEvent, format_event_message
from tui.services.search_service import SearchService
from tui.state import CandidateBatch, RunConfig, RunViewState, VerificationRecord


class AppController:
    def __init__(
        self,
        config: RunConfig,
        on_state_change: Callable[[RunViewState], None],
    ) -> None:
        self.config = config
        self.on_state_change = on_state_change
        self.state = RunViewState()
        self.service = SearchService(config)
        self.current_task: asyncio.Task | None = None

    def load_problems(self):
        return self.service.list_problems()

    def start_search(self, problem_id: str | None, question: str) -> None:
        if self.current_task is not None and not self.current_task.done():
            self.state.logs.append("Search already running")
            self._publish_state()
            return
        self.state = RunViewState(status="running", problem_id=problem_id, question=question)
        self._publish_state()
        self.current_task = asyncio.create_task(self._run_search(question))

    def cancel_search(self) -> None:
        if self.current_task is None or self.current_task.done():
            return
        self.current_task.cancel()
        self.state.status = "cancelled"
        self.state.logs.append("Search cancelled")
        self._publish_state()

    async def _run_search(self, question: str) -> None:
        try:
            result = await self.service.run_problem(question, on_event=self.handle_event)
        except asyncio.CancelledError:
            self.state.status = "cancelled"
            self._publish_state()
            raise
        except Exception as exc:
            self.state.status = "failed"
            self.state.failure = str(exc)
            self.state.logs.append(f"Search crashed: {exc}")
            self._publish_state()
            return
        finally:
            self.current_task = None

        self.state.result_payload = result
        self.state.solved = bool(result.get("solved"))
        self.state.failure = result.get("failure")
        self.state.proof_steps = list(result.get("proof_steps") or [])
        self.state.expanded_nodes = int(result.get("expanded_nodes") or 0)
        self.state.verified_tactics = int(result.get("verified_tactics") or 0)
        self.state.visited_states = int(result.get("visited_states") or 0)
        self.state.layer_summaries = list(result.get("layer_summaries") or [])
        self.state.status = "solved" if self.state.solved else "finished"
        self._publish_state()

    def handle_event(self, event: SearchEvent) -> None:
        self.state.logs.append(format_event_message(event))

        if event.type == "root_state_loaded":
            self.state.root_goal_text = str(event.payload.get("goal_text") or "")

        elif event.type == "layer_started":
            self.state.current_depth = int(event.payload.get("depth") or 0)
            self.state.expanded_nodes = int(event.payload.get("expanded_nodes") or 0)
            self.state.verified_tactics = int(event.payload.get("verified_tactics") or 0)

        elif event.type == "layer_generation_completed":
            self.state.generation_errors = list(event.payload.get("generation_errors") or [])
            self.state.candidate_batches = [
                CandidateBatch(
                    depth=int(batch.get("depth") or 0),
                    proof_steps=list(batch.get("proof_steps") or []),
                    goal_text=str(batch.get("goal_text") or ""),
                    candidates=list(batch.get("candidates") or []),
                )
                for batch in event.payload.get("node_batches") or []
            ]

        elif event.type == "node_verification_completed":
            replay_error = event.payload.get("replay_error")
            if replay_error:
                self.state.verification_errors.append(str(replay_error))
            next_prefix = list(event.payload.get("proof_steps") or [])
            batch = CandidateBatch(
                depth=int(event.payload.get("depth") or 0),
                proof_steps=next_prefix,
                goal_text=str(event.payload.get("goal_text") or ""),
                replay_error=str(replay_error) if replay_error else None,
                verification_results=[
                    VerificationRecord(
                        rank=int(result.get("rank") or 0),
                        tactic=str(result.get("tactic") or ""),
                        source=str(result.get("source") or ""),
                        ok=bool(result.get("ok")),
                        closed=bool(result.get("closed")),
                        error=result.get("error"),
                        next_goal_text=result.get("next_goal_text"),
                    )
                    for result in event.payload.get("results") or []
                ],
            )
            self.state.candidate_batches = [batch] + self.state.candidate_batches[:9]

        elif event.type == "layer_completed":
            self.state.layer_summaries.append(dict(event.payload))

        elif event.type == "solution_found":
            self.state.proof_steps = list(event.payload.get("proof_steps") or [])
            self.state.solved = True

        elif event.type == "search_finished":
            self.state.failure = event.payload.get("failure")

        self._publish_state()

    def _publish_state(self) -> None:
        self.on_state_change(self.state)
