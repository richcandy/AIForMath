from __future__ import annotations

from textual.widgets import Static

from tui.state import RunViewState


class TacticPanel(Static):
    def update_state(self, state: RunViewState) -> None:
        if not state.candidate_batches:
            self.update("Tactics: no data")
            return
        batch = state.candidate_batches[0]
        lines = [f"Latest depth={batch.depth}"]
        if batch.replay_error:
            lines.append(f"replay_error={batch.replay_error}")
        if batch.verification_results:
            for result in batch.verification_results[:12]:
                status = "ok" if result.ok else "fail"
                closed = " closed" if result.closed else ""
                lines.append(f"[{result.rank}] {result.source} {status}{closed}: {result.tactic}")
        else:
            for candidate in batch.candidates[:12]:
                lines.append(
                    f"[{candidate.get('rank')}] {candidate.get('source')}: {candidate.get('tactic')}"
                )
        self.update("\n".join(lines))
