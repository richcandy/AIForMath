from __future__ import annotations

from textual.widgets import Static

from tui.state import RunViewState


class BeamPanel(Static):
    def update_state(self, state: RunViewState) -> None:
        if not state.candidate_batches:
            self.update("Beam: no active nodes")
            return
        lines = ["Beam / Candidate Batches"]
        for batch in state.candidate_batches[:5]:
            lines.append(
                f"depth={batch.depth} steps={len(batch.proof_steps)} candidates={len(batch.candidates)}"
            )
            lines.append(batch.goal_text[:200] or "<empty goal>")
        self.update("\n\n".join(lines))
