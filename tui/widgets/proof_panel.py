from __future__ import annotations

from textual.widgets import Static

from tui.state import RunViewState


class ProofPanel(Static):
    DEFAULT_CSS = """
    ProofPanel {
        overflow-y: auto;
    }
    """

    def update_state(self, state: RunViewState) -> None:
        lines = ["Proof", ""]
        if state.proof_steps:
            lines.append("Final proof:")
            lines.extend(state.proof_steps)
        elif state.status in {"finished", "failed", "cancelled"}:
            lines.append("Final proof:")
            lines.append("<no complete proof produced>")
        else:
            lines.append("Final proof:")
            lines.append("<search still running>")

        if state.candidate_batches:
            latest = state.candidate_batches[0]
            if latest.proof_steps:
                lines.append("")
                lines.append("Latest explored prefix:")
                lines.extend(latest.proof_steps)
        self.update("\n".join(lines))
