from __future__ import annotations

from textual.widgets import Static

from tui.state import RunViewState


class ResultPanel(Static):
    DEFAULT_CSS = """
    ResultPanel {
        overflow-y: auto;
    }
    """

    def update_state(self, state: RunViewState) -> None:
        lines = [
            "Summary",
            "",
            f"status={state.status}",
            f"solved={state.solved}",
            f"failure={state.failure}",
            f"visited_states={state.visited_states}",
            f"expanded_nodes={state.expanded_nodes}",
            f"verified_tactics={state.verified_tactics}",
        ]
        if state.generation_errors:
            lines.append("")
            lines.append("Generation errors:")
            lines.extend(state.generation_errors[-3:])
        if state.verification_errors:
            lines.append("")
            lines.append("Verification errors:")
            lines.extend(state.verification_errors[-3:])
        self.update("\n".join(lines))
