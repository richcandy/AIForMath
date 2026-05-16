from __future__ import annotations

from textual.widgets import Static

from tui.state import RunViewState


class HeaderBar(Static):
    def update_state(self, state: RunViewState) -> None:
        problem = state.problem_id or "no problem"
        self.update(
            f"AIForMath TUI | status={state.status} | problem={problem} | depth={state.current_depth} | "
            f"expanded={state.expanded_nodes} | verified={state.verified_tactics}"
        )
