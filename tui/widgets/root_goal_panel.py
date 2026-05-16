from __future__ import annotations

from textual.widgets import Static

from tui.state import RunViewState


class RootGoalPanel(Static):
    DEFAULT_CSS = """
    RootGoalPanel {
        overflow-y: auto;
    }
    """

    def update_state(self, state: RunViewState) -> None:
        lines = ["Root Goal", ""]
        if state.root_goal_text:
            lines.append(state.root_goal_text)
        else:
            lines.append("<root goal not loaded yet>")
        self.update("\n".join(lines))
