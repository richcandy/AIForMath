from __future__ import annotations

from textual.widgets import Static

from tui.state import RunViewState


class LogPanel(Static):
    def update_state(self, state: RunViewState) -> None:
        logs = state.logs[-40:]
        if not logs:
            self.update("Logs: idle")
            return
        self.update("\n".join(logs))
