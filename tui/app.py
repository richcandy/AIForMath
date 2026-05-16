from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Button, Select

from tui.controller import AppController
from tui.screens import WorkbenchScreen
from tui.state import ProblemOption, RunConfig, RunViewState
from tui.widgets.beam_panel import BeamPanel
from tui.widgets.header_bar import HeaderBar
from tui.widgets.log_panel import LogPanel
from tui.widgets.proof_panel import ProofPanel
from tui.widgets.problem_panel import ProblemPanel
from tui.widgets.result_panel import ResultPanel
from tui.widgets.root_goal_panel import RootGoalPanel
from tui.widgets.tactic_panel import TacticPanel


class LeanSearchApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    WorkbenchScreen {
        layout: vertical;
    }
    #main-columns {
        height: 1fr;
    }
    #left-column {
        width: 33%;
        height: 1fr;
    }
    #right-column {
        width: 67%;
    }
    #header-bar, #beam-panel, #tactic-panel, #log-panel, #result-panel {
        border: solid $primary;
        padding: 1;
    }
    #problem-panel {
        height: 14;
        min-height: 14;
    }
    #left-tabs {
        height: 1fr;
        min-height: 4;
    }
    #root-goal-panel, #proof-panel, #result-panel {
        border: solid $primary;
        padding: 1;
        height: 1fr;
        min-height: 0;
    }
    #beam-panel, #tactic-panel, #log-panel, #result-panel {
        height: 1fr;
    }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "start_search", "Run"),
        Binding("s", "stop_search", "Stop"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.base_config = RunConfig(project_dir=Path("."), input_root=Path("MiniF2F_Highschool"))
        self.controller = AppController(self.base_config, on_state_change=self._on_state_change)
        self.problem_index: dict[str, ProblemOption] = {}

    def compose(self) -> ComposeResult:
        if False:
            yield

    async def on_mount(self) -> None:
        await self.push_screen(WorkbenchScreen())
        problems = self.controller.load_problems()
        self.problem_index = {option.problem_id: option for option in problems}
        self.screen.query_one(ProblemPanel).set_problem_options(problems)
        self._on_state_change(self.controller.state)

    def _on_state_change(self, state: RunViewState) -> None:
        self.screen.query_one(HeaderBar).update_state(state)
        self.screen.query_one(BeamPanel).update_state(state)
        self.screen.query_one(TacticPanel).update_state(state)
        self.screen.query_one(LogPanel).update_state(state)
        self.screen.query_one(RootGoalPanel).update_state(state)
        self.screen.query_one(ProofPanel).update_state(state)
        self.screen.query_one(ResultPanel).update_state(state)

    def action_start_search(self) -> None:
        problem_panel = self.screen.query_one(ProblemPanel)
        selected_problem_id = problem_panel.selected_problem_id()
        if selected_problem_id is None or selected_problem_id not in self.problem_index:
            self.notify("Select a problem first", severity="warning")
            return
        self.controller.config = problem_panel.build_run_config(self.base_config)
        self.controller.service.config = self.controller.config
        option = self.problem_index[selected_problem_id]
        self.controller.start_search(selected_problem_id, option.question)

    def action_stop_search(self) -> None:
        self.controller.cancel_search()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-search":
            self.action_start_search()
        elif event.button.id == "stop-search":
            self.action_stop_search()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "problem-select":
            return


def main() -> None:
    LeanSearchApp().run()


if __name__ == "__main__":
    main()
