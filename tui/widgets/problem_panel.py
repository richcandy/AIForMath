from __future__ import annotations

from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Select, Static

from tui.state import ProblemOption, RunConfig


class ProblemPanel(Vertical):
    DEFAULT_CSS = """
    ProblemPanel {
        border: solid $primary;
        padding: 1;
    }
    ProblemPanel > #problem-actions {
        width: 100%;
        align-horizontal: center;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._options_by_id: dict[str, ProblemOption] = {}

    def compose(self):
        yield Static("Problem")
        yield Static("Keys: `r` run, `s` stop, `q` quit")
        yield Select([], prompt="Select a problem", id="problem-select")
        with Horizontal(id="problem-actions"):
            yield Button("Start", id="start-search", variant="success")
            yield Button("Stop", id="stop-search", variant="error")

    def set_problem_options(self, options: list[ProblemOption]) -> None:
        self._options_by_id = {option.problem_id: option for option in options}
        select = self.query_one("#problem-select", Select)
        select.set_options([(option.label, option.problem_id) for option in options])
        if options:
            select.value = options[0].problem_id

    def selected_problem_id(self) -> str | None:
        value = self.query_one("#problem-select", Select).value
        return value if isinstance(value, str) else None

    def build_run_config(self, base_config: RunConfig) -> RunConfig:
        return RunConfig(**base_config.__dict__)
