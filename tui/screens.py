from __future__ import annotations

from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, TabbedContent, TabPane

from tui.widgets.beam_panel import BeamPanel
from tui.widgets.header_bar import HeaderBar
from tui.widgets.log_panel import LogPanel
from tui.widgets.proof_panel import ProofPanel
from tui.widgets.problem_panel import ProblemPanel
from tui.widgets.result_panel import ResultPanel
from tui.widgets.root_goal_panel import RootGoalPanel
from tui.widgets.tactic_panel import TacticPanel


class WorkbenchScreen(Screen):
    def compose(self):
        yield HeaderBar(id="header-bar")
        with Horizontal(id="main-columns"):
            with Vertical(id="left-column"):
                yield ProblemPanel(id="problem-panel")
                with TabbedContent(id="left-tabs"):
                    with TabPane("Root Goal", id="root-goal-tab"):
                        yield RootGoalPanel(id="root-goal-panel")
                    with TabPane("Proof", id="proof-tab"):
                        yield ProofPanel(id="proof-panel")
                    with TabPane("Summary", id="summary-tab"):
                        yield ResultPanel(id="result-panel")
            with Vertical(id="right-column"):
                yield BeamPanel(id="beam-panel")
                yield TacticPanel(id="tactic-panel")
                yield LogPanel(id="log-panel")
        yield Footer()
