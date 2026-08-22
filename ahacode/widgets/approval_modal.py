"""A modal that asks the user to approve a side-effecting tool call before it runs.

Roo Code and Claude Code both gate shell commands behind an explicit approve/reject
prompt; this is the same idea as a Textual ModalScreen. It returns a bool via
dismiss() — collected by the (worker-thread) caller through a threading.Event —
and can be answered by clicking a button or by the y/n keys.
"""

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ApprovalModal(ModalScreen[bool]):
    """Confirm one tool call. dismiss(True) = run, dismiss(False) = skip."""

    BINDINGS = [
        ("y", "approve", "Yes"),
        ("n", "deny", "No"),
        ("escape", "deny", "No"),
    ]

    def __init__(self, tool_name: str, summary: str) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._summary = summary

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-box"):
            yield Static(f"Run the {self._tool_name} tool?", id="approval-title")
            yield Static(self._summary, id="approval-cmd")
            with Horizontal(id="approval-buttons"):
                yield Button("Run  (y)", variant="success", id="approve-btn")
                yield Button("Skip  (n)", variant="error", id="deny-btn")

    @on(Button.Pressed, "#approve-btn")
    def _click_approve(self, event: Button.Pressed) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#deny-btn")
    def _click_deny(self, event: Button.Pressed) -> None:
        self.dismiss(False)

    def action_approve(self) -> None:  # y key
        self.dismiss(True)

    def action_deny(self) -> None:  # n / escape key
        self.dismiss(False)
