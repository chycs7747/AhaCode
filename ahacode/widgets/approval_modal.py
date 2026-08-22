"""A modal that asks the user to approve a side-effecting tool call before it runs.

Roo Code and Claude Code both gate shell commands behind an explicit approve/reject
prompt; this is the same idea as a Textual ModalScreen. It returns a bool via
dismiss(), which the (worker-thread) caller collects through a threading.Event.
"""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static


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
            yield Static("[y] run     [n] skip", id="approval-hint")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)
