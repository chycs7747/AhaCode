"""A modal that asks the user to approve a side-effecting tool call before it runs.

Side-effecting tools are gated behind an explicit approve/reject prompt (as in
Claude Code); this is a Textual ModalScreen. It returns
a bool via dismiss() — collected by the (worker-thread) caller through a
threading.Event — and can be answered by clicking a button or the y/n keys.

The body shows a *formatted preview* of what the tool will do (write → the file
content as code, edit → a -/+ diff, bash → the command) instead of a raw repr of
the arguments, and scrolls when that preview is long (the same formatted preview
the tool result card shows).
"""

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ahacode.render import tool_preview


class ApprovalModal(ModalScreen[bool]):
    """Confirm one tool call. dismiss(True) = run, dismiss(False) = skip."""

    BINDINGS = [
        ("y", "approve", "Yes"),
        ("n", "deny", "No"),
        ("escape", "deny", "No"),
    ]

    def __init__(self, tool_name: str, arguments: dict) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._arguments = arguments

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-box"):
            yield Static(f"Run the {self._tool_name} tool?", id="approval-title")
            # A scrollable, syntax-aware preview — long content scrolls instead of
            # overflowing the dialog.
            with VerticalScroll(id="approval-preview"):
                yield Static(tool_preview(self._tool_name, self._arguments), markup=False)
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
