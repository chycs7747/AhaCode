"""A modal that asks the user to approve a side-effecting tool call before it runs.

The body shows a formatted preview of what the tool will do; the answer comes back
through dismiss(bool), which the worker thread waits on.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ahacode.render import tool_preview


class ApprovalModal(ModalScreen[bool]):
    """Confirm one tool call: dismiss(True) runs it, dismiss(False) skips it."""

    # Escape denies this one call, as closing a dialog does everywhere. While the
    # modal is up it also shadows the app's escape=stop binding, so the run gets an
    # explicit way out: s, or the Stop button.
    BINDINGS = [
        ("y", "approve", "Yes"),
        ("n", "deny", "No"),
        ("escape", "deny", "No"),
        ("s", "stop_run", "Stop the run"),
    ]

    def __init__(self, tool_name: str, arguments: dict) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._arguments = arguments

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-box"):
            yield Static(f"Run the {self._tool_name} tool?", id="approval-title")
            with VerticalScroll(id="approval-preview"):
                yield Static(tool_preview(self._tool_name, self._arguments), markup=False)
            with Horizontal(id="approval-buttons"):
                yield Button("Run  (y)", variant="success", id="approve-btn")
                yield Button("Skip  (n)", variant="error", id="deny-btn")
                yield Button("Stop  (s)", variant="warning", id="stop-btn")

    @on(Button.Pressed, "#approve-btn")
    def _click_approve(self, event: Button.Pressed) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#deny-btn")
    def _click_deny(self, event: Button.Pressed) -> None:
        self.dismiss(False)

    def action_approve(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#stop-btn")
    def _click_stop(self, event: Button.Pressed) -> None:
        self.action_stop_run()

    def action_deny(self) -> None:
        self.dismiss(False)

    def action_stop_run(self) -> None:
        """Skip this call and end the whole run. Cancel first, then dismiss: the
        worker wakes on the dismissal and must find the flag already set."""
        self.app.action_stop()
        self.dismiss(False)
