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

    # escape stays "deny THIS call" — closing a dialog is what escape means everywhere
    # else, and changing that would surprise. But while this modal is up it also
    # shadows the app's own escape=stop binding, and the modal screen swallows clicks
    # on the Stop button underneath, so the run became unstoppable exactly when a tool
    # (often a sub-agent's) was waiting to be approved: escape denied one call and the
    # loop simply asked for the next. Hence a third, explicit way out.
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
            # A scrollable, syntax-aware preview — long content scrolls instead of
            # overflowing the dialog.
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

    def action_approve(self) -> None:  # y key
        self.dismiss(True)

    @on(Button.Pressed, "#stop-btn")
    def _click_stop(self, event: Button.Pressed) -> None:
        self.action_stop_run()

    def action_deny(self) -> None:  # n / escape key
        self.dismiss(False)

    def action_stop_run(self) -> None:  # s key / Stop button
        """Skip this call AND end the whole run.

        Order matters: cancel first, then dismiss. The worker is blocked on the
        Event this dismissal sets, so it wakes the moment we dismiss — and it must
        find the cancellation flag already set, or it will run one more turn.
        """
        self.app.action_stop()
        self.dismiss(False)
