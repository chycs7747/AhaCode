"""The multi-line prompt: Enter sends; Shift+Enter, Alt+Enter and Ctrl+J insert a newline."""

from __future__ import annotations

from dataclasses import dataclass

from textual import events
from textual.message import Message
from textual.widgets import TextArea


class PromptInput(TextArea):
    """The prompt box. Shift+Enter needs a terminal that speaks the Kitty keyboard
    protocol; the other two newline combos work everywhere."""

    @dataclass
    class Submitted(Message):
        """Posted on Enter or Send. `text` may be empty: a bare Enter is the app's
        to interpret (it answers an open plan gate)."""

        text: str

    _NEWLINE_KEYS = ("shift+enter", "alt+enter", "ctrl+j")

    def on_mount(self) -> None:
        self.show_line_numbers = False
        self.border_subtitle = "Enter to send · Shift+Enter for newline"

    def submit(self) -> None:
        """Post the current text and clear the box when it was not empty."""
        text = self.text.strip()
        self.post_message(self.Submitted(text))
        if text:
            self.clear()

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.submit()
            return
        if event.key in self._NEWLINE_KEYS:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)
