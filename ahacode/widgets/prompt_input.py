from dataclasses import dataclass

from textual import events
from textual.message import Message
from textual.widgets import TextArea


class PromptInput(TextArea):
    """A multi-line prompt box. Enter sends; Shift+Enter / Alt+Enter / Ctrl+J
    insert a newline.

    Textual's Input is single-line, so — like elia
    (reference/elia/elia_chat/widgets/prompt_input.py) — we subclass TextArea for
    multi-line editing. elia submits on Ctrl+J; we bind Enter to send so it feels
    like Claude Code, keeping the other combos for newlines.

    Terminal note (a CS gotcha): historically a terminal sends the SAME bytes for
    Enter and Shift+Enter, so it can't tell them apart. Shift+Enter-for-newline
    only works where the Kitty keyboard protocol is active (kitty / WezTerm /
    Ghostty / recent others); Alt+Enter and Ctrl+J insert a newline everywhere.
    """

    @dataclass
    class Submitted(Message):
        """Posted when the user presses Enter with non-empty text."""

        text: str

    _NEWLINE_KEYS = ("shift+enter", "alt+enter", "ctrl+j")

    def on_mount(self) -> None:
        self.show_line_numbers = False
        self.border_subtitle = "Enter to send · Shift+Enter for newline"

    async def _on_key(self, event: events.Key) -> None:
        # Enter sends; the newline combos insert a line break; everything else is
        # normal TextArea editing (printable insert, cursor moves, …).
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
                self.clear()
            return
        if event.key in self._NEWLINE_KEYS:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)
