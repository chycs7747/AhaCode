"""A foldable reasoning block: expanded while streaming, folded once the answer begins."""

from __future__ import annotations

from textual.widgets import Collapsible

from ahacode.widgets.chatbox import Chatbox


class ThinkingBlock(Collapsible):
    """The 🤔 thinking block. Reasoning streams as plain text and is rendered as
    Markdown once, when the block folds, so code fences in the reasoning show as
    code blocks when a click reopens it."""

    def __init__(self) -> None:
        self._box = Chatbox("", role="thinking")
        super().__init__(self._box, title="🤔 thinking", collapsed=False)

    def append_chunk(self, chunk: str) -> None:
        """Forward a streamed reasoning delta to the inner bubble."""
        self._box.append_chunk(chunk)

    def done(self) -> None:
        """Fold the block away and render its text as Markdown; a click on the
        title reopens it."""
        self._box.set_markdown(True)
        self.collapsed = True
