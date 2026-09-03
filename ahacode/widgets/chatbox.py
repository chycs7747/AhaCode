"""One chat bubble, styled by role."""

from __future__ import annotations

from rich.markdown import Markdown
from rich.text import Text
from textual.widgets import Static

from ahacode.text import keep_line_breaks

_CODE_THEME = "nord"  # low-contrast fences; headings are themed in app.MARKDOWN_THEME


class Chatbox(Static):
    """A chat bubble. Assistant answers render as Rich Markdown, everything else
    as plain Text. Returning a renderable rather than a str sidesteps console
    markup, so a '[' in tool output is never mistaken for a tag."""

    def __init__(self, content: str = "", role: str = "user", markdown: bool = False) -> None:
        super().__init__(markup=False)
        self._content = content
        self._rich = None  # a renderable set by set_rich(), e.g. a diff
        self._markdown = markdown
        self.add_class(f"chatbox--{role}")

    def render(self):
        if self._rich is not None:
            return self._rich
        if self._markdown and self._content:
            # keep_line_breaks: markdown would fold the author's single newlines.
            return Markdown(keep_line_breaks(self._content), code_theme=_CODE_THEME)
        return Text(self._content)

    def append_chunk(self, chunk: str) -> None:
        """Append a streamed delta and re-render.

        Args:
            chunk: The next piece of text.
        """
        if not self.display:  # a bubble born hidden reveals itself on the first delta
            self.display = True
        self._content += chunk
        self._rich = None
        self.refresh(layout=True)

    def set_rich(self, renderable, plain: str) -> None:
        """Display a Rich renderable, keeping a plain-text mirror for tests and logic.

        Args:
            renderable: What to draw.
            plain: The same content as text.
        """
        self._content = plain
        self._rich = renderable
        self.refresh(layout=True)
