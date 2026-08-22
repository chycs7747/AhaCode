from rich.markdown import Markdown
from rich.text import Text
from textual.widgets import Static

# Syntax theme for fenced code blocks — "nord" is soft/low-contrast (easier on the
# eyes than Rich's default "monokai"). Heading/inline-code colours are themed
# separately on the app console (see app.MARKDOWN_THEME).
_CODE_THEME = "nord"


class Chatbox(Static):
    """A single chat bubble, styled by role (user / assistant / thinking / …).

    Rendering is driven by render() (elia's Chatbox model,
    reference/elia/elia_chat/widgets/chatbox.py:358): assistant answers render as
    Rich Markdown so ```code``` fences become highlighted code blocks; everything
    else renders as plain Rich Text. Returning a Rich renderable (never a raw str)
    also sidesteps Textual's console-markup parsing — '[' in tool output or code
    can never be mistaken for a markup tag (the old MarkupError crash).
    """

    def __init__(self, content: str = "", role: str = "user", markdown: bool = False) -> None:
        super().__init__(markup=False)
        self._content = content
        self._rich = None          # a Rich renderable set by set_rich() (e.g. a diff)
        self._markdown = markdown  # render _content as Markdown (assistant answers)
        self.add_class(f"chatbox--{role}")

    def render(self):
        # A set_rich() renderable wins; then Markdown for answers; else plain text.
        if self._rich is not None:
            return self._rich
        if self._markdown and self._content:
            return Markdown(self._content, code_theme=_CODE_THEME)
        return Text(self._content)

    def append_chunk(self, chunk: str) -> None:
        """Append a streamed delta and re-render. Called from the worker via
        call_from_thread; Markdown bubbles re-parse live, as in elia."""
        if not self.display:  # a bubble born hidden reveals itself on the first delta
            self.display = True
        self._content += chunk
        self._rich = None
        self.refresh(layout=True)

    def set_rich(self, renderable, plain: str) -> None:
        """Display a Rich renderable (e.g. a coloured diff) while keeping a
        plain-text mirror in _content so logic/tests stay text-based."""
        self._content = plain
        self._rich = renderable
        self.refresh(layout=True)
