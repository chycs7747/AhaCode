from textual.widgets import Collapsible

from ahacode.widgets.chatbox import Chatbox

# One icon per tool, shown in the result card's header. Unknown tools fall back
# to a generic wrench.
_ICONS = {
    "read": "👓",
    "bash": "🖥",
    "grep": "🔍",
    "glob": "🗂",
    "list": "📂",
    "write": "📝",
    "edit": "✏",
    "todo_write": "🗒",
}

# Results longer than this fold by default so the chat stays scannable
# (over N lines → click to expand, keeps the chat scrollable).
_COLLAPSE_OVER = 12


class ToolResultBlock(Collapsible):
    """A tool result as a foldable card: a one-line header (icon · tool · size)
    over the output body.

    Long results and failures start collapsed; short successes stay open — a per-tool
    header plus a collapsible body, with long output auto-collapsing and a failure
    folding down to a dot. Reuses the same Textual Collapsible as ThinkingBlock, so
    the toggle/▼▶ come for free.
    """

    def __init__(
        self, name: str, output: str, is_error: bool = False, summary: str = ""
    ) -> None:
        icon = _ICONS.get(name, "🔧")
        lines = output.count("\n") + 1 if output else 0
        role = "tool-error" if is_error else "tool-result"
        prefix = "✘ " if is_error else ""
        # Title shows the INPUT (command / path) when known — that's the "IN" of a
        # Claude Code-style card — else the output size. Errors append " · failed".
        detail = summary or f"{lines} line{'s' if lines != 1 else ''}"
        if is_error:
            detail = f"{summary} · failed" if summary else "failed"
        # Keep a handle to the inner bubble so tests/logic stay text-based.
        self._box = Chatbox(output or "(no output)", role=role)
        collapsed = is_error or lines > _COLLAPSE_OVER
        super().__init__(
            self._box,
            title=f"{prefix}{icon} {name} · {detail}",
            collapsed=collapsed,
            classes="tool-block--error" if is_error else "tool-block--ok",
        )
