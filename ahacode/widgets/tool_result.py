"""A tool result as a foldable card: icon · tool · input (or size) over the output."""

from __future__ import annotations

from textual.widgets import Collapsible

from ahacode.widgets.chatbox import Chatbox

_ICONS = {  # unknown tools get a wrench
    "read": "👓",
    "bash": "🖥",
    "grep": "🔍",
    "glob": "🗂",
    "write": "📝",
    "edit": "✏",
    "webfetch": "🌐",
    "todo_write": "🗒",
}

_COLLAPSE_OVER = 12  # results longer than this fold by default


class ToolResultBlock(Collapsible):
    """The result card. Long results and failures start collapsed; short successes
    stay open."""

    def __init__(
        self, name: str, output: str, is_error: bool = False, summary: str = ""
    ) -> None:
        icon = _ICONS.get(name, "🔧")
        lines = output.count("\n") + 1 if output else 0
        role = "tool-error" if is_error else "tool-result"
        prefix = "✘ " if is_error else ""
        detail = summary or f"{lines} line{'s' if lines != 1 else ''}"
        if is_error:
            detail = f"{summary} · failed" if summary else "failed"
        self._box = Chatbox(output or "(no output)", role=role)  # tests read its text
        collapsed = is_error or lines > _COLLAPSE_OVER
        super().__init__(
            self._box,
            title=f"{prefix}{icon} {name} · {detail}",
            collapsed=collapsed,
            classes="tool-block--error" if is_error else "tool-block--ok",
        )
