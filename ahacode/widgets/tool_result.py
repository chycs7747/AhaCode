from textual.widgets import Collapsible

from ahacode.widgets.chatbox import Chatbox

# One icon per tool — the Textual analogue of kilocode's getToolInfo() switch
# (reference/kilocode/packages/kilo-ui/src/components/message-part.tsx:203) and
# hmm-code's buildToolCallBlock() (reference/hmm-code-vscode-main/webview/tools.ts:43).
# Unknown tools fall back to a generic wrench.
_ICONS = {
    "read": "👓",
    "bash": "🖥",
    "grep": "🔍",
    "list": "📂",
    "write": "📝",
    "edit": "✏",
    "todo_write": "🗒",
}

# Results longer than this fold by default so the chat stays scannable — the
# "over N lines… click to expand, keeps the chat scrollable" rule from hmm-code
# (reference/hmm-code-vscode-main/webview/tools.ts:35).
_COLLAPSE_OVER = 12


class ToolResultBlock(Collapsible):
    """A tool result as a foldable card: a one-line header (icon · tool · size)
    over the output body.

    Long results and failures start collapsed; short successes stay open. This is
    the pattern shared by every reference harness — a per-tool header plus a
    collapsible body: kilocode's Collapsible + getToolInfo, roocode's CodeAccordion
    (onToggleExpand), and hmm-code's long-output auto-collapse with a
    failure-collapses-to-a-dot (reference/hmm-code .../CHANGELOG.md:216). Reuses the
    same Textual Collapsible as ThinkingBlock, so the toggle/▼▶ come for free.
    """

    def __init__(self, name: str, output: str, is_error: bool = False) -> None:
        icon = _ICONS.get(name, "🔧")
        lines = output.count("\n") + 1 if output else 0
        role = "tool-error" if is_error else "tool-result"
        size = "failed" if is_error else f"{lines} line{'s' if lines != 1 else ''}"
        prefix = "✘ " if is_error else ""
        # Keep a handle to the inner bubble so tests/logic stay text-based and so
        # streamed content could later append here.
        self._box = Chatbox(output or "(no output)", role=role)
        collapsed = is_error or lines > _COLLAPSE_OVER
        super().__init__(
            self._box,
            title=f"{prefix}{icon} {name} · {size}",
            collapsed=collapsed,
            classes="tool-block--error" if is_error else "tool-block--ok",
        )
