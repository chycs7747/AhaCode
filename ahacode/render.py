"""Rich previews for tool calls — shared by the chat (edit diffs) and the approval
modal (write content / edit diff / bash command). Widget-free so both can import
it with no cycle.

Mirrors the reference pattern of one per-tool formatted preview reused everywhere:
hmm-code's renderEditOrWriteBody (reference/hmm-code-vscode-main/webview/tools.ts:128)
feeds both the tool card and the permission prompt; roocode shows the same content
in a scrollable CodeAccordion rather than dumping raw arguments.
"""

import difflib
from pathlib import Path

from rich.console import Group, RenderableType
from rich.syntax import Syntax
from rich.text import Text

# Diff line colours (GitHub-ish green/red), matching the chat's edit diff.
_DIFF_STYLE = {"+": "#2ea043", "-": "#f85149", " ": "dim"}

# Syntax theme for previews — same low-contrast "nord" the chat uses for fences.
_CODE_THEME = "nord"

_LEXERS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx",
    ".jsx": "jsx", ".json": "json", ".md": "markdown", ".sh": "bash",
    ".bash": "bash", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
    ".css": "css", ".tcss": "css", ".html": "html", ".sql": "sql",
    ".rs": "rust", ".go": "go", ".c": "c", ".cpp": "cpp",
}


def lexer_for(path: str) -> str:
    """Pygments lexer guessed from a path's extension (default: plain text)."""
    return _LEXERS.get(Path(path).suffix.lower(), "text")


def diff_rows(old: str, new: str) -> list[tuple[str, str]]:
    """LCS line diff -> list of (" " | "-" | "+", line)."""
    o, n = old.splitlines(), new.splitlines()
    rows: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=o, b=n).get_opcodes():
        if tag == "equal":
            rows += [(" ", ln) for ln in o[i1:i2]]
        elif tag == "delete":
            rows += [("-", ln) for ln in o[i1:i2]]
        elif tag == "insert":
            rows += [("+", ln) for ln in n[j1:j2]]
        else:  # replace
            rows += [("-", ln) for ln in o[i1:i2]]
            rows += [("+", ln) for ln in n[j1:j2]]
    return rows


def edit_diff(path: str, old: str, new: str) -> tuple[Text, str]:
    """Return (rich_text, plain_text) for an edit's coloured -/+ diff."""
    header = f"🔧 edit · {path}"
    text = Text(header + "\n", style="bold")
    plain = [header]
    for sign, line in diff_rows(old, new):
        prefix = (sign + " ") if sign != " " else "  "
        text.append(prefix + line + "\n", style=_DIFF_STYLE[sign])
        plain.append(prefix + line)
    return text, "\n".join(plain)


def tool_preview(name: str, args: dict) -> RenderableType:
    """A readable preview of what a tool call will do — the approval-modal body.

    write -> path header + content as syntax-highlighted code (real newlines, not a
    repr with literal \\n); edit -> the -/+ diff; bash -> the command; anything
    else -> readable "key: value" lines. This is what makes the prompt legible.
    """
    if name == "write":
        path = args.get("path", "?")
        content = args.get("content", "")
        return Group(
            Text(f"📝 {path}", style="bold"),
            Syntax(content, lexer_for(path), theme=_CODE_THEME, word_wrap=True),
        )
    if name == "edit":
        text, _ = edit_diff(
            args.get("path", "?"), args.get("old_string", ""), args.get("new_string", "")
        )
        return text
    if name == "bash":
        return Text(f"$ {args.get('command', '')}", style="bold")
    return Text("\n".join(f"{k}: {v}" for k, v in args.items()))
