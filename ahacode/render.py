"""Widget-free Rich previews of tool calls, shared by the chat and the approval modal."""

from __future__ import annotations

import difflib
from pathlib import Path

from rich.console import Group, RenderableType
from rich.syntax import Syntax
from rich.text import Text

from ahacode import permissions

# Diff line colours with a subtle line background, so added and removed lines read
# as highlighted rows rather than coloured text.
_DIFF_STYLE = {
    "+": "#3fb950 on #0d2818",
    "-": "#f0665a on #2d1418",
    " ": "dim",
}

_CODE_THEME = "nord"  # the same low-contrast theme the chat uses for fences

_LEXERS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx",
    ".jsx": "jsx", ".json": "json", ".md": "markdown", ".sh": "bash",
    ".bash": "bash", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
    ".css": "css", ".tcss": "css", ".html": "html", ".sql": "sql",
    ".rs": "rust", ".go": "go", ".c": "c", ".cpp": "cpp",
}


def lexer_for(path: str) -> str:
    """The Pygments lexer for a path's extension, or plain text."""
    return _LEXERS.get(Path(path).suffix.lower(), "text")


def diff_rows(old: str, new: str) -> list[tuple[str, str]]:
    """A line diff of `old` against `new`.

    Args:
        old: The text before.
        new: The text after.

    Returns:
        (" " | "-" | "+", line) rows, in order.
    """
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


def tool_summary(name: str, args: dict) -> str:
    """A one-line summary of a tool call's input, for a result card's title.

    The subject is permissions.subject, the same string an allow rule is matched
    against, so a rule the user writes reads like the line they saw on screen.

    Args:
        name: The tool name.
        args: The call's arguments.

    Returns:
        The subject's first line, cut to 60 characters; "" when there is none.
    """
    raw = permissions.subject(name, args)
    first = raw.splitlines()[0] if raw else ""
    return first if len(first) <= 60 else first[:57] + "…"


def diff_stats(old: str, new: str) -> tuple[int, int]:
    """(added, removed) line counts for an edit, shown as the card's chip."""
    rows = diff_rows(old, new)
    return sum(s == "+" for s, _ in rows), sum(s == "-" for s, _ in rows)


def edit_diff_lines(old: str, new: str) -> tuple[Text, str]:
    """The coloured -/+ diff lines of an edit, without a path header.

    Args:
        old: The snippet replaced.
        new: Its replacement.

    Returns:
        (Rich text, the same lines as plain text).
    """
    text = Text()
    plain: list[str] = []
    for sign, line in diff_rows(old, new):
        prefix = (sign + " ") if sign != " " else "  "
        text.append(prefix + line + "\n", style=_DIFF_STYLE[sign])
        plain.append(prefix + line)
    return text, "\n".join(plain)


def edit_diff(path: str, old: str, new: str) -> tuple[Text, str]:
    """An edit's diff with a path header, for the approval modal.

    Args:
        path: The file being edited.
        old: The snippet replaced.
        new: Its replacement.

    Returns:
        (Rich text, the same as plain text).
    """
    header = f"🔧 edit · {path}"
    lines_text, lines_plain = edit_diff_lines(old, new)
    text = Text(header + "\n", style="bold")
    text.append(lines_text)
    return text, header + "\n" + lines_plain


def tool_preview(name: str, args: dict) -> RenderableType:
    """A readable preview of what a tool call will do: the approval modal's body.

    Args:
        name: The tool name.
        args: The call's arguments.

    Returns:
        write → a path header and the content as highlighted code; edit → the diff;
        bash → the command; anything else → "key: value" lines.
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
