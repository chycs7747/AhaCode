"""Rich previews for tool calls — shared by the chat (edit diffs) and the approval
modal (write content / edit diff / bash command). Widget-free so both can import
it with no cycle.

One per-tool formatted preview is reused everywhere: the same rendered content
feeds both the tool card and the permission prompt, rather than dumping raw
arguments into either.
"""

import difflib
from pathlib import Path

from rich.console import Group, RenderableType
from rich.syntax import Syntax
from rich.text import Text

# Diff line colours (GitHub-ish green/red) + a subtle line background, so added
# and removed lines read as highlighted rows (like Claude Code's / the artifact's
# edit diff), not just coloured text.
_DIFF_STYLE = {
    "+": "#3fb950 on #0d2818",
    "-": "#f0665a on #2d1418",
    " ": "dim",
}

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


def tool_summary(name: str, args: dict) -> str:
    """One-line summary of a tool call's INPUT for the result card's title —
    bash → the command, read/write/list → the path, grep/glob → the pattern. This
    is what turns a card into a Claude Code-style IN (title) / OUT (body) block."""
    if name == "bash":
        raw = args.get("command", "")
    elif name in ("grep", "glob"):
        raw = args.get("pattern", "")
    else:
        raw = args.get("path", "")
    raw = raw.strip()
    first = raw.splitlines()[0] if raw else ""
    return first if len(first) <= 60 else first[:57] + "…"


def diff_stats(old: str, new: str) -> tuple[int, int]:
    """(added, removed) line counts for an edit — shown as the card's chip."""
    rows = diff_rows(old, new)
    return sum(s == "+" for s, _ in rows), sum(s == "-" for s, _ in rows)


def edit_diff_lines(old: str, new: str) -> tuple[Text, str]:
    """Return (rich_text, plain_text) for just the coloured -/+ diff lines
    (no path header — the chat card carries that in its border title)."""
    text = Text()
    plain: list[str] = []
    for sign, line in diff_rows(old, new):
        prefix = (sign + " ") if sign != " " else "  "
        text.append(prefix + line + "\n", style=_DIFF_STYLE[sign])
        plain.append(prefix + line)
    return text, "\n".join(plain)


def edit_diff(path: str, old: str, new: str) -> tuple[Text, str]:
    """Return (rich_text, plain_text) for an edit's diff with a path header — used
    by the approval modal (the chat renders the header in the card border instead)."""
    header = f"🔧 edit · {path}"
    lines_text, lines_plain = edit_diff_lines(old, new)
    text = Text(header + "\n", style="bold")
    text.append(lines_text)
    return text, header + "\n" + lines_plain


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
