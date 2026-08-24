"""Text helpers shared by everything that has to keep something small.

`elide` in particular: several places cap a long string, and every one of them
wants the SAME shape — keep both ends, say how much went missing. Both ends,
because which one matters depends on the text: a test run's verdict is on the
last line, a directory listing's header is on the first.
"""

from __future__ import annotations

ELISION = "\n... [{n:,} chars elided] ...\n"


def elide(text: str, limit: int) -> str:
    """Shorten `text` to about `limit` chars by removing the middle.

    A no-op when it already fits, so callers can apply it unconditionally.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + ELISION.format(n=len(text) - limit) + text[-half:]


def line_count(text: str) -> int:
    """Lines in a blob, counting a missing trailing newline as a line."""
    return text.count("\n") + 1 if text else 0


# --- markdown line breaks --------------------------------------------------
# CommonMark treats a single newline inside a paragraph as a SPACE, so three lines
# the model wrote separately are re-wrapped into one solid block. Beside the grey
# bubbles — plain Text, every newline kept — the answer reads noticeably denser, and
# the difference is not spacing but lines being merged. Rich honours the two-space
# hard break (measured), so restoring the author's breaks costs no extra rows.
#
# Deliberately conservative: only ordinary prose lines are touched. Anything that is
# a block construct in its own right (fence, table, heading, list, quote, indented
# code) is left exactly as written, because there a trailing hard break either does
# nothing or changes how the block parses.

_FENCE = ("```", "~~~")
# Lines that open a block: a hard break before or after them is meaningless or unsafe.
_BLOCK_STARTS = ("|", "#", ">", "-", "*", "+", "=")


def _is_prose(line: str) -> bool:
    """True if the line is ordinary paragraph text that markdown would re-wrap."""
    stripped = line.strip()
    if not stripped:
        return False
    if line.startswith("    ") or line.startswith("\t"):
        return False  # indented code block
    if stripped[0] in _BLOCK_STARTS:
        return False
    first = stripped.split(".", 1)[0]
    if first.isdigit():
        return False  # ordered list item
    return True


def keep_line_breaks(text: str) -> str:
    """Make the author's single newlines survive markdown rendering.

    Appends a two-space hard break to a prose line that is followed by another prose
    line. Content inside fenced code blocks is passed through untouched, and a line
    that already ends in a hard break (two spaces or a backslash) is left alone.
    """
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith(_FENCE):
            in_fence = not in_fence
            out.append(line)
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if (
            not in_fence
            and _is_prose(line)
            and _is_prose(nxt)
            and not line.endswith("  ")
            and not line.endswith("\\")
        ):
            out.append(line + "  ")
        else:
            out.append(line)
    return "\n".join(out)
