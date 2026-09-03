"""Text helpers shared by everything that has to keep a string small or readable."""

from __future__ import annotations

ELISION = "\n... [{n:,} chars elided] ...\n"


def elide(text: str, limit: int) -> str:
    """Shorten `text` to about `limit` characters by removing the middle.

    Both ends are kept because which one matters depends on the text: a test run's
    verdict is on the last line, a listing's header on the first.

    Args:
        text: The text.
        limit: The target length; 0 or less disables eliding.

    Returns:
        The text, unchanged when it already fits.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + ELISION.format(n=len(text) - limit) + text[-half:]


def line_count(text: str) -> int:
    """Lines in a blob, counting a missing trailing newline as a line."""
    return text.count("\n") + 1 if text else 0


# --- markdown line breaks --------------------------------------------------
# CommonMark folds a single newline inside a paragraph into a space, so lines the
# model wrote separately are re-wrapped into one block. A two-space hard break
# keeps them apart at no extra rows. Only ordinary prose lines are touched: a
# block construct (fence, table, heading, list, quote, indented code) is left as
# written, where a hard break either does nothing or changes how it parses.

_FENCE = ("```", "~~~")
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

    Args:
        text: Markdown text.

    Returns:
        The text with a two-space hard break appended to every prose line that is
        followed by another prose line. Fenced code and lines that already end in
        a hard break are left alone.
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
