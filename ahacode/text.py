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
