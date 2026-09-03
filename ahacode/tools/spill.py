"""Spill oversized tool output to a file beside the session instead of truncating
it: the file is ordinary text, so `read` and `grep` can go back to it."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ahacode import workspace
from ahacode.text import elide, line_count

# Past this many characters a result is written to a file and only a preview of
# both ends comes back; with nowhere to write, it is elided in the middle instead.
SPILL_OVER_CHARS = 4_000
PREVIEW_CHARS = 2_000
MAX_INLINE_CHARS = 30_000
# The spilled file itself is capped, so a runaway command cannot fill the disk.
MAX_FILE_CHARS = 1_000_000

_session_dir: Path | None = None


def set_session(session_path: Path | None) -> None:
    """Point spills at the session that will own them (sessions/<id>-out/).

    Args:
        session_path: The open session's file, or None for the shared fallback.
    """
    global _session_dir
    _session_dir = (
        session_path.with_suffix("").with_name(session_path.stem + "-out")
        if session_path
        else None
    )


def target_dir() -> Path:
    """Where spills go; created on demand."""
    return _session_dir or (workspace.SESSIONS_DIR / "tool-output")


def write(text: str, prefix: str = "out") -> Path | None:
    """Save `text` to a uniquely named file.

    Args:
        text: The text; cut at MAX_FILE_CHARS.
        prefix: The file name prefix.

    Returns:
        The file's path, or None when it could not be written.
    """
    try:
        directory = target_dir()
        directory.mkdir(parents=True, exist_ok=True)
        # mkstemp claims a unique name atomically; parallel sub-agents may spill at once.
        fd, name = tempfile.mkstemp(dir=directory, prefix=f"{prefix}-", suffix=".txt")
        path = Path(name)
        with open(fd, "w", encoding="utf-8") as f:
            f.write(text[:MAX_FILE_CHARS])
        return path
    except OSError:
        return None


def preview(text: str, *, prefix: str, noun: str) -> str:
    """Hand `text` back whole when it is small, else spill it and return a preview.

    Args:
        text: The full tool output.
        prefix: The spill file's name prefix (the tool's name).
        noun: What the header calls the text ("output", "page").

    Returns:
        The text itself; or a header naming the saved file plus both ends of the
        text; or, when the file could not be written, the text elided in the middle.
    """
    if len(text) <= SPILL_OVER_CHARS:
        return text
    path = write(text, prefix=prefix)
    if path is None:
        return elide(text, MAX_INLINE_CHARS)
    where = workspace.display_path(path)
    header = (
        f"[{noun} was {len(text):,} chars / {line_count(text):,} lines — saved in full to {where}\n"
        f" read it with read(path=\"{where}\", offset=…, limit=…), "
        f"or search it with grep(pattern=…, path=\"{where}\")]\n"
    )
    return header + elide(text, PREVIEW_CHARS)
