"""Spill oversized tool output to a file instead of throwing the middle away.

Truncation loses information permanently: the model that needed the elided part has
no way back to it, and re-running the command elides the same stretch again. Writing
the whole thing to a file and handing back a preview plus the path loses nothing —
and it costs no model call, unlike summarizing.

It fits this project in particular because the tools to go back already exist: the
spilled file is an ordinary text file, so `read` pages through it with offset/limit
and `grep` searches it. A large result stops being a context problem and becomes a
file problem, which the agent already knows how to solve.

Files live beside the session that produced them (sessions/<id>-out/) so they share
its lifetime, and sessions/ is git-ignored and skipped by the search tools already.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ahacode import storage
from ahacode.tools import base

# The spilled file itself is capped too: an unbounded command (`yes`, a runaway
# build log) must not be able to fill the disk.
MAX_FILE_CHARS = 1_000_000

_session_dir: Path | None = None


def set_session(session_path: Path | None) -> None:
    """Point spills at the session that will own them. Called by the app whenever
    the open session changes; `None` falls back to a shared directory."""
    global _session_dir
    _session_dir = (
        session_path.with_suffix("").with_name(session_path.stem + "-out")
        if session_path
        else None
    )


def target_dir() -> Path:
    """Where spills go. Created on demand — most sessions never spill at all."""
    return _session_dir or (storage.SESSIONS_DIR / "tool-output")


def write(text: str, prefix: str = "out") -> Path | None:
    """Save `text` and return its path, or None if it could not be written.

    mkstemp, not a hand-rolled name: several sub-agents can spill at the same
    moment, and it claims a unique name atomically. A failure here must never take
    down the tool — the caller falls back to plain truncation.
    """
    try:
        directory = target_dir()
        directory.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=directory, prefix=f"{prefix}-", suffix=".txt")
        path = Path(name)
        with open(fd, "w", encoding="utf-8") as f:  # utf-8 explicit (cp949 default on KR Windows)
            f.write(text[:MAX_FILE_CHARS])
        return path
    except OSError:
        return None


def relative(path: Path) -> str:
    """The path as the model should refer to it.

    Measured against the SAME root the tools resolve against (tools.base), not
    storage's — the whole point is that `read`/`grep` take this string straight back.
    Accessed through the module so the root stays swappable in tests.
    """
    root = base.PROJECT_ROOT
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
