"""Shared file traversal for the search tools (grep, glob).

Both tools need the same thing: walk the project, but never wander into the
directories that would drown a search — a virtualenv, a vendored clone, a build
cache. Kept in one module so the skip rules can't drift between them.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

# Directories that are never worth searching: caches, build output, and vendored
# third-party source (a search hit inside a dependency is noise, and one clone can
# outweigh the whole project). Every dot-directory is skipped too — that covers
# .git/.venv/.pytest_cache without naming each one.
SKIP_DIRS = {
    "__pycache__", "node_modules", "venv", "dist", "build", "target",
    "reference",  # vendored reference clones, like node_modules for this project
    "sessions",   # private conversation transcripts; never search them
}

# A file bigger than this is data, not source — reading it would stall the search.
MAX_FILE_BYTES = 2_000_000


def is_skipped_dir(name: str) -> bool:
    """Should traversal refuse to descend into a directory of this name?"""
    return name in SKIP_DIRS or name.startswith(".")


def iter_files(root: Path, pattern: str = "**/*") -> Iterator[Path]:
    """Yield files under `root` matching a glob pattern, skipping the noise dirs.

    The skip list applies to directories *discovered* while walking, never to
    `root` itself — so an explicit search inside e.g. reference/ still works, while
    a project-wide search never wanders in.
    """
    for path in root.glob(pattern):
        if any(is_skipped_dir(part) for part in path.relative_to(root).parts[:-1]):
            continue
        if path.is_dir():
            if is_skipped_dir(path.name):
                continue
        elif not path.is_file():
            continue
        yield path


def read_text_or_none(path: Path) -> str | None:
    """A file's text, or None when it isn't searchable text.

    Binary files raise UnicodeDecodeError on decode — that IS the binary test here,
    rather than guessing from the extension. Unreadable files (permissions, a
    dangling symlink) are skipped the same way: a search must not crash on one file.
    """
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
