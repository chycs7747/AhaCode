"""JSONL session storage — one file per session, one message per line, append-only.

Sessions live under the project root (./sessions/), kept out of git.
"""

import datetime
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = PROJECT_ROOT / "sessions"


def new_session_path(base_dir: Path | None = None) -> Path:
    """Return a path for a new session file (the file is created on first append)."""
    base_dir = base_dir or SESSIONS_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    # No colons in the timestamp — Windows forbids them in file names.
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return base_dir / f"{stamp}.jsonl"


def append_message(path: Path, message: dict) -> None:
    """Append one message as a single JSON line."""
    # Explicit utf-8: the platform default may differ (e.g. cp949 on Korean Windows).
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(message, ensure_ascii=False) + "\n")


def load_messages(path: Path) -> list[dict]:
    """Read a session file back into a messages list."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_session(base_dir: Path | None = None) -> Path | None:
    """Most recent session file, or None — used to resume on startup."""
    base_dir = base_dir or SESSIONS_DIR
    if not base_dir.exists():
        return None
    # File names are timestamps, so lexical order == chronological order.
    files = sorted(base_dir.glob("*.jsonl"))
    return files[-1] if files else None
