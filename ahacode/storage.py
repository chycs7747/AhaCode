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
    path = base_dir / f"{stamp}.jsonl"
    # Two sessions in the same second would collide — bump with a suffix.
    n = 2
    while path.exists():
        path = base_dir / f"{stamp}_{n}.jsonl"
        n += 1
    return path


def append_message(path: Path, message: dict) -> None:
    """Append one message as a single JSON line."""
    # Explicit utf-8: the platform default may differ (e.g. cp949 on Korean Windows).
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(message, ensure_ascii=False) + "\n")


def load_messages(path: Path) -> list[dict]:
    """Read a session file back into a messages list.

    Metadata lines (the header, later title updates) carry a "type" field and are
    skipped — only chat messages (which have "role", not "type") are returned.
    """
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("type"):  # header / title / other metadata, not a message
                continue
            out.append(obj)
    return out


def latest_session(base_dir: Path | None = None) -> Path | None:
    """Most recent session file, or None — used to resume on startup."""
    base_dir = base_dir or SESSIONS_DIR
    if not base_dir.exists():
        return None
    # File names are timestamps, so lexical order == chronological order.
    files = sorted(base_dir.glob("*.jsonl"))
    return files[-1] if files else None


# --- session headers & hierarchy ------------------------------------------
# Each session file's FIRST line is a header carrying its place in the tree:
# {"type":"header","version":1,"id","parent_id","kind","depth","model","cwd","title"}
# A child points to its parent by id (pi's parentSessionId model); the tree is
# derived by scanning headers — a parent never stores a child list.

HEADER_VERSION = 1


def make_header(
    session_id: str,
    *,
    parent_id: str | None = None,
    kind: str = "main",
    depth: int = 0,
    model: str = "",
    title: str = "",
    cwd: Path | str | None = None,
) -> dict:
    """Build a session header. kind is "main" | "subagent" | "fork"."""
    return {
        "type": "header",
        "version": HEADER_VERSION,
        "id": session_id,
        "parent_id": parent_id,
        "kind": kind,
        "depth": depth,
        "model": model,
        "cwd": str(cwd or PROJECT_ROOT),
        "title": title,
    }


def write_header(path: Path, header: dict) -> None:
    """Write the header as the file's first line. Call once, on a fresh session."""
    with path.open("a", encoding="utf-8") as f:  # append == first line on a new file
        f.write(json.dumps(header, ensure_ascii=False) + "\n")


def read_header(path: Path) -> dict | None:
    """Return a session's header (first line), or None for a headerless/legacy file."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        first = f.readline()
    if not first.strip():
        return None
    try:
        obj = json.loads(first)
    except json.JSONDecodeError:
        return None
    return obj if obj.get("type") == "header" else None


def set_title(path: Path, title: str) -> None:
    """Record/replace a session's title as an append-only metadata line (last wins)."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "title", "title": title}, ensure_ascii=False) + "\n")


def read_session_meta(path: Path) -> dict | None:
    """The header with its title overridden by the latest {type:"title"} line.

    None for a headerless/legacy file (caller falls back to a synthesized header).
    """
    header = read_header(path)
    if header is None:
        return None
    title = header.get("title", "")
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "title" and isinstance(obj.get("title"), str):
                title = obj["title"]
    return {**header, "title": title}


def _legacy_header(path: Path) -> dict:
    """Synthesize a header for a pre-header file so old sessions still list/tree."""
    msgs = load_messages(path)
    title = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
    return make_header(path.stem, kind="main", title=(title[:40] or path.stem))


def list_sessions(base_dir: Path | None = None) -> list[dict]:
    """Every session's header (newest last), synthesizing one for legacy files."""
    base_dir = base_dir or SESSIONS_DIR
    if not base_dir.exists():
        return []
    return [
        read_session_meta(path) or _legacy_header(path)
        for path in sorted(base_dir.glob("*.jsonl"))
    ]


def build_tree(sessions: list[dict]) -> list[dict]:
    """Nest a flat header list into a tree by parent_id.

    Returns the root nodes; each node is its header plus a sorted "children" list.
    A header whose parent_id is missing/unknown is treated as a root (orphan-safe).
    """
    by_id = {s["id"]: {**s, "children": []} for s in sessions}
    roots: list[dict] = []
    for s in sorted(sessions, key=lambda h: h["id"]):
        node = by_id[s["id"]]
        parent = s.get("parent_id")
        if parent and parent in by_id:
            by_id[parent]["children"].append(node)
        else:
            roots.append(node)
    return roots
