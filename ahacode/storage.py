"""JSONL session storage — one file per session, one message per line, append-only.

Sessions live under the project root (./sessions/), kept out of git.
"""

import datetime
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = PROJECT_ROOT / "sessions"


def new_session_path(base_dir: Path | None = None) -> Path:
    """Return a path for a new session file, atomically claiming the name by creating
    an empty file. The claim closes a race where two threads spawning sub-agents in
    the same second would otherwise compute the same name and clobber each other
    (parallel task fan-out); the header/messages are appended right after."""
    base_dir = base_dir or SESSIONS_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    # No colons in the timestamp — Windows forbids them in file names.
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = base_dir / f"{stamp}.jsonl"
    n = 2
    while True:
        try:
            # O_CREAT|O_EXCL: create-and-claim in one atomic step, so two racing
            # callers can never be handed the same path.
            path.touch(exist_ok=False)
            return path
        except FileExistsError:  # taken (existing session, or a concurrent claim) — bump
            path = base_dir / f"{stamp}_{n}.jsonl"
            n += 1


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
    """Most recent MAIN session to resume on startup, or None.

    Skips sub-agent (and fork) sessions: those are spawned by an agent as children,
    not conversations the user opened. A sub-agent session is depth-gated out of the
    `task` tool, so silently resuming into one (it may well be the newest file) makes
    delegation look broken. Legacy headerless files count as main and stay resumable.
    """
    base_dir = base_dir or SESSIONS_DIR
    if not base_dir.exists():
        return None
    # File names are timestamps, so reverse order == newest first.
    for path in sorted(base_dir.glob("*.jsonl"), reverse=True):
        header = read_header(path)
        if header is None or header.get("kind") == "main":
            return path
    return None


# --- session headers & hierarchy ------------------------------------------
# Each session file's FIRST line is a header carrying its place in the tree:
# {"type":"header","version":1,"id","parent_id","kind","depth","model","cwd","title"}
# A child points to its parent by id; the tree is
# derived by scanning headers — a parent never stores a child list.

HEADER_VERSION = 1


def make_header(
    session_id: str,
    *,
    parent_id: str | None = None,
    kind: str = "main",
    relation: str | None = None,
    depth: int = 0,
    model: str = "",
    title: str = "",
    cwd: Path | str | None = None,
) -> dict:
    """Build a session header.

    kind is the node's role: "main" | "plan" | "impl" | "subagent" | "fork".
    relation is the edge to the parent: "handoff" (control passed down a chain —
    plan → impl; the parent stops working) or "delegate" (a task fanned out while
    the parent waits). None for a root. depth counts delegate edges only — a
    handoff inherits the parent's depth, so the sub-agent cap keys off it.
    """
    return {
        "type": "header",
        "version": HEADER_VERSION,
        "id": session_id,
        "parent_id": parent_id,
        "kind": kind,
        "relation": relation,
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
