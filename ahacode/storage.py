"""JSONL session storage — one file per session, one message per line, append-only.

All generated data lives under one hidden folder, ./.ahacode/ (sessions, plans,
scratch, and config.toml), kept out of git as a single entry. A dot-prefixed name
also means walk.py skips it for free — private transcripts never turn up in a search.
"""

import datetime
import json
import shutil
from pathlib import Path

from ahacode import workspace


def new_session_path() -> Path:
    """Claim a path for a new session file by creating it empty.

    Creating the file is what makes the name unique: two sub-agents spawned in the
    same second would otherwise compute the same timestamp.

    Returns:
        The claimed, empty .jsonl path under workspace.SESSIONS_DIR.
    """
    workspace.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")  # no colons: Windows
    path = workspace.SESSIONS_DIR / f"{stamp}.jsonl"
    n = 2
    while True:
        try:
            path.touch(exist_ok=False)
            return path
        except FileExistsError:
            path = workspace.SESSIONS_DIR / f"{stamp}_{n}.jsonl"
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


# Session kinds the user drives (and so may be resumed into on startup). A
# sub-agent transcript is machine-authored and view-only.
RESUMABLE_KINDS = frozenset({"main", "impl"})


def latest_session() -> Path | None:
    """Most recent session the user was driving, to resume on startup, or None.

    An impl session counts: after a Ctrl+D mid-plan the newest file IS the impl
    session, and reopening its planning parent instead would invite "이어서 해"
    in plan mode — which cannot act and would spawn a fresh sibling instead.
    Skips sub-agent sessions: those are spawned by an agent as children,
    not conversations the user opened, and are depth-gated out of the `task` tool.
    Legacy headerless files count as main and stay resumable.
    """
    if not workspace.SESSIONS_DIR.exists():
        return None
    # File names are timestamps, so reverse order == newest first.
    for path in sorted(workspace.SESSIONS_DIR.glob("*.jsonl"), reverse=True):
        header = read_header(path)
        if header is None or header.get("kind") in RESUMABLE_KINDS:
            return path
    return None


# --- session headers & hierarchy ------------------------------------------
# Each session file's first line is a header carrying its place in the tree:
# {"type":"header","id","parent_id","kind","relation","depth","model","cwd","title"}
# A child points to its parent by id; a parent never stores a child list.


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

    kind is the node's role: "main" | "impl" | "subagent".
    relation is the edge to the parent: "handoff" (control passed down a chain —
    plan → impl; the parent stops working) or "delegate" (a task fanned out while
    the parent waits). None for a root. depth counts delegate edges only — a
    handoff inherits the parent's depth, so the sub-agent cap keys off it.
    """
    return {
        "type": "header",
        "id": session_id,
        "parent_id": parent_id,
        "kind": kind,
        "relation": relation,
        "depth": depth,
        "model": model,
        "cwd": str(cwd or workspace.PROJECT_ROOT),
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


def list_sessions() -> list[dict]:
    """Every session's header (newest last), synthesizing one for legacy files."""
    if not workspace.SESSIONS_DIR.exists():
        return []
    return [
        read_session_meta(path) or _legacy_header(path)
        for path in sorted(workspace.SESSIONS_DIR.glob("*.jsonl"))
    ]


def descendants(session_id: str, sessions: list[dict]) -> list[str]:
    """`session_id` and every session below it (children, grandchildren…), BFS.
    A parent never stores a child list, so this walks parent_id pointers."""
    by_parent: dict[str | None, list[str]] = {}
    for s in sessions:
        by_parent.setdefault(s.get("parent_id"), []).append(s["id"])
    out, queue = [], [session_id]
    while queue:
        sid = queue.pop(0)
        out.append(sid)
        queue.extend(by_parent.get(sid, []))
    return out


def delete_session(session_id: str) -> list[str]:
    """Delete a session and everything that hangs off it: its descendants (a child
    transcript without its parent is noise), each one's spilled tool output, and
    the plan / result files named after it. Returns the ids removed."""
    ids = descendants(session_id, list_sessions())
    for sid in ids:
        (workspace.SESSIONS_DIR / f"{sid}.jsonl").unlink(missing_ok=True)
        shutil.rmtree(workspace.SESSIONS_DIR / f"{sid}-out", ignore_errors=True)
        for extra in (workspace.PLANS_DIR / f"{sid}.md", workspace.PLANS_DIR / f"{sid}.result.md"):
            extra.unlink(missing_ok=True)
    return ids


def dangling_tool_calls(messages: list[dict]) -> list[dict]:
    """The last assistant turn's tool calls that never got a result — the mark of a
    turn cut off mid-tool (a Stop or Ctrl+D while a tool ran). Each is returned with
    its id, name and parsed arguments, so a resume can fill a result AND name which
    call it was (a bare id says nothing when three bash calls ran at once). The API
    requires a result for every tool_call, so these must be filled before continuing."""
    answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    last = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
    out: list[dict] = []
    for c in (last.get("tool_calls") or []) if last else []:
        if c["id"] in answered:
            continue
        fn = c.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        out.append({"id": c["id"], "name": fn.get("name", "tool"), "arguments": args})
    return out

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


def plan_path(session_path: Path) -> Path:
    """Where the plan written in `session_path` lives: plans/{session}.md."""
    return workspace.PLANS_DIR / f"{session_path.stem}.md"


def result_path(plan: Path) -> Path:
    """The progress/result file that sits beside a plan: plans/{session}.result.md."""
    return plan.with_name(f"{plan.stem}.result.md")


def write_result(
    path: Path, *, plan: Path, session_id: str, steps: list[str], done: int, summary: str,
) -> None:
    """Snapshot an impl session's progress beside its plan.

    Rewritten after every turn, so the file always says where the plan stands. Kept
    apart from the plan so the plan stays the baseline a review compares against.

    Args:
        path: The result file (see result_path).
        plan: The plan file the session is carrying out.
        session_id: The impl session's id.
        steps: The checklist, one rendered line per step (glyph + text).
        done: How many of `steps` are finished.
        summary: The model's latest answer, or "".
    """
    head = "완료" if done == len(steps) else f"진행 중 {done}/{len(steps)}"
    lines = [
        f"# {head} — {plan.name}",
        "",
        f"- plan: {workspace.display_path(plan)}",
        f"- session: {session_id}",
        "",
        "## Steps",
        "",
        *steps,
    ]
    if summary:
        lines += ["", "## Latest summary", "", summary]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plan_title(path: Path) -> str:
    """The plan's summary line ("# …"), for naming the session that carries it out."""
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return ""
    return first.removeprefix("#").strip()


def write_plan(
    path: Path, *, summary: str, steps: list[str], validation: list[str], body: str
) -> None:
    """Render a plan as Markdown and write it (whole file, last write wins — a
    revised plan from the same session replaces the previous one)."""
    lines = [f"# {summary or 'Plan'}", "", "## Steps", ""]
    # Plain numbers, no checkboxes: the plan is the spec a review compares against
    # and is never ticked. Progress goes to the sibling result file (write_result).
    lines += [f"{i}. {s}" for i, s in enumerate(steps, 1)]
    if validation:
        lines += ["", "## Validation", ""]
        lines += [f"- {v}" for v in validation]
    if body:
        lines += ["", "## Notes", "", body]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
