"""JSONL session files, and the plan files that sit beside them."""

from __future__ import annotations

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
    """Append one message as a single JSON line.

    Args:
        path: The session file.
        message: An OpenAI-shaped chat message.
    """
    with path.open("a", encoding="utf-8") as f:  # explicit utf-8: cp949 on Korean Windows
        f.write(json.dumps(message, ensure_ascii=False) + "\n")


def load_messages(path: Path) -> list[dict]:
    """Read a session file back into a messages list.

    Metadata lines (the header, title updates) carry a "type" field and are skipped.

    Args:
        path: The session file; a missing file reads as empty.

    Returns:
        The chat messages, in order.
    """
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("type"):
                continue
            out.append(obj)
    return out


# Session kinds the user drives, and so may be resumed into on startup.
RESUMABLE_KINDS = frozenset({"main", "impl"})


def latest_session() -> Path | None:
    """The newest session the user was driving, to resume on startup.

    An impl session counts: after a quit mid-plan it is the one to reopen. A
    sub-agent session is a machine-authored child and is skipped. A headerless
    (legacy) file counts as main.

    Returns:
        The session file, or None when there is none.
    """
    if not workspace.SESSIONS_DIR.exists():
        return None
    for path in sorted(workspace.SESSIONS_DIR.glob("*.jsonl"), reverse=True):  # newest first
        header = read_header(path)
        if header is None or header.get("kind") in RESUMABLE_KINDS:
            return path
    return None


# --- session headers & hierarchy ------------------------------------------
# A session file's first line is a header carrying its place in the tree: a child
# points to its parent by id, and a parent never stores a child list.


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

    Args:
        session_id: The file stem.
        parent_id: The session this one hangs off, or None for a root.
        kind: "main" | "impl" | "subagent".
        relation: The edge to the parent: "handoff" (plan → impl; the parent stops
            working) or "delegate" (a task fanned out while the parent waits).
            None for a root.
        depth: Delegate edges above this session. A handoff keeps its parent's
            depth, so the sub-agent cap keys off it.
        model: The model name at creation.
        title: The display title, if known.
        cwd: The project root; workspace.PROJECT_ROOT when omitted.

    Returns:
        The header dict, ready for write_header.
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
    """Write the header as a fresh session file's first line.

    Args:
        path: The empty session file.
        header: From make_header.
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")


def read_header(path: Path) -> dict | None:
    """A session's header line.

    Args:
        path: The session file.

    Returns:
        The header dict, or None for a missing or headerless file.
    """
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
    """Record a session's title as an append-only metadata line; the last one wins.

    Args:
        path: The session file.
        title: The new title.
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "title", "title": title}, ensure_ascii=False) + "\n")


def read_session_meta(path: Path) -> dict | None:
    """The header with its title overridden by the latest title line.

    Args:
        path: The session file.

    Returns:
        The header dict, or None for a headerless file.
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
    """Synthesize a header for a pre-header file so old sessions still list."""
    msgs = load_messages(path)
    title = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
    return make_header(path.stem, kind="main", title=(title[:40] or path.stem))


def list_sessions() -> list[dict]:
    """Every session's header, oldest first, synthesizing one for legacy files.

    Returns:
        The header dicts.
    """
    if not workspace.SESSIONS_DIR.exists():
        return []
    return [
        read_session_meta(path) or _legacy_header(path)
        for path in sorted(workspace.SESSIONS_DIR.glob("*.jsonl"))
    ]


def descendants(session_id: str, sessions: list[dict]) -> list[str]:
    """`session_id` and every session below it, breadth first.

    Args:
        session_id: The root of the subtree.
        sessions: The headers to walk (see list_sessions).

    Returns:
        The ids, the root first.
    """
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
    """Delete a session with its descendants, spilled output, and plan files.

    Args:
        session_id: The session to delete.

    Returns:
        The ids removed.
    """
    ids = descendants(session_id, list_sessions())
    for sid in ids:
        (workspace.SESSIONS_DIR / f"{sid}.jsonl").unlink(missing_ok=True)
        shutil.rmtree(workspace.SESSIONS_DIR / f"{sid}-out", ignore_errors=True)
        for extra in (workspace.PLANS_DIR / f"{sid}.md", workspace.PLANS_DIR / f"{sid}.result.md"):
            extra.unlink(missing_ok=True)
    return ids


def dangling_tool_calls(messages: list[dict]) -> list[dict]:
    """The last assistant turn's tool calls that never got a result.

    The mark of a turn cut off mid-tool. The API requires a result for every call,
    so these must be filled in before the conversation continues.

    Args:
        messages: The session's messages.

    Returns:
        One {"id", "name", "arguments"} per unanswered call.
    """
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

    Args:
        sessions: The headers (see list_sessions).

    Returns:
        The root nodes; each node is its header plus a sorted "children" list. A
        header whose parent is unknown becomes a root.
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


# --- plan files -------------------------------------------------------------


def plan_path(session_path: Path) -> Path:
    """Where the plan written in a session lives: plans/{session}.md.

    Args:
        session_path: The planning session's file.

    Returns:
        The plan file path.
    """
    return workspace.PLANS_DIR / f"{session_path.stem}.md"


def result_path(plan: Path) -> Path:
    """The progress file beside a plan: plans/{session}.result.md.

    Args:
        plan: The plan file.

    Returns:
        The result file path.
    """
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
    """The plan's summary line, for naming the session that carries it out.

    Args:
        path: The plan file.

    Returns:
        The first line without its "#", or "" when the file cannot be read.
    """
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return ""
    return first.removeprefix("#").strip()


def write_plan(
    path: Path, *, summary: str, steps: list[str], validation: list[str], body: str
) -> None:
    """Render a plan as Markdown and write it; a resubmitted plan replaces the file.

    Steps are numbered, not checkboxes: the plan is the spec a review compares
    against and is never ticked. Progress goes to the sibling result file.

    Args:
        path: The plan file (see plan_path).
        summary: The one-line goal; "Plan" when empty.
        steps: The executable steps, in order.
        validation: How to confirm the result, if any.
        body: Free-form notes for the executor, if any.
    """
    lines = [f"# {summary or 'Plan'}", "", "## Steps", ""]
    lines += [f"{i}. {s}" for i, s in enumerate(steps, 1)]
    if validation:
        lines += ["", "## Validation", ""]
        lines += [f"- {v}" for v in validation]
    if body:
        lines += ["", "## Notes", "", body]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
