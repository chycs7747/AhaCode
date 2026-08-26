"""read: return the contents of a text file (read-only, no approval needed)."""

from __future__ import annotations

from ahacode.tools.base import Tool, resolve_path

_MAX_LINES = 2000  # guard rail so one read can't flood the model's context


def _read(args: dict) -> str:
    target = resolve_path(args["path"])
    # utf-8 explicit: the platform default may differ (cp949 on Korean Windows).
    lines = target.read_text(encoding="utf-8").splitlines()

    offset = int(args.get("offset", 1))  # 1-indexed, like an editor's line numbers
    start = max(offset - 1, 0)
    limit = int(args.get("limit", _MAX_LINES))
    window = lines[start : start + min(limit, _MAX_LINES)]

    body = "\n".join(window)
    shown_end = start + len(window)
    if shown_end < len(lines):  # tell the model the file continues
        body += f"\n... ({len(lines) - shown_end} more lines; use offset={shown_end + 1})"
    return body or "(empty file)"


READ = Tool(
    name="read",
    description=(
        "Read the contents of a text file, relative to the project root. "
        "Use offset/limit (1-indexed) to page through large files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (relative or absolute)"},
            "offset": {"type": "integer", "description": "1-indexed line to start at"},
            "limit": {"type": "integer", "description": "Max number of lines to read"},
        },
        "required": ["path"],
    },
    execute=_read,
    parallelizable=True,  # pure read, no side effects — safe to batch in one turn
)
