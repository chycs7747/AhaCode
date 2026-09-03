"""read: the contents of a text file, paged with offset/limit."""

from __future__ import annotations

from ahacode import workspace
from ahacode.tools.base import Tool

_MAX_LINES = 2000  # so one read cannot flood the model's context


def _read(args: dict) -> str:
    """Read a window of lines, telling the model when the file continues."""
    target = workspace.resolve_path(args["path"])
    lines = target.read_text(encoding="utf-8").splitlines()

    offset = int(args.get("offset", 1))  # 1-indexed, like an editor
    start = max(offset - 1, 0)
    limit = int(args.get("limit", _MAX_LINES))
    window = lines[start : start + min(limit, _MAX_LINES)]

    body = "\n".join(window)
    shown_end = start + len(window)
    if shown_end < len(lines):
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
    parallelizable=True,
)
