"""write: create or overwrite a text file; approval-gated."""

from __future__ import annotations

from ahacode import workspace
from ahacode.tools.base import Tool


def _write(args: dict) -> str:
    """Write the file, creating parent directories as needed."""
    target = workspace.resolve_path(args["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    content = args.get("content", "")
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {args['path']}"


WRITE = Tool(
    name="write",
    description="Create or overwrite a text file (parent directories are created).",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (relative or absolute)"},
            "content": {"type": "string", "description": "Full file contents to write"},
        },
        "required": ["path", "content"],
    },
    execute=_write,
    requires_approval=True,
)
