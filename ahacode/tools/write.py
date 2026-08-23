"""write: create or overwrite a text file. requires_approval=True — it changes
the filesystem, so it goes through the same confirmation modal as bash.

A dedicated tool (like Claude Code's Write) instead of asking
the model to build files through bash heredocs, which is fragile for multi-line
code (quoting/escaping/indent mistakes)."""

from __future__ import annotations

from ahacode.tools.base import Tool, resolve_path


def _write(args: dict) -> str:
    target = resolve_path(args["path"])
    target.parent.mkdir(parents=True, exist_ok=True)  # create intermediate dirs
    content = args.get("content", "")
    target.write_text(content, encoding="utf-8")  # utf-8 explicit (cp949 default on KR Windows)
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
    requires_approval=True,  # filesystem change -> confirm first
)
