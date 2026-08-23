"""edit: replace a unique snippet in an existing file.

Like Claude Code's Edit: the old_string must match exactly once, so
the change is unambiguous — 0 matches or >1 matches are errors that push the model
to supply more context. requires_approval=True (it changes the filesystem)."""

from __future__ import annotations

from ahacode.tools.base import Tool, resolve_path


def _edit(args: dict) -> str:
    target = resolve_path(args["path"])
    text = target.read_text(encoding="utf-8")
    old, new = args["old_string"], args["new_string"]
    count = text.count(old)
    if count == 0:
        raise ValueError(f"old_string not found in {args['path']}")
    if count > 1:
        raise ValueError(
            f"old_string appears {count}× in {args['path']} — add surrounding "
            "context so it matches exactly once"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    return f"edited {args['path']}"


EDIT = Tool(
    name="edit",
    description=(
        "Replace an exact snippet in an existing file. old_string must appear "
        "exactly once (include enough surrounding lines to be unique)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File to edit (relative or absolute)"},
            "old_string": {"type": "string", "description": "Exact text to replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
        },
        "required": ["path", "old_string", "new_string"],
    },
    execute=_edit,
    requires_approval=True,
)
