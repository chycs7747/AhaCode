"""edit: replace a snippet that appears exactly once in a file; approval-gated."""

from __future__ import annotations

from ahacode import workspace
from ahacode.tools.base import Tool


def _edit(args: dict) -> str:
    """Replace old_string with new_string.

    Raises:
        ValueError: When old_string is absent or ambiguous, so the model supplies
            more context.
    """
    target = workspace.resolve_path(args["path"])
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
