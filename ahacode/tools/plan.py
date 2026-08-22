"""todo_write: record a structured plan (task list). Read-only in effect — it
just formats the steps back as a checklist, so it is safe in plan mode.

Stateless by design: like Claude Code's TodoWrite, the model sends the whole list
each time and we render it fresh. No shared-state mutation from the worker thread,
so no ctx is needed (that seam is deferred until subagents require it)."""

from __future__ import annotations

from ahacode.tools.base import Tool

_MARKS = {"done": "☑", "in_progress": "▶", "pending": "☐"}


def _todo_write(args: dict) -> str:
    items = args.get("items", [])
    lines = [
        f"{_MARKS.get(it.get('status', 'pending'), '☐')} {it['content']}"
        for it in items
    ]
    return "\n".join(lines) or "(empty plan)"


TODO_WRITE = Tool(
    name="todo_write",
    description=(
        "Record or update the plan as a task list. Call this to lay out the steps "
        "before acting; send the full list each time (status: pending/in_progress/done)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "description": "The full task list, in order",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "What the step does"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "done"],
                        },
                    },
                    "required": ["content"],
                },
            },
        },
        "required": ["items"],
    },
    execute=_todo_write,
)
