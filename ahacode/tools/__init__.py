"""Tool registry — assembles the individual tool modules into one lookup table
and the OpenAI `tools=[...]` payload. This is the only place that knows the full
tool set; adding a tool means writing its module and registering it here.
"""

from ahacode.tools.base import Tool
from ahacode.tools.bash import BASH
from ahacode.tools.edit import EDIT
from ahacode.tools.plan import TODO_WRITE
from ahacode.tools.read import READ
from ahacode.tools.write import WRITE

# Name -> Tool. The agent looks tools up here when the model calls one by name.
REGISTRY: dict[str, Tool] = {t.name: t for t in (READ, WRITE, EDIT, BASH, TODO_WRITE)}


def specs(registry: dict[str, Tool] | None = None) -> list[dict]:
    """The `tools=[...]` payload sent to chat.completions (OpenAI function schema).

    Defaults to the global REGISTRY; a subset can be passed (e.g. plan mode
    exposes only read-only tools)."""
    reg = REGISTRY if registry is None else registry
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in reg.values()
    ]


__all__ = ["Tool", "READ", "WRITE", "EDIT", "BASH", "TODO_WRITE", "REGISTRY", "specs"]
