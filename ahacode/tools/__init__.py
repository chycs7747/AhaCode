"""Tool registry — assembles the individual tool modules into one lookup table
and the OpenAI `tools=[...]` payload. This is the only place that knows the full
tool set; adding a tool means writing its module and registering it here.
"""

from ahacode.tools.base import Tool
from ahacode.tools.bash import BASH
from ahacode.tools.edit import EDIT
from ahacode.tools.glob import GLOB
from ahacode.tools.grep import GREP
from ahacode.tools.plan import TODO_WRITE
from ahacode.tools.read import READ
from ahacode.tools.task import TASK
from ahacode.tools.write import WRITE

# Name -> Tool. The agent looks tools up here when the model calls one by name.
# `task` is deliberately NOT here: it needs a spawning context and a depth check,
# so it is added per-session by registry_for() rather than offered globally.
REGISTRY: dict[str, Tool] = {
    t.name: t for t in (READ, GLOB, GREP, WRITE, EDIT, BASH, TODO_WRITE)
}


def registry_for(depth: int, subagent_depth: int, base: dict | None = None) -> dict:
    """The tool set for a session at `depth`: the base tools plus `task`, but only
    while depth < subagent_depth — so a sub-agent at the limit has no task tool and
    therefore cannot recurse (the vertical guard against runaway spawning)."""
    reg = dict(REGISTRY if base is None else base)
    if depth < subagent_depth:
        reg[TASK.name] = TASK
    return reg


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


__all__ = [
    "Tool", "READ", "GLOB", "GREP", "WRITE", "EDIT", "BASH", "TODO_WRITE", "TASK",
    "REGISTRY", "specs", "registry_for",
]
