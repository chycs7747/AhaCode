"""The tool registry: every tool by name, and the depth gate that hands out `task`."""

from __future__ import annotations

from ahacode.tools.base import Tool
from ahacode.tools.bash import BASH
from ahacode.tools.edit import EDIT
from ahacode.tools.glob import GLOB
from ahacode.tools.grep import GREP
from ahacode.tools.plan import TODO_WRITE
from ahacode.tools.plan_submit import PLAN_SUBMIT
from ahacode.tools.read import READ
from ahacode.tools.task import TASK
from ahacode.tools.webfetch import WEBFETCH
from ahacode.tools.write import WRITE

# `task` is absent: it needs a spawning context and a depth check, so registry_for
# adds it per session. `plan_submit` is absent too: it belongs to plan mode alone,
# and a sub-agent must never be able to ask for approval.
REGISTRY: dict[str, Tool] = {
    t.name: t for t in (READ, GLOB, GREP, WRITE, EDIT, BASH, WEBFETCH, TODO_WRITE)
}


def registry_for(depth: int, subagent_depth: int) -> dict[str, Tool]:
    """The tool set for a session at `depth`: the base tools, plus `task` while
    depth < subagent_depth — so a sub-agent at the limit cannot recurse.

    Args:
        depth: The session's depth in the tree (0 = main).
        subagent_depth: How many generations of sub-agents may nest.

    Returns:
        A fresh name → Tool dict.
    """
    reg = dict(REGISTRY)
    if depth < subagent_depth:
        reg[TASK.name] = TASK
    return reg


def specs(registry: dict[str, Tool]) -> list[dict]:
    """The `tools=[...]` payload sent to chat.completions.

    Args:
        registry: The tools to offer this turn.

    Returns:
        One OpenAI function schema per tool.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in registry.values()
    ]
