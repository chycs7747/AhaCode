"""task: delegate a subtask to a fresh sub-agent.

A thin shim over ctx.run_subagent, which the app fills in. Approval-gated, and
offered only while depth < subagent_depth (see tools.registry_for), so a child at
the limit cannot recurse.
"""

from __future__ import annotations

from ahacode.tools.base import Tool


def _task(args: dict, ctx) -> str:
    """Hand the prompt to the running context; fail soft without one."""
    if ctx is None or getattr(ctx, "run_subagent", None) is None:
        return "error: sub-agents are not available in this context"
    return ctx.run_subagent(args["prompt"], args.get("description", ""))


TASK = Tool(
    name="task",
    description=(
        "Delegate a self-contained subtask to a fresh sub-agent. Use it for work that "
        "is INDEPENDENT of your other work: a different file, and a result that does "
        "not depend on another task's output. Launch several at once by putting "
        "multiple task calls in a SINGLE message — they run concurrently. Never let "
        "two tasks touch the same file, and keep dependent steps in order (do them "
        "yourself, or one task after the previous result). In each prompt, say exactly "
        "what to build, whether to write code or only investigate, and what to return."
    ),
    parameters={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Short label for the subtask (a few words)",
            },
            "prompt": {
                "type": "string",
                "description": "The full, self-contained instructions for the sub-agent",
            },
        },
        "required": ["prompt"],
    },
    execute=_task,
    requires_approval=True,
    wants_ctx=True,
    parallelizable=True,  # a fan-out of task calls runs concurrently
)
