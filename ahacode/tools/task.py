"""task: delegate a subtask to a fresh sub-agent and return its result.

Sub-agent-as-a-tool: spawning a child agent is exposed to the model as one ordinary
tool call, so it rides the existing native tool-calling path with no special
machinery in the loop. This tool is a thin shim —
it hands the delegated prompt to ctx.run_subagent (filled in by the app, which owns
the child session file + nested rendering) and returns whatever the child concluded;
the loop injects that back as this call's result.

Three guards, per the design discussion:
- wants_ctx=True — unlike read/bash, it needs the running context to spawn a child.
- requires_approval=True — each spawn is gated by the human (defense against runaway
  delegation); auto-approve bypasses it like any other tool.
- offered only where depth < subagent_depth (the app builds the registry via
  tools.registry_for), so a child at the limit cannot recurse.
"""

from __future__ import annotations

from ahacode.tools.base import Tool


def _task(args: dict, ctx) -> str:
    # ctx is the AgentContext the loop forwards; without a run_subagent (e.g. a bare
    # unit test, or a context that doesn't support spawning) fail soft so the model
    # sees an error rather than the agent crashing.
    if ctx is None or getattr(ctx, "run_subagent", None) is None:
        return "error: sub-agents are not available in this context"
    return ctx.run_subagent(args["prompt"], args.get("description", ""))


TASK = Tool(
    name="task",
    description=(
        # Conservative, decision-local guidance: bias toward solving directly.
        # qwen never self-delegates (measured 0/120), so the real risk is a wrong
        # delegation fragmenting one tight reasoning chain — hence "usually yourself".
        "Delegate a self-contained subtask to a fresh sub-agent. Use ONLY when the "
        "task spans multiple INDEPENDENT areas or files, or the scope is uncertain "
        "and would flood your context. Prefer solving directly — usually do the "
        "work yourself. A single, tightly-reasoned problem must NOT be delegated."
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
    requires_approval=True,  # each spawn is gated by the human
    wants_ctx=True,          # needs the running context to spawn a child
    parallelizable=True,     # a fan-out of task calls runs concurrently
)
