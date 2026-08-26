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
        # The model decides, judging independence by two axes — different files, and
        # results that do not depend on each other. The harness supplies the
        # parallelism (several task calls in one turn run concurrently) and this
        # rule tells the model when to use it.
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
    requires_approval=True,  # each spawn is gated by the human
    wants_ctx=True,          # needs the running context to spawn a child
    parallelizable=True,     # a fan-out of task calls runs concurrently
)
