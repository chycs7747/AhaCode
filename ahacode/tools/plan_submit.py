"""plan_submit: the model declares its plan finished.

The harness writes plans/{session}.md from the arguments and opens the approval
gate. Only an empty plan is refused; steps that do not read as executable are
noted rather than rejected, because the check is a heuristic. Offered in plan
mode only, never to a sub-agent.
"""

from __future__ import annotations

from ahacode import storage, workspace
from ahacode.tools.base import Tool
from ahacode.tools.plan import coerce_items, non_actionable


class PlanRejected(ValueError):
    """The submitted plan cannot be accepted as-is; the message says how to fix it."""


def _clean(items) -> list[str]:
    """Steps as plain strings, normalised the same way todo_write normalises items."""
    normalised, _ = coerce_items(items)
    return [it["content"] for it in normalised]


def check(steps: list[str]) -> str | None:
    """Why `steps` cannot be accepted at all.

    Args:
        steps: The cleaned steps.

    Returns:
        The reason, or None when the plan is acceptable.
    """
    if not steps:
        return "steps is empty. Lay out the executable steps of the plan."
    return None


def note(steps: list[str]) -> str:
    """A soft warning naming steps that may not be executable.

    Args:
        steps: The cleaned steps.

    Returns:
        The note, or "" when every step reads as a task.
    """
    vague = [(i, s) for i, s in enumerate(steps, 1) if non_actionable(s)]
    if not vague:
        return ""
    listed = "; ".join(f"step {i} '{s[:60]}'" for i, s in vague)
    return (
        f" Note: {listed} may not be executable — a step usually starts with an "
        "imperative verb and names a concrete artifact or checkable outcome."
    )


def _plan_submit(args: dict, ctx) -> str:
    """Write the plan file for the session in `ctx` and report where it went."""
    steps = _clean(args.get("steps"))
    problem = check(steps)
    if problem:
        raise PlanRejected(problem)
    session_path = getattr(ctx, "session_path", None)
    if session_path is None:
        raise PlanRejected("no session to attach the plan to")
    path = storage.plan_path(session_path)
    storage.write_plan(
        path,
        summary=str(args.get("summary", "")).strip(),
        steps=steps,
        validation=_clean(args.get("validation")),
        body=str(args.get("body", "")).strip(),
    )
    return (
        f"Plan saved to {workspace.display_path(path)} ({len(steps)} steps)."
        f"{note(steps)} Planning is complete — stop here and wait for the user's decision."
    )


PLAN_SUBMIT = Tool(
    name="plan_submit",
    description=(
        "Submit the finished plan for the user's approval. This ends the planning "
        "turn. Call it once the plan is complete and you have no unanswered "
        "questions — not before, and not again unless the user asks for changes."
    ),
    parameters={
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "One line: what the plan achieves",
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "The steps in execution order. Each one EXECUTABLE, imperative: a "
                    "verb plus a concrete artifact or checkable outcome (\"Write "
                    "solver.py with solve()\", \"Run the 4 examples and confirm "
                    "40/14/27/9\"). Not a topic, a formula, or an idea."
                ),
            },
            "validation": {
                "type": "array",
                "items": {"type": "string"},
                "description": "How to confirm the result is correct (commands, checks)",
            },
            "body": {
                "type": "string",
                "description": "Optional detail the executor needs: decisions, paths, gotchas",
            },
        },
        "required": ["summary", "steps"],
    },
    execute=_plan_submit,
    wants_ctx=True,  # the plan file is named after the session that planned it
)
