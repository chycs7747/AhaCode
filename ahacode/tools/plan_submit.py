"""plan_submit: the model declares its plan finished and hands it to the user.

The plan-mode counterpart of attempt_completion: an explicit "I am done planning"
signal, so the harness knows the exact moment to open the approval gate instead
of guessing from the shape of a todo list. The harness — not the model — writes
the plan file (plans/{session}.md) from the tool's arguments, so there is nothing
to verify on disk: a successful call IS the file.

Only an EMPTY plan is refused. Steps that do not read as executable are accepted
and named in the result as a note: the check is a word-list heuristic, and as a
hard gate it rejected a perfectly good Korean step three times in a row (it ended
in "print"). The human reading the gate card is the real validator.
Offered in plan mode only; a sub-agent never sees it — approval is the root
session's to ask for.
"""

from __future__ import annotations

from ahacode import storage
from ahacode.tools.base import Tool
from ahacode.tools.plan import non_actionable


class PlanRejected(ValueError):
    """The submitted plan cannot be accepted as-is; the message says how to fix it."""


def _clean(items) -> list[str]:
    return [s.strip() for s in (items or []) if isinstance(s, str) and s.strip()]


def check(steps: list[str]) -> str | None:
    """Why `steps` cannot be accepted at all, or None if it can."""
    if not steps:
        return "steps is empty. Lay out the executable steps of the plan."
    return None


def note(steps: list[str]) -> str:
    """A soft warning naming steps that may not be executable ("" if none)."""
    vague = [(i, s) for i, s in enumerate(steps, 1) if non_actionable(s)]
    if not vague:
        return ""
    listed = "; ".join(f"step {i} '{s[:60]}'" for i, s in vague)
    return (
        f" Note: {listed} may not be executable — a step usually starts with an "
        "imperative verb and names a concrete artifact or checkable outcome."
    )


def _plan_submit(args: dict, ctx) -> str:
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
        f"Plan saved to {storage.display_path(path)} ({len(steps)} steps)."
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
