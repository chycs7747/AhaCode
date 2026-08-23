"""Structural plan execution: run a plan's steps as fresh sub-agents, in order.

The problem this solves: on a hard, single problem the model spirals — one session
accumulates a growing *reasoning* history and re-verifies it every turn (measured:
9-15 min, 0 tool calls). Delegation would fix it (each phase in a fresh, small
context), but the model never *chooses* to delegate (measured 0/120). So the harness
delegates STRUCTURALLY: given an ordered plan, this drives a plain `for` loop that
hands each step to its own sub-agent — the decision to split is code, not a model
tool call.

Why the spiral doesn't return: each step runs in a fresh sub-agent
(`subagent.run` = `[system, task]`, no parent history), so it never sees the prior
steps' *reasoning*. Only each step's concise, self-contained RESULT is threaded
forward — N short summaries stay tiny next to one turn's spiralling context.

Kept pure and UI-free: the sub-agent runner is injected as `delegate` (in the app,
the same `run_subagent` closure the `task` tool uses — child session file + nested
card), so this is unit-testable offline with a fake delegate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# (prompt, description) -> the sub-agent's final result. Matches
# AgentContext.run_subagent exactly, so the app reuses that closure verbatim.
DelegateFn = Callable[[str, str], str]


@dataclass
class PhaseResult:
    description: str  # the step label, taken from the plan
    result: str       # the concise result the step's sub-agent handed back


@dataclass
class PlanResult:
    phases: list[PhaseResult] = field(default_factory=list)
    result: str = ""  # the final answer — the last phase's result


def _phase_prompt(task: str, step: str, prior: list[PhaseResult]) -> str:
    """Curated context for one phase: the overall task, THIS step, and the concise
    results of the phases before it — nothing else. Forwarding the results (not the
    reasoning) keeps each fresh context small while still letting a later step (e.g.
    "fix if a test fails") see what earlier steps produced."""
    parts = [f"# Overall task\n{task}", f"# Your phase\n{step}"]
    if prior:
        so_far = "\n\n".join(f"## {p.description}\n{p.result}" for p in prior)
        parts.append(f"# Results of earlier phases\n{so_far}")
    parts.append(
        "Complete THIS phase end to end with the tools available. Finish with a "
        "concise, self-contained result the next phase can use directly."
    )
    return "\n\n".join(parts)


def run_plan(
    task: str,
    steps: list[str],
    delegate: DelegateFn,
    *,
    is_cancelled: Callable[[], bool] | None = None,
) -> PlanResult:
    """Execute `steps` in order, each in its own fresh sub-agent, threading each
    step's concise result forward. Returns every phase's result plus the final one.

    `task` is the original user problem; `steps` is the ordered plan (the contents of
    a todo_write list); `delegate(prompt, description)` runs one fresh sub-agent and
    returns its result — the structural replacement for the model choosing to call
    `task` itself.
    """
    is_cancelled = is_cancelled or (lambda: False)
    out = PlanResult()
    for step in steps:
        if is_cancelled():
            break
        prompt = _phase_prompt(task, step, out.phases)
        result = delegate(prompt, step)
        out.phases.append(PhaseResult(description=step, result=result))
        out.result = result
    return out
