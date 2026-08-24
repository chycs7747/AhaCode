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

from ahacode.text import elide

# (prompt, description) -> the sub-agent's final result. Matches
# AgentContext.run_subagent exactly, so the app reuses that closure verbatim.
DelegateFn = Callable[[str, str], str]

# (task, phases) -> the final answer. The reduce step: one LLM pass over the concise
# phase results (NOT their reasoning), so the "main" produces a combined answer while
# its context stays small. Injected so the core stays pure/testable; the app's version
# streams it into the main turn (references do this by re-invoking the parent LLM).
SynthesizeFn = Callable[[str, "list[PhaseResult]"], str]


# A phase result is read TWICE — threaded into every later phase's prompt, and
# again by the synthesis at the end — so an unbounded one is paid for many times
# over: N phases carry N(N-1)/2 copies between them. Capping it at the source is
# the only single place that bounds both readers. Same guard rail the tools already
# have (read 2000 lines, grep 100 matches, bash its output): as far as the context
# is concerned a phase result is just another piece of tool output.
MAX_RESULT_CHARS = 4_000


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
    "fix if a test fails") see what earlier steps produced.

    The results arrive already capped (MAX_RESULT_CHARS), so this prompt cannot grow
    without bound however verbose a phase turns out to be."""
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
    synthesize: SynthesizeFn | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> PlanResult:
    """Execute `steps` in order, each in its own fresh sub-agent, threading each
    step's concise result forward. Returns every phase's result plus the final one.

    `task` is the original user problem; `steps` is the ordered plan (the contents of
    a todo_write list); `delegate(prompt, description)` runs one fresh sub-agent and
    returns its result — the structural replacement for the model choosing to call
    `task` itself.

    If `synthesize` is given (and any phase ran), it is the reduce step: the final
    answer becomes a combined synthesis of the phase results rather than just the last
    phase's output. Without it, `result` stays the last phase's result (map-only).
    """
    is_cancelled = is_cancelled or (lambda: False)
    out = PlanResult()
    for step in steps:
        if is_cancelled():
            break
        prompt = _phase_prompt(task, step, out.phases)
        result = elide(delegate(prompt, step), MAX_RESULT_CHARS)
        out.phases.append(PhaseResult(description=step, result=result))
        out.result = result
    if synthesize and out.phases and not is_cancelled():
        out.result = synthesize(task, out.phases)
    return out
