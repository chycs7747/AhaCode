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


# How a finished phase is written into the parent transcript, and therefore how it is
# recognised again when resuming. One constant so the writer (app.PhaseComplete) and
# the reader (completed_phases) cannot drift apart.
PHASE_HEADING = "## {description}\n"


def phase_message(phase: "PhaseResult") -> str:
    """Render a phase result as the parent's assistant message."""
    return PHASE_HEADING.format(description=phase.description) + phase.result


def completed_phases(messages: list[dict], steps: list[str]) -> list[PhaseResult]:
    """Recover, from the parent transcript, the phases of `steps` already carried out.

    Resume needs the earlier RESULTS, not just the knowledge that a step is done: a
    later phase is prompted with what the ones before it produced, so a resumed run
    that forgot them would hand step 4 an empty history and it would redo step 3's
    work. The transcript is the source of truth (it survives a restart, unlike any
    in-memory record), and phases are matched by the exact step text.

    Returns them in PLAN order, not transcript order — the plan is what defines the
    sequence. A step run more than once keeps its latest result.
    """
    latest: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        for step in steps:
            head = PHASE_HEADING.format(description=step)
            if content.startswith(head):
                latest[step] = content[len(head):]
    return [PhaseResult(description=s, result=latest[s]) for s in steps if s in latest]


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
    on_phase: Callable[[PhaseResult], None] | None = None,
    prior: list[PhaseResult] | None = None,
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

    `on_phase(result)` fires the moment a phase finishes, before the next one starts.
    It exists because a run is LONG: the caller has to be able to commit each phase as
    it lands, rather than holding everything until the end. That single seam fixes two
    bugs at once — the parent session no longer answers "나 아무것도 안 했는데" from a
    context that is minutes stale, and a cancelled run keeps the phases that had
    already completed instead of discarding the lot.

    `prior` RESUMES a run that was stopped: pass the phases already carried out (see
    completed_phases) together with the WHOLE plan, and the finished steps are skipped
    while their results are still threaded into the ones that follow. Resuming is
    therefore just running the plan again — the caller does not compute a remainder,
    which is what keeps a plan edited between the two runs from silently shifting.
    """
    is_cancelled = is_cancelled or (lambda: False)
    out = PlanResult()
    out.phases = list(prior or [])
    if out.phases:
        out.result = out.phases[-1].result
    # Consumed as a LIST, not a set: a plan may legitimately repeat a step ("Run the
    # tests" twice), and one recovered result must satisfy exactly one occurrence.
    already = [p.description for p in out.phases]
    for step in steps:
        if step in already:
            already.remove(step)
            continue
        if is_cancelled():
            break
        prompt = _phase_prompt(task, step, out.phases)
        result = elide(delegate(prompt, step), MAX_RESULT_CHARS)
        phase = PhaseResult(description=step, result=result)
        out.phases.append(phase)
        out.result = result
        if on_phase:
            # Deliberately NOT guarded by is_cancelled: this phase really did run, so
            # its result is real work and must be committed even if the user stops the
            # run a moment later.
            on_phase(phase)
    if synthesize and out.phases and not is_cancelled():
        out.result = synthesize(task, out.phases)
    return out
