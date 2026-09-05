"""User turns the harness writes into the conversation on the user's behalf, each
at the moment its name describes."""

from __future__ import annotations

# What an impl session does when the plan turns out wrong or a step cannot be done
# as written: `stop` hands the problem back to the user, `adapt` lets the session
# rewrite its own checklist. The harness picks one; stop is the default.
GAP_STOP = "stop"
GAP_ADAPT = "adapt"

# The first user turn of an impl session. A user message rather than the system
# prompt: it names this session's plan file, while the system prompt is the constant
# prefix every session shares.
_HANDOFF_HEAD = """You are in an implementation session, handed off from an approved plan. Reply in the language the plan is written in.

A plan was saved at {path}. Read it first, then call todo_write with one item per plan step and keep it current as you work.

Do not re-plan, expand scope, refactor adjacent code, or add features the plan did not ask for."""

_HANDOFF_GAP = {
    GAP_STOP: (
        "If the plan has a real gap — a missing step, a contradiction with the code, a "
        "wrong path — or a step cannot be done as written, stop: leave the step "
        "in_progress, report the problem as text, and do not start any other step. The "
        "user will revise the plan."
    ),
    GAP_ADAPT: (
        "If the plan has a real gap — a missing step, a contradiction with the code, a "
        "wrong path — or a step cannot be done as written, revise the checklist yourself "
        "with todo_write: cancel the step with the reason in its content, add the steps "
        "that close the gap, say in one line what changed and why, then continue. Stay "
        "inside the plan's goal — no new features, no refactoring beyond what the gap "
        "requires."
    ),
}

_HANDOFF_TAIL = (
    "When every step is done and the plan's validation passes, finish with a concise "
    "summary of what was done and how it was verified."
)


def handoff(plan_path: str, *, gap: str = GAP_STOP) -> str:
    """The seed message of an impl session.

    Args:
        plan_path: The plan file, as the model should refer to it.
        gap: GAP_STOP or GAP_ADAPT — what the session does when the plan turns out wrong.

    Returns:
        The message text.
    """
    return "\n\n".join([_HANDOFF_HEAD.format(path=plan_path), _HANDOFF_GAP[gap], _HANDOFF_TAIL])


# Carries an impl session on by itself between turns. Says nothing about WHAT to
# do: the plan and the checklist are already in the session.
_CONTINUE = (
    "Continue with the plan: work on the next unfinished step in the checklist and "
    "keep it current with todo_write. "
)

_CONTINUE_GAP = {
    GAP_STOP: (
        "If that step is blocked, or a gap you already reported is still open, restate "
        "it in one line and stop — do not redo work that is already done."
    ),
    GAP_ADAPT: (
        "If a step is blocked or already satisfied, cancel or complete it with the "
        "reason and move on — do not redo work that is already done."
    ),
}


def auto_continue(*, gap: str = GAP_STOP) -> str:
    """The turn that carries an impl session on without asking.

    Args:
        gap: GAP_STOP or GAP_ADAPT — the policy the handoff was sent with.

    Returns:
        The message text.
    """
    return _CONTINUE + _CONTINUE_GAP[gap]


# Sent when the loop hits its turn cap, with no tools, so the model must answer.
MAX_TURNS = (
    "You've reached the turn limit for this task and tools are no longer available. "
    "Give your best final answer now, as text only: briefly summarize what you "
    "accomplished, what remains unfinished, and the recommended next step."
)

# Sent when a session is reopened after its last turn was cut off mid-tool, right
# after the stub results that close the dangling calls.
INTERRUPTED = (
    "The previous turn was interrupted before it finished. The project may have "
    "changed on disk — re-check the actual state (files, tests) and, if a checklist "
    "is in progress, bring it into line before continuing. If the interrupted call's "
    "work did not complete, redo it."
)

# Opens the message that stands in for the condensed stretch of history.
SUMMARY_PREFIX = "# Condensed summary of the earlier conversation\n\n"
