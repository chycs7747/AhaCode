"""User turns the harness writes into the conversation on the user's behalf, each
at the moment its name describes."""

from __future__ import annotations

# The first user turn of an impl session. A user message rather than the system
# prompt: it names this session's plan file, while the system prompt is the constant
# prefix every session shares.
_HANDOFF = """You are in an implementation session, handed off from an approved plan. Edit, write and bash tools are available.

A plan was saved at {path}. Read it first, then call todo_write with one item per plan step. Work through them one-by-one, marking in_progress before starting each and done immediately after finishing.

Do not re-plan, expand scope, refactor adjacent code, or add features the plan did not ask for.

If the plan has a real gap — a missing step, a contradiction with the code, a wrong path — stop and report the gap as text instead of improvising. The user will revise the plan.

When every step is done and the plan's validation passes, finish with a concise summary of what was done and how it was verified."""


def handoff(plan_path: str) -> str:
    """The seed message of an impl session.

    Args:
        plan_path: The plan file, as the model should refer to it.

    Returns:
        The message text.
    """
    return _HANDOFF.format(path=plan_path)


# Carries an impl session on by itself between turns. Says nothing about WHAT to
# do: the plan and the checklist are already in the session.
AUTO_CONTINUE = (
    "Continue with the plan. Work on the next unfinished step in the checklist, and "
    "mark steps done with todo_write as you complete them. If a step turns out to be "
    "blocked or already satisfied, say so and move on to the next one rather than "
    "repeating work you have already done."
)

# Sent when the loop hits its turn cap, with no tools, so the model must answer.
MAX_TURNS = (
    "You've reached the step limit for this task and tools are no longer available. "
    "Give your best final answer now, as text only: briefly summarize what you "
    "accomplished, what remains unfinished, and the recommended next step."
)

# Sent when a session is reopened after its last turn was cut off mid-tool, right
# after the stub results that close the dangling calls.
INTERRUPTED = (
    "[system] The previous turn was interrupted before it finished. The project "
    "may have changed on disk — re-check the actual state (files, tests) and bring "
    "the plan's checklist into line with it before continuing. If the last step did "
    "not complete, redo it."
)

# Opens the message that stands in for the condensed stretch of history.
SUMMARY_PREFIX = "# Condensed summary of the earlier conversation\n\n"
