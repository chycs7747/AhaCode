"""Run a delegated task as a fresh sub-agent and hand its result back to the caller.

A sub-agent is "just another agent loop": the same agent.run drives it, only the
framing differs — a focused system prompt plus the delegated task as the opening
user turn (the sub-agent-as-a-tool model). Because agent.run is synchronous, the
parent naturally *pauses* here until the child finishes (a sequential delegate →
resume), and the Python call depth mirrors the session depth in the tree.

Kept pure and UI-free so it is unit-testable with a fake stream: the seams
(emit / approve / stream / registry) are injected, and the app supplies the real
session file + nested rendering through AgentContext.run_subagent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ahacode import agent, prompts
from ahacode.events import Event
from ahacode.prompts import SUBAGENT_SYSTEM  # re-exported for callers/tests


@dataclass
class AgentContext:
    """The running context handed to wants_ctx tools (currently just `task`).

    The pure loop forwards it opaquely; only the app fills run_subagent in — it is
    the closure that creates the child session file, renders the child's events into
    a nested card, and persists the transcript. Signature: (prompt, description) ->
    the child's final result string.
    """

    run_subagent: Callable[[str, str], str] | None = None


@dataclass
class SubagentResult:
    messages: list[dict]  # the child's full transcript (system + task + loop)
    result: str           # the final answer text handed back to the parent


def _final_text(messages: list[dict]) -> str:
    """The child's last assistant answer — the terminating turn with no tool calls."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return "(sub-agent produced no result)"


def run(
    task_prompt: str,
    *,
    emit: Callable[[Event], None],
    approve=None,
    stream=None,
    registry: dict | None = None,
    ctx: object | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    max_turns: int = 10,
    system: str | None = None,
    summarize=None,
) -> SubagentResult:
    """Drive a child agent loop for one delegated task and return its result.

    `registry` is passed in already built (the caller applies the depth gate via
    tools.registry_for), so this stays agnostic about how deep it may go. `ctx` is
    forwarded for the depth>1 case where a child may itself spawn; at the default
    depth limit the child simply has no task tool and never touches it.
    """
    # Resolved at CALL time, not bound as a default: a default argument freezes the
    # bare SUBAGENT_SYSTEM constant at import, which silently bypassed the assembly in
    # prompts.subagent_system() — so every child ran without the shared CODING_RULES
    # (and the ROLE_ADDENDA seam was dead on arrival). The function is the seam.
    seed = [
        {"role": "system", "content": system or prompts.subagent_system()},
        {"role": "user", "content": task_prompt},
    ]
    # agent.run mutates its list — and may CONDENSE it if the child's own context
    # grows — so the transcript we hand back is the seed plus what the run actually
    # produced, never the live list. The caller writes this to the child's session
    # file, which must stay complete even when the request in flight was compacted.
    live = list(seed)
    produced = agent.run(
        live,
        emit=emit,
        approve=approve,
        stream=stream,
        registry=registry,
        ctx=ctx,
        is_cancelled=is_cancelled,
        max_turns=max_turns,
        summarize=summarize,
    )
    messages = [*seed, *produced]
    return SubagentResult(messages=messages, result=_final_text(messages))
