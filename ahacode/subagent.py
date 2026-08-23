"""Run a delegated task as a fresh sub-agent and hand its result back to the caller.

A sub-agent is "just another agent loop": the same agent.run drives it, only the
framing differs — a focused system prompt plus the delegated task as the opening
user turn. This is the sub-agent-as-a-tool model of Kilo Code (opencode's
tool/task.ts) and Roo Code's Orchestrator (new_task). Because agent.run is
synchronous, the parent naturally *pauses* here until the child finishes (Roo's
sequential delegate → resume), and the Python call depth mirrors the session depth
in the tree.

Kept pure and UI-free so it is unit-testable with a fake stream: the seams
(emit / approve / stream / registry) are injected, and the app supplies the real
session file + nested rendering through AgentContext.run_subagent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ahacode import agent
from ahacode.events import Event


@dataclass
class AgentContext:
    """The running context handed to wants_ctx tools (currently just `task`).

    The pure loop forwards it opaquely; only the app fills run_subagent in — it is
    the closure that creates the child session file, renders the child's events into
    a nested card, and persists the transcript. Signature: (prompt, description) ->
    the child's final result string.
    """

    run_subagent: Callable[[str, str], str] | None = None


# A worker sub-agent's framing. Deliberately short: it inherits the same tools, so
# it only needs to know its job is one delegated task and to end with a
# self-contained result. Kept at the very front of the message list so that, when
# many sub-agents share it, the gateway's prefix cache reuses this prefill (we
# measured ~67% prefill savings on a shared ~10K-token prefix).
SUBAGENT_SYSTEM = (
    "You are a focused sub-agent spawned to complete ONE delegated task. "
    "Work autonomously with the tools available, then finish with a concise, "
    "self-contained result the caller can use directly — no filler, no questions."
)


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
    system: str = SUBAGENT_SYSTEM,
) -> SubagentResult:
    """Drive a child agent loop for one delegated task and return its result.

    `registry` is passed in already built (the caller applies the depth gate via
    tools.registry_for), so this stays agnostic about how deep it may go. `ctx` is
    forwarded for the depth>1 case where a child may itself spawn; at the default
    depth limit the child simply has no task tool and never touches it.
    """
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task_prompt},
    ]
    agent.run(
        messages,
        emit=emit,
        approve=approve,
        stream=stream,
        registry=registry,
        ctx=ctx,
        is_cancelled=is_cancelled,
        max_turns=max_turns,
    )
    return SubagentResult(messages=messages, result=_final_text(messages))
