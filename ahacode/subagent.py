"""Run a delegated task as a fresh sub-agent loop and hand its result back.

The same agent.run drives it; only the framing differs. agent.run is synchronous,
so the parent pauses here until the child finishes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ahacode import agent, client, prompts
from ahacode.events import Event


@dataclass
class AgentContext:
    """The running context handed to wants_ctx tools (`task`, `plan_submit`).

    Only the app fills it in; the loop forwards it opaquely.
    """

    run_subagent: Callable[[str, str], str] | None = None  # (prompt, description) -> the child's result
    session_path: Path | None = None  # the session the loop runs in; plan_submit names the plan after it


@dataclass
class SubagentResult:
    messages: list[dict]  # the child's full transcript (system + task + loop)
    result: str  # the final answer handed back to the parent


def _final_text(messages: list[dict]) -> str:
    """The child's last assistant answer, or a placeholder when there is none."""
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
    max_turns: int = agent.DEFAULT_MAX_TURNS,
    system: str | None = None,
    summarize=None,
) -> SubagentResult:
    """Drive a child agent loop for one delegated task.

    Args:
        task_prompt: The delegated task, sent as the opening user turn.
        emit: Receives the child's events.
        approve: Asked before a tool that requires approval runs.
        stream: The model call; client.stream_chat when omitted.
        registry: The child's tools, already depth-gated by the caller.
        ctx: Forwarded for a child that may itself delegate.
        is_cancelled: Polled between events.
        max_turns: Tool-call rounds before the wrap-up turn.
        system: A system prompt to use instead of prompts.subagent_system().
        summarize: The compaction summarizer.

    Returns:
        The child's transcript and its final answer.
    """
    # Resolved at call time: a default argument would freeze the bare framing
    # constant and skip the assembly that adds the shared CODING_RULES.
    seed = [
        {"role": "system", "content": system or prompts.subagent_system()},
        {"role": "user", "content": task_prompt},
    ]
    # agent.run mutates (and may condense) its list; the transcript handed back is
    # the seed plus what the run produced, so the child's session file stays complete.
    live = list(seed)
    with client.mode("subagent"):  # restores the parent's mode on return
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
