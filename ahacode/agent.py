"""The agent loop: stream a turn, run any tools the model asked for, feed the
results back, repeat — until a turn arrives with no tool calls (the final
answer). This is the classic harness loop shared by Roo Code, Kilo Code
(opencode runLoop) and Pi (agent-loop.ts): termination is "the assistant
produced no tool calls", with a max-turns backstop against runaways.

Kept widget-free and pure so it is testable without a terminal: the UI is
reached only through the injected `emit` callback (one canonical Event at a
time), and the LLM/tools are injectable for offline tests.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator

from ahacode import client, tools
from ahacode.events import Event, TextDelta, ToolCall, ToolResult

# Type aliases spelling out the seams the app (and tests) plug into.
StreamFn = Callable[[list[dict], list[dict] | None], Iterator[Event]]
EmitFn = Callable[[Event], None]
ApproveFn = Callable[[ToolCall], bool]


def _assistant_message(text: str, tool_calls: list[ToolCall]) -> dict:
    """Build the OpenAI `assistant` history entry for a turn.

    tool_calls must be re-serialised: we parsed arguments to a dict for the UI,
    but the wire format wants the arguments back as a JSON string.
    """
    msg: dict = {"role": "assistant", "content": text or None}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in tool_calls
        ]
    return msg


def _tool_message(call_id: str, output: str) -> dict:
    """The `tool` history entry that carries a result back to the model."""
    return {"role": "tool", "tool_call_id": call_id, "content": output}


def _run_tool(
    call: ToolCall, registry: dict, approve: ApproveFn | None
) -> ToolResult:
    """Execute one tool call, converting any failure into an error ToolResult so
    the loop never crashes on a bad call — the model sees the error and adapts."""
    tool = registry.get(call.name)
    if tool is None:
        return ToolResult(call.id, call.name, f"unknown tool: {call.name}", is_error=True)
    # Safety gate first: a hard-blocked call never runs and is never even offered
    # for approval (e.g. `rm -rf /`).
    if tool.validate:
        reason = tool.validate(call.arguments)
        if reason:
            return ToolResult(call.id, call.name, f"blocked (dangerous): {reason}", is_error=True)
    if tool.requires_approval and not (approve and approve(call)):
        return ToolResult(call.id, call.name, "denied by user", is_error=True)
    try:
        return ToolResult(call.id, call.name, tool.execute(call.arguments), is_error=False)
    except Exception as exc:  # a broken tool must not take down the agent
        return ToolResult(call.id, call.name, f"{type(exc).__name__}: {exc}", is_error=True)


def run(
    messages: list[dict],
    *,
    emit: EmitFn,
    is_cancelled: Callable[[], bool] | None = None,
    approve: ApproveFn | None = None,
    stream: StreamFn | None = None,
    registry: dict | None = None,
    max_turns: int = 10,
) -> list[dict]:
    """Drive the loop and return the messages appended to history this run.

    `messages` is mutated in place (assistant + tool entries are appended); the
    return value is just the tail so the caller can persist exactly what's new.
    """
    registry = tools.REGISTRY if registry is None else registry
    stream = stream or client.stream_chat  # resolved now, not at def time
    is_cancelled = is_cancelled or (lambda: False)
    specs = tools.specs(registry)
    start = len(messages)

    for _ in range(max_turns):
        if is_cancelled():
            break

        # --- one LLM turn: stream deltas live, collect the tool calls ---
        text = ""
        calls: list[ToolCall] = []
        for event in stream(messages, specs):
            if is_cancelled():
                return messages[start:]
            if isinstance(event, ToolCall):
                calls.append(event)
            elif isinstance(event, TextDelta):
                text += event.text
            emit(event)  # thinking / text / tool-call shown as it arrives

        messages.append(_assistant_message(text, calls))

        # Termination: a turn with no tool calls is the final answer.
        if not calls:
            return messages[start:]

        # --- run each requested tool, feed results back, then loop ---
        for call in calls:
            result = _run_tool(call, registry, approve)
            emit(result)
            messages.append(_tool_message(call.id, result.output))

    else:  # loop fell through without returning -> hit the backstop
        emit(TextDelta("\n[reached max tool turns — stopping]"))

    return messages[start:]
