"""The agent loop: stream a turn, run the tools it asked for, feed the results
back, until a turn arrives with no tool calls. Widget-free: the UI is reached only
through the injected `emit` callback."""

from __future__ import annotations

import difflib
import itertools
import json
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from ahacode import client, context, prompts, tools
from ahacode.events import (
    Event, Notice, Phase, TextDelta, ToolCall, ToolResult, Usage,
)

# The seams the app and the tests plug into.
StreamFn = Callable[[list[dict], list[dict] | None], Iterator[Event]]
EmitFn = Callable[[Event], None]
ApproveFn = Callable[[ToolCall], bool]

# Tool-call rounds one user message may take before the tool-free wrap-up turn.
DEFAULT_MAX_TURNS = 10

# What the status line shows during compaction. Short: the bar leaves ~18 cells on
# an 80-wide terminal and Korean costs two each.
COMPACTING = "압축 중"

# Names models invent for tools this project spells differently. Not aliases: the
# call still fails, but the error names the real tool so the failure costs one turn.
_INSTEAD_OF = {
    "bash": ("run_python", "python", "python3", "code_interpreter", "run_code",
             "execute_code", "execute", "exec", "shell", "terminal", "run_command"),
    "read": ("read_file", "open_file", "view_file", "get_file", "cat"),
    "grep": ("search", "search_files", "find_in_files", "ripgrep", "rg", "search_code"),
    "glob": ("find", "find_files", "list_files", "list_directory", "ls"),
    "write": ("write_file", "create_file", "save_file"),
    "edit": ("apply_patch", "str_replace", "replace_in_file", "patch", "edit_file"),
    "webfetch": ("fetch", "fetch_url", "http_get", "browse", "open_url"),
    "task": ("delegate", "spawn_agent", "subagent"),
    "todo_write": ("todo", "update_todos", "set_todos", "task_list"),
}


def _instead_of(name: str, registry: dict) -> str:
    """The sentence that turns an unknown-tool rejection into a next move."""
    lowered = name.lower()
    for real, invented in _INSTEAD_OF.items():
        if lowered == real or lowered in invented:
            if real in registry:
                return f" Use `{real}` for that."
            return (f" What you are reaching for is `{real}`, and it is NOT available "
                    "this turn — the list above is the whole set, so do it another "
                    "way or say why you cannot.")
    close = difflib.get_close_matches(lowered, list(registry), n=1, cutoff=0.6)
    return f" Did you mean `{close[0]}`?" if close else ""


@contextmanager
def _streaming(events: Iterator[Event]) -> Iterator[Iterator[Event]]:
    """Close the stream on every way out of the loop.

    client.stream_chat holds its concurrency permit until the generator is closed;
    leaving that to garbage collection can hold it until a GC pass, and a few
    leaked permits freeze every later request. An injected stream may be a plain
    iterator without close(), hence the getattr.
    """
    try:
        yield events
    finally:
        close = getattr(events, "close", None)
        if close is not None:
            close()


def _compaction_note(done: context.Compaction) -> str:
    """What to tell the user about a compaction; losing history silently is worse."""
    if done.pruned_chars:
        return f"🗜 컨텍스트 확보를 위해 오래된 도구 출력 {done.pruned_chars:,}자를 비웠어요."
    return f"🗜 컨텍스트 한계에 가까워 이전 메시지 {done.summarized}개를 요약으로 압축했어요."


def _announced_summarize(
    summarize: context.SummarizeFn | None, emit: EmitFn
) -> context.SummarizeFn:
    """Wrap the summarizer so the UI is told while it runs: this is the one slow,
    blocking half of compaction."""

    def run(older: list[dict]) -> str:
        emit(Phase(COMPACTING))
        try:
            return (summarize or context.llm_summarize)(older)
        finally:
            emit(Phase(COMPACTING, done=True))

    return run


def _assistant_message(text: str, tool_calls: list[ToolCall]) -> dict:
    """The OpenAI `assistant` history entry for a turn; arguments go back to JSON."""
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


def _gate_tool(call: ToolCall, registry: dict, approve: ApproveFn | None):
    """The approval and safety phase for one call; runs sequentially so approval
    modals never race.

    Args:
        call: The model's call.
        registry: This turn's tools.
        approve: Asked before a tool that requires approval runs.

    Returns:
        The Tool to execute, or a ToolResult (unknown, unparseable, dangerous or
        denied) that stands in for execution.
    """
    # Feed a parse failure back as a result so the model resends the call, instead
    # of the loop ending on an empty turn.
    if call.parse_error:
        return ToolResult(
            call.id, call.name,
            f"{call.parse_error}: the call did not run. Resend it as a real tool "
            "call with valid JSON arguments (not a text block).",
            is_error=True,
        )
    tool = registry.get(call.name)
    if tool is None:
        # Name this turn's actual set: it varies by mode, so the model cannot know
        # it unless told.
        return ToolResult(
            call.id, call.name,
            f"unknown tool: {call.name}. This turn's tools are: "
            f"{', '.join(sorted(registry))} — that is the complete set, and the "
            "schema and description of each one came with this request; read them "
            f"there rather than guessing a name.{_instead_of(call.name, registry)}",
            is_error=True,
        )
    # The denylist runs before approval: a hard-blocked call is never even offered.
    if tool.validate:
        reason = tool.validate(call.arguments)
        if reason:
            return ToolResult(call.id, call.name, f"blocked (dangerous): {reason}", is_error=True)
    if tool.requires_approval and not (approve and approve(call)):
        return ToolResult(call.id, call.name, "denied by user", is_error=True)
    return tool


def _exec_tool(call: ToolCall, tool, ctx: object | None = None) -> ToolResult:
    """Run one gated tool; any failure becomes an error ToolResult so the loop
    never crashes on a bad call."""
    try:
        output = tool.execute(call.arguments, ctx) if tool.wants_ctx else tool.execute(call.arguments)
        return ToolResult(call.id, call.name, output, is_error=False)
    except Exception as exc:
        return ToolResult(call.id, call.name, f"{type(exc).__name__}: {exc}", is_error=True)


def run(
    messages: list[dict],
    *,
    emit: EmitFn,
    is_cancelled: Callable[[], bool] | None = None,
    approve: ApproveFn | None = None,
    stream: StreamFn | None = None,
    registry: dict | None = None,
    ctx: object | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    summarize: context.SummarizeFn | None = None,
    prompt_tokens: int | None = None,
    should_pause: Callable[[], bool] | None = None,
) -> list[dict]:
    """Drive the loop and return the messages it appended to the history.

    `messages` is mutated in place, and may be condensed in place when it nears the
    context window; the return value is a separate accumulator of exactly what this
    run produced, so the caller persists the real messages even when the in-flight
    copy was compacted.

    Args:
        messages: The history, beginning with the system prompt.
        emit: Receives every event as it happens.
        is_cancelled: Polled between events; a True stops the loop like Stop does.
        approve: Asked before a tool that requires approval runs.
        stream: The model call; client.stream_chat when omitted.
        registry: The tools this turn may use; tools.REGISTRY when omitted.
        ctx: An opaque bag for wants_ctx tools (task → sub-agents).
        max_turns: Tool-call rounds before a tool-free wrap-up; 0 or less is uncapped.
        summarize: The compaction summarizer; a model call when omitted.
        prompt_tokens: The server's count for the previous request, seeding compaction.
        should_pause: Polled between turns; a True stops the loop cleanly, the way a
            finished answer does. The plan gate and the stall backstop use it.

    Returns:
        The assistant and tool messages this run appended.
    """
    registry = tools.REGISTRY if registry is None else registry
    stream = stream or client.stream_chat
    is_cancelled = is_cancelled or (lambda: False)
    specs = tools.specs(registry)
    appended: list[dict] = []

    def add(msg: dict) -> None:
        messages.append(msg)
        appended.append(msg)

    rounds = itertools.count() if max_turns <= 0 else range(max_turns)
    for _ in rounds:
        if is_cancelled():
            break
        # A pause is not a cancellation: the loop stops between turns, so everything
        # produced so far is complete, and the caller resumes with the same history.
        if should_pause and should_pause():
            break

        # Condense before sending, so this turn's request fits.
        done = context.maybe_compact(
            messages, prompt_tokens, summarize=_announced_summarize(summarize, emit)
        )
        if done:
            prompt_tokens = None  # the old count no longer describes this prompt
            emit(Notice(_compaction_note(done)))

        text = ""
        calls: list[ToolCall] = []
        with _streaming(stream(messages, specs)) as events:
            for event in events:
                if is_cancelled():
                    return appended
                if isinstance(event, ToolCall):
                    calls.append(event)
                elif isinstance(event, TextDelta):
                    text += event.text
                elif isinstance(event, Usage):
                    prompt_tokens = event.prompt_tokens
                emit(event)

        add(_assistant_message(text, calls))

        if not calls:  # a turn with no tool calls is the final answer
            return appended

        # Approval runs sequentially so modals never race; execution runs in parallel
        # when every runnable tool allows it. Results are emitted and appended in call
        # order, so the tool messages line up with the assistant's tool_calls.
        gated = [(call, _gate_tool(call, registry, approve)) for call in calls]
        runnable = [(call, t) for call, t in gated if not isinstance(t, ToolResult)]
        if len(runnable) > 1 and all(tool.parallelizable for _, tool in runnable):
            with ThreadPoolExecutor(max_workers=len(runnable)) as pool:
                futures = {
                    call.id: pool.submit(_exec_tool, call, tool, ctx)
                    for call, tool in runnable
                }
            done = {cid: fut.result() for cid, fut in futures.items()}
        else:
            done = {call.id: _exec_tool(call, tool, ctx) for call, tool in runnable}

        for call, gate in gated:
            result = gate if isinstance(gate, ToolResult) else done[call.id]
            emit(result)
            add(_tool_message(call.id, result.output))

    else:
        # The turn cap: one tool-free wrap-up turn, so the model must answer with
        # what is done, what remains, and what comes next.
        if not is_cancelled():
            add({"role": "user", "content": prompts.injected.MAX_TURNS})
            text = ""
            with _streaming(stream(messages, None)) as events:
                for event in events:
                    if is_cancelled():
                        return appended
                    if isinstance(event, TextDelta):
                        text += event.text
                    emit(event)
            add(_assistant_message(text, []))

    return appended
