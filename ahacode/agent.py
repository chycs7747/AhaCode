"""The agent loop: stream a turn, run any tools the model asked for, feed the
results back, repeat — until a turn arrives with no tool calls (the final
answer). This is the classic agent harness loop: termination is "the assistant
produced no tool calls", with a max-turns backstop against runaways.

Kept widget-free and pure so it is testable without a terminal: the UI is
reached only through the injected `emit` callback (one canonical Event at a
time), and the LLM/tools are injectable for offline tests.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

from ahacode import client, context, prompts, tools
from ahacode.events import Event, Notice, TextDelta, ToolCall, ToolResult, Usage

# Type aliases spelling out the seams the app (and tests) plug into.
StreamFn = Callable[[list[dict], list[dict] | None], Iterator[Event]]
EmitFn = Callable[[Event], None]
ApproveFn = Callable[[ToolCall], bool]


# Tool-call rounds one user message may take before the tool-free wrap-up turn.
DEFAULT_MAX_TURNS = 10


def _compaction_note(done: context.Compaction) -> str:
    """What to tell the user about a compaction. Losing history silently is the bad
    failure mode — they should know why the agent may have forgotten something."""
    if done.pruned_chars:
        return f"🗜 컨텍스트 확보를 위해 오래된 도구 출력 {done.pruned_chars:,}자를 비웠어요."
    return f"🗜 컨텍스트 한계에 가까워 이전 메시지 {done.summarized}개를 요약으로 압축했어요."


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


def _gate_tool(call: ToolCall, registry: dict, approve: ApproveFn | None):
    """Approval/safety phase — runs SEQUENTIALLY so approval modals never race.
    Returns the Tool to execute, or a blocking ToolResult (unknown / dangerous /
    denied / unparseable) that skips execution."""
    # The model's argument JSON did not parse. Feed the failure back as a result so
    # the model resends the call, instead of the loop stopping on an empty turn.
    if call.parse_error:
        return ToolResult(
            call.id, call.name,
            f"{call.parse_error}: the call did not run. Resend it as a real tool "
            "call with valid JSON arguments (not a text block).",
            is_error=True,
        )
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
    return tool


def _exec_tool(call: ToolCall, tool, ctx: object | None = None) -> ToolResult:
    """Execution phase — safe to run in PARALLEL for parallelizable tools. Converts
    any failure into an error ToolResult so the loop never crashes on a bad call."""
    try:
        # wants_ctx tools (task) need the running context to spawn a sub-agent;
        # the rest keep the simple execute(args) contract.
        output = tool.execute(call.arguments, ctx) if tool.wants_ctx else tool.execute(call.arguments)
        return ToolResult(call.id, call.name, output, is_error=False)
    except Exception as exc:  # a broken tool must not take down the agent
        return ToolResult(call.id, call.name, f"{type(exc).__name__}: {exc}", is_error=True)


def _run_tool(
    call: ToolCall, registry: dict, approve: ApproveFn | None, ctx: object | None = None
) -> ToolResult:
    """Gate then execute one tool call — the sequential path (and direct callers)."""
    gated = _gate_tool(call, registry, approve)
    if isinstance(gated, ToolResult):
        return gated
    return _exec_tool(call, gated, ctx)


def run(
    messages: list[dict],
    *,
    emit: EmitFn,
    is_cancelled: Callable[[], bool] | None = None,
    approve: ApproveFn | None = None,
    stream: StreamFn | None = None,
    registry: dict | None = None,
    ctx: object | None = None,  # opaque bag for wants_ctx tools (task → sub-agents)
    max_turns: int = DEFAULT_MAX_TURNS,
    summarize: context.SummarizeFn | None = None,
    prompt_tokens: int | None = None,
    should_pause: Callable[[], bool] | None = None,
) -> list[dict]:
    """Drive the loop and return the messages appended to history this run.

    `should_pause` is checked between turns. It stops the loop the way a finished
    answer would — cleanly, with everything so far returned — rather than the way a
    cancellation does. Used by the plan gate: the model lays out a multi-step plan,
    and the harness holds execution until the user approves it.

    `messages` is mutated in place (assistant + tool entries are appended) and may
    also be CONDENSED in place when it approaches the context window. The return
    value is a separate accumulator of exactly what this run produced, so the
    caller persists the real messages even when the in-flight copy was compacted —
    the transcript on disk stays complete while the request stays inside the window.
    (It is also why this is an accumulator rather than a `messages[start:]` slice:
    compaction changes the list's length, which would invalidate any saved index.)
    """
    registry = tools.REGISTRY if registry is None else registry
    stream = stream or client.stream_chat  # resolved now, not at def time
    is_cancelled = is_cancelled or (lambda: False)
    specs = tools.specs(registry)
    appended: list[dict] = []
    # The server's own count for the last request — the accurate compaction signal.
    # Seeded by the caller because one run is one user turn: without carrying the
    # previous turn's count across the boundary, a growing conversation would only
    # ever be measured by the estimate.

    def add(msg: dict) -> None:
        """Record a message both in the live history and in what we hand back."""
        messages.append(msg)
        appended.append(msg)

    for _ in range(max_turns):
        if is_cancelled():
            break
        # A pause is NOT a cancellation: something outside the loop (the plan gate)
        # wants the user to decide before the next turn runs. We stop between turns,
        # so everything produced so far is complete and gets persisted; the caller
        # resumes by simply running again with the same history.
        if should_pause and should_pause():
            break

        # Condense BEFORE sending, so this turn's request fits. `appended` is
        # untouched by this, so a compacted run still persists its real messages.
        done = context.maybe_compact(messages, prompt_tokens, summarize=summarize)
        if done:
            prompt_tokens = None  # the old count no longer describes this prompt
            emit(Notice(_compaction_note(done)))

        # --- one LLM turn: stream deltas live, collect the tool calls ---
        text = ""
        calls: list[ToolCall] = []
        for event in stream(messages, specs):
            if is_cancelled():
                return appended
            if isinstance(event, ToolCall):
                calls.append(event)
            elif isinstance(event, TextDelta):
                text += event.text
            elif isinstance(event, Usage):
                prompt_tokens = event.prompt_tokens
            emit(event)  # thinking / text / tool-call shown as it arrives

        add(_assistant_message(text, calls))

        # Termination: a turn with no tool calls is the final answer.
        if not calls:
            return appended

        # --- run the requested tools, feed results back, then loop ---
        # Approval/safety runs sequentially (so modals never race); execution then
        # runs in PARALLEL when there is more than one runnable tool and all are
        # parallelizable (a task fan-out). The global gate in client.py still bounds
        # true gateway concurrency. Results are emitted/appended in call order so the
        # tool messages line up with the assistant's tool_calls.
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

    else:  # loop fell through without returning -> hit the turn cap
        # Force ONE tool-free wrap-up turn instead of truncating mid-work: withhold the
        # tools entirely (stream(..., None)) so the model CANNOT call another tool and
        # MUST answer, primed to summarize done/remaining/next. Beats a bare stop — the
        # user gets a usable close, and a runaway loop still can't keep calling tools.
        if not is_cancelled():
            add({"role": "user", "content": prompts.max_turns_prompt()})
            text = ""
            for event in stream(messages, None):  # tools off for this turn
                if is_cancelled():
                    return appended
                if isinstance(event, TextDelta):
                    text += event.text
                emit(event)
            add(_assistant_message(text, []))

    return appended
