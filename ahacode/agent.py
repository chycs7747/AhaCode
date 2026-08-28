"""The agent loop: stream a turn, run any tools the model asked for, feed the
results back, repeat — until a turn arrives with no tool calls (the final
answer). This is the classic agent harness loop: termination is "the assistant
produced no tool calls", with a max-turns backstop against runaways.

Kept widget-free and pure so it is testable without a terminal: the UI is
reached only through the injected `emit` callback (one canonical Event at a
time), and the LLM/tools are injectable for offline tests.
"""

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

# Type aliases spelling out the seams the app (and tests) plug into.
StreamFn = Callable[[list[dict], list[dict] | None], Iterator[Event]]
EmitFn = Callable[[Event], None]
ApproveFn = Callable[[ToolCall], bool]


# Tool-call rounds one user message may take before the tool-free wrap-up turn.
DEFAULT_MAX_TURNS = 10

# What the status line calls the compaction pass while it blocks the loop. Short
# because it has to be: the composer bar leaves the status 18 cells on an 80-wide
# terminal and Korean costs two each, so "컨텍스트 압축" (23 with the clock) loses
# exactly the seconds it is there to show. The Notice afterwards says the rest.
COMPACTING = "압축 중"

# Names models reach for that this project spells differently — mapped to the tool
# that actually does the job. Not aliases: the call still fails. This exists so the
# failure costs ONE turn instead of a run of them, because a model that invented
# `run_python` answers a bare rejection by inventing `python`, then
# `code_interpreter`. Only names with an unambiguous local equivalent belong here.
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
    """The one sentence that turns a rejection into a next move.

    Three cases, most specific first: a known invented name whose equivalent IS
    available, the same but NOT available this turn (plan mode has no bash, a
    sub-agent at the depth limit has no task) — which is worth saying outright so
    the model stops hunting — and finally a plain near-miss on spelling.
    """
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
    """Own the stream's lifetime: close it on every way out of the loop.

    client.stream_chat holds the process-wide concurrency permit with a `with` INSIDE
    its generator, so the permit comes back only once that generator is exhausted or
    closed. Stopping a turn mid-stream leaves the loop early; letting the frame's
    refcount finalize the generator works right up until the frame is kept alive by a
    traceback or a reference cycle, and then the permit is held until a GC pass. Leak
    max_parallel_agents of them and every later request blocks on acquire() forever —
    no CPU, no network, an app that simply looks frozen.

    contextlib.closing would do, except `stream` is an injected seam typed
    Iterator[Event]: the real one is a generator, and a plain iterator has no close().
    """
    try:
        yield events
    finally:
        close = getattr(events, "close", None)
        if close is not None:
            close()


def _compaction_note(done: context.Compaction) -> str:
    """What to tell the user about a compaction. Losing history silently is the bad
    failure mode — they should know why the agent may have forgotten something."""
    if done.pruned_chars:
        return f"🗜 컨텍스트 확보를 위해 오래된 도구 출력 {done.pruned_chars:,}자를 비웠어요."
    return f"🗜 컨텍스트 한계에 가까워 이전 메시지 {done.summarized}개를 요약으로 압축했어요."


def _announced_summarize(
    summarize: context.SummarizeFn | None, emit: EmitFn
) -> context.SummarizeFn:
    """Wrap the summarizer so the UI is told while it runs.

    The bracket goes HERE rather than around maybe_compact because only this half
    is slow, and only sometimes: most calls return under the threshold without
    doing anything, and pruning is pure string work. Announcing the whole of
    maybe_compact would flash an indicator on every turn for work that already
    finished. This fires exactly when a model call is about to block the loop.
    """

    def run(older: list[dict]) -> str:
        emit(Phase(COMPACTING))
        try:
            return (summarize or context.llm_summarize)(older)
        finally:
            emit(Phase(COMPACTING, done=True))

    return run


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
        # "unknown tool: X" alone is a dead end. Point at the two things that
        # resolve it: the set this turn actually has (it varies by mode, so the
        # model cannot know it without being told) and the schemas already sitting
        # in this request. Same principle as the parse_error branch above — say what
        # to do next, not just what went wrong.
        return ToolResult(
            call.id, call.name,
            f"unknown tool: {call.name}. This turn's tools are: "
            f"{', '.join(sorted(registry))} — that is the complete set, and the "
            "schema and description of each one came with this request; read them "
            f"there rather than guessing a name.{_instead_of(call.name, registry)}",
            is_error=True,
        )
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

    # max_turns <= 0 means no cap. The cap exists to stop a runaway loop, but on a
    # session carrying out a plan it was stopping ORDINARY work: one measured turn
    # spent all 30 rounds and 25 minutes and finished 0 of 3 steps, so every turn
    # ended in the wrap-up below instead of at a completed step. Uncapped, the
    # backstop moves to the caller, which can see something the round counter never
    # could — whether the checklist is actually advancing (see app._auto_continue).
    rounds = itertools.count() if max_turns <= 0 else range(max_turns)
    for _ in rounds:
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
        done = context.maybe_compact(
            messages, prompt_tokens, summarize=_announced_summarize(summarize, emit)
        )
        if done:
            prompt_tokens = None  # the old count no longer describes this prompt
            emit(Notice(_compaction_note(done)))

        # --- one LLM turn: stream deltas live, collect the tool calls ---
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
            with _streaming(stream(messages, None)) as events:  # tools off this turn
                for event in events:
                    if is_cancelled():
                        return appended
                    if isinstance(event, TextDelta):
                        text += event.text
                    emit(event)
            add(_assistant_message(text, []))

    return appended
