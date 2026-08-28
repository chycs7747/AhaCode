import json
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from openai import OpenAI

from ahacode import config, prompts
from ahacode.events import Event, TextDelta, ThinkingDelta, ToolCall, ToolCallDelta, Usage

# The mode of the turn currently being sent, per THREAD — so plan/impl/sub-agent
# each pick their own thinking budget (config.thinking_budget_for). Thread-local
# because parallel sub-agents run on separate pool threads; the context manager
# saves and restores, so a sub-agent run on the SAME thread as its parent (the
# sequential path) leaves the parent's mode intact when it returns.
_mode = threading.local()


def current_mode() -> str | None:
    return getattr(_mode, "value", None)


@contextmanager
def mode(name: str | None):
    """Set the active mode for stream_chat/complete calls on this thread."""
    prev = getattr(_mode, "value", None)
    _mode.value = name
    try:
        yield
    finally:
        _mode.value = prev

# The UI never sees provider-specific shapes: this module converts them into the
# canonical events in events.py. A plain turn emits TextDelta / ThinkingDelta;
# when tools are offered, completed ToolCalls are emitted once reassembled.

# --- sampling profiles ------------------------------------------------------
# How a model should be SAMPLED is a property of the model, not of the task, so it
# belongs here beside the other provider-specific knobs rather than in config.toml.
#
# Until now nothing was sent at all, which meant the server's own default governed
# every request. That is worse than a wrong value: it is an INVISIBLE one. It changes
# when the vLLM container is restarted with different flags or a different
# generation_config.json, the app has no way to know, and an experiment run today
# cannot be reproduced tomorrow. Measured on the gateway: with nothing sent, five
# identical requests produced five different answers; with temperature=0 they were
# byte-identical — so a per-request value does override the load-time default.
#
# The values are Qwen's published recommendation, and they differ BY MODE — which
# matters here because this app switches modes within a single conversation (see
# no_think below). One server-side default cannot satisfy both.
#
# A/B measured across 48 runs: no effect on solve rate, turns or tokens (48/48 either
# way). This is not here to make the model better; it is here so the app stops
# depending on a setting it cannot see.
#
# Standard OpenAI fields go in the request body; top_k / min_p / repetition_penalty
# are vLLM extensions and must ride in extra_body — which is also why the profile is
# keyed by family: sending them to Anthropic or OpenAI would be rejected.
SAMPLING: dict[str, dict[str, dict]] = {
    "qwen": {
        "think":   {"kwargs": {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0},
                    "extra":  {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}},
        "nothink": {"kwargs": {"temperature": 0.7, "top_p": 0.80, "presence_penalty": 1.5},
                    "extra":  {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}},
    },
}


def sampling_for(model: str, *, no_think: bool) -> tuple[dict, dict]:
    """(request kwargs, extra_body) for this model in this mode.

    An unknown family gets nothing — better to let that provider apply its own
    default than to send it parameters it may not understand.
    """
    profile = SAMPLING.get(prompts.family(model))
    if not profile:
        return {}, {}
    slot = profile["nothink" if no_think else "think"]
    return dict(slot["kwargs"]), dict(slot["extra"])


# How long a "does this address answer?" probe may take (see list_models). Short on
# purpose: it runs while someone waits on a settings screen.
PROBE_TIMEOUT = 10.0

_client: OpenAI | None = None
_cfg: config.ModelConfig | None = None
# Process-wide concurrency gate — see _ensure_gate. Sized from config on first use.
_gate: threading.BoundedSemaphore | None = None
# Guards the lazy construction of the three globals above. Without it a fan-out that
# starts several requests at once can have two threads both find _gate None and both
# build a semaphore: the loser's permits are already held, so the cap is briefly
# exceeded by exactly the thing it exists to bound.
_init_lock = threading.Lock()


def reset() -> None:
    """Forget the cached client, config, and concurrency gate; the next request
    reloads from disk (so an edited max_parallel_agents resizes the gate)."""
    global _client, _cfg, _gate
    _client = None
    _cfg = None
    _gate = None


def _ensure_gate() -> threading.BoundedSemaphore:
    """The one gate every request funnels through, so total concurrency against the
    single-GPU gateway is bounded no matter how the sub-agent tree fans out. Sized
    from max_parallel_agents; a permit is held only for a request's lifetime (never
    across a sub-agent delegation), so nested spawning cannot deadlock."""
    global _gate
    if _gate is None:
        with _init_lock:
            if _gate is None:  # re-checked: another thread may have built it meanwhile
                _gate = threading.BoundedSemaphore(config.load().max_parallel_agents)
    return _gate


def _ensure_client() -> tuple[OpenAI, config.ModelConfig]:
    global _client, _cfg
    if _client is None:
        with _init_lock:
            if _client is None:
                cfg = config.load()
                # Published together, and _cfg first: a reader that sees a non-None
                # _client must never find the config that belongs to it still unset.
                _cfg = cfg
                _client = OpenAI(
                    base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout
                )
    return _client, _cfg


def _iter_events(chunks: Iterable) -> Iterator[Event]:
    """Convert raw OpenAI stream chunks into canonical events.

    Pure (no network) so the reassembly logic is unit-testable with synthetic
    chunks. Tool-call arguments arrive as JSON fragments spread across many
    chunks (`{"path": "` -> `Se` -> `oul` -> `"}`), keyed by an index; we buffer
    per index and only parse once the stream ends — the classic "message framing"
    problem: a byte stream carries no record boundaries, so the receiver must
    reassemble whole messages itself.
    """
    pending: dict[int, dict] = {}  # index -> {"id", "name", "args"}
    finish_reason: str | None = None

    for chunk in chunks:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            yield Usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            )
        if not chunk.choices:  # usage-only trailer chunk (choices is empty here)
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        delta = choice.delta

        # Thinking: some servers stream it under "reasoning" (verified on the wire
        # against vLLM); not in the SDK's typed model, hence the defensive getattr.
        reasoning = getattr(delta, "reasoning", None)
        if isinstance(reasoning, str) and reasoning:
            yield ThinkingDelta(reasoning)
        if isinstance(delta.content, str) and delta.content:
            yield TextDelta(delta.content)

        for frag in getattr(delta, "tool_calls", None) or []:
            slot = pending.setdefault(frag.index, {"id": "", "name": "", "args": ""})
            if frag.id:
                slot["id"] = frag.id
            fn = getattr(frag, "function", None)
            if fn and fn.name:
                slot["name"] = fn.name
            piece = fn.arguments if (fn and fn.arguments) else ""
            if piece:
                slot["args"] += piece
            # Live fragment for the UI (a streaming delta); the final parsed
            # ToolCall is still emitted after the stream for the loop to execute.
            yield ToolCallDelta(index=frag.index, name=slot["name"], fragment=piece)

    # A "length" finish means the model was cut off at the token limit, so any
    # tool call it was mid-way through emitting is half-built and unsafe to run.
    # Skip them rather than execute garbage.
    if finish_reason == "length" and pending:
        yield TextDelta("\n[response truncated at token limit — tool call(s) skipped]")
        return

    for slot in pending.values():
        try:
            arguments = json.loads(slot["args"] or "{}")
        except json.JSONDecodeError:
            # Don't drop it. A dropped call can leave the turn with no tool call at
            # all, which the agent loop reads as "final answer" and stops mid-task
            # (measured: a local model emitting a malformed call twice ended the run
            # each time). Emit the call carrying the parse failure so the loop feeds
            # an error result back and the model resends it — the same way an
            # execution error is surfaced, not a silent skip.
            yield ToolCall(id=slot["id"], name=slot["name"], arguments={},
                           parse_error="arguments were not valid JSON")
            continue
        yield ToolCall(id=slot["id"], name=slot["name"], arguments=arguments)


def stream_chat(messages: list[dict], tools: list[dict] | None = None) -> Iterator[Event]:
    """Send the conversation (optionally with tool specs) and yield canonical events."""
    client, cfg = _ensure_client()
    kwargs: dict = {
        "model": cfg.name,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},  # ask for the token-usage trailer
    }
    if tools:
        kwargs["tools"] = tools
        # Explicit even though "auto" is the API default: it states plainly that
        # the *model* decides whether to call a tool (vs "required"/"none"/a named
        # tool, which would force its hand). Only sent alongside tools — some
        # servers reject tool_choice without a tools list.
        kwargs["tool_choice"] = "auto"
    # Reasoning controls (vendor extensions passed via extra_body). Two mutually
    # exclusive modes for this request:
    #  - no-think turn: the last message is a tool result, so this turn just acts on
    #    it. Disable thinking entirely (enable_thinking=False) — no budget/effort, they
    #    are meaningless with thinking off. This stops the per-turn re-deliberation that
    #    stacks into a multi-turn spiral (the budget only caps ONE turn).
    #  - normal turn: thinking on, capped by thinking_token_budget; reasoning_effort is
    #    a named hint. A server without its reasoning-config refuses the budget — handled
    #    by the fallback below.
    extra = {}
    no_think = (
        cfg.no_think_after_tools
        and bool(messages)
        and messages[-1].get("role") == "tool"
    )
    if no_think:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        if cfg.reasoning_effort:
            extra["reasoning_effort"] = cfg.reasoning_effort
        budget = cfg.thinking_budget_for(current_mode())
        if budget:
            extra["thinking_token_budget"] = budget
    # Sampling rides the SAME branch: the mode was just decided above, and the two
    # profiles differ, so deciding it twice is how they would drift apart.
    sample_kwargs, sample_extra = sampling_for(cfg.name, no_think=no_think)
    kwargs.update(sample_kwargs)
    extra.update(sample_extra)
    if extra:
        kwargs["extra_body"] = extra
    # Hold a global concurrency permit for the request's lifetime: acquired here,
    # released when this generator is exhausted or closed (the outer `with` exits).
    # All agents funnel through here, so this bounds true gateway concurrency (see
    # _ensure_gate). The inner `with` closes the HTTP connection on every exit path —
    # including when the caller abandons the generator mid-stream (GeneratorExit).
    with _ensure_gate():
        yield from _stream_with_budget_fallback(client, kwargs)


def _stream_with_budget_fallback(client, kwargs: dict) -> Iterator[Event]:
    """Stream the request. If the server refuses thinking_token_budget because its
    reasoning-config isn't set up, retry ONCE without the budget so the app still
    works (uncapped) instead of erroring on every turn. The refusal is a request-time
    400 (before any event is yielded), so the retry can't double-emit."""
    try:
        with client.chat.completions.create(**kwargs) as response:
            yield from _iter_events(response)
    except Exception as exc:
        extra = kwargs.get("extra_body") or {}
        if "thinking_token_budget" not in extra or "reasoning_config" not in str(exc):
            raise
        extra = {k: v for k, v in extra.items() if k != "thinking_token_budget"}
        retry = {**kwargs, "extra_body": extra} if extra else {
            k: v for k, v in kwargs.items() if k != "extra_body"
        }
        with client.chat.completions.create(**retry) as response:
            yield from _iter_events(response)


def complete(messages: list[dict]) -> str:
    """One-shot, non-streaming completion — for short utility calls (e.g. titling).

    Separate from stream_chat: no tools, no streaming, just the text back.
    """
    client, cfg = _ensure_client()
    # Same reason as stream_chat: never leave the sampling to whatever the server
    # happens to default to. A title or a summary is not a thinking task, so it takes
    # the non-thinking profile. (Thinking itself is left alone here — this helper does
    # not switch it, and turning it off would be a separate decision.)
    sample_kwargs, sample_extra = sampling_for(cfg.name, no_think=True)
    resp = client.chat.completions.create(
        model=cfg.name, messages=messages, stream=False,
        extra_body=sample_extra or None, **sample_kwargs,
    )
    return (resp.choices[0].message.content or "").strip()


def list_models(base_url: str | None = None, api_key: str | None = None) -> list[str]:
    """Model ids offered by an endpoint (GET /v1/models).

    With no arguments: the configured endpoint, through the cached client. Passing
    base_url probes a *different* endpoint with a throwaway client — what the
    settings modal needs, since the address being typed there is not saved yet and
    must not disturb the client the running session is using.

    Listing is a plain GET and does not load a model, so it is safe to call against
    a gateway that starts an engine on demand; only an inference request does that.
    The probe gets its own short timeout: an address typed with a typo should come
    back as an error in seconds, not hold the settings modal for the configured
    request timeout (minutes).
    """
    if base_url is None:
        client, _ = _ensure_client()
    else:
        client = OpenAI(base_url=base_url,
                        api_key=api_key or config.load().api_key,
                        timeout=PROBE_TIMEOUT)
    return [m.id for m in client.models.list()]


FAKE_THINKING = "The user greeted me. Keep the reply short."
FAKE_RESPONSE = "Hello! How can I help you today?"


def stream_chat_fake(messages: list[dict], tools: list[dict] | None = None) -> Iterator[Event]:
    """Offline fake stream emitting the same canonical events — for tests."""
    for word in FAKE_THINKING.split(" "):
        time.sleep(0.01)
        yield ThinkingDelta(word + " ")
    for word in FAKE_RESPONSE.split(" "):
        time.sleep(0.01)
        yield TextDelta(word + " ")
