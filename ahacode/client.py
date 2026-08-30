import json
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from openai import OpenAI

from ahacode import config, prompts
from ahacode.events import (
    Event, Notice, TextDelta, ThinkingDelta, ToolCall, ToolCallDelta, Usage,
)

# The mode of the turn being sent, per THREAD, so plan/impl/sub-agent each pick
# their own thinking budget (config.thinking_budget_for). Thread-local because
# parallel sub-agents run on separate pool threads; the context manager saves and
# restores, so a child on its parent's own thread leaves the parent's mode intact.
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
# How a model is SAMPLED is a property of the model, not of the task, so it lives
# here beside the other provider knobs rather than in config.toml.
#
# Sending nothing is worse than sending a wrong value: it hands the request to the
# server's default, which moves with the vLLM container's flags and which the app
# cannot see. Measured on the gateway — nothing sent: five identical requests, five
# different answers; temperature=0: byte-identical. So a per-request value does
# override the load-time default. A/B over 48 runs found no effect on solve rate,
# turns or tokens, which is the point: this buys reproducibility, not quality.
#
# The values are Qwen's published recommendation and differ BY MODE, because this
# app switches modes within one conversation (see no_think) and a single
# server-side default cannot satisfy both. top_k / min_p / repetition_penalty are
# vLLM extensions that must ride in extra_body — which is why the profile is keyed
# by family: Anthropic or OpenAI would reject them.
SAMPLING: dict[str, dict[str, dict]] = {
    "qwen": {
        "think":   {"kwargs": {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0},
                    "extra":  {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}},
        "nothink": {"kwargs": {"temperature": 0.7, "top_p": 0.80, "presence_penalty": 1.5},
                    "extra":  {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}},
    },
}


def _sampling_family(model: str) -> str | None:
    """The sampling profile that fits this model, or None.

    Deliberately NOT prompts.family(), which falls back to "qwen" for anything it
    does not recognise — a prompt has to pick something, so a default is right
    there. Sampling has no such obligation, and the values are model-specific: the
    Qwen3 profile below (temperature 1.0, top_k 20, min_p 0) is a worse guess for
    Llama or DeepSeek than sending nothing and letting the server apply its own.
    """
    name = (model or "").lower()
    return "qwen" if "qwen" in name else None


def sampling_for(model: str, *, no_think: bool) -> tuple[dict, dict]:
    """(request kwargs, extra_body) for this model in this mode.

    An unknown family gets nothing — better to let that provider apply its own
    default than to send it parameters it may not understand.
    """
    profile = SAMPLING.get(_sampling_family(model))
    if not profile:
        return {}, {}
    slot = profile["nothink" if no_think else "think"]
    return dict(slot["kwargs"]), dict(slot["extra"])


# How long a "does this address answer?" probe may take (see list_models). Short on
# purpose: it runs while someone waits on a settings screen.
PROBE_TIMEOUT = 10.0

# Endpoints that refused our vendor extensions, by base_url. Everything outside the
# plain OpenAI shape — enable_thinking, thinking_token_budget, top_k, min_p — is an
# extension some servers reject outright, so the failed round trip that discovers it
# is paid once. reset() clears this, which is how you re-probe a reconfigured server.
_NO_EXTRAS: set[str] = set()

_client: OpenAI | None = None
_cfg: config.ModelConfig | None = None
# Process-wide concurrency gate — see _ensure_gate. Sized from config on first use.
_gate: threading.BoundedSemaphore | None = None
# Guards the lazy construction of the three globals above. Without it a fan-out that
# starts several requests at once can have two threads both find _gate None and both
# build a semaphore: the loser's permits are already held, so the cap is briefly
# exceeded by exactly the thing it exists to bound.
_init_lock = threading.Lock()
# When a permit last entered or left the gate. A queue — however deep — keeps
# changing hands; a gate whose permits were taken by requests that are now gone
# never does. That difference is what tells a busy backend apart from a deadlock
# without guessing at a deadline (see _acquire_permit).
_gate_clock = threading.Lock()
_last_gate_change = time.monotonic()
# Grace on top of the request timeout before an unmoving gate counts as stuck.
GATE_STUCK_MARGIN = 60.0
_GATE_POLL = 1.0
# How long a queue may be silent before it is worth a line on screen. Waiting is
# normal — max_parallel_agents = 1 is the recommended setting for a single GPU, so
# every fan-out queues — but on screen a normal wait and a hang look identical, and
# that is the whole reason a stuck app was hard to recognise as stuck.
WAIT_NOTICE_AFTER = 3.0


def _touch_gate() -> None:
    global _last_gate_change
    with _gate_clock:
        _last_gate_change = time.monotonic()


def _gate_idle_seconds() -> float:
    with _gate_clock:
        return time.monotonic() - _last_gate_change


def _reset_gate() -> None:
    """Throw the gate away so the next caller builds a fresh one."""
    global _gate
    with _init_lock:
        _gate = None
    _touch_gate()


def _wait_for_permit(timeout: float, limit: int):
    """Take a concurrency permit — saying so when the wait is long enough to look
    like a hang, and rebuilding a gate whose permits have leaked.

    A permit is held for one request, and no request outlives the configured client
    timeout — so if nothing has entered or left the gate for longer than that, the
    permits are held by requests that no longer exist and waiting on them is waiting
    forever. Rebuild in that case. A real queue keeps the clock moving and is left
    alone however long it takes, which is the point: the app must not cut off a cold
    model load, and it must not sit frozen on permits nobody holds.

    A generator, so it can report through the same event channel as everything else:
    client.py has no way to reach a widget and should not grow one. Callers take the
    result with `yield from`. Returns the semaphore the permit came from (release
    THAT one, not whatever _ensure_gate hands out later) and whether a rebuild was
    needed.
    """
    stuck_after = timeout + GATE_STUCK_MARGIN
    started = time.monotonic()
    told = healed = False
    while True:
        gate = _ensure_gate()
        if gate.acquire(timeout=_GATE_POLL):
            _touch_gate()
            if told:  # close the loop on a wait the user was told about
                yield Notice("▶ 자리가 나서 요청을 시작합니다.")
            return gate, healed
        if not told and time.monotonic() - started > WAIT_NOTICE_AFTER:
            told = True
            yield Notice(
                f"⏳ 다른 요청이 끝나기를 기다리는 중입니다 (동시 실행 한도 {limit}개). "
                "멈춘 것이 아니라 차례를 기다리는 중입니다."
            )
        if _gate_idle_seconds() > stuck_after:
            _reset_gate()
            healed = True


def reset() -> None:
    """Forget the cached client, config, and concurrency gate; the next request
    reloads from disk (so an edited max_parallel_agents resizes the gate)."""
    global _client, _cfg, _gate
    _client = None
    _cfg = None
    _gate = None
    _NO_EXTRAS.clear()  # re-probe: the endpoint or its config may have just changed
    _touch_gate()  # a fresh gate has not been sitting still — don't inherit an old idle


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

        # Thinking arrives under a different key per server: reasoning_content on
        # vLLM's parser and DeepSeek-shaped APIs, plain reasoning elsewhere. Neither
        # is in the SDK's typed model, hence the getattr. Read BOTH — looking for
        # the wrong one is indistinguishable from a model that never thinks: no
        # block, no error, just a long silence before the answer.
        for key in ("reasoning_content", "reasoning"):
            piece = getattr(delta, key, None)
            if isinstance(piece, str) and piece:
                yield ThinkingDelta(piece)
                break
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
            # Don't drop it: a turn left with no tool call reads as "final answer"
            # to the agent loop, which then stops mid-task (measured — a malformed
            # call ended the run both times). Emit the call carrying its parse
            # failure so the loop feeds an error back and the model resends.
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
    # Reasoning controls (vendor extensions, via extra_body). Two exclusive modes:
    #  - no-think: the last message is a tool result, so this turn acts on it rather
    #    than re-deliberating. enable_thinking=False, and no budget/effort — both are
    #    meaningless with thinking off. The budget caps ONE turn; this is what stops
    #    re-deliberation stacking into a multi-turn spiral.
    #  - normal: thinking on, capped by thinking_token_budget, reasoning_effort as a
    #    hint. A server without a reasoning-config refuses the budget (see fallback).
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
    # Skip the extras entirely once this endpoint has told us it refuses them: the
    # discovery costs one rejected request, and repeating it every turn would make a
    # working server feel broken.
    if extra and cfg.base_url not in _NO_EXTRAS:
        kwargs["extra_body"] = extra
    # Hold a global concurrency permit for the request's lifetime: taken here,
    # released when this generator is exhausted or closed. Every agent funnels
    # through here, so this is what bounds real gateway concurrency. The inner
    # `with` closes the connection on every exit path, GeneratorExit included.
    gate, healed = yield from _wait_for_permit(cfg.timeout, cfg.max_parallel_agents)
    if healed:
        # Say it out loud rather than recovering in silence: a gate that had to be
        # rebuilt means requests ended without giving their permit back, and that is
        # a defect worth seeing rather than one worth smoothing over.
        yield Notice("동시 요청 게이트가 멈춰 있어 초기화했습니다 — "
                     "이전 요청이 자리를 반납하지 않았습니다.")
    try:
        yield from _stream_with_budget_fallback(client, kwargs, cfg.base_url)
    finally:
        gate.release()
        _touch_gate()


def _stream_with_budget_fallback(client, kwargs: dict, base_url: str = "") -> Iterator[Event]:
    """Stream the request, degrading once if the server refuses our extensions.

    Two steps, narrowest first. A server whose reasoning-config is not set up refuses
    thinking_token_budget in particular, and dropping only that keeps the sampling
    profile. Any other request rejection means this endpoint does not take vendor
    extensions at all — vLLM, Ollama, llama.cpp and the rest disagree about which of
    them exist — so drop them wholesale and remember the endpoint, or the failed
    round trip is paid again on every single turn.

    Retried only while nothing has been yielded: past the first event the server has
    accepted the request, and retrying would replay what the user already saw.
    """
    started = False
    retry_extra: dict | None = None
    note: Notice | None = None
    try:
        with client.chat.completions.create(**kwargs) as response:
            for event in _iter_events(response):
                started = True
                yield event
        return
    except Exception as exc:
        extra = kwargs.get("extra_body") or {}
        status = getattr(exc, "status_code", None)
        if started or not extra:
            raise
        if "thinking_token_budget" in extra and "reasoning_config" in str(exc):
            retry_extra = {k: v for k, v in extra.items() if k != "thinking_token_budget"}
        elif isinstance(status, int) and 400 <= status < 500:
            retry_extra = {}
            if base_url:
                _NO_EXTRAS.add(base_url)
            note = Notice(
                "이 서버는 사고·샘플링 확장 옵션을 받지 않아 기본 설정으로 요청합니다 "
                "(사고 예산과 reasoning_effort는 이 엔드포인트에서 무시됩니다)."
            )
        else:
            raise

    if note is not None:
        yield note
    retry = {**kwargs, "extra_body": retry_extra} if retry_extra else {
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
    # the non-thinking profile.
    sample_kwargs, sample_extra = sampling_for(cfg.name, no_think=True)
    # ...and thinking is switched off to match, not just the sampling — otherwise
    # the sampling says "be decisive" while the budget says "deliberate for 4096
    # tokens". Compaction runs here, so a thinking model paid a reasoning pass before
    # the summary's first word: one measured six-minute compaction, no stream behind
    # it, indistinguishable from a frozen app. Nothing here needs deliberation — a
    # title and a condensed transcript both restate text already in the prompt.
    sample_extra["chat_template_kwargs"] = {"enable_thinking": False}
    if cfg.base_url in _NO_EXTRAS:
        sample_extra = {}
    try:
        resp = client.chat.completions.create(
            model=cfg.name, messages=messages, stream=False,
            extra_body=sample_extra or None, **sample_kwargs,
        )
    except Exception as exc:
        # Same degrade as the streaming path, and it matters as much: titling and
        # context compaction run through here, so an endpoint that rejects the extras
        # would fail every compaction — and compaction failing is how a long session
        # stops working entirely.
        status = getattr(exc, "status_code", None)
        if not sample_extra or not (isinstance(status, int) and 400 <= status < 500):
            raise
        _NO_EXTRAS.add(cfg.base_url)
        resp = client.chat.completions.create(
            model=cfg.name, messages=messages, stream=False, **sample_kwargs,
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
