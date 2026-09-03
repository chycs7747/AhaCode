"""The one module that talks to the model provider: streaming turns, one-shot
completions, the model list, and the process-wide concurrency gate."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from openai import OpenAI

from ahacode import config
from ahacode.events import (
    Event, Notice, TextDelta, ThinkingDelta, ToolCall, ToolCallDelta, Usage,
)

# The mode of the turn being sent, per thread, so plan / impl / sub-agent each pick
# their own thinking budget. Parallel sub-agents run on separate threads.
_mode = threading.local()


def current_mode() -> str | None:
    """The mode set by `mode()` on this thread, or None."""
    return getattr(_mode, "value", None)


@contextmanager
def mode(name: str | None):
    """Set the mode for stream_chat / complete calls on this thread, restoring the
    previous one on exit.

    Args:
        name: "plan", "impl", "subagent", or None for a plain act turn.
    """
    prev = getattr(_mode, "value", None)
    _mode.value = name
    try:
        yield
    finally:
        _mode.value = prev


# --- sampling profiles ------------------------------------------------------
# How a model is sampled is a property of the model, so it lives here rather than in
# config.toml. A value is always sent: the server's default moves with its launch
# flags and the app cannot see it. The profiles are Qwen's published recommendation
# and differ by mode, because this app switches modes within one conversation.
# top_k / min_p / repetition_penalty are vLLM extensions and ride in extra_body.
SAMPLING: dict[str, dict[str, dict]] = {
    "qwen": {
        "think":   {"kwargs": {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0},
                    "extra":  {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}},
        "nothink": {"kwargs": {"temperature": 0.7, "top_p": 0.80, "presence_penalty": 1.5},
                    "extra":  {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}},
    },
}


def _sampling_family(model: str) -> str | None:
    """The sampling profile that fits this model, or None: the values are
    model-specific, and a wrong guess is worse than the server's own default."""
    name = (model or "").lower()
    return "qwen" if "qwen" in name else None


def sampling_for(model: str, *, no_think: bool) -> tuple[dict, dict]:
    """The sampling parameters for this model in this mode.

    Args:
        model: The model name.
        no_think: Whether the turn runs with thinking off.

    Returns:
        (request kwargs, extra_body); both empty for a model with no profile, so the
        server applies its own default.
    """
    profile = SAMPLING.get(_sampling_family(model))
    if not profile:
        return {}, {}
    slot = profile["nothink" if no_think else "think"]
    return dict(slot["kwargs"]), dict(slot["extra"])


# How long a "does this address answer?" probe may take (see list_models).
PROBE_TIMEOUT = 10.0

# Endpoints that refused our vendor extensions (enable_thinking,
# thinking_token_budget, top_k, min_p), so the failed round trip is paid once.
_NO_EXTRAS: set[str] = set()

_client: OpenAI | None = None
_cfg: config.ModelConfig | None = None
_gate: threading.BoundedSemaphore | None = None  # see _ensure_gate
_init_lock = threading.Lock()  # guards the lazy construction of the three above
# When a permit last entered or left the gate. A queue keeps changing hands; a gate
# whose permits leaked never does (see _wait_for_permit).
_gate_clock = threading.Lock()
_last_gate_change = time.monotonic()
# Grace on top of the request timeout before an unmoving gate counts as stuck.
GATE_STUCK_MARGIN = 60.0
_GATE_POLL = 1.0
# Seconds a queued request waits silently before saying so on screen.
WAIT_NOTICE_AFTER = 3.0


def _touch_gate() -> None:
    """Record that a permit just changed hands."""
    global _last_gate_change
    with _gate_clock:
        _last_gate_change = time.monotonic()


def _gate_idle_seconds() -> float:
    """Seconds since a permit last changed hands."""
    with _gate_clock:
        return time.monotonic() - _last_gate_change


def _reset_gate() -> None:
    """Throw the gate away so the next caller builds a fresh one."""
    global _gate
    with _init_lock:
        _gate = None
    _touch_gate()


def _wait_for_permit(timeout: float, limit: int):
    """Take a concurrency permit, reporting a long wait and healing a leaked gate.

    No request outlives the client timeout, so a gate that has not moved for longer
    than that holds permits nobody will return, and is rebuilt. A real queue keeps
    the clock moving and is waited on however long it takes.

    A generator, so it can report through the event channel; take the result with
    `yield from`.

    Args:
        timeout: The configured request timeout, in seconds.
        limit: The gate's size, for the notice.

    Returns:
        (the semaphore the permit came from, whether the gate had to be rebuilt).
    """
    stuck_after = timeout + GATE_STUCK_MARGIN
    started = time.monotonic()
    told = healed = False
    while True:
        gate = _ensure_gate()
        if gate.acquire(timeout=_GATE_POLL):
            _touch_gate()
            if told:
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
    """Forget the cached client, config, gate and refused-extras memory, so the next
    request reloads everything from disk."""
    global _client, _cfg, _gate
    _client = None
    _cfg = None
    _gate = None
    _NO_EXTRAS.clear()
    _touch_gate()


def _ensure_gate() -> threading.BoundedSemaphore:
    """The one gate every request funnels through, sized from max_parallel_agents.

    A permit is held only for a request's lifetime, never across a sub-agent
    delegation, so nested spawning cannot deadlock.
    """
    global _gate
    if _gate is None:
        with _init_lock:
            if _gate is None:
                _gate = threading.BoundedSemaphore(config.load().max_parallel_agents)
    return _gate


def _ensure_client() -> tuple[OpenAI, config.ModelConfig]:
    """The cached OpenAI client and the config it was built from."""
    global _client, _cfg
    if _client is None:
        with _init_lock:
            if _client is None:
                cfg = config.load()
                # _cfg first: a reader that sees a client must never find its config unset.
                _cfg = cfg
                _client = OpenAI(
                    base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout
                )
    return _client, _cfg


def _iter_events(chunks: Iterable) -> Iterator[Event]:
    """Convert raw OpenAI stream chunks into canonical events.

    Tool-call arguments arrive as JSON fragments spread across chunks, keyed by an
    index; they are buffered per index and parsed once the stream ends.

    Args:
        chunks: The SDK's stream chunks.

    Returns:
        Usage, ThinkingDelta, TextDelta and ToolCallDelta events as they arrive,
        then one ToolCall per completed call.
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
        if not chunk.choices:  # the usage-only trailer chunk
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        delta = choice.delta

        # Thinking arrives under a different key per server (reasoning_content on
        # vLLM and DeepSeek-shaped APIs, reasoning elsewhere), outside the SDK's
        # typed model. Reading only one looks like a model that never thinks.
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
            yield ToolCallDelta(index=frag.index, name=slot["name"], fragment=piece)

    # Cut off at the token limit: a half-built tool call is unsafe to run.
    if finish_reason == "length" and pending:
        yield TextDelta("\n[response truncated at token limit — tool call(s) skipped]")
        return

    for slot in pending.values():
        try:
            arguments = json.loads(slot["args"] or "{}")
        except json.JSONDecodeError:
            # Emit the call with its parse failure so the loop feeds an error back
            # and the model resends; dropping it would read as a final answer.
            yield ToolCall(id=slot["id"], name=slot["name"], arguments={},
                           parse_error="arguments were not valid JSON")
            continue
        yield ToolCall(id=slot["id"], name=slot["name"], arguments=arguments)


def stream_chat(messages: list[dict], tools: list[dict] | None = None) -> Iterator[Event]:
    """Send the conversation and yield canonical events as they stream.

    Args:
        messages: The history to send.
        tools: The function schemas to offer, or None for a tool-free turn.

    Returns:
        The events. The concurrency permit is held until the iterator is exhausted
        or closed.
    """
    client, cfg = _ensure_client()
    kwargs: dict = {
        "model": cfg.name,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        kwargs["tools"] = tools
        # Only alongside tools: some servers reject tool_choice without a tools list.
        kwargs["tool_choice"] = "auto"
    # After a tool result the turn acts on it rather than re-deliberating: thinking
    # off, and no budget or effort, which are meaningless with thinking off.
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
    sample_kwargs, sample_extra = sampling_for(cfg.name, no_think=no_think)
    kwargs.update(sample_kwargs)
    extra.update(sample_extra)
    if extra and cfg.base_url not in _NO_EXTRAS:
        kwargs["extra_body"] = extra
    gate, healed = yield from _wait_for_permit(cfg.timeout, cfg.max_parallel_agents)
    if healed:
        yield Notice("동시 요청 게이트가 멈춰 있어 초기화했습니다 — "
                     "이전 요청이 자리를 반납하지 않았습니다.")
    try:
        yield from _stream_with_budget_fallback(client, kwargs, cfg.base_url)
    finally:
        gate.release()
        _touch_gate()


def _stream_with_budget_fallback(client, kwargs: dict, base_url: str = "") -> Iterator[Event]:
    """Stream the request, degrading once if the server refuses our extensions.

    Narrowest first: a server without a reasoning config refuses
    thinking_token_budget alone; any other 4xx means it takes no vendor extensions
    at all, which is remembered per endpoint. Retried only while nothing has been
    yielded, since past the first event a retry would replay what the user saw.

    Args:
        client: The OpenAI client.
        kwargs: The request.
        base_url: The endpoint, for remembering a refusal.

    Returns:
        The events.
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
    """A one-shot, non-streaming completion for short utility calls (titles, summaries).

    Thinking is switched off and the non-thinking sampling profile applied: nothing
    here needs deliberation, and a reasoning pass before a summary is a minutes-long
    silence on screen.

    Args:
        messages: The prompt.

    Returns:
        The answer text, stripped.
    """
    client, cfg = _ensure_client()
    sample_kwargs, sample_extra = sampling_for(cfg.name, no_think=True)
    sample_extra["chat_template_kwargs"] = {"enable_thinking": False}
    if cfg.base_url in _NO_EXTRAS:
        sample_extra = {}
    try:
        resp = client.chat.completions.create(
            model=cfg.name, messages=messages, stream=False,
            extra_body=sample_extra or None, **sample_kwargs,
        )
    except Exception as exc:
        # The same degrade as the streaming path: compaction runs through here, and
        # compaction failing is how a long session stops working entirely.
        status = getattr(exc, "status_code", None)
        if not sample_extra or not (isinstance(status, int) and 400 <= status < 500):
            raise
        _NO_EXTRAS.add(cfg.base_url)
        resp = client.chat.completions.create(
            model=cfg.name, messages=messages, stream=False, **sample_kwargs,
        )
    return (resp.choices[0].message.content or "").strip()


def list_models(base_url: str | None = None, api_key: str | None = None) -> list[str]:
    """The model ids an endpoint offers (GET /v1/models).

    Listing is a plain GET and loads no model. With no arguments the configured
    endpoint is asked through the cached client; with a base_url a throwaway client
    probes that address on a short timeout, which is what the settings screen needs.

    Args:
        base_url: An endpoint to probe instead of the configured one.
        api_key: The key for that endpoint; the configured one when omitted.

    Returns:
        The model ids.
    """
    if base_url is None:
        client, _ = _ensure_client()
    else:
        client = OpenAI(base_url=base_url,
                        api_key=api_key or config.load().api_key,
                        timeout=PROBE_TIMEOUT)
    return [m.id for m in client.models.list()]
