import json
import time
from collections.abc import Iterable, Iterator

from openai import OpenAI

from ahacode import config
from ahacode.events import Event, TextDelta, ThinkingDelta, ToolCall, ToolCallDelta, Usage

# The UI never sees provider-specific shapes: this module converts them into the
# canonical events in events.py. A plain turn emits TextDelta / ThinkingDelta;
# when tools are offered, completed ToolCalls are emitted once reassembled.

_client: OpenAI | None = None
_cfg: config.ModelConfig | None = None


def reset() -> None:
    """Forget the cached client and config; the next request reloads from disk."""
    global _client, _cfg
    _client = None
    _cfg = None


def _ensure_client() -> tuple[OpenAI, config.ModelConfig]:
    global _client, _cfg
    if _client is None:
        _cfg = config.load()
        _client = OpenAI(
            base_url=_cfg.base_url, api_key=_cfg.api_key, timeout=_cfg.timeout
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
            # Live fragment for the UI (kilocode-style delta); the final parsed
            # ToolCall is still emitted after the stream for the loop to execute.
            yield ToolCallDelta(index=frag.index, name=slot["name"], fragment=piece)

    # A "length" finish means the model was cut off at the token limit, so any
    # tool call it was mid-way through emitting is half-built and unsafe to run
    # (Roo Code / Pi both refuse these). Skip them rather than execute garbage.
    if finish_reason == "length" and pending:
        yield TextDelta("\n[response truncated at token limit — tool call(s) skipped]")
        return

    for slot in pending.values():
        try:
            arguments = json.loads(slot["args"] or "{}")
        except json.JSONDecodeError:
            yield TextDelta(f"\n[tool call '{slot['name']}' had unparseable arguments — skipped]")
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
    # `with` closes the HTTP connection on every exit path — including when the
    # caller abandons this generator mid-stream (GeneratorExit) — so a cancelled
    # request stops occupying the single-slot gateway.
    with client.chat.completions.create(**kwargs) as response:
        yield from _iter_events(response)


def complete(messages: list[dict]) -> str:
    """One-shot, non-streaming completion — for short utility calls (e.g. titling).

    Separate from stream_chat: no tools, no streaming, just the text back.
    """
    client, cfg = _ensure_client()
    resp = client.chat.completions.create(model=cfg.name, messages=messages, stream=False)
    return (resp.choices[0].message.content or "").strip()


def list_models() -> list[str]:
    """Model ids offered by the endpoint (GET /v1/models)."""
    client, _ = _ensure_client()
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
