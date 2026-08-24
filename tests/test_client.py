"""Unit tests for client._iter_events — the stream-reassembly logic, exercised
with synthetic OpenAI-shaped chunks (no network)."""

from types import SimpleNamespace

from ahacode import client
from ahacode.events import TextDelta, ThinkingDelta, ToolCall


def _delta(content=None, tool_calls=None, reasoning=None):
    d = SimpleNamespace(content=content, tool_calls=tool_calls)
    if reasoning is not None:
        d.reasoning = reasoning
    return d


def _chunk(delta=None, finish_reason=None, no_choices=False):
    if no_choices:  # usage-only trailer chunk
        return SimpleNamespace(choices=[])
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    )


def _frag(index, id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def test_text_and_thinking_deltas():
    chunks = [
        _chunk(_delta(reasoning="Think")),
        _chunk(_delta(content="Hello")),
        _chunk(_delta(content=" world"), finish_reason="stop"),
    ]
    assert list(client._iter_events(chunks)) == [
        ThinkingDelta("Think"),
        TextDelta("Hello"),
        TextDelta(" world"),
    ]


def test_tool_call_reassembled_across_fragments():
    chunks = [
        _chunk(_delta(tool_calls=[_frag(0, id="call_1", name="read", arguments='{"path": "')])),
        _chunk(_delta(tool_calls=[_frag(0, arguments="ahacode/app.py")])),
        _chunk(_delta(tool_calls=[_frag(0, arguments='"}')]), finish_reason="tool_calls"),
    ]
    finals = [e for e in client._iter_events(chunks) if isinstance(e, ToolCall)]
    assert finals == [
        ToolCall(id="call_1", name="read", arguments={"path": "ahacode/app.py"})
    ]


def test_two_tool_calls_by_index():
    chunks = [
        _chunk(_delta(tool_calls=[_frag(0, id="a", name="read", arguments='{"path":"x"}')])),
        _chunk(_delta(tool_calls=[_frag(1, id="b", name="bash", arguments='{"command":"ls"}')]),
               finish_reason="tool_calls"),
    ]
    finals = [e for e in client._iter_events(chunks) if isinstance(e, ToolCall)]
    assert finals == [
        ToolCall(id="a", name="read", arguments={"path": "x"}),
        ToolCall(id="b", name="bash", arguments={"command": "ls"}),
    ]


def test_length_truncation_skips_partial_tool_call():
    chunks = [
        _chunk(_delta(tool_calls=[_frag(0, id="c", name="read", arguments='{"path": "aha')]),
               finish_reason="length"),
    ]
    events = list(client._iter_events(chunks))
    assert not any(isinstance(e, ToolCall) for e in events)  # never run a half-built call
    assert any(isinstance(e, TextDelta) and "truncated" in e.text for e in events)


def test_usage_only_trailer_chunk_is_ignored():
    chunks = [_chunk(_delta(content="hi")), _chunk(no_choices=True)]
    assert list(client._iter_events(chunks)) == [TextDelta("hi")]


def test_usage_trailer_becomes_a_usage_event():
    from ahacode.events import Usage
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=7, total_tokens=19)
    chunks = [
        _chunk(_delta(content="hi"), finish_reason="stop"),
        SimpleNamespace(choices=[], usage=usage),  # the include_usage trailer
    ]
    events = list(client._iter_events(chunks))
    assert TextDelta("hi") in events
    u = next(e for e in events if isinstance(e, Usage))
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (12, 7, 19)


def test_tool_call_streams_deltas_then_final():
    from ahacode.events import ToolCall, ToolCallDelta
    chunks = [
        _chunk(_delta(tool_calls=[_frag(0, id="c1", name="write", arguments='{"path": "')])),
        _chunk(_delta(tool_calls=[_frag(0, arguments='a.py", "content": "x')])),
        _chunk(_delta(tool_calls=[_frag(0, arguments='=1"}')]), finish_reason="tool_calls"),
    ]
    events = list(client._iter_events(chunks))
    deltas = [e for e in events if isinstance(e, ToolCallDelta)]
    assert deltas and all(d.name == "write" for d in deltas)          # name known from first frag
    assert "".join(d.fragment for d in deltas) == '{"path": "a.py", "content": "x=1"}'
    final = [e for e in events if isinstance(e, ToolCall)]
    assert len(final) == 1 and final[0].arguments == {"path": "a.py", "content": "x=1"}


# --- stream_chat reasoning params + budget fallback ------------------------
import contextlib

from ahacode import config


class _FakeStream:
    def __enter__(self):
        return iter([])  # no chunks -> _iter_events yields nothing

    def __exit__(self, *a):
        return False


def _fake_client(create):
    class C:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return create(**kw)
    return C()


def _cfg(**kw):
    base = dict(base_url="u", name="m", api_key="k", timeout=1.0,
                thinking_token_budget=4096, reasoning_effort="medium")
    base.update(kw)
    return config.ModelConfig(**base)


def test_stream_chat_sends_reasoning_extra_body(monkeypatch):
    """budget + effort ride in extra_body when set."""
    seen = {}
    fc = _fake_client(lambda **kw: (seen.update(kw), _FakeStream())[1])
    monkeypatch.setattr(client, "_ensure_client", lambda: (fc, _cfg()))
    monkeypatch.setattr(client, "_ensure_gate", lambda: contextlib.nullcontext())
    list(client.stream_chat([{"role": "user", "content": "x"}]))
    assert seen["extra_body"] == {"reasoning_effort": "medium", "thinking_token_budget": 4096}


def test_stream_chat_omits_budget_when_zero(monkeypatch):
    seen = {}
    fc = _fake_client(lambda **kw: (seen.update(kw), _FakeStream())[1])
    monkeypatch.setattr(client, "_ensure_client", lambda: (fc, _cfg(thinking_token_budget=0)))
    monkeypatch.setattr(client, "_ensure_gate", lambda: contextlib.nullcontext())
    list(client.stream_chat([{"role": "user", "content": "x"}]))
    assert seen["extra_body"] == {"reasoning_effort": "medium"}  # no budget key


def test_budget_fallback_on_reasoning_config_error(monkeypatch):
    """A server without its reasoning-config refuses the budget; we retry once
    without it (keeping effort) so the turn still runs."""
    calls = []

    def create(**kw):
        calls.append(kw)
        if "thinking_token_budget" in (kw.get("extra_body") or {}):
            raise RuntimeError("thinking_token_budget is set but reasoning_config is not configured")
        return _FakeStream()

    monkeypatch.setattr(client, "_ensure_client", lambda: (_fake_client(create), _cfg()))
    monkeypatch.setattr(client, "_ensure_gate", lambda: contextlib.nullcontext())
    list(client.stream_chat([{"role": "user", "content": "x"}]))
    assert len(calls) == 2
    assert "thinking_token_budget" in calls[0]["extra_body"]      # first tried with budget
    assert calls[1]["extra_body"] == {"reasoning_effort": "medium"}  # retry dropped only the budget


def test_no_think_after_tool_result(monkeypatch):
    """When the last message is a tool result, thinking is disabled for that turn —
    enable_thinking=False rides in extra_body and the budget/effort are dropped."""
    seen = {}
    fc = _fake_client(lambda **kw: (seen.update(kw), _FakeStream())[1])
    monkeypatch.setattr(client, "_ensure_client", lambda: (fc, _cfg()))
    monkeypatch.setattr(client, "_ensure_gate", lambda: contextlib.nullcontext())
    list(client.stream_chat([
        {"role": "user", "content": "solve it"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ran"},
    ]))
    assert seen["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "thinking_token_budget" not in seen["extra_body"]  # meaningless with thinking off


def test_no_think_off_keeps_thinking_on_tool_turn(monkeypatch):
    """With the flag off, a tool-result turn is a normal thinking turn (budget sent)."""
    seen = {}
    fc = _fake_client(lambda **kw: (seen.update(kw), _FakeStream())[1])
    monkeypatch.setattr(client, "_ensure_client", lambda: (fc, _cfg(no_think_after_tools=False)))
    monkeypatch.setattr(client, "_ensure_gate", lambda: contextlib.nullcontext())
    list(client.stream_chat([{"role": "tool", "tool_call_id": "c1", "content": "ran"}]))
    assert seen["extra_body"] == {"reasoning_effort": "medium", "thinking_token_budget": 4096}


def test_unrelated_error_is_not_retried(monkeypatch):
    calls = []

    def create(**kw):
        calls.append(kw)
        raise RuntimeError("some other 500")

    monkeypatch.setattr(client, "_ensure_client", lambda: (_fake_client(create), _cfg()))
    monkeypatch.setattr(client, "_ensure_gate", lambda: contextlib.nullcontext())
    import pytest
    with pytest.raises(RuntimeError, match="some other 500"):
        list(client.stream_chat([{"role": "user", "content": "x"}]))
    assert len(calls) == 1  # no retry
