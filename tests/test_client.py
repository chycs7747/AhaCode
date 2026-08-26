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


def _thinking_keys(extra: dict) -> dict:
    """The thinking-related part of extra_body, without the sampling profile that
    now shares the dict (see client.SAMPLING)."""
    return {k: v for k, v in extra.items()
            if k in ("reasoning_effort", "thinking_token_budget", "chat_template_kwargs")}


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
    assert _thinking_keys(seen["extra_body"]) == {"reasoning_effort": "medium", "thinking_token_budget": 4096}


def test_stream_chat_omits_budget_when_zero(monkeypatch):
    seen = {}
    fc = _fake_client(lambda **kw: (seen.update(kw), _FakeStream())[1])
    monkeypatch.setattr(client, "_ensure_client", lambda: (fc, _cfg(thinking_token_budget=0)))
    monkeypatch.setattr(client, "_ensure_gate", lambda: contextlib.nullcontext())
    list(client.stream_chat([{"role": "user", "content": "x"}]))
    assert _thinking_keys(seen["extra_body"]) == {"reasoning_effort": "medium"}  # no budget key


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
    assert _thinking_keys(calls[1]["extra_body"]) == {"reasoning_effort": "medium"}  # retry dropped only the budget


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
    assert _thinking_keys(seen["extra_body"]) == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "thinking_token_budget" not in seen["extra_body"]  # meaningless with thinking off


def test_no_think_off_keeps_thinking_on_tool_turn(monkeypatch):
    """With the flag off, a tool-result turn is a normal thinking turn (budget sent)."""
    seen = {}
    fc = _fake_client(lambda **kw: (seen.update(kw), _FakeStream())[1])
    monkeypatch.setattr(client, "_ensure_client", lambda: (fc, _cfg(no_think_after_tools=False)))
    monkeypatch.setattr(client, "_ensure_gate", lambda: contextlib.nullcontext())
    list(client.stream_chat([{"role": "tool", "tool_call_id": "c1", "content": "ran"}]))
    assert _thinking_keys(seen["extra_body"]) == {"reasoning_effort": "medium", "thinking_token_budget": 4096}


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


# --- sampling profiles -------------------------------------------------------
# Nothing was sent before, so the server's own default governed every request — an
# invisible setting that changes when the container is restarted and makes a run
# impossible to reproduce. See client.SAMPLING for the full reasoning.

def test_sampling_differs_by_mode():
    """The app switches thinking on and off within one conversation, and Qwen
    publishes different parameters for the two modes."""
    think_kwargs, think_extra = client.sampling_for("qwen38", no_think=False)
    act_kwargs, act_extra = client.sampling_for("qwen38", no_think=True)
    assert think_kwargs == {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0}
    assert act_kwargs == {"temperature": 0.7, "top_p": 0.80, "presence_penalty": 1.5}
    assert think_extra == act_extra == {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}


def test_an_unknown_family_gets_no_sampling_params():
    """top_k / min_p / repetition_penalty are vLLM extensions; sending them to a
    provider that does not know them would be rejected outright."""
    assert client.sampling_for("claude-sonnet-5", no_think=False) == ({}, {})
    assert client.sampling_for("gpt-5", no_think=True) == ({}, {})


def test_the_request_carries_the_profile_for_its_mode(monkeypatch):
    """Standard fields in the body, vendor extensions in extra_body — and the mode
    must match the one the thinking switch just chose."""
    seen = {}
    fc = _fake_client(lambda **kw: (seen.update(kw), _FakeStream())[1])
    monkeypatch.setattr(client, "_ensure_client", lambda: (fc, _cfg(name="qwen38")))
    monkeypatch.setattr(client, "_ensure_gate", lambda: contextlib.nullcontext())

    list(client.stream_chat([{"role": "user", "content": "hi"}]))
    assert seen["temperature"] == 1.0 and seen["top_p"] == 0.95
    assert seen["extra_body"]["top_k"] == 20
    assert "chat_template_kwargs" not in seen["extra_body"]      # thinking is on

    seen.clear()
    list(client.stream_chat([{"role": "user", "content": "hi"},
                             {"role": "tool", "tool_call_id": "1", "content": "x"}]))
    assert seen["temperature"] == 0.7 and seen["presence_penalty"] == 1.5
    assert seen["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_complete_also_pins_its_sampling(monkeypatch):
    """The utility path (titles, summaries) had the same invisible dependency."""
    seen = {}
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="t"))])
    fc = _fake_client(lambda **kw: (seen.update(kw), reply)[1])
    monkeypatch.setattr(client, "_ensure_client", lambda: (fc, _cfg(name="qwen38")))
    assert client.complete([{"role": "user", "content": "title this"}]) == "t"
    assert seen["temperature"] == 0.7          # a title is not a thinking task
    assert seen["extra_body"]["top_k"] == 20


def test_stream_chat_uses_the_active_mode_thinking_budget(monkeypatch):
    """The mode context picks the per-mode budget: plan deep, impl shallow, and it
    restores on exit so nesting (a sub-agent inside an impl turn) is correct."""
    from dataclasses import replace
    from ahacode import client, config

    config.save(replace(config.DEFAULTS, thinking_token_budget=4096,
                        plan_thinking_budget=8192, impl_thinking_budget=2048,
                        subagent_thinking_budget=1024))
    client.reset()

    seen = []

    class FakeStream:
        def __enter__(self): return iter([])
        def __exit__(self, *a): return False

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    seen.append((kw.get("extra_body") or {}).get("thinking_token_budget"))
                    return FakeStream()

    monkeypatch.setattr(client, "_ensure_client", lambda: (FakeClient(), config.load()))

    def budget_under(mode):
        seen.clear()
        with client.mode(mode):
            list(client.stream_chat([{"role": "user", "content": "hi"}]))
        return seen[0]

    assert budget_under("plan") == 8192
    assert budget_under("impl") == 2048
    assert budget_under("subagent") == 1024
    assert budget_under(None) == 4096          # a plain act turn → global

    # nesting restores: impl outside, subagent inside, impl again after
    with client.mode("impl"):
        assert budget_under("subagent") == 1024
        assert client.current_mode() == "impl"
