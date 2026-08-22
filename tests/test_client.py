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
    assert list(client._iter_events(chunks)) == [
        ToolCall(id="call_1", name="read", arguments={"path": "ahacode/app.py"})
    ]


def test_two_tool_calls_by_index():
    chunks = [
        _chunk(_delta(tool_calls=[_frag(0, id="a", name="read", arguments='{"path":"x"}')])),
        _chunk(_delta(tool_calls=[_frag(1, id="b", name="bash", arguments='{"command":"ls"}')]),
               finish_reason="tool_calls"),
    ]
    assert list(client._iter_events(chunks)) == [
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
