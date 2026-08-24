"""Context-window management: when to condense, where it is safe to cut, and that
the transcript on disk stays complete when the in-flight copy is compacted."""

import pytest

from ahacode import agent, config, context, subagent
from ahacode.events import Notice, TextDelta, ToolCall, Usage

CFG = config.DEFAULTS


def _cfg(**kw):
    from dataclasses import replace
    return replace(CFG, **kw)


def _turn(*msgs):
    return list(msgs)


# --- the boundary rule ----------------------------------------------------

def test_split_never_orphans_a_tool_message():
    """The cut must land on a `user` turn: separating a `tool` message from the
    assistant tool_calls that introduced it makes the server reject the request."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "read", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "1", "content": "file body"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "ok"},
    ]
    split = context.find_split(messages, keep_recent=2)
    assert messages[split]["role"] == "user"
    kept = messages[split:]
    # every tool message left behind still has its assistant tool_calls above it
    introduced = {
        c["id"] for m in kept for c in (m.get("tool_calls") or [])
    }
    assert all(m["tool_call_id"] in introduced for m in kept if m["role"] == "tool")


def test_split_returns_zero_when_there_is_no_legal_boundary():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "only turn"},
        {"role": "assistant", "content": "reply"},
    ]
    assert context.find_split(messages, keep_recent=2) == 0


def test_split_prefers_the_newest_legal_boundary():
    """Condense as little as the threshold allows — the newest user turn wins."""
    messages = [{"role": "system", "content": "s"}]
    for i in range(6):
        messages += [
            {"role": "user", "content": f"u{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]
    split = context.find_split(messages, keep_recent=4)
    assert messages[split]["content"] == "u4"


# --- the trigger ----------------------------------------------------------

def test_below_the_threshold_nothing_happens():
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    called = []
    n = context.maybe_compact(
        messages, 100, summarize=lambda m: called.append(m) or "s", cfg=_cfg(context_window=1000)
    )
    assert n == 0 and called == []


def test_server_reported_tokens_drive_the_trigger():
    """The prompt_tokens from the stream's usage trailer are trusted over the
    character estimate — they are the server's own count."""
    messages = [{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"}]
    messages += [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    # tiny text, but the server says the prompt was huge
    n = context.maybe_compact(
        messages, 9_000, summarize=lambda m: "summary",
        cfg=_cfg(context_window=10_000, compact_threshold=0.8, keep_recent_messages=2),
    )
    assert n == 1  # u0+a0 replaced by one summary


def test_context_window_zero_disables_compaction():
    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}] * 5
    assert context.maybe_compact(messages, 10**9, summarize=lambda m: "s",
                                 cfg=_cfg(context_window=0)) == 0


def test_estimate_is_used_before_any_usage_arrives():
    messages = [{"role": "system", "content": "s"}]
    for i in range(6):
        messages += [
            {"role": "user", "content": "x" * 3000},
            {"role": "assistant", "content": "y" * 3000},
        ]
    n = context.maybe_compact(messages, None, summarize=lambda m: "summary",
                              cfg=_cfg(context_window=4000, keep_recent_messages=4))
    assert n > 0


# --- the replacement ------------------------------------------------------

def test_summary_replaces_the_old_stretch_and_keeps_the_system_prompt():
    messages = [{"role": "system", "content": "sys"}]
    for i in range(6):
        messages += [
            {"role": "user", "content": f"u{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]
    before = len(messages)
    removed = context.maybe_compact(
        messages, 10**6, summarize=lambda m: "they agreed to use utf-8",
        cfg=_cfg(context_window=1000, keep_recent_messages=4),
    )
    assert removed > 0
    assert len(messages) == before - removed
    assert messages[0] == {"role": "system", "content": "sys"}  # system survives
    assert context.SUMMARY_PREFIX in messages[1]["content"]
    assert "utf-8" in messages[1]["content"]
    assert messages[-1]["content"] == "a5"  # the newest turns are untouched


def test_a_failed_summary_deletes_nothing():
    """An empty summary must not be allowed to erase the conversation."""
    messages = [{"role": "system", "content": "s"}]
    for i in range(6):
        messages += [{"role": "user", "content": f"u{i}"},
                     {"role": "assistant", "content": f"a{i}"}]
    before = list(messages)
    assert context.maybe_compact(messages, 10**6, summarize=lambda m: "   ",
                                 cfg=_cfg(context_window=1000)) == 0
    assert messages == before


def test_transcript_elides_a_huge_message_from_the_middle():
    body = "START" + "z" * 50_000 + "END"
    out = context.render_transcript([{"role": "user", "content": body}])
    assert "START" in out and "END" in out and "elided" in out
    assert len(out) < 5_000


# --- integration with the loop -------------------------------------------

def test_loop_compacts_and_still_returns_every_real_message(monkeypatch):
    """The in-flight list is condensed, but the run hands back the real messages —
    so the session file on disk stays complete."""
    monkeypatch.setattr(config, "load", lambda: _cfg(context_window=100, keep_recent_messages=2))
    sent: list[int] = []

    turns = iter([
        [ToolCall(id="1", name="read", arguments={"path": "a"}), Usage(9_999, 1, 10_000)],
        [TextDelta("final answer")],
    ])

    def stream(messages, tools=None):
        sent.append(len(messages))
        yield from next(turns)

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
    ]
    events: list = []
    produced = agent.run(
        messages, emit=events.append, stream=stream,
        registry={"read": _fake_read()}, summarize=lambda m: "condensed",
    )

    # the loop's own output is intact: the tool turn and the final answer
    assert [m["role"] for m in produced] == ["assistant", "tool", "assistant"]
    assert produced[-1]["content"] == "final answer"
    # the second request was sent with a condensed history, and the user was told
    assert sent[1] < sent[0] + 2
    assert any(isinstance(e, Notice) for e in events)
    assert any(context.SUMMARY_PREFIX in str(m.get("content")) for m in messages)


def test_subagent_transcript_is_complete_after_compaction(monkeypatch):
    """A child's session file must hold its real transcript, not the compacted copy."""
    monkeypatch.setattr(config, "load", lambda: _cfg(context_window=1, keep_recent_messages=1))
    turns = iter([
        [ToolCall(id="1", name="read", arguments={"path": "a"})],
        [TextDelta("child result")],
    ])
    res = subagent.run(
        "do the thing", emit=lambda e: None,
        stream=lambda m, tools=None: iter(next(turns)),
        registry={"read": _fake_read()}, summarize=lambda m: "condensed",
    )
    assert [m["role"] for m in res.messages] == [
        "system", "user", "assistant", "tool", "assistant"
    ]
    assert res.messages[1]["content"] == "do the thing"  # the real task, not a summary
    assert res.result == "child result"


def _fake_read():
    from ahacode.tools.base import Tool
    return Tool(name="read", description="", parameters={}, execute=lambda a: "contents")
