"""Context-window management: when to condense, where it is safe to cut, and that
the transcript on disk stays complete when the in-flight copy is compacted."""

import re

import pytest

from ahacode import agent, config, context, subagent
from ahacode.events import Notice, Phase, TextDelta, ToolCall, Usage

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
    done = context.maybe_compact(
        messages, 100, summarize=lambda m: called.append(m) or "s", cfg=_cfg(context_window=1000)
    )
    assert not done and called == []


def test_server_reported_tokens_drive_the_trigger():
    """The prompt_tokens from the stream's usage trailer are trusted over the
    character estimate — they are the server's own count."""
    messages = [{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"}]
    messages += [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    # tiny text, but the server says the prompt was huge
    done = context.maybe_compact(
        messages, 9_000, summarize=lambda m: "summary",
        cfg=_cfg(context_window=10_000, compact_threshold=0.8, keep_recent_messages=2),
    )
    assert done.summarized == 1  # u0+a0 replaced by one summary


def test_context_window_zero_disables_compaction():
    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}] * 5
    assert not context.maybe_compact(messages, 10**9, summarize=lambda m: "s",
                                     cfg=_cfg(context_window=0))


def test_estimate_is_used_before_any_usage_arrives():
    messages = [{"role": "system", "content": "s"}]
    for i in range(6):
        messages += [
            {"role": "user", "content": "x" * 3000},
            {"role": "assistant", "content": "y" * 3000},
        ]
    done = context.maybe_compact(messages, None, summarize=lambda m: "summary",
                                 cfg=_cfg(context_window=4000, keep_recent_messages=4))
    assert done.summarized > 0


# --- the replacement ------------------------------------------------------

def test_summary_replaces_the_old_stretch_and_keeps_the_system_prompt():
    messages = [{"role": "system", "content": "sys"}]
    for i in range(6):
        messages += [
            {"role": "user", "content": f"u{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]
    before = len(messages)
    done = context.maybe_compact(
        messages, 10**6, summarize=lambda m: "they agreed to use utf-8",
        cfg=_cfg(context_window=1000, keep_recent_messages=4),
    )
    assert done.summarized > 0
    assert len(messages) == before - done.summarized
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
    assert not context.maybe_compact(messages, 10**6, summarize=lambda m: "   ",
                                     cfg=_cfg(context_window=1000))
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


# --- compaction has to say it is running, not only that it ran --------------

def _phases(events):
    return [(e.name, e.done) for e in events if isinstance(e, Phase)]


def test_compaction_announces_itself_while_it_blocks(monkeypatch):
    """The Notice arrives when compaction is OVER. Between the user's message and
    that Notice sits one synchronous model call over the whole history — minutes,
    with nothing on screen changing. That gap is what made a working app and a
    deadlocked one look identical, so the slow half brackets itself."""
    monkeypatch.setattr(config, "load", lambda: _cfg(context_window=1, keep_recent_messages=1))
    events: list = []

    def summarize(older):
        # Mid-call: the UI must already know, because this is the part that blocks.
        assert _phases(events) == [(agent.COMPACTING, False)]
        return "condensed"

    agent.run(
        [{"role": "system", "content": "s"},
         {"role": "user", "content": "old"},
         {"role": "assistant", "content": "a"},
         {"role": "user", "content": "new"}],
        emit=events.append,
        stream=lambda m, tools=None: iter([TextDelta("done")]),
        summarize=summarize,
    )
    assert _phases(events) == [(agent.COMPACTING, False), (agent.COMPACTING, True)]
    # and it still closes with the after-the-fact Notice
    assert any(isinstance(e, Notice) for e in events)


def test_a_turn_that_does_not_compact_announces_nothing(monkeypatch):
    """maybe_compact runs before EVERY turn and almost always returns under the
    threshold. Bracketing the whole of it would flash an indicator on each one."""
    monkeypatch.setattr(config, "load", lambda: _cfg(context_window=100_000))
    events: list = []
    agent.run(
        [{"role": "user", "content": "hi"}],
        emit=events.append,
        stream=lambda m, tools=None: iter([TextDelta("hello")]),
        summarize=lambda m: "never called",
    )
    assert _phases(events) == []


def test_a_pruned_turn_announces_nothing(monkeypatch):
    """Pruning is pure string work — it returns before the slow half is reached,
    so there is no wait to explain."""
    monkeypatch.setattr(config, "load", lambda: _cfg(context_window=1, keep_recent_messages=1))
    events: list = []
    agent.run(
        _subagent_history(20_000),
        emit=events.append,
        stream=lambda m, tools=None: iter([TextDelta("done")]),
        summarize=lambda m: "never called",
    )
    assert _phases(events) == []


def test_the_phase_closes_even_when_summarizing_fails(monkeypatch):
    """A crashed summarizer must not leave the indicator up forever — that would
    be a permanent 'still working' on an app that stopped."""
    monkeypatch.setattr(config, "load", lambda: _cfg(context_window=1, keep_recent_messages=1))
    events: list = []

    def boom(older):
        raise RuntimeError("gateway said no")

    with pytest.raises(RuntimeError):
        agent.run(
            [{"role": "system", "content": "s"},
             {"role": "user", "content": "old"},
             {"role": "assistant", "content": "a"},
             {"role": "user", "content": "new"}],
            emit=events.append,
            stream=lambda m, tools=None: iter([TextDelta("x")]),
            summarize=boom,
        )
    assert _phases(events) == [(agent.COMPACTING, False), (agent.COMPACTING, True)]


# --- what the summarizer is actually shown ---------------------------------

def _long_session(turns: int = 40, rounds: int = 8) -> list[dict]:
    msgs = [{"role": "system", "content": "sys"}]
    for t in range(turns):
        msgs.append({"role": "user", "content": f"[TURN {t}] next thing"})
        for r in range(rounds):
            msgs.append({"role": "assistant", "content": f"[TURN {t} R{r}] working"})
            msgs.append({"role": "tool", "tool_call_id": f"{t}-{r}",
                         "content": f"[TURN {t} R{r}] " + "output " * 250})
    return msgs


def test_the_budget_follows_the_window():
    """A flat cap was survivable at 32K and ruinous above it — the stretch grows
    with the window but the budget did not."""
    small = context.transcript_budget(_cfg(context_window=32768))
    large = context.transcript_budget(_cfg(context_window=262144))
    assert small >= context._MIN_TRANSCRIPT_CHARS
    assert large > small * 4
    assert large <= context._MAX_TRANSCRIPT_CHARS


def test_every_turn_reaches_the_summarizer():
    """The real defect: filling from the oldest end and stopping at the cap meant
    a 663-message stretch was summarized from its first three turns and nothing
    else. Coverage of the whole stretch is the thing being bought here — detail
    per message is what pays for it."""
    msgs = _long_session()
    split = context.find_split(msgs, keep_recent=6)
    older = msgs[1:split]
    rendered = context.render_transcript(
        older, context.transcript_budget(_cfg(context_window=262144)))
    seen = {int(t) for t in re.findall(r"\[TURN (\d+)", rendered)}
    # Derived from the stretch, not hardcoded: find_split keeps the newest turns
    # verbatim, so which ones are condensed is its business, not this test's.
    present = {int(t) for m in older
               for t in re.findall(r"\[TURN (\d+)", str(m.get("content") or ""))}
    assert seen == present, f"only turns {sorted(seen)} of {len(present)} survived"


def test_a_budget_too_small_for_the_stretch_keeps_both_ends():
    """When even the per-message floor will not fit, the middle goes — not the
    end. Where the session STANDS lives at the end of it."""
    msgs = _long_session()
    split = context.find_split(msgs, keep_recent=6)
    older = msgs[1:split]
    rendered = context.render_transcript(older, budget=8_000)
    turns = [int(t) for t in re.findall(r"\[TURN (\d+)", rendered)]
    present = sorted({int(t) for m in older
                      for t in re.findall(r"\[TURN (\d+)", str(m.get("content") or ""))})
    assert turns, "nothing survived at all"
    assert min(turns) == present[0] and max(turns) == present[-1]
    assert "omitted from the middle" in rendered


def test_an_empty_stretch_renders_nothing():
    assert context.render_transcript([]) == ""


# --- prune: the cheap pass, and the only one a sub-agent can use ------------

def _subagent_history(tool_chars: int, turns: int = 8):
    """What a sub-agent's history actually looks like: exactly ONE user message
    (its task), then assistant/tool pairs."""
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "THE DELEGATED TASK"}]
    for i in range(turns):
        msgs += [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": str(i), "type": "function",
                 "function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": str(i), "content": "x" * tool_chars},
        ]
    return msgs


def test_a_subagent_cannot_be_summarized_at_all():
    """Documents WHY prune exists: the only legal cut is a user message, and a
    sub-agent has exactly one — its task, which must survive. So find_split has
    nowhere to go and summarizing can never help there."""
    msgs = _subagent_history(100)
    assert context.find_split(msgs, keep_recent=2) == 0


def test_prune_frees_a_subagent_that_summarizing_could_not():
    msgs = _subagent_history(20_000)
    done = context.maybe_compact(msgs, 10**6, summarize=lambda m: "never called",
                                 cfg=_cfg(context_window=1000))
    assert done.pruned_chars > 0
    assert done.summarized == 0            # the expensive path was not reached
    assert msgs[1]["content"] == "THE DELEGATED TASK"   # the task always survives


def test_prune_never_breaks_the_assistant_tool_pairing():
    """Only the CONTENT goes; the message stays. That is what makes it safe
    without any boundary — the server's pairing check sees no change."""
    msgs = _subagent_history(20_000)
    before = [m["role"] for m in msgs]
    ids_before = [m.get("tool_call_id") for m in msgs if m["role"] == "tool"]
    context.prune_tool_output(msgs, _cfg())
    assert [m["role"] for m in msgs] == before
    assert [m.get("tool_call_id") for m in msgs if m["role"] == "tool"] == ids_before


def test_prune_protects_the_newest_tool_output():
    msgs = _subagent_history(20_000)
    context.prune_tool_output(msgs, _cfg())
    kept = [m["content"] for m in msgs if m["role"] == "tool"
            and m["content"] != context.PRUNED_STUB]
    assert kept, "the recent results must survive"
    # everything kept is at the END of the history
    tools = [m for m in msgs if m["role"] == "tool"]
    stubs = [i for i, m in enumerate(tools) if m["content"] == context.PRUNED_STUB]
    fresh = [i for i, m in enumerate(tools) if m["content"] != context.PRUNED_STUB]
    assert max(stubs) < min(fresh)


def test_prune_does_nothing_below_the_floor():
    """Blanking a few hundred chars is churn, not relief."""
    msgs = _subagent_history(50)
    assert context.prune_tool_output(msgs, _cfg()) == 0
    assert all(m["content"] != context.PRUNED_STUB for m in msgs if m["role"] == "tool")


def test_prune_is_idempotent():
    msgs = _subagent_history(20_000)
    first = context.prune_tool_output(msgs, _cfg())
    assert first > 0
    assert context.prune_tool_output(msgs, _cfg()) == 0   # nothing left to take


def test_prune_runs_before_summarizing_when_both_are_possible():
    """Cheapest first: a history with a user boundary AND big tool output should
    spend no model call while pruning still frees enough."""
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u0"}]
    for i in range(6):
        msgs += [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": str(i), "type": "function",
                 "function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": str(i), "content": "x" * 20_000},
            {"role": "user", "content": f"u{i+1}"},
        ]
    called = []
    done = context.maybe_compact(msgs, 10**6,
                                 summarize=lambda m: called.append(1) or "s",
                                 cfg=_cfg(context_window=1000))
    assert done.pruned_chars > 0
    assert called == []   # no request was made
