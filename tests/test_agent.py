"""Unit tests for the agent loop — offline, with an injected stream and a fake
tool registry (no network, no real filesystem)."""

import json

from ahacode import agent
from ahacode.events import TextDelta, ToolCall, ToolResult
from ahacode.tools.base import Tool


def registry():
    """A fresh fake registry: read runs freely, bash needs approval."""
    return {
        "read": Tool("read", "", {}, execute=lambda a: f"content:{a['path']}"),
        "bash": Tool("bash", "", {}, execute=lambda a: f"ran:{a['command']}",
                     requires_approval=True),
    }


def make_stream(turns):
    """Script one list of events per turn; each call to stream() replays the next."""
    it = iter(turns)

    def stream(messages, specs):
        return iter(next(it))

    return stream


def test_tool_then_final_answer():
    emitted = []
    turns = [
        [ToolCall(id="1", name="read", arguments={"path": "app.py"})],
        [TextDelta("done reading")],
    ]
    new = agent.run([{"role": "user", "content": "read it"}], emit=emitted.append,
                    stream=make_stream(turns), registry=registry())

    assert [type(e).__name__ for e in emitted] == ["ToolCall", "ToolResult", "TextDelta"]
    result = next(e for e in emitted if isinstance(e, ToolResult))
    assert result.output == "content:app.py" and not result.is_error

    assert [m["role"] for m in new] == ["assistant", "tool", "assistant"]
    call = new[0]["tool_calls"][0]
    assert call["function"]["name"] == "read"
    assert json.loads(call["function"]["arguments"]) == {"path": "app.py"}  # re-serialised
    assert new[1] == {"role": "tool", "tool_call_id": "1", "content": "content:app.py"}
    assert new[2]["content"] == "done reading"


def test_bash_denied_without_approval():
    emitted = []
    turns = [
        [ToolCall(id="1", name="bash", arguments={"command": "rm -rf /"})],
        [TextDelta("ok, skipped")],
    ]
    new = agent.run([{"role": "user", "content": "x"}], emit=emitted.append,
                    stream=make_stream(turns), registry=registry(), approve=lambda c: False)
    result = next(e for e in emitted if isinstance(e, ToolResult))
    assert result.is_error and "denied" in result.output
    assert new[1]["content"] == "denied by user"  # the command never ran


def test_bash_runs_when_approved():
    emitted = []
    turns = [[ToolCall(id="1", name="bash", arguments={"command": "ls"})], [TextDelta("done")]]
    agent.run([{"role": "user", "content": "x"}], emit=emitted.append,
              stream=make_stream(turns), registry=registry(), approve=lambda c: True)
    result = next(e for e in emitted if isinstance(e, ToolResult))
    assert result.output == "ran:ls" and not result.is_error


def test_unknown_tool_is_error():
    emitted = []
    turns = [[ToolCall(id="1", name="nope", arguments={})], [TextDelta("ok")]]
    agent.run([{"role": "user", "content": "x"}], emit=emitted.append,
              stream=make_stream(turns), registry=registry())
    result = next(e for e in emitted if isinstance(e, ToolResult))
    assert result.is_error and "unknown tool" in result.output


def test_max_turns_backstop():
    emitted = []

    def stream(messages, specs):  # always asks for a tool -> never terminates
        return iter([ToolCall(id="1", name="read", arguments={"path": "x"})])

    agent.run([{"role": "user", "content": "x"}], emit=emitted.append,
              stream=stream, registry=registry(), max_turns=3)
    assert sum(isinstance(e, ToolCall) for e in emitted) == 3  # capped
    assert any(isinstance(e, TextDelta) and "max tool turns" in e.text for e in emitted)


def test_plain_answer_no_tools():
    emitted = []
    turns = [[TextDelta("hi"), TextDelta(" there")]]
    new = agent.run([{"role": "user", "content": "x"}], emit=emitted.append,
                    stream=make_stream(turns), registry=registry())
    assert [m["role"] for m in new] == ["assistant"]
    assert new[0]["content"] == "hi there"
    assert "tool_calls" not in new[0]


def test_cancellation_stops_loop():
    emitted = []
    turns = [
        [ToolCall(id="1", name="read", arguments={"path": "x"})],
        [TextDelta("should not reach")],
    ]
    state = {"n": 0}

    def cancelled():
        state["n"] += 1
        return state["n"] > 1  # allow the turn to start, then cancel

    new = agent.run([{"role": "user", "content": "x"}], emit=emitted.append,
                    stream=make_stream(turns), registry=registry(), is_cancelled=cancelled)
    assert not any(isinstance(e, TextDelta) and "should not reach" in e.text for e in emitted)
    assert len(new) == 0  # bailed before recording anything


def test_dangerous_command_blocked_before_approval():
    from ahacode.tools.bash import _check_dangerous

    # Real denylist, but a harmless execute so a logic slip can never run `rm`.
    safe_bash = Tool("bash", "", {}, execute=lambda a: "SHOULD NOT RUN",
                     requires_approval=True, validate=_check_dangerous)
    approve_calls = []

    def approve(call):
        approve_calls.append(call)  # would approve — but must never be reached
        return True

    turns = [[ToolCall(id="1", name="bash", arguments={"command": "rm -rf /"})],
             [TextDelta("ok")]]
    new = agent.run([{"role": "user", "content": "x"}], emit=lambda e: None,
                    stream=make_stream(turns), registry={"bash": safe_bash}, approve=approve)

    assert approve_calls == []                       # never even prompted
    tool_msg = next(m for m in new if m["role"] == "tool")
    assert "blocked" in tool_msg["content"]          # and it did not run
