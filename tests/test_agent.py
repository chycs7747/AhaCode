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


def test_a_batch_of_pure_reads_runs_in_parallel():
    """Several independent reads in one assistant message execute concurrently. Each
    tool blocks on a shared barrier that only releases once BOTH are inside it, so
    the turn can finish only if the two ran at the same time — a serial path would
    deadlock the first call until it times out."""
    import threading

    barrier = threading.Barrier(2, timeout=3)
    threads: dict[str, str] = {}

    def blocking(name):
        def run(a):
            barrier.wait()  # proceeds only when the second parallel call arrives
            threads[name] = threading.current_thread().name
            return f"{name}:ok"
        return run

    reg = {
        "read": Tool("read", "", {}, execute=blocking("read"), parallelizable=True),
        "grep": Tool("grep", "", {}, execute=blocking("grep"), parallelizable=True),
    }
    turns = [
        [ToolCall(id="1", name="read", arguments={"path": "a"}),
         ToolCall(id="2", name="grep", arguments={"pattern": "x"})],
        [TextDelta("done")],
    ]
    emitted = []
    agent.run([{"role": "user", "content": "x"}], emit=emitted.append,
              stream=make_stream(turns), registry=reg)

    results = [e for e in emitted if isinstance(e, ToolResult)]
    assert [r.output for r in results] == ["read:ok", "grep:ok"]  # both released, in call order
    assert not any(r.is_error for r in results)
    # ran off the calling thread, on two distinct workers
    assert threads["read"] != threads["grep"]


def test_a_batch_mixing_a_writer_falls_back_to_serial():
    """The all()-parallelizable guard: one non-parallelizable call in the batch
    forces the WHOLE batch onto the serial path, so nothing races a side effect.
    Proven by thread identity — serial execution stays on the calling thread."""
    import threading

    threads: dict[str, str] = {}

    def record(name):
        def run(a):
            threads[name] = threading.current_thread().name
            return f"{name}:ok"
        return run

    reg = {
        "read": Tool("read", "", {}, execute=record("read"), parallelizable=True),
        "write": Tool("write", "", {}, execute=record("write")),  # parallelizable=False
    }
    turns = [
        [ToolCall(id="1", name="read", arguments={"path": "a"}),
         ToolCall(id="2", name="write", arguments={"path": "b"})],
        [TextDelta("done")],
    ]
    agent.run([{"role": "user", "content": "x"}], emit=lambda e: None,
              stream=make_stream(turns), registry=reg)

    main = threading.current_thread().name
    assert threads["read"] == main and threads["write"] == main  # both serial, no worker pool


def test_a_malformed_tool_call_is_reported_back_and_the_loop_continues():
    """A tool call whose arguments did not parse must not end the run. The loop
    feeds an error result back (so the model can resend) and keeps going, instead
    of reading the empty turn as a final answer and stopping mid-task."""
    emitted = []
    turns = [
        [ToolCall(id="1", name="read", arguments={},
                  parse_error="arguments were not valid JSON")],
        [TextDelta("resent and done")],
    ]
    new = agent.run([{"role": "user", "content": "x"}], emit=emitted.append,
                    stream=make_stream(turns), registry=registry())
    result = next(e for e in emitted if isinstance(e, ToolResult))
    assert result.is_error and "valid JSON" in result.output  # the failure is fed back
    # the run did NOT stop on the bad call: a second turn ran and produced the answer
    assert [m["role"] for m in new] == ["assistant", "tool", "assistant"]
    assert new[-1]["content"] == "resent and done"


def test_unknown_tool_is_error():
    emitted = []
    turns = [[ToolCall(id="1", name="nope", arguments={})], [TextDelta("ok")]]
    agent.run([{"role": "user", "content": "x"}], emit=emitted.append,
              stream=make_stream(turns), registry=registry())
    result = next(e for e in emitted if isinstance(e, ToolResult))
    assert result.is_error and "unknown tool" in result.output


def _unknown_output(name, reg):
    emitted = []
    turns = [[ToolCall(id="1", name=name, arguments={})], [TextDelta("ok")]]
    agent.run([{"role": "user", "content": "x"}], emit=emitted.append,
              stream=make_stream(turns), registry=reg)
    return next(e for e in emitted if isinstance(e, ToolResult)).output


def test_unknown_tool_names_what_is_available():
    """A bare "unknown tool: X" is a dead end — a model with a code-interpreter prior
    answers it by trying run_python, then python, then code_interpreter. The result
    carries the actual registry and points at the schemas already in the request."""
    reg = registry()
    out = _unknown_output("run_python", reg)
    for name in reg:
        assert name in out                    # every real option is named
    assert "came with this request" in out    # and where to read about them


def test_an_invented_name_is_answered_with_the_real_one():
    """The whole point: one turn, not a run of them. run_python must come back
    pointing at bash, not just at a list the model has to re-derive."""
    reg = registry()                          # this fake holds read + bash
    assert "Use `bash` for that." in _unknown_output("run_python", reg)
    assert "Use `read` for that." in _unknown_output("read_file", reg)


def test_an_invented_name_whose_tool_is_absent_says_so():
    """Plan mode has no bash. Naming it without saying it is missing would send the
    model straight into another dead end."""
    reg = {k: v for k, v in registry().items() if k != "bash"}
    out = _unknown_output("run_python", reg)
    assert "`bash`" in out and "NOT available" in out


def test_a_misspelling_gets_a_near_match():
    """A typo is not an invented name, so the alias table cannot help — spelling
    distance can."""
    assert "Did you mean `read`?" in _unknown_output("reads", registry())


def test_max_turns_backstop_forces_tool_free_summary():
    """Hitting the turn cap forces ONE final tool-free turn: tools are withheld
    (specs is None) and the model is made to produce a text wrap-up, instead of a
    bare truncation."""
    emitted = []
    specs_seen = []

    def stream(messages, specs):
        specs_seen.append(specs)
        if specs is None:                     # the forced wrap-up turn — tools withheld
            return iter([TextDelta("summary: did X; Y remains; next do Z")])
        return iter([ToolCall(id="1", name="read", arguments={"path": "x"})])  # never terminates

    msgs = agent.run([{"role": "user", "content": "x"}], emit=emitted.append,
                     stream=stream, registry=registry(), max_turns=3)
    assert sum(isinstance(e, ToolCall) for e in emitted) == 3   # capped at max_turns
    assert specs_seen[-1] is None                                # wrap-up sent no tools
    # a wrap-up user prompt was injected, and the final message is the text summary
    assert msgs[-2]["role"] == "user"
    assert msgs[-1] == {"role": "assistant", "content": "summary: did X; Y remains; next do Z"}


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
