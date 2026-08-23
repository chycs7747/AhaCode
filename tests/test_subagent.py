"""Step 1 of sub-agents (roadmap ⑤): the pure core — the ctx seam, the sub-agent
runner, and the task tool — all exercised with a scripted fake stream, no gateway
or UI. Mirrors how agent.py itself is tested: inject the stream, assert on events
and the resulting transcript.
"""

from ahacode import agent, subagent, tools
from ahacode.events import TextDelta, ToolCall, ToolResult
from ahacode.tools.base import Tool


def scripted_stream(*turns):
    """A fake StreamFn that plays one scripted list of events per turn() call."""
    it = iter(turns)

    def _stream(messages, specs):
        yield from next(it)

    return _stream


ECHO = Tool(
    name="echo",
    description="echo the argument back",
    parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
    execute=lambda a: f"echoed {a['x']}",
)


def test_subagent_returns_final_answer():
    seen = []
    res = subagent.run(
        "do the thing",
        emit=seen.append,
        stream=scripted_stream([TextDelta("Hello "), TextDelta("world")]),
        registry={},
    )
    assert res.result == "Hello world"
    # transcript framing: system prompt, then the delegated task, then the answer.
    assert res.messages[0]["role"] == "system"
    assert res.messages[1] == {"role": "user", "content": "do the thing"}
    assert res.messages[-1]["role"] == "assistant"


def test_subagent_runs_a_tool_then_answers():
    seen = []
    res = subagent.run(
        "use echo",
        emit=seen.append,
        stream=scripted_stream(
            [ToolCall(id="c1", name="echo", arguments={"x": "hi"})],  # turn 1: call a tool
            [TextDelta("done")],                                      # turn 2: final answer
        ),
        registry={"echo": ECHO},
    )
    assert res.result == "done"
    assert any(isinstance(e, ToolResult) and e.name == "echo" and not e.is_error for e in seen)
    # the tool's output was fed back into the child transcript
    assert any(m.get("role") == "tool" and "echoed hi" in m.get("content", "") for m in res.messages)


def test_task_tool_delegates_to_ctx():
    class FakeCtx:
        def __init__(self):
            self.calls = []

        def run_subagent(self, prompt, description):
            self.calls.append((prompt, description))
            return "child result"

    ctx = FakeCtx()
    out = tools.TASK.execute({"prompt": "analyze x", "description": "analysis"}, ctx)
    assert out == "child result"
    assert ctx.calls == [("analyze x", "analysis")]


def test_task_tool_without_ctx_is_graceful():
    # No spawning context -> a soft error string, not a crash.
    assert "not available" in tools.TASK.execute({"prompt": "x"}, None)


def test_agent_forwards_ctx_to_wants_ctx_tool():
    received = {}

    def _exec(args, ctx):
        received["ctx"] = ctx
        return "ok"

    ctxtool = Tool(
        name="ctxtool",
        description="needs ctx",
        parameters={"type": "object", "properties": {}},
        execute=_exec,
        wants_ctx=True,
    )
    sentinel = object()
    agent.run(
        [{"role": "user", "content": "go"}],
        emit=lambda e: None,
        stream=scripted_stream(
            [ToolCall(id="c1", name="ctxtool", arguments={})],
            [TextDelta("done")],
        ),
        registry={"ctxtool": ctxtool},
        ctx=sentinel,
    )
    assert received["ctx"] is sentinel


def test_registry_for_depth_gates_task():
    r0 = tools.registry_for(depth=0, subagent_depth=1)
    assert "task" in r0 and "read" in r0                                # depth 0 < 1 -> can spawn
    assert "task" not in tools.registry_for(depth=1, subagent_depth=1)  # at limit -> cannot recurse
    assert "task" in tools.registry_for(depth=1, subagent_depth=2)      # deeper limit -> still can
    assert "task" not in tools.registry_for(depth=2, subagent_depth=2)
    # base tools are always present regardless of depth
    assert "read" in tools.registry_for(depth=9, subagent_depth=1)


def test_final_text_helper():
    assert subagent._final_text([{"role": "user", "content": "x"}]) == "(sub-agent produced no result)"
    assert subagent._final_text(
        [{"role": "assistant", "content": "A"}, {"role": "assistant", "content": "B"}]
    ) == "B"


def test_parallel_tool_calls_overlap():
    """A turn of parallelizable calls runs them concurrently, and their results are
    appended in call order (so they line up with the assistant's tool_calls)."""
    import threading
    import time as _time

    counter, peak, lock = [0], [0], threading.Lock()

    def slow(args):
        with lock:
            counter[0] += 1
            peak[0] = max(peak[0], counter[0])
        _time.sleep(0.1)
        with lock:
            counter[0] -= 1
        return "ok"

    tool = Tool(name="slow", description="", parameters={"type": "object", "properties": {}},
                execute=slow, parallelizable=True)
    msgs = agent.run(
        [{"role": "user", "content": "go"}],
        emit=lambda e: None,
        stream=scripted_stream(
            [ToolCall(id="a", name="slow", arguments={}),
             ToolCall(id="b", name="slow", arguments={}),
             ToolCall(id="c", name="slow", arguments={})],
            [TextDelta("done")],
        ),
        registry={"slow": tool},
    )
    assert peak[0] >= 2  # the three calls overlapped
    assert [m["tool_call_id"] for m in msgs if m.get("role") == "tool"] == ["a", "b", "c"]


def test_non_parallelizable_calls_stay_sequential():
    """When any runnable tool is not parallelizable, the turn runs sequentially."""
    import threading
    import time as _time

    counter, peak, lock = [0], [0], threading.Lock()

    def slow(args):
        with lock:
            counter[0] += 1
            peak[0] = max(peak[0], counter[0])
        _time.sleep(0.1)
        with lock:
            counter[0] -= 1
        return "ok"

    tool = Tool(name="s", description="", parameters={"type": "object", "properties": {}},
                execute=slow, parallelizable=False)
    agent.run(
        [{"role": "user", "content": "go"}],
        emit=lambda e: None,
        stream=scripted_stream(
            [ToolCall(id="a", name="s", arguments={}), ToolCall(id="b", name="s", arguments={})],
            [TextDelta("done")],
        ),
        registry={"s": tool},
    )
    assert peak[0] == 1  # never overlapped
