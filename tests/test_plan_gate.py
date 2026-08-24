"""The plan gate: a fresh multi-step plan pauses the loop until the user decides."""

from dataclasses import replace

import pytest
from textual.widgets import Button, Select

from ahacode import client, config, storage
from ahacode.app import AhaCodeApp
from ahacode.events import TextDelta, ToolCall
from ahacode.widgets.chatbox import Chatbox
from ahacode.widgets.plan_gate import PlanGate
from ahacode.widgets.prompt_input import PromptInput
from ahacode.widgets.subagent_card import SubagentCard
from ahacode.widgets.todo_panel import TodoPanel


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(client, "list_models", lambda: ["qwen38"])
    monkeypatch.setattr(client, "complete", lambda messages: "")
    monkeypatch.setattr(AhaCodeApp, "generate_title", lambda self, *a, **k: None)
    client.reset()
    yield
    client.reset()


def _plan(*steps, status="pending"):
    return ToolCall(id="1", name="todo_write", arguments={
        "items": [{"content": s, "status": status} for s in steps]
    })


def _stream_turns(monkeypatch, turns):
    """Serve one prepared event list per stream_chat call."""
    it = iter(turns)
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter(next(it)))


async def _ask(pilot, app, text):
    app.query_one("#prompt", PromptInput).text = text
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


# --- the gate opens (and holds) -------------------------------------------

@pytest.mark.asyncio
async def test_three_step_plan_pauses_before_anything_runs(monkeypatch):
    """The model plans, then would immediately act. The gate stops it in between:
    the second turn is never requested until a button is pressed."""
    calls = []

    def stream(messages, tools=None):
        calls.append(len(messages))
        if len(calls) == 1:
            yield _plan("design", "implement", "verify")
        else:
            yield TextDelta("ran everything")

    monkeypatch.setattr(client, "stream_chat", stream)

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _ask(pilot, app, "리팩터링 해줘")
        assert len(app.query(PlanGate)) == 1
        assert app.query_one(PlanGate).steps == ["design", "implement", "verify"]
        assert len(calls) == 1          # the loop stopped; no second turn was sent
        assert app._plan_gate_pending is True
        # the planning turn itself is complete and persisted
        assert [m["role"] for m in app.session.messages] == ["user", "assistant", "tool"]


@pytest.mark.asyncio
async def test_two_step_plan_does_not_pause(monkeypatch):
    """Below the threshold the loop runs straight through — today's behaviour."""
    _stream_turns(monkeypatch, [[_plan("design", "implement")], [TextDelta("done")]])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _ask(pilot, app, "작은 수정")
        assert list(app.query(PlanGate)) == []
        assert app._plan_gate_pending is False
        assert app.session.messages[-1]["content"] == "done"


@pytest.mark.asyncio
async def test_status_update_is_not_a_new_plan(monkeypatch):
    """todo_write is called repeatedly to update status. A list that is already in
    progress is not a plan awaiting approval, so the gate must ignore it."""
    _stream_turns(monkeypatch, [
        [_plan("a", "b", "c", status="in_progress")],
        [TextDelta("carried on")],
    ])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _ask(pilot, app, "계속해")
        assert list(app.query(PlanGate)) == []


@pytest.mark.asyncio
async def test_gate_off_when_threshold_is_zero(monkeypatch):
    monkeypatch.setattr(
        config, "load", lambda *a, **k: replace(config.DEFAULTS, plan_gate_min_steps=0)
    )
    _stream_turns(monkeypatch, [[_plan("a", "b", "c")], [TextDelta("done")]])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _ask(pilot, app, "해줘")
        assert list(app.query(PlanGate)) == []


@pytest.mark.asyncio
async def test_plan_mode_does_not_gate(monkeypatch):
    """Nothing can execute in plan mode, so there is nothing to hold."""
    _stream_turns(monkeypatch, [[_plan("a", "b", "c")], [TextDelta("here is the plan")]])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        app.query_one("#mode-select", Select).value = "plan"
        await pilot.pause()
        await _ask(pilot, app, "계획 세워줘")
        assert list(app.query(PlanGate)) == []


# --- the two ways out ------------------------------------------------------

@pytest.mark.asyncio
async def test_run_button_delegates_each_step_and_switches_to_act(monkeypatch):
    """▶ 실행 hands the plan to the structural runner — and flips the bar to act,
    because that is what the sub-agents were always doing."""
    turns = iter([[_plan("design", "implement", "verify")]] + [
        [TextDelta(f"phase {i}")] for i in range(10)
    ])
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter(next(turns)))

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.auto_approve = True  # sub-agents act; skip the modal
        await _ask(pilot, app, "리팩터링 해줘")
        assert app._plan_gate_pending is True

        await pilot.click("#plan-gate-run")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app._plan_gate_pending is False
        assert app.mode == "act"
        assert len(app.query(SubagentCard)) == 3      # one fresh sub-agent per step
        assert app.session.messages[-1]["role"] == "assistant"
        assert list(app.query(PlanGate))[0].has_class("plan-gate--settled")


@pytest.mark.asyncio
async def test_continue_button_resumes_the_same_loop(monkeypatch):
    """계속 re-enters the agent loop with the history the pause left behind — the
    model carries on in one session, with no sub-agents."""
    seen = []

    def stream(messages, tools=None):
        seen.append([m["role"] for m in messages])
        if len(seen) == 1:
            yield _plan("design", "implement", "verify")
        else:
            yield TextDelta("carried on in one go")

    monkeypatch.setattr(client, "stream_chat", stream)

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _ask(pilot, app, "리팩터링 해줘")
        await pilot.click("#plan-gate-continue")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app._plan_gate_pending is False
        assert list(app.query(SubagentCard)) == []     # no delegation on this path
        # the resumed request continued from the paused history (ends with the
        # todo_write result), rather than starting over
        assert seen[1][-1] == "tool"
        assert app.session.messages[-1]["content"] == "carried on in one go"


@pytest.mark.asyncio
async def test_typing_instead_of_choosing_dismisses_the_gate(monkeypatch):
    """A new instruction answers the gate: the plan is dropped, not executed."""
    _stream_turns(monkeypatch, [
        [_plan("a", "b", "c")],
        [TextDelta("ok, different thing")],
    ])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _ask(pilot, app, "리팩터링 해줘")
        assert app._plan_gate_pending is True

        await _ask(pilot, app, "아니 그냥 이거 알려줘")
        assert app._plan_gate_pending is False
        assert list(app.query(SubagentCard)) == []
        assert list(app.query(PlanGate))[0].has_class("plan-gate--settled")


@pytest.mark.asyncio
async def test_same_plan_is_not_asked_about_twice(monkeypatch):
    """After 계속, re-sending the identical list must not re-open the gate."""
    plan = _plan("a", "b", "c")
    _stream_turns(monkeypatch, [[plan], [plan], [TextDelta("done")]])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _ask(pilot, app, "해줘")
        await pilot.click("#plan-gate-continue")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(app.query(PlanGate)) == 1           # still just the settled one
        assert app._plan_gate_pending is False


@pytest.mark.asyncio
async def test_new_session_forgets_the_gate(monkeypatch):
    _stream_turns(monkeypatch, [[_plan("a", "b", "c")]])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _ask(pilot, app, "해줘")
        assert app._plan_gate_pending is True

        app.query_one("#prompt", PromptInput).text = "/new"
        await pilot.press("enter")
        await pilot.pause()
        assert app._plan_gate_pending is False
        assert app._gated_plan is None
        assert app.query_one(TodoPanel).items == []


@pytest.mark.asyncio
async def test_a_subagents_plan_never_pauses_its_parent(monkeypatch):
    """A sub-agent has no user to ask, and its parent is blocked waiting for it —
    so a child's todo_write must not open the gate."""
    turns = iter([
        # the parent delegates once...
        [ToolCall(id="t1", name="task",
                  arguments={"description": "sub", "prompt": "do it"})],
        [_plan("a", "b", "c")],          # ...and the CHILD lays out a 3-step plan
        [TextDelta("child done")],
        [TextDelta("parent done")],
    ])
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter(next(turns)))

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.auto_approve = True
        await _ask(pilot, app, "위임해줘")
        assert list(app.query(PlanGate)) == []
        assert app._plan_gate_pending is False
        assert len(app.query(SubagentCard)) == 1
        assert app.session.messages[-1]["content"] == "parent done"
