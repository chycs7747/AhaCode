"""The plan gate: plan_submit ends a planning turn and pauses the loop until the
user decides — run it, or keep revising."""

import pytest
from textual.widgets import Select

from ahacode import client, config, storage
from ahacode.app import AhaCodeApp
from ahacode.events import TextDelta, ToolCall
from ahacode.widgets.plan_gate import PlanGate
from ahacode.widgets.prompt_input import PromptInput
from ahacode.widgets.subagent_card import SubagentCard
from ahacode.widgets.todo_panel import TodoPanel
from ahacode.widgets.tool_result import ToolResultBlock


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(storage, "PLANS_DIR", tmp_path / "plans")
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(client, "list_models", lambda: ["qwen38"])
    monkeypatch.setattr(client, "complete", lambda messages: "")
    monkeypatch.setattr(AhaCodeApp, "generate_title", lambda self, *a, **k: None)
    client.reset()
    yield
    client.reset()


STEPS = ["Write solver.py with solve()", "Add tests/test_solver.py", "Run pytest and confirm 3 passed"]


def _submit(*steps, summary="Solve it", call_id="1"):
    return ToolCall(id=call_id, name="plan_submit",
                    arguments={"summary": summary, "steps": list(steps or STEPS)})


def _todo(*steps, status="pending"):
    return ToolCall(id="1", name="todo_write", arguments={
        "items": [{"content": s, "status": status} for s in steps]
    })


def _stream_turns(monkeypatch, turns):
    """Serve one prepared event list per stream_chat call; returns the call log."""
    it = iter(turns)
    calls = []

    def stream(messages, tools=None):
        calls.append([m["role"] for m in messages])
        return iter(next(it))

    monkeypatch.setattr(client, "stream_chat", stream)
    return calls


# Pilot.click's default offset is the widget's TOP-LEFT cell — on a bordered Button
# that is the border corner, and when the card sits flush against the top of the chat
# viewport the scroll container claims that exact cell, so the click misses. Aim one
# cell inside instead; a human clicks the middle of the button anyway.
_INSIDE = (2, 1)


async def _plan_mode(app, pilot):
    await app.workers.wait_for_complete()
    app.query_one("#mode-select", Select).value = "plan"
    await pilot.pause()


async def _ask(pilot, app, text):
    """Send a message and let the screen settle before the test clicks anything.

    Two pauses, not one: opening the gate mounts the card AND reveals the pinned plan
    panel, so the layout reflows twice. Clicking on the first frame's coordinates
    misses the button — which is a test-timing artefact, not something a human hits.
    """
    app.query_one("#prompt", PromptInput).text = text
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()
    await pilot.pause()


def _plan_file(app):
    return storage.PLANS_DIR / f"{app.session_path.stem}.md"


# --- opening ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_plan_submit_opens_the_gate_and_pauses(monkeypatch):
    """The model submits; the harness writes the file, fills the panel, and holds
    the loop — no second turn is requested until a button is pressed."""
    calls = _stream_turns(monkeypatch, [[_submit()], [TextDelta("never sent")]])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        gate = app.query_one(PlanGate)
        assert gate.steps == STEPS and gate.summary == "Solve it"
        assert app._plan_gate_pending is True
        assert len(calls) == 1                      # the loop stopped
        assert [m["role"] for m in app.session.messages] == ["user", "assistant", "tool"]
        assert [it["content"] for it in app.query_one(TodoPanel).items] == STEPS
        assert _plan_file(app).read_text(encoding="utf-8").startswith("# Solve it\n")
        assert gate.path.endswith(f"{app.session_path.stem}.md")


@pytest.mark.asyncio
async def test_a_rejected_submission_does_not_open_the_gate(monkeypatch):
    """A plan the harness refuses goes back to the model as an error result; the
    gate opens only for the corrected resubmission."""
    calls = _stream_turns(monkeypatch, [
        [_submit("Algorithm: subtree sums", call_id="1")],   # a topic, not a step
        [_submit(*STEPS, call_id="2")],
        [TextDelta("never sent")],
    ])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        assert len(app.query(PlanGate)) == 1
        assert app._plan_gate_pending is True
        assert len(calls) == 2
        first_result = app.session.messages[2]
        assert first_result["role"] == "tool" and "PlanRejected: step 1" in first_result["content"]
        # the refusal is visible to the user too, as an error card
        assert any("plan_submit" in b.title for b in app.query(ToolResultBlock))
        assert app.query_one(PlanGate).steps == STEPS


@pytest.mark.asyncio
async def test_act_mode_todo_write_never_gates(monkeypatch):
    """In act mode todo_write is a working checklist, not a plan awaiting approval:
    the loop runs straight through."""
    _stream_turns(monkeypatch, [[_todo("a", "b", "c")], [TextDelta("done")]])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _ask(pilot, app, "리팩터링 해줘")          # act is the default mode
        assert list(app.query(PlanGate)) == []
        assert app._plan_gate_pending is False
        assert app.session.messages[-1]["content"] == "done"


# --- the two ways out ------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_button_delegates_each_step_and_switches_to_act(monkeypatch):
    """▶ 실행 hands the plan to the structural runner and flips the bar to act."""
    _stream_turns(monkeypatch, [[_submit()]] + [[TextDelta(f"phase {i}")] for i in range(10)])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.auto_approve = True  # sub-agents act; skip the modal
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        assert app._plan_gate_pending is True

        await pilot.click("#plan-gate-run", offset=_INSIDE)
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app._plan_gate_pending is False
        assert app.mode == "act"
        assert len(app.query(SubagentCard)) == 3      # one fresh sub-agent per step
        assert app.session.messages[-1]["role"] == "assistant"
        assert app.query_one(PlanGate).has_class("plan-gate--settled")


@pytest.mark.asyncio
async def test_revise_button_settles_without_resuming(monkeypatch):
    """✎ 수정 keeps the user in plan mode: nothing is re-requested; the next message
    is a fresh turn that can revise the plan."""
    calls = _stream_turns(monkeypatch, [[_submit()], [_submit("Write b.py with g()", summary="v2")]])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        await pilot.click("#plan-gate-continue", offset=_INSIDE)
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app._plan_gate_pending is False
        assert len(calls) == 1                         # no resume request
        assert app.mode == "plan"
        assert app.query_one(PlanGate).has_class("plan-gate--settled")

        await _ask(pilot, app, "b.py 하나로 줄여")
        gates = list(app.query(PlanGate))
        assert len(gates) == 2 and gates[-1].steps == ["Write b.py with g()"]
        assert app._plan_gate_pending is True
        assert _plan_file(app).read_text(encoding="utf-8").startswith("# v2\n")
        assert [it["content"] for it in app.query_one(TodoPanel).items] == ["Write b.py with g()"]


@pytest.mark.asyncio
async def test_typing_instead_of_choosing_dismisses_the_gate(monkeypatch):
    """A new instruction answers the gate: the plan is dropped, not executed."""
    _stream_turns(monkeypatch, [[_submit()], [TextDelta("ok, different thing")]])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        assert app._plan_gate_pending is True

        await _ask(pilot, app, "아니 그냥 이거 알려줘")
        assert app._plan_gate_pending is False
        assert list(app.query(SubagentCard)) == []
        assert list(app.query(PlanGate))[0].has_class("plan-gate--settled")
        assert app.session.messages[-1]["content"] == "ok, different thing"


@pytest.mark.asyncio
async def test_new_session_forgets_the_gate(monkeypatch):
    _stream_turns(monkeypatch, [[_submit()]])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        assert app._plan_gate_pending is True

        app.query_one("#prompt", PromptInput).text = "/new"
        await pilot.press("enter")
        await pilot.pause()
        assert app._plan_gate_pending is False
        assert app.query_one(TodoPanel).items == []


# --- ownership -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_subagents_plan_never_pauses_its_parent(monkeypatch):
    """A sub-agent has no user to ask, and its parent is blocked waiting for it —
    so a child's checklist must not open the gate or touch the parent's panel."""
    _stream_turns(monkeypatch, [
        [ToolCall(id="t1", name="task", arguments={"description": "sub", "prompt": "do it"})],
        [_todo("a", "b", "c")],          # the CHILD lays out a 3-step list
        [TextDelta("child done")],
        [TextDelta("parent done")],
    ])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.auto_approve = True
        await _ask(pilot, app, "위임해줘")
        assert list(app.query(PlanGate)) == []
        assert app._plan_gate_pending is False
        assert len(app.query(SubagentCard)) == 1
        assert app.query_one(TodoPanel).items == []
        assert app.session.messages[-1]["content"] == "parent done"


@pytest.mark.asyncio
async def test_a_reloaded_session_shows_the_submitted_plan_in_the_panel(monkeypatch):
    """History replay treats plan_submit like the live turn did: its steps go to the
    pinned panel, and its result stays as the card that names the file."""
    _stream_turns(monkeypatch, [[_submit()]])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        await app._render_history()
        await pilot.pause()
        assert [it["content"] for it in app.query_one(TodoPanel).items] == STEPS
        cards = [b for b in app.query(ToolResultBlock) if "plan_submit" in b.title]
        assert len(cards) == 1
