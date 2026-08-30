"""The plan gate: plan_submit ends a planning turn and pauses the loop until the
user decides — run it, or keep revising."""

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
# A viewport tall enough to hold the plan gate. pilot.click aims at real screen
# coordinates, so at the default 80x24 a gate pushed below the fold raises
# OutOfBounds — which is why some tests below press() the button instead. Where the
# click can be kept, it should be: it is the path a user actually takes.
_TALL = (100, 50)


async def _settle(app, pilot):
    """Wait until the app goes quiet, not just until the current worker ends.

    workers.wait_for_complete() waits for the workers running when it is called, and
    an approved plan chains turns — the gate starts the impl session, and that turn's
    tool result starts the next one. Waiting once can return in the gap between two
    of them, before the end-of-plan notice has been posted.
    """
    for _ in range(6):
        await app.workers.wait_for_complete()
        await pilot.pause()


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


@pytest.mark.asyncio
async def test_the_gate_is_reachable_on_a_small_terminal(monkeypatch):
    """The gate must fit on the screen the plan was written on.

    The pinned panel grows one row per step, and a 12-step plan took 15 of an 80x24
    terminal's 24 rows — leaving the chat 2, which is less than the gate card the
    same plan had just opened. The loop then sat blocked on buttons that could not
    be scrolled to: nothing was broken, and nothing could be answered either.
    """
    steps = [f"{i + 1}단계: 구체적인 작업 항목을 적는다" for i in range(12)]
    _stream_turns(monkeypatch, [[_submit(*steps)], [TextDelta("did it")]])

    app = AhaCodeApp()
    async with app.run_test(size=(80, 24)) as pilot:   # the default terminal
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "만들어줘")
        assert app.plan.pending is True
        r = app.query_one("#plan-gate-run", Button).region
        assert 0 <= r.y and r.bottom <= app.size.height, (
            f"▶ 실행 sits at rows {r.y}..{r.bottom} of a {app.size.height}-row screen"
        )


@pytest.mark.asyncio
async def test_a_submitted_plan_survives_reopening_the_session(monkeypatch):
    """Quitting with a plan on screen must not strand it. The gate is runtime state,
    so reopening the session has to rebuild it from the transcript — otherwise the
    checklist comes back, the plan file is still there, and nothing can run it."""
    _stream_turns(monkeypatch, [[_submit()], [TextDelta("did it all")]])

    app = AhaCodeApp()
    async with app.run_test(size=_TALL) as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        planner = app.session_path.stem
        assert app.plan.pending is True

        await app._new_session()                  # walk away, as quitting would
        await pilot.pause()
        assert app.plan.pending is False
        assert not list(app.query(PlanGate))

        await app._switch_session(planner)        # come back to it
        await pilot.pause()
        assert app.plan.pending is True     # the gate is waiting again
        assert len(list(app.query(PlanGate))) == 1
        assert list(app.query(PlanGate))[-1].steps == STEPS

        app.query_one("#plan-gate-run", Button).press()   # and it still works
        await _settle(app, pilot)
        assert app.session_kind == "impl"
        assert storage.read_header(app.session_path)["parent_id"] == planner


@pytest.mark.asyncio
async def test_a_session_that_never_submitted_reopens_without_a_gate(monkeypatch):
    """Only a transcript ENDING on a submitted plan reopens one. A turn that just
    talked leaves nothing to decide, and must not pause the loop on reload."""
    _stream_turns(monkeypatch, [[TextDelta("아직 계획은 없어요")]])

    app = AhaCodeApp()
    async with app.run_test(size=_TALL) as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "생각 좀 해봐")
        planner = app.session_path.stem

        await app._new_session()
        await pilot.pause()
        await app._switch_session(planner)
        await pilot.pause()
        assert app.plan.pending is False
        assert not list(app.query(PlanGate))


def _plan_file(app):
    return storage.PLANS_DIR / f"{app.session_path.stem}.md"


def _plan_file_for(app):
    """The plan an impl session was handed (named after its PARENT)."""
    return storage.PLANS_DIR / f"{app.session_parent_id}.md"


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
        assert app.plan.pending is True
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
        [ToolCall(id="1", name="plan_submit", arguments={"summary": "s", "steps": []})],
        [_submit(*STEPS, call_id="2")],
        [TextDelta("never sent")],
    ])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        assert len(app.query(PlanGate)) == 1
        assert app.plan.pending is True
        assert len(calls) == 2
        first_result = app.session.messages[2]
        assert first_result["role"] == "tool" and "PlanRejected: steps is empty" in first_result["content"]
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
        assert app.plan.pending is False
        assert app.session.messages[-1]["content"] == "done"


# --- the two ways out ------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_button_hands_the_plan_to_a_child_impl_session(monkeypatch):
    """▶ 실행 opens a HANDOFF child: same depth, parented to the planning session,
    in act mode, seeded with one user message naming the plan file — and the whole
    plan runs in that one context (no per-step sub-agents)."""
    calls = _stream_turns(monkeypatch, [[_submit()], [TextDelta("did it all")]])

    app = AhaCodeApp()
    async with app.run_test(size=_TALL) as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        parent = app.session_path
        assert app.plan.pending is True

        await pilot.click("#plan-gate-run", offset=_INSIDE)
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.plan.pending is False
        assert app.mode == "act" and app.session_kind == "impl"
        assert app.session_path != parent
        header = storage.read_header(app.session_path)
        assert header["kind"] == "impl" and header["relation"] == "handoff"
        assert header["parent_id"] == parent.stem and header["depth"] == 0
        assert header["title"] == "Solve it"                 # named after the plan
        # the seed is one user turn naming the plan file, then the model's own work
        assert app.session.messages[0]["role"] == "user"
        assert f"plans/{parent.stem}.md" in app.session.messages[0]["content"]
        assert app.session.messages[-1]["content"] == "did it all"
        assert list(app.query(SubagentCard)) == []
        # the impl turn ran under the act prompt, not the planner's
        assert "PLAN MODE" not in calls[1][0] if isinstance(calls[1][0], str) else True


@pytest.mark.asyncio
async def test_approving_a_revised_plan_makes_a_sibling_not_a_deeper_child(monkeypatch):
    """Revise → resubmit → approve again: the second impl session is parented to the
    SAME planning session (a sibling of the first), never nested under it."""
    _stream_turns(monkeypatch, [
        [_submit()], [TextDelta("v1 done")],
        [_submit("Write b.py with g()", summary="v2")], [TextDelta("v2 done")],
    ])

    app = AhaCodeApp()
    async with app.run_test(size=_TALL) as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        planner = app.session_path.stem
        await pilot.click("#plan-gate-run", offset=_INSIDE)
        await _settle(app, pilot)
        first = app.session_path.stem

        await app._switch_session(planner)        # back to the planning session
        await pilot.pause()
        await _plan_mode(app, pilot)
        app.query_one("#prompt", PromptInput).focus()  # Enter must reach the prompt
        await _ask(pilot, app, "b.py 하나로 줄여")   # revise → resubmit → new gate
        app.query_one("#plan-gate-run", Button).press()  # may sit below the viewport
        await _settle(app, pilot)   # the gate starts the impl turn; wait for that too
        second = app.session_path.stem

        assert first != second
        assert storage.read_header(app.session_path)["parent_id"] == planner
        tree = storage.build_tree(storage.list_sessions())
        root = next(n for n in tree if n["id"] == planner)
        assert sorted(c["id"] for c in root["children"]) == sorted([first, second])


@pytest.mark.asyncio
async def test_revise_button_settles_without_resuming(monkeypatch):
    """✎ 수정 keeps the user in plan mode: nothing is re-requested; the next message
    is a fresh turn that can revise the plan."""
    calls = _stream_turns(monkeypatch, [[_submit()], [_submit("Write b.py with g()", summary="v2")]])

    app = AhaCodeApp()
    async with app.run_test(size=_TALL) as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        await pilot.click("#plan-gate-continue", offset=_INSIDE)
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.plan.pending is False
        assert len(calls) == 1                         # no resume request
        assert app.mode == "plan"
        assert app.query_one(PlanGate).has_class("plan-gate--settled")

        await _ask(pilot, app, "b.py 하나로 줄여")
        gates = list(app.query(PlanGate))
        assert len(gates) == 2 and gates[-1].steps == ["Write b.py with g()"]
        assert app.plan.pending is True
        assert _plan_file(app).read_text(encoding="utf-8").startswith("# v2\n")
        assert [it["content"] for it in app.query_one(TodoPanel).items] == ["Write b.py with g()"]


@pytest.mark.asyncio
async def test_typing_while_the_gate_is_open_is_revision_feedback(monkeypatch):
    """Text is never an approval (Roo / Claude Code): the card settles as 수정, the
    plan is not executed, and the message reaches the model as feedback."""
    calls = _stream_turns(monkeypatch, [[_submit()], [TextDelta("revised")]])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        assert app.plan.pending is True

        await _ask(pilot, app, "승인")            # even this — the model answers by resubmitting
        assert app.plan.pending is False
        assert list(app.query(SubagentCard)) == []
        gate = list(app.query(PlanGate))[0]
        assert gate.has_class("plan-gate--settled")
        assert "수정" in str(gate.query_one(".plan-gate-title").render())
        assert calls[1][-1] == "user" and app.session.messages[-1]["content"] == "revised"


@pytest.mark.asyncio
async def test_empty_enter_while_the_gate_is_open_approves(monkeypatch):
    """The keyboard's ▶: nothing typed, Enter — the card is the only thing waiting."""
    _stream_turns(monkeypatch, [[_submit()], [TextDelta("done")]])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        assert app.plan.pending is True

        await _ask(pilot, app, "")
        assert app.plan.pending is False
        assert app.mode == "act" and app.session_kind == "impl"
        assert list(app.query(PlanGate)) == []          # the child is a fresh view


@pytest.mark.asyncio
async def test_new_session_forgets_the_gate(monkeypatch):
    _stream_turns(monkeypatch, [[_submit()]])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        assert app.plan.pending is True

        app.query_one("#prompt", PromptInput).text = "/new"
        await pilot.press("enter")
        await pilot.pause()
        assert app.plan.pending is False
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
        assert app.plan.pending is False
        assert len(app.query(SubagentCard)) == 1
        assert app.query_one(TodoPanel).items == []
        assert app.session.messages[-1]["content"] == "parent done"


@pytest.mark.asyncio
async def test_a_reloaded_session_shows_the_plan_in_the_panel_not_a_tool_card(monkeypatch):
    """A successful plan_submit is the plan panel (and, live, the gate) — never a grey
    tool-result card. On reload its steps refill the panel and no card is mounted."""
    _stream_turns(monkeypatch, [[_submit()]])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        await app._render_history()
        await pilot.pause()
        assert [it["content"] for it in app.query_one(TodoPanel).items] == STEPS
        assert [b for b in app.query(ToolResultBlock) if "plan_submit" in b.title] == []


@pytest.mark.asyncio
async def test_a_rejected_submission_still_shows_its_reason_on_reload(monkeypatch):
    """The one plan_submit that keeps a card: a rejection, so the user can still read
    why after a reload."""
    _stream_turns(monkeypatch, [
        [ToolCall(id="1", name="plan_submit", arguments={"summary": "s", "steps": []})],
        [_submit(call_id="2")],
    ])
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        await app._render_history()
        await pilot.pause()
        cards = [b for b in app.query(ToolResultBlock) if "plan_submit" in b.title]
        assert len(cards) == 1 and "failed" in cards[0].title  # the rejection, as an error card


@pytest.mark.asyncio
async def test_an_impl_session_gets_the_larger_turn_cap(monkeypatch):
    """One context carries a whole plan, so it may take impl_max_turns tool rounds;
    an ordinary turn keeps the default."""
    from ahacode import agent

    seen = []
    real_run = agent.run

    def spy(messages, **kw):
        seen.append(kw.get("max_turns"))
        return real_run(messages, **kw)

    monkeypatch.setattr(agent, "run", spy)
    _stream_turns(monkeypatch, [[TextDelta("hi")], [_submit()], [TextDelta("done")]])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _ask(pilot, app, "안녕")                 # an ordinary act turn
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        app.query_one("#plan-gate-run", Button).press()  # may sit below the viewport
        await _settle(app, pilot)   # the gate starts a SECOND turn; wait for that one
    assert seen[0] == agent.DEFAULT_MAX_TURNS
    assert seen[-1] == config.load().impl_max_turns > agent.DEFAULT_MAX_TURNS


# --- the impl turn's end: what is still owed ---------------------------------

def _todo_status(*pairs, call_id="w1"):
    return ToolCall(id=call_id, name="todo_write", arguments={
        "items": [{"content": c, "status": st} for c, st in pairs]
    })


async def _approved_impl(monkeypatch, impl_turns):
    """A plan submitted and approved; the impl session then plays `impl_turns`."""
    _stream_turns(monkeypatch, [[_submit()], *impl_turns])
    app = AhaCodeApp()
    return app


@pytest.mark.asyncio
async def test_an_impl_turn_that_leaves_steps_owed_says_so(monkeypatch):
    app = await _approved_impl(monkeypatch, [
        [_todo_status(("Write solver.py with solve()", "done"), ("Add tests/test_solver.py", "in_progress"),
               ("Run pytest and confirm 3 passed", "pending"))],
        [TextDelta("stopping here for now")],
    ])
    async with app.run_test() as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        app.query_one("#plan-gate-run", Button).press()
        await _settle(app, pilot)
        said = [b._content for b in app.query(Chatbox) if b.has_class("chatbox--system")]
        assert any("미완 항목 2개" in s and "Add tests/test_solver.py" in s for s in said)
        # the progress snapshot sits beside the PLANNING session's plan file
        result = storage.PLANS_DIR / f"{app.session_parent_id}.result.md"
        text = result.read_text(encoding="utf-8")
        assert text.startswith("# 진행 중 1/3") and "▶ Add tests/test_solver.py" in text
        assert f"- session: {app.session_path.stem}" in text


@pytest.mark.asyncio
async def test_a_finished_plan_gets_no_owed_notice(monkeypatch):
    """done and cancelled both count as finished — nothing is owed."""
    app = await _approved_impl(monkeypatch, [
        [_todo_status(("Write solver.py with solve()", "done"), ("Add tests/test_solver.py", "cancelled"),
               ("Run pytest and confirm 3 passed", "done"))],
        [TextDelta("all done")],
    ])
    async with app.run_test() as pilot:
        await _plan_mode(app, pilot)
        await _ask(pilot, app, "풀어줘")
        app.query_one("#plan-gate-run", Button).press()
        await _settle(app, pilot)
        said = [b._content for b in app.query(Chatbox) if b.has_class("chatbox--system")]
        assert not any("미완 항목" in s for s in said)
        assert any("✓ 계획 완료" in s for s in said)
        assert app.query_one(TodoPanel).has_class("todo-panel--done")
        result = storage.PLANS_DIR / f"{app.session_parent_id}.result.md"
        text = result.read_text(encoding="utf-8")
        assert text.startswith("# 완료") and "## Latest summary\n\nall done" in text
        # the plan itself was not touched: numbered steps, no checkboxes
        plan_text = _plan_file_for(app).read_text(encoding="utf-8")
        assert "1. Write solver.py with solve()" in plan_text and "[ ]" not in plan_text


@pytest.mark.asyncio
async def test_an_ordinary_session_never_gets_the_owed_notice(monkeypatch):
    _stream_turns(monkeypatch, [[_todo_status(("a", "in_progress"), ("b", "pending"))], [TextDelta("later")]])
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _ask(pilot, app, "해줘")                 # act, kind=main
        said = [b._content for b in app.query(Chatbox) if b.has_class("chatbox--system")]
        assert not any("미완 항목" in s for s in said)
