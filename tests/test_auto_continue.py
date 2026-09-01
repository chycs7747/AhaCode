"""An impl session carries itself, and knows when to stop trying.

A plan is a list of steps, and the loop was stopping after every turn to ask
whether to go on — including the turns that ended because they ran out of rounds
rather than because they finished anything. One measured turn spent 30 rounds and
25 minutes and completed 0 of 3 steps, then asked. These tests pin the two halves
of the fix: keep going while steps are being completed, and give up when they are
not.
"""

from dataclasses import replace

import pytest

from ahacode import agent, client, config, storage
from ahacode.app import AhaCodeApp
from ahacode.events import TextDelta, ToolCall, Usage
from ahacode.tools import plan
from ahacode.tools.base import Tool
from ahacode.widgets.todo_panel import TodoPanel

METRICS = {"prompt": 100, "gen": 10, "gen_seconds": 1.0, "ttft": 0.5, "model": "m"}


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(client, "list_models", lambda *a, **k: ["m"])
    monkeypatch.setattr(client, "complete", lambda m: "")
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter([TextDelta("ok")]))
    monkeypatch.setattr(AhaCodeApp, "generate_title", lambda self, *a, **k: None)


def _cfg(**kw):
    return lambda: replace(config.DEFAULTS, **kw)


def _steps(done: int, total: int = 3) -> list[dict]:
    # The status vocabulary comes from plan.py rather than being spelled out here:
    # a literal that drifts from it makes every step look unfinished, which reads
    # as "the run never stops" instead of "the test is wrong".
    return [{"content": f"step {i}", "status": plan.DONE if i < done else plan.PENDING}
            for i in range(total)]


async def _impl_app(pilot, app, done: int, total: int = 3):
    """An impl session showing `done` of `total` steps finished."""
    app.session_kind = "impl"
    app.query_one(TodoPanel).update_todos(_steps(done, total))
    await pilot.pause()


def _turns_started(app, monkeypatch) -> list:
    """Record _start_turn calls instead of running the agent loop."""
    started: list = []

    async def fake_start_turn():
        started.append(True)

    monkeypatch.setattr(app, "_start_turn", fake_start_turn)
    return started


@pytest.mark.asyncio
async def test_a_turn_that_finished_a_step_carries_on(monkeypatch):
    monkeypatch.setattr(config, "load", _cfg(auto_continue_stall=3))
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _impl_app(pilot, app, done=1)
        started = _turns_started(app, monkeypatch)
        await app.plan.auto_continue()
        assert started, "a step was completed; the run should continue by itself"
        assert app.plan.stalled == 0


@pytest.mark.asyncio
async def test_a_finished_plan_stops(monkeypatch):
    monkeypatch.setattr(config, "load", _cfg(auto_continue_stall=3))
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _impl_app(pilot, app, done=3, total=3)
        started = _turns_started(app, monkeypatch)
        await app.plan.auto_continue()
        assert not started


@pytest.mark.asyncio
async def test_turns_that_finish_nothing_eventually_stop(monkeypatch):
    """The stall, not the round count, is what ends a run. Three turns that each
    produce plenty and complete nothing are the case a turn cap was reaching for
    and kept missing."""
    monkeypatch.setattr(config, "load", _cfg(auto_continue_stall=3))
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _impl_app(pilot, app, done=1)
        started = _turns_started(app, monkeypatch)
        await app.plan.auto_continue()          # 1 done: progress, continue
        assert len(started) == 1
        for expected in (2, 3):             # nothing new completes
            await app.plan.auto_continue()
            assert len(started) == expected, "still under the stall limit"
        await app.plan.auto_continue()          # third barren turn: give up
        assert len(started) == 3
        assert app.plan.stalled == 3


@pytest.mark.asyncio
async def test_progress_resets_the_stall_counter(monkeypatch):
    """A run that is slow but moving must never be cut off: two barren turns
    followed by a completed step start the count over."""
    monkeypatch.setattr(config, "load", _cfg(auto_continue_stall=3))
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _impl_app(pilot, app, done=1, total=5)
        started = _turns_started(app, monkeypatch)
        await app.plan.auto_continue()
        await app.plan.auto_continue()
        assert app.plan.stalled == 1
        app.query_one(TodoPanel).update_todos(_steps(2, 5))   # a step lands
        await app.plan.auto_continue()
        assert app.plan.stalled == 0
        assert len(started) == 3


@pytest.mark.asyncio
async def test_a_stopped_run_is_not_restarted(monkeypatch):
    """Esc has to mean stop. Auto-continue runs from the same handler the stop
    path posts through, so without this check it would immediately undo it."""
    monkeypatch.setattr(config, "load", _cfg(auto_continue_stall=3))
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _impl_app(pilot, app, done=1)
        started = _turns_started(app, monkeypatch)
        app._stopping = True
        await app.plan.auto_continue()
        assert not started


@pytest.mark.asyncio
async def test_the_plan_gate_is_not_talked_over(monkeypatch):
    monkeypatch.setattr(config, "load", _cfg(auto_continue_stall=3))
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _impl_app(pilot, app, done=1)
        started = _turns_started(app, monkeypatch)
        app.plan.pending = True
        await app.plan.auto_continue()
        assert not started


@pytest.mark.asyncio
async def test_zero_turns_it_off(monkeypatch):
    """0 keeps the old behaviour: report progress and wait to be told."""
    monkeypatch.setattr(config, "load", _cfg(auto_continue_stall=0))
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _impl_app(pilot, app, done=1)
        started = _turns_started(app, monkeypatch)
        await app.plan.auto_continue()
        assert not started


@pytest.mark.asyncio
async def test_a_session_with_no_checklist_is_left_alone(monkeypatch):
    """Progress is read off the checklist, so with no checklist there is nothing
    to read — continuing would be running blind, not running autonomously."""
    monkeypatch.setattr(config, "load", _cfg(auto_continue_stall=3))
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.session_kind = "impl"
        started = _turns_started(app, monkeypatch)
        await app.plan.auto_continue()
        assert not started


@pytest.mark.asyncio
async def test_typing_clears_the_stall(monkeypatch):
    """A typed instruction is a fresh start — the turns that went nowhere before
    it must not count against the ones after it."""
    monkeypatch.setattr(config, "load", _cfg(auto_continue_stall=3))
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _impl_app(pilot, app, done=1)
        app.plan.stalled = 2
        monkeypatch.setattr(app, "_start_turn", lambda: _noop())
        app.query_one("#prompt").text = "여기부터 다시 해줘"
        await pilot.press("enter")
        await pilot.pause()
        assert app.plan.stalled == 0


async def _noop():
    return None


# --- the round backstop, which is what makes an uncapped run leavable -------

@pytest.mark.asyncio
async def test_an_uncapped_turn_ends_when_no_step_lands(monkeypatch):
    """The case the whole backstop exists for: no turn cap, nobody watching, and
    a model that will keep calling tools all night. should_pause ends the turn
    between rounds, which hands it to the turn-level detector above."""
    monkeypatch.setattr(config, "load",
                        _cfg(impl_max_turns=0, stall_rounds=5, auto_continue_stall=3))
    rounds = []

    def stream(messages, tools=None):
        rounds.append(1)
        return iter([Usage(10, 5, 15),
                     ToolCall(id=str(len(rounds)), name="read", arguments={"path": "a"})])

    monkeypatch.setattr(client, "stream_chat", stream)
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _impl_app(pilot, app, done=0)
        monkeypatch.setattr(app, "_registry_for_mode", lambda: {
            "read": Tool(name="read", description="", parameters={},
                         execute=lambda a: "body")})
        monkeypatch.setattr(app.runner, "approve_tool", lambda call: True)
        monkeypatch.setattr(app.plan, "auto_continue", _noop)  # judge one turn only
        app.query_one("#prompt").text = "go"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert len(rounds) == 5, f"ran {len(rounds)} rounds against a limit of 5"
    assert app.plan.round_stalled


@pytest.mark.asyncio
async def test_a_completed_step_buys_more_rounds(monkeypatch):
    """A slow but moving run must never be cut. Completing a step resets the
    counter, so the limit bounds barren rounds — not rounds."""
    monkeypatch.setattr(config, "load",
                        _cfg(impl_max_turns=0, stall_rounds=3, auto_continue_stall=3))
    rounds = []

    def stream(messages, tools=None):
        rounds.append(1)
        # On round 3, a step lands — which should buy 3 more rounds.
        if len(rounds) == 3:
            return iter([Usage(10, 5, 15),
                         ToolCall(id="p", name="todo_write",
                                  arguments={"items": _steps(1)})])
        return iter([Usage(10, 5, 15),
                     ToolCall(id=str(len(rounds)), name="read", arguments={"path": "a"})])

    monkeypatch.setattr(client, "stream_chat", stream)
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _impl_app(pilot, app, done=0)
        monkeypatch.setattr(app, "_registry_for_mode", lambda: {
            "read": Tool(name="read", description="", parameters={},
                         execute=lambda a: "body"),
            "todo_write": Tool(name="todo_write", description="", parameters={},
                               execute=lambda a: "saved")})
        monkeypatch.setattr(app.runner, "approve_tool", lambda call: True)
        monkeypatch.setattr(app.plan, "auto_continue", _noop)
        app.query_one("#prompt").text = "go"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
    # 3 barren, then the step resets, then 3 more barren: 6, not 3.
    assert len(rounds) == 6, f"the step bought no rounds back (ran {len(rounds)})"


# --- the round cap ---------------------------------------------------------

def test_zero_max_turns_means_no_cap():
    """The cap was ending ordinary work, not runaway work. Uncapped, the loop runs
    until the model stops calling tools; the caller's stall detector is the
    backstop that replaces it."""
    rounds = []

    def stream(messages, tools=None):
        rounds.append(1)
        if len(rounds) > 25:          # well past the old cap of 10 / 30
            return iter([TextDelta("done")])
        return iter([agent.ToolCall(id=str(len(rounds)), name="read",
                                    arguments={"path": "a"})])

    from ahacode.tools.base import Tool
    agent.run(
        [{"role": "user", "content": "go"}],
        emit=lambda e: None, stream=stream, max_turns=0,
        registry={"read": Tool(name="read", description="", parameters={},
                               execute=lambda a: "body")},
    )
    assert len(rounds) == 26


def test_a_cap_still_caps():
    rounds = []

    def stream(messages, tools=None):
        rounds.append(1)
        return iter([agent.ToolCall(id=str(len(rounds)), name="read",
                                    arguments={"path": "a"})])

    from ahacode.tools.base import Tool
    agent.run(
        [{"role": "user", "content": "go"}],
        emit=lambda e: None, stream=stream, max_turns=4,
        registry={"read": Tool(name="read", description="", parameters={},
                               execute=lambda a: "body")},
    )
    assert len(rounds) == 5  # 4 tool rounds + the forced tool-free wrap-up
