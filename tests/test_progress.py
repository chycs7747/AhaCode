"""Long work has to look different from stuck work.

A status line reading `running bash…` that never changes is the same picture as a
frozen app — which is how a 120-second test run and a real deadlock came to look
alike. Everything here is about the elapsed second that tells them apart.
"""

import time

import pytest
from rich.cells import cell_len

from ahacode import agent, client, config, storage
from ahacode.app import AhaCodeApp
from ahacode.turn_view import _PHASE_ID, TurnBoxes
from ahacode.events import Phase, TextDelta, ToolCall, ToolResult
from ahacode.widgets.subagent_card import SubagentCard


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(client, "list_models", lambda *a, **k: ["m"])
    monkeypatch.setattr(client, "complete", lambda m: "")
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter([TextDelta("hi")]))
    monkeypatch.setattr(AhaCodeApp, "generate_title", lambda self, *a, **k: None)


def _boxes(gate=True):
    return TurnBoxes(owns_session=gate)


@pytest.mark.asyncio
async def test_a_running_tool_counts_up():
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        container = app.query_one("#chat-container")
        await app.turn_view.render(
            ToolCall(id="1", name="bash", arguments={"command": "pytest"}),
            _boxes(), container)
        app._running_tools["1"] = ("bash", time.monotonic() - 87)
        app._tick_progress()
        assert "bash" in app._last_status and "87초" in app._last_status


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [80, 100, 120])
@pytest.mark.parametrize("label, ago", [
    ("bash", 87),
    # Korean costs two cells per character, so a label that reads fine in source
    # can still lose the seconds at the end. "컨텍스트 압축" needed 23 of the 18
    # cells the bar leaves at width 80 and truncated to "● 컨텍스트 압축 ·" —
    # everything except the number it exists to show. 1220s covers a 20-minute
    # compaction, the longest run actually observed.
    (agent.COMPACTING, 1220),
])
async def test_the_status_fits_the_bar_it_lives_in(width, label, ago):
    """The counter was invisible on an 80-wide terminal: the composer's fixed
    controls left the flexible status 7 columns, enough for the bullet and nothing
    else. Counting seconds nobody can read is the same as not counting."""
    from textual.widgets import Static

    app = AhaCodeApp()
    async with app.run_test(size=(width, 30)) as pilot:
        await pilot.pause()
        app._running_tools["1"] = (label, time.monotonic() - ago)
        app._tick_progress()
        await pilot.pause()
        status = app.query_one("#status", Static)
        rows = [status.render_line(y).text.strip() for y in range(status.region.height)]
    # cell_len, not len: Korean is two cells per character, which is the whole
    # reason a label can measure "short" in source and still not fit.
    assert app._last_status in rows, (
        f"{width} columns: status area is {status.region.width} wide, "
        f"needs {cell_len(app._last_status)} for {app._last_status!r}"
    )


@pytest.mark.asyncio
async def test_parallel_tools_are_counted_not_listed():
    """Three reads racing must not take turns overwriting each other's name."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        container = app.query_one("#chat-container")
        boxes = _boxes()
        for i in range(3):
            await app.turn_view.render(
                ToolCall(id=str(i), name="read", arguments={"path": f"{i}.py"}),
                boxes, container)
        app._tick_progress()
        assert "3개" in app._last_status


@pytest.mark.asyncio
async def test_a_finished_tool_stops_being_counted():
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        container = app.query_one("#chat-container")
        boxes = _boxes()
        await app.turn_view.render(
            ToolCall(id="1", name="bash", arguments={"command": "echo hi"}),
            boxes, container)
        assert app._running_tools
        await app.turn_view.render(ToolResult("1", "bash", "hi"), boxes, container)
        assert not app._running_tools


@pytest.mark.asyncio
async def test_a_subagents_tool_does_not_claim_the_status_line():
    """A fan-out runs several children at once; letting their tools drive the one
    status line makes it flicker between them and speak for none."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        container = app.query_one("#chat-container")
        await app.turn_view.render(
            ToolCall(id="9", name="grep", arguments={"pattern": "x"}),
            _boxes(gate=False), container)          # a child's boxes carry no gate
        assert not app._running_tools


@pytest.mark.asyncio
async def test_compaction_counts_up_like_a_tool():
    """Compaction was the last silent stretch: one model call over the whole
    history, no stream, no tool, minutes long. It reported only once it was over,
    by which time the user had already read the still screen as a freeze."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        container = app.query_one("#chat-container")
        await app.turn_view.render(Phase(agent.COMPACTING), _boxes(), container)
        assert agent.COMPACTING in app._last_status
        app._running_tools[_PHASE_ID] = (agent.COMPACTING, time.monotonic() - 122)
        app._tick_progress()
        assert "122초" in app._last_status


@pytest.mark.asyncio
async def test_a_finished_compaction_stops_being_counted():
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        container = app.query_one("#chat-container")
        boxes = _boxes()
        await app.turn_view.render(Phase(agent.COMPACTING), boxes, container)
        assert app._running_tools
        await app.turn_view.render(Phase(agent.COMPACTING, done=True), boxes, container)
        assert not app._running_tools


@pytest.mark.asyncio
async def test_a_subagents_compaction_does_not_claim_the_status_line():
    """A child compacts its own history; its card already carries a ticking clock,
    and letting it drive the one status line would speak over the parent."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        container = app.query_one("#chat-container")
        await app.turn_view.render(Phase(agent.COMPACTING), _boxes(gate=False), container)
        assert not app._running_tools


@pytest.mark.asyncio
async def test_a_stopped_turn_stops_the_clock():
    """Esc during a tool never delivers its ToolResult, so the entry that the
    result would have retired stays — and the counter goes on claiming work is
    running for an app that stopped. Ending the turn is what clears it."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app._running_tools["1"] = ("bash", time.monotonic() - 30)
        app._set_send_running(False)
        assert not app._running_tools
        app._tick_progress()  # and nothing revives it
        assert not app._running_tools


def test_a_card_counts_up_while_its_child_works():
    card = SubagentCard("요약", "m")
    card._t0 = time.monotonic() - 42
    card.tick()
    assert "42초" in card.title


def test_a_finished_card_freezes_its_time():
    card = SubagentCard("요약", "m")
    card._t0 = time.monotonic() - 10
    card.done(tool_count=3)
    frozen = card.title
    card._t0 = time.monotonic() - 999
    card.tick()
    assert card.title == frozen and "3개 도구" in frozen
