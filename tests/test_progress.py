"""Long work has to look different from stuck work.

A status line reading `running bash…` that never changes is the same picture as a
frozen app — which is how a 120-second test run and a real deadlock came to look
alike. Everything here is about the elapsed second that tells them apart.
"""

import time

import pytest

from ahacode import client, config, storage
from ahacode.app import AhaCodeApp
from ahacode.events import TextDelta, ToolCall, ToolResult
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
    b = {"thinking": None, "answer": None, "tool": {}, "tool_buf": {}, "call_args": {}}
    if gate:
        b["gate"] = True
    return b


@pytest.mark.asyncio
async def test_a_running_tool_counts_up():
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        container = app.query_one("#chat-container")
        await app._render_event(
            ToolCall(id="1", name="bash", arguments={"command": "pytest"}),
            _boxes(), container)
        app._running_tools["1"] = ("bash", time.monotonic() - 87)
        app._tick_progress()
        assert "bash" in app._last_status and "87초" in app._last_status


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [80, 100, 120])
async def test_the_status_fits_the_bar_it_lives_in(width):
    """The counter was invisible on an 80-wide terminal: the composer's fixed
    controls left the flexible status 7 columns, enough for the bullet and nothing
    else. Counting seconds nobody can read is the same as not counting."""
    from textual.widgets import Static

    app = AhaCodeApp()
    async with app.run_test(size=(width, 30)) as pilot:
        await pilot.pause()
        app._running_tools["1"] = ("bash", time.monotonic() - 87)
        app._tick_progress()
        await pilot.pause()
        status = app.query_one("#status", Static)
        rows = [status.render_line(y).text.strip() for y in range(status.region.height)]
    assert app._last_status in rows, (
        f"{width} columns: status area is {status.region.width} wide, "
        f"needs {len(app._last_status)} for {app._last_status!r}"
    )


@pytest.mark.asyncio
async def test_parallel_tools_are_counted_not_listed():
    """Three reads racing must not take turns overwriting each other's name."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        container = app.query_one("#chat-container")
        boxes = _boxes()
        for i in range(3):
            await app._render_event(
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
        await app._render_event(
            ToolCall(id="1", name="bash", arguments={"command": "echo hi"}),
            boxes, container)
        assert app._running_tools
        await app._render_event(ToolResult("1", "bash", "hi"), boxes, container)
        assert not app._running_tools


@pytest.mark.asyncio
async def test_a_subagents_tool_does_not_claim_the_status_line():
    """A fan-out runs several children at once; letting their tools drive the one
    status line makes it flicker between them and speak for none."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        container = app.query_one("#chat-container")
        await app._render_event(
            ToolCall(id="9", name="grep", arguments={"pattern": "x"}),
            _boxes(gate=False), container)          # a child's boxes carry no gate
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
