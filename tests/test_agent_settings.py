"""The agent-settings modal: set max parallel / depth from a button, persisted to
config without editing the file by hand."""

import pytest
from dataclasses import replace
from textual.widgets import Button, Select

from ahacode import client, config, storage
from ahacode.app import AhaCodeApp
from ahacode.events import TextDelta
from ahacode.widgets.agent_settings import AgentSettings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(client, "list_models", lambda: ["qwen38"])
    monkeypatch.setattr(client, "complete", lambda m: "")
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter([TextDelta("hi")]))
    monkeypatch.setattr(AhaCodeApp, "generate_title", lambda self, *a, **k: None)
    config.save(config.DEFAULTS)
    client.reset()
    yield
    client.reset()


@pytest.mark.asyncio
async def test_button_opens_the_modal_seeded_from_config():
    config.save(replace(config.DEFAULTS, max_parallel_agents=8, subagent_depth=1))
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.click("#agent-settings-btn")
        await pilot.pause()
        assert isinstance(app.screen, AgentSettings)
        assert app.screen.query_one("#agent-parallel", Select).value == 8
        assert app.screen.query_one("#agent-depth", Select).value == 1


@pytest.mark.asyncio
async def test_saving_persists_and_resizes_the_gate(monkeypatch):
    resized = []
    real_reset = client.reset
    monkeypatch.setattr(client, "reset", lambda: (resized.append(True), real_reset())[1])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.click("#agent-settings-btn")
        await pilot.pause()
        app.screen.query_one("#agent-parallel", Select).value = 1     # serialise
        app.screen.query_one("#agent-depth", Select).value = 2
        await pilot.pause()
        app.screen.query_one("#agent-settings-save", Button).press()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not isinstance(app.screen, AgentSettings)             # closed
        cfg = config.load()
        assert cfg.max_parallel_agents == 1 and cfg.subagent_depth == 2
        assert resized                                               # the gate was reset


@pytest.mark.asyncio
async def test_cancel_changes_nothing():
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.click("#agent-settings-btn")
        await pilot.pause()
        app.screen.query_one("#agent-parallel", Select).value = 1
        await pilot.pause()
        app.screen.query_one("#agent-settings-cancel", Button).press()
        await pilot.pause()
        assert not isinstance(app.screen, AgentSettings)
        assert config.load().max_parallel_agents == config.DEFAULTS.max_parallel_agents


def test_max_parallel_one_serialises_the_client_gate():
    """The knob's effect: a gate of 1 admits one gateway request at a time, so two
    sub-agents cannot hit the GPU together."""
    config.save(replace(config.DEFAULTS, max_parallel_agents=1))
    client.reset()
    gate = client._ensure_gate()
    assert gate._initial_value == 1
