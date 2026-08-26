"""The agent-settings modal: set max parallel / depth from a button, persisted to
config without editing the file by hand."""

import pytest
from dataclasses import replace
from textual.widgets import Button, Select

from ahacode import client, config, storage
from ahacode.app import AhaCodeApp
from ahacode.events import TextDelta
from ahacode.widgets.agent_settings import AgentSettings, AgentSettingsResult


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
    config.save(replace(config.DEFAULTS, max_parallel_agents=8, subagent_depth=1,
                        context_window=32768, compact_threshold=0.8,
                        plan_thinking_budget=8192, impl_thinking_budget=None,
                        no_think_after_tools=True))
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.click("#agent-settings-btn")
        await pilot.pause()
        assert isinstance(app.screen, AgentSettings)
        assert app.screen.query_one("#agent-parallel", Select).value == 8
        assert app.screen.query_one("#agent-depth", Select).value == 1
        assert app.screen.query_one("#agent-window", Select).value == 32768
        assert app.screen.query_one("#agent-threshold", Select).value == 0.8
        assert app.screen.query_one("#agent-think-plan", Select).value == 8192
        assert app.screen.query_one("#agent-think-impl", Select).value == -1  # None → 전역 sentinel
        assert app.screen.query_one("#agent-after-tools", Select).value is True


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
        app.screen.query_one("#agent-window", Select).value = 16384
        app.screen.query_one("#agent-threshold", Select).value = 0.9
        app.screen.query_one("#agent-think-plan", Select).value = 8192
        app.screen.query_one("#agent-think-subagent", Select).value = 1024
        app.screen.query_one("#agent-after-tools", Select).value = False  # reference-style
        await pilot.pause()
        app.screen.query_one("#agent-settings-save", Button).press()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not isinstance(app.screen, AgentSettings)             # closed
        cfg = config.load()
        assert cfg.max_parallel_agents == 1 and cfg.subagent_depth == 2
        assert cfg.context_window == 16384 and cfg.compact_threshold == 0.9
        assert cfg.plan_thinking_budget == 8192 and cfg.subagent_thinking_budget == 1024
        assert cfg.impl_thinking_budget is None                      # left at 전역
        assert cfg.no_think_after_tools is False                     # toggled to reference-style
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


@pytest.mark.asyncio
async def test_a_hand_set_window_snaps_to_the_nearest_option():
    """A context_window edited by hand to an off-list value still lands on a real
    Select option rather than leaving it blank."""
    config.save(replace(config.DEFAULTS, context_window=30000))     # not a listed value
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.click("#agent-settings-btn")
        await pilot.pause()
        assert app.screen.query_one("#agent-window", Select).value == 32768  # nearest
