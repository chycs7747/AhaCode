"""The settings modal: every config field reachable from one button, grouped into
tabs, persisted without editing config.toml by hand."""

import pytest
from dataclasses import replace
from textual.widgets import Button, ContentSwitcher, Input, Label, Select

from ahacode import client, config, storage
from ahacode.app import AhaCodeApp
from ahacode.events import TextDelta
from ahacode.widgets.settings import Settings, SettingsResult


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(client, "list_models", lambda *a, **k: ["qwen38"])
    monkeypatch.setattr(client, "complete", lambda m: "")
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter([TextDelta("hi")]))
    monkeypatch.setattr(AhaCodeApp, "generate_title", lambda self, *a, **k: None)
    config.save(config.DEFAULTS)
    client.reset()
    yield
    client.reset()


async def _open(pilot):
    await pilot.click("#settings-btn")
    await pilot.pause()


@pytest.mark.asyncio
async def test_button_opens_the_modal_seeded_from_config():
    """Every pane is seeded from config, and every pane is queryable at once —
    ContentSwitcher keeps them all mounted, so a save reads them whichever tab shows."""
    config.save(replace(config.DEFAULTS, base_url="http://localhost:8888/v1",
                        name="qwen38", api_key="sk-test", timeout=300.0,
                        max_parallel_agents=8, subagent_depth=1,
                        context_window=32768, compact_threshold=0.8,
                        plan_thinking_budget=8192, impl_thinking_budget=None,
                        no_think_after_tools=True))
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _open(pilot)
        assert isinstance(app.screen, Settings)
        s = app.screen
        assert s.query_one("#settings-base-url", Input).value == "http://localhost:8888/v1"
        assert s.query_one("#settings-api-key", Input).value == "sk-test"
        assert s.query_one("#settings-model", Select).value == "qwen38"
        assert s.query_one("#settings-timeout", Select).value == 300.0
        assert s.query_one("#settings-parallel", Select).value == 8
        assert s.query_one("#settings-depth", Select).value == 1
        assert s.query_one("#settings-window", Select).value == 32768
        assert s.query_one("#settings-threshold", Select).value == 0.8
        assert s.query_one("#settings-think-plan", Select).value == 8192
        assert s.query_one("#settings-think-impl", Select).value == -1  # None → 전역
        assert s.query_one("#settings-after-tools", Select).value is True


@pytest.mark.asyncio
async def test_the_rail_switches_panes():
    """Clicking a tab shows its pane and marks it active — one active at a time."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _open(pilot)
        panes = app.screen.query_one("#settings-panes", ContentSwitcher)
        assert panes.current == "settings-pane-connection"   # opens on 연결

        app.screen.query_one("#settings-tab-thinking", Button).press()
        await pilot.pause()
        assert panes.current == "settings-pane-thinking"
        active = [b.id for b in app.screen.query(".settings-tab").results(Button)
                  if b.has_class("-active")]
        assert active == ["settings-tab-thinking"]


@pytest.mark.asyncio
async def test_saving_persists_every_pane_and_resets_the_client(monkeypatch):
    reset = []
    real_reset = client.reset
    monkeypatch.setattr(client, "reset", lambda: (reset.append(True), real_reset())[1])

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _open(pilot)
        s = app.screen
        s.query_one("#settings-base-url", Input).value = "http://localhost:9000/v1"
        s.query_one("#settings-api-key", Input).value = "sk-new"
        s.query_one("#settings-timeout", Select).value = 60.0
        s.query_one("#settings-parallel", Select).value = 1        # serialise
        s.query_one("#settings-depth", Select).value = 2
        s.query_one("#settings-impl-turns", Select).value = 50
        s.query_one("#settings-window", Select).value = 16384
        s.query_one("#settings-threshold", Select).value = 0.9
        s.query_one("#settings-keep-recent", Select).value = 12
        s.query_one("#settings-think-global", Select).value = 2048
        s.query_one("#settings-effort", Select).value = "high"
        s.query_one("#settings-think-plan", Select).value = 8192
        s.query_one("#settings-think-subagent", Select).value = 1024
        s.query_one("#settings-after-tools", Select).value = False
        s.query_one("#settings-stall", Select).value = 5
        await pilot.pause()
        s.query_one("#settings-save", Button).press()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert not isinstance(app.screen, Settings)          # closed
        cfg = config.load()
        assert cfg.base_url == "http://localhost:9000/v1" and cfg.api_key == "sk-new"
        assert cfg.timeout == 60.0
        assert cfg.max_parallel_agents == 1 and cfg.subagent_depth == 2
        assert cfg.impl_max_turns == 50
        assert cfg.context_window == 16384 and cfg.compact_threshold == 0.9
        assert cfg.keep_recent_messages == 12
        assert cfg.thinking_token_budget == 2048 and cfg.reasoning_effort == "high"
        assert cfg.plan_thinking_budget == 8192 and cfg.subagent_thinking_budget == 1024
        assert cfg.impl_thinking_budget is None              # left at 전역
        assert cfg.no_think_after_tools is False
        assert cfg.auto_continue_stall == 5
        assert reset                                         # client + gate rebuilt


@pytest.mark.asyncio
async def test_the_window_picker_offers_what_the_server_can_hold():
    """The picker stopped at 128K while the gateway advertises max_model_len
    524288, so the windows worth choosing could not be chosen. A setting that has
    to be hand-written into TOML is a setting the screen does not really have."""
    from ahacode.widgets.settings import _IMPL_TURNS, _WINDOW

    windows = [v for _, v in _WINDOW]
    assert 196608 in windows and 262144 in windows
    # ...and a plan run can be told to stop counting rounds at all.
    assert 0 in [v for _, v in _IMPL_TURNS]


@pytest.mark.asyncio
async def test_cancel_changes_nothing():
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _open(pilot)
        app.screen.query_one("#settings-parallel", Select).value = 1
        await pilot.pause()
        app.screen.query_one("#settings-cancel", Button).press()
        await pilot.pause()
        assert not isinstance(app.screen, Settings)
        assert config.load().max_parallel_agents == config.DEFAULTS.max_parallel_agents


@pytest.mark.asyncio
async def test_an_empty_address_keeps_the_configured_one():
    """A blank box is 'unchanged', not 'clear it' — saving an empty base_url would
    otherwise leave the app with no endpoint at all."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _open(pilot)
        app.screen.query_one("#settings-base-url", Input).value = ""
        await pilot.pause()
        app.screen.query_one("#settings-save", Button).press()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert config.load().base_url == config.DEFAULTS.base_url


@pytest.mark.asyncio
async def test_fetch_offers_what_the_endpoint_reports(monkeypatch):
    """모델 불러오기 asks the address in the box — not the saved one — and the Select
    then offers exactly what came back."""
    asked = {}

    def fake_list(base_url=None, api_key=None):
        asked["base_url"], asked["api_key"] = base_url, api_key
        return ["qwen3.8-flash-next", "qwen3-4b"]

    monkeypatch.setattr(client, "list_models", fake_list)
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _open(pilot)
        s = app.screen
        s.query_one("#settings-base-url", Input).value = "http://localhost:8888/v1"
        s.query_one("#settings-api-key", Input).value = "sk-probe"
        await pilot.pause()
        s.query_one("#settings-fetch-models", Button).press()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert asked == {"base_url": "http://localhost:8888/v1", "api_key": "sk-probe"}
        select = s.query_one("#settings-model", Select)
        assert [v for _, v in select._options if v is not None] == \
               ["qwen3.8-flash-next", "qwen3-4b"]
        assert select.value == "qwen3.8-flash-next"   # config's name is not on offer


@pytest.mark.asyncio
async def test_a_failed_fetch_says_so_and_keeps_the_current_model(monkeypatch):
    """A typo in the address must not empty the model list — the configured model
    is still the one in use."""
    def boom(base_url=None, api_key=None):
        raise ConnectionError("nope")

    monkeypatch.setattr(client, "list_models", boom)
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _open(pilot)
        app.screen.query_one("#settings-fetch-models", Button).press()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "실패" in str(app.screen.query_one("#settings-fetch-status", Label).content)
        assert app.screen.query_one("#settings-model", Select).value == config.DEFAULTS.name


@pytest.mark.asyncio
async def test_a_hand_set_window_snaps_to_the_nearest_option():
    """A context_window edited by hand to an off-list value still lands on a real
    Select option rather than leaving it blank."""
    config.save(replace(config.DEFAULTS, context_window=30000))     # not listed
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await _open(pilot)
        assert app.screen.query_one("#settings-window", Select).value == 32768


def test_max_parallel_one_serialises_the_client_gate():
    """The knob's effect: a gate of 1 admits one gateway request at a time, so two
    sub-agents cannot hit the GPU together."""
    config.save(replace(config.DEFAULTS, max_parallel_agents=1))
    client.reset()
    assert client._ensure_gate()._initial_value == 1
