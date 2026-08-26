"""Deleting sessions from the picker: cascade, confirm, and the open session."""

import pytest
from textual.widgets import ListView, Static

from ahacode import client, config, storage
from ahacode.app import AhaCodeApp
from ahacode.events import TextDelta
from ahacode.widgets.prompt_input import PromptInput
from ahacode.widgets.session_picker import SessionPicker


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(storage, "PLANS_DIR", tmp_path / "plans")
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(client, "list_models", lambda: ["qwen38"])
    monkeypatch.setattr(client, "complete", lambda messages: "")
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter([TextDelta("hi")]))
    monkeypatch.setattr(AhaCodeApp, "generate_title", lambda self, *a, **k: None)
    client.reset()
    yield
    client.reset()


def _session(tmp_path, sid, **header):
    p = tmp_path / f"{sid}.jsonl"
    storage.write_header(p, storage.make_header(sid, **header))
    return p


# --- storage ---------------------------------------------------------------------

def test_delete_cascades_over_descendants_and_side_files(tmp_path):
    _session(tmp_path, "root", title="root")
    _session(tmp_path, "impl", parent_id="root", kind="impl", relation="handoff")
    _session(tmp_path, "sub", parent_id="impl", kind="subagent", relation="delegate", depth=1)
    _session(tmp_path, "other", title="other")
    (tmp_path / "impl-out").mkdir()
    (tmp_path / "impl-out" / "x.txt").write_text("spill", encoding="utf-8")
    plans = tmp_path / "plans"; plans.mkdir()
    (plans / "root.md").write_text("# plan", encoding="utf-8")
    (plans / "root.result.md").write_text("# 완료", encoding="utf-8")

    gone = storage.delete_session("root")

    assert set(gone) == {"root", "impl", "sub"}
    assert not (tmp_path / "root.jsonl").exists()
    assert not (tmp_path / "impl.jsonl").exists() and not (tmp_path / "sub.jsonl").exists()
    assert not (tmp_path / "impl-out").exists()
    assert not (plans / "root.md").exists() and not (plans / "root.result.md").exists()
    assert (tmp_path / "other.jsonl").exists()          # unrelated, untouched


def test_deleting_a_child_leaves_its_parent(tmp_path):
    _session(tmp_path, "root")
    _session(tmp_path, "impl", parent_id="root", kind="impl")
    assert storage.delete_session("impl") == ["impl"]
    assert (tmp_path / "root.jsonl").exists()


# --- picker ----------------------------------------------------------------------

async def _open_picker(app, pilot):
    app.query_one("#prompt", PromptInput).text = "/sessions"
    await pilot.press("enter")
    await pilot.pause()
    assert isinstance(app.screen, SessionPicker)
    return app.screen


def _row_of(picker, sid):
    lv = picker.query_one("#picker-list", ListView)
    for i, item in enumerate(lv.children):
        if getattr(item, "session_id", None) == sid:
            return i
    raise AssertionError(f"no row for {sid}")


@pytest.mark.asyncio
async def test_delete_takes_two_presses_and_esc_withdraws(tmp_path):
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        # created AFTER the app opened its own session, so it is another row, not
        # the one the app resumed into (latest_session would have picked it)
        _session(tmp_path, "victim", title="victim")
        picker = await _open_picker(app, pilot)
        lv = picker.query_one("#picker-list", ListView)
        lv.index = _row_of(picker, "victim")
        await pilot.pause()

        await pilot.press("d")                      # first press: the question
        await pilot.pause()
        assert "삭제할까요? victim" in str(picker.query_one("#picker-title", Static).render())
        assert (tmp_path / "victim.jsonl").exists()

        await pilot.press("escape")                 # withdraws, does not close
        await pilot.pause()
        assert isinstance(app.screen, SessionPicker)
        assert (tmp_path / "victim.jsonl").exists()

        await pilot.press("d")
        await pilot.press("d")                      # second press: gone
        await pilot.pause()
        assert not (tmp_path / "victim.jsonl").exists()
        assert all(getattr(i, "session_id", None) != "victim" for i in lv.children)
        assert isinstance(app.screen, SessionPicker)  # still browsing


@pytest.mark.asyncio
async def test_the_new_row_cannot_be_deleted(tmp_path):
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        picker = await _open_picker(app, pilot)
        picker.query_one("#picker-list", ListView).index = 0   # "＋ new session"
        await pilot.press("d")
        await pilot.press("d")
        await pilot.pause()
        assert picker._pending is None
        assert isinstance(app.screen, SessionPicker)


@pytest.mark.asyncio
async def test_deleting_the_open_session_moves_the_app_to_a_fresh_one(tmp_path):
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        current = app.session_path
        picker = await _open_picker(app, pilot)
        picker.query_one("#picker-list", ListView).index = _row_of(picker, current.stem)
        await pilot.press("d")
        await pilot.press("d")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not current.exists()
        assert app.session_path != current and app.session_path.exists()


@pytest.mark.asyncio
async def test_a_session_whose_turn_is_running_is_not_deletable(monkeypatch, tmp_path):
    import time

    def slow(messages, tools=None):
        for _ in range(60):
            time.sleep(0.02)
            yield TextDelta("x")

    monkeypatch.setattr(client, "stream_chat", slow)
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        current = app.session_path
        app.query_one("#prompt", PromptInput).text = "go"
        await pilot.press("enter")
        for _ in range(50):
            w = getattr(app, "_response_worker", None)
            if w is not None and w.is_running:
                break
            await pilot.pause(0.02)
        app._open_picker()
        await pilot.pause()
        picker = app.screen
        picker.query_one("#picker-list", ListView).index = _row_of(picker, current.stem)
        await pilot.press("d")
        await pilot.press("d")
        await pilot.pause()
        assert current.exists()
        assert "진행 중" in str(picker.query_one("#picker-title", Static).render())
        await pilot.press("escape")
        await pilot.pause()
        app.action_stop()
        await app.workers.wait_for_complete()
