"""Deleting sessions from the picker: cascade, confirm, and the open session."""

import pytest
from textual.widgets import Button, Input, Label, ListView, Static

from ahacode import client, config, storage
from ahacode.app import AhaCodeApp
from ahacode.events import TextDelta
from ahacode.widgets.prompt_input import PromptInput
from ahacode.widgets.session_picker import SessionPicker, SessionRow


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
        assert not any(r.armed for r in picker.query(SessionRow))
        assert isinstance(app.screen, SessionPicker)


@pytest.mark.asyncio
async def test_deleting_the_open_session_moves_the_app_to_a_fresh_one(tmp_path):
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        current = app.session_path
        picker = await _open_picker(app, pilot)
        row = next(r for r in picker.query(SessionRow) if r.session_id == current.stem)                 # deterministic: press its button
        row.query_one(".picker-delete", Button).press()
        await pilot.pause()
        row.query_one(".picker-delete", Button).press()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        # The app left the deleted transcript for a fresh, drivable session. (The file
        # NAME can recur: new_session_path stamps to the second, and a mock clock makes
        # delete+create land in the same tick — so assert freshness, not path.)
        assert app.session.messages == [] and app.session_kind == "main"
        assert not app.view_only
        assert storage.read_header(app.session_path) is not None


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
        app.sessions.open_picker()
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
        app.action_stop()                       # tidy up; the worker ends cancelled
        await pilot.pause(0.1)


# --- per-row buttons ---------------------------------------------------------------

def _row(picker, sid) -> SessionRow:
    return next(r for r in picker.query(SessionRow) if r.session_id == sid)


@pytest.mark.asyncio
async def test_the_row_delete_button_needs_two_clicks_and_never_opens_the_session(tmp_path):
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        _session(tmp_path, "victim", title="victim")
        picker = await _open_picker(app, pilot)
        row = _row(picker, "victim")
        row.query_one(".picker-delete", Button).press()
        await pilot.pause()
        assert isinstance(app.screen, SessionPicker)          # a button click is not a row click
        assert row.armed and str(row.query_one(".picker-delete", Button).label) == "확인?"
        assert (tmp_path / "victim.jsonl").exists()
        row.query_one(".picker-delete", Button).press()
        await pilot.pause()
        assert not (tmp_path / "victim.jsonl").exists()
        assert isinstance(app.screen, SessionPicker)


@pytest.mark.asyncio
async def test_arming_one_row_then_another_disarms_the_first(tmp_path):
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        _session(tmp_path, "a", title="a")
        _session(tmp_path, "b", title="b")
        picker = await _open_picker(app, pilot)
        _row(picker, "a").query_one(".picker-delete", Button).press()
        await pilot.pause()
        _row(picker, "b").query_one(".picker-delete", Button).press()
        await pilot.pause()
        assert not _row(picker, "a").armed and _row(picker, "b").armed
        await pilot.press("escape")                            # withdraws b's question
        await pilot.pause()
        assert not _row(picker, "b").armed and isinstance(app.screen, SessionPicker)
        assert (tmp_path / "a.jsonl").exists() and (tmp_path / "b.jsonl").exists()


@pytest.mark.asyncio
async def test_rename_inline_saves_on_enter(tmp_path):
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        p = _session(tmp_path, "s", title="old name")
        picker = await _open_picker(app, pilot)
        row = _row(picker, "s")
        row.query_one(".picker-rename", Button).press()
        await pilot.pause()
        box = row.query_one(Input)
        assert box.value == "old name" and box.has_focus
        box.value = "new name"
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, SessionPicker)          # Enter saved, did not open
        assert storage.read_session_meta(p)["title"] == "new name"
        assert "new name" in str(row.query_one(".picker-row-title", Label).render())
        assert list(row.query(Input)) == []


@pytest.mark.asyncio
async def test_rename_esc_keeps_the_old_name(tmp_path):
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        p = _session(tmp_path, "s", title="old name")
        picker = await _open_picker(app, pilot)
        picker.query_one("#picker-list", ListView).index = _row_of(picker, "s")
        await pilot.press("r")                                 # keyboard mirror of ✎
        await pilot.pause()
        row = _row(picker, "s")
        row.query_one(Input).value = "typo"
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, SessionPicker)          # Esc ended the rename only
        assert storage.read_session_meta(p)["title"] == "old name"
        assert list(row.query(Input)) == []


@pytest.mark.asyncio
async def test_renaming_the_open_session_updates_the_header(tmp_path):
    from ahacode.widgets.header_bar import HeaderBar

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        current = app.session_path.stem
        picker = await _open_picker(app, pilot)
        row = _row(picker, current)
        row.query_one(".picker-rename", Button).press()
        await pilot.pause()
        row.query_one(Input).value = "renamed here"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("escape")                            # close the picker
        await pilot.pause()
        assert app.query_one(HeaderBar)._title_text == "AhaCode · renamed here"
        assert storage.read_session_meta(app.session_path)["title"] == "renamed here"


@pytest.mark.asyncio
async def test_the_close_button_dismisses_the_picker(tmp_path):
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        picker = await _open_picker(app, pilot)
        current = app.session_path
        picker.query_one("#picker-close", Button).press()
        await pilot.pause()
        assert not isinstance(app.screen, SessionPicker)   # closed
        assert app.session_path == current                 # nothing opened


@pytest.mark.asyncio
async def test_the_close_button_withdraws_a_pending_delete_first(tmp_path):
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        _session(tmp_path, "victim", title="victim")
        picker = await _open_picker(app, pilot)
        row = _row(picker, "victim")
        row.query_one(".picker-delete", Button).press()
        await pilot.pause()
        assert row.armed
        picker.query_one("#picker-close", Button).press()   # first press: withdraw
        await pilot.pause()
        assert not row.armed and isinstance(app.screen, SessionPicker)
        assert (tmp_path / "victim.jsonl").exists()
        picker.query_one("#picker-close", Button).press()   # now it closes
        await pilot.pause()
        assert not isinstance(app.screen, SessionPicker)
