import pytest
from textual.widgets import Input, Select

from ahacode import client, config, storage
from ahacode.app import AhaCodeApp
from ahacode.widgets.chatbox import Chatbox


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    """Isolate all global state (~/.ahacode) from every test.

    Sessions and config both live under the home directory, and the client
    caches its config — every app test must run against private temporaries
    with a fresh client cache.
    """
    monkeypatch.setattr(storage, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    # The model bar fetches /v1/models on mount — keep tests offline.
    monkeypatch.setattr(client, "list_models", lambda: ["qwen38", "qwen3-4b"])
    client.reset()
    yield
    client.reset()


@pytest.fixture
def fake_llm(monkeypatch):
    """Swap the real stream_chat for the offline fake for the duration of a test."""
    monkeypatch.setattr(client, "stream_chat", client.stream_chat_fake)


@pytest.mark.asyncio
async def test_enter_creates_three_bubbles(fake_llm):
    """One Enter → user bubble + pre-mounted thinking and assistant bubbles."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.press("h", "i")
        await pilot.press("enter")
        await pilot.pause()
        assert len(app.query(Chatbox)) == 3


@pytest.mark.asyncio
async def test_deltas_routed_to_right_boxes(fake_llm):
    """Thinking deltas land in the thinking box, text deltas in the answer box."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.press("h", "i")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        user, thinking, response = list(app.query(Chatbox))
        assert thinking.display is True
        assert thinking._content.strip() == client.FAKE_THINKING
        assert response._content.strip() == client.FAKE_RESPONSE


@pytest.mark.asyncio
async def test_thinking_box_stays_hidden_without_thinking(monkeypatch):
    """A model that never thinks must leave the thinking box hidden forever."""
    def text_only(messages):
        yield ("text", "Hi!")
    monkeypatch.setattr(client, "stream_chat", text_only)

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.press("h", "i")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        _, thinking, response = list(app.query(Chatbox))
        assert thinking.display is False  # present in the tree, never shown
        assert response._content == "Hi!"


@pytest.mark.asyncio
async def test_second_turn_carries_history(monkeypatch):
    """The second request must carry the [user, assistant, user] history."""
    captured = []

    def recording_fake(messages):
        captured.append(list(messages))  # spy: record what the client received
        yield ("text", "answer")

    monkeypatch.setattr(client, "stream_chat", recording_fake)

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.press("a")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()  # let ResponseComplete be processed
        await pilot.press("b")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert [m["role"] for m in captured[1]] == ["user", "assistant", "user"]
    assert captured[1][1]["content"] == "answer"  # turn 1's reply rides in turn 2
    assert len(app.session.messages) == 4


@pytest.mark.asyncio
async def test_messages_persisted_to_file(fake_llm, tmp_path):
    """One full turn must be written to the JSONL session file."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.press("h", "i")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    assert [m["role"] for m in storage.load_messages(files[0])] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_slash_command_is_not_a_chat_message():
    """/commands answer locally: no LLM call, nothing recorded in the session."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "/help"
        await pilot.press("enter")
        await pilot.pause()
        boxes = list(app.query(Chatbox))
        assert len(boxes) == 1  # a single system bubble — no user/thinking/answer
        assert "Commands:" in boxes[0]._content
    assert app.session.messages == []


@pytest.mark.asyncio
async def test_model_command_switches_and_persists():
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "/model my-new-model"
        await pilot.press("enter")
        await pilot.pause()
        boxes = list(app.query(Chatbox))
        assert "my-new-model" in boxes[0]._content
    assert config.load().name == "my-new-model"  # persisted to config.toml


@pytest.mark.asyncio
async def test_unknown_command_suggests_help():
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "/nonsense"
        await pilot.press("enter")
        await pilot.pause()
        assert "/help" in list(app.query(Chatbox))[0]._content


@pytest.mark.asyncio
async def test_restore_on_startup(tmp_path):
    """An existing session file must be restored as bubbles on startup."""
    p = storage.new_session_path(base_dir=tmp_path)
    storage.append_message(p, {"role": "user", "content": "earlier message"})
    storage.append_message(p, {"role": "assistant", "content": "earlier reply"})

    app = AhaCodeApp()  # __init__ should discover the file above
    async with app.run_test() as pilot:
        await pilot.pause()
        assert len(app.query(Chatbox)) == 2
        assert app.session.messages[0]["content"] == "earlier message"


@pytest.mark.asyncio
async def test_model_bar_shows_current_model():
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()  # model list fetched
        await pilot.pause()
        assert app.query_one("#model-select", Select).value == config.DEFAULT_MODEL


@pytest.mark.asyncio
async def test_picking_a_model_persists_and_announces():
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one("#model-select", Select).value = "qwen3-4b"  # user picks
        await pilot.pause()
        assert "qwen3-4b" in list(app.query(Chatbox))[-1]._content  # system bubble
    assert config.load().name == "qwen3-4b"  # persisted


@pytest.mark.asyncio
async def test_slash_model_updates_the_bar():
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        app.query_one("#prompt", Input).value = "/model custom-model"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#model-select", Select).value == "custom-model"
