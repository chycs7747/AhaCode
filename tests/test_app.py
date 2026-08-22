import time

import pytest
from textual.widgets import Checkbox, Input, Select

from ahacode import client, config, storage
from ahacode.app import AhaCodeApp
from ahacode.events import TextDelta
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
    """One Enter → user bubble + lazily-mounted thinking and assistant bubbles."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.press("h", "i")
        await pilot.press("enter")
        await app.workers.wait_for_complete()  # bubbles appear as deltas arrive
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
async def test_no_thinking_bubble_when_model_does_not_think(monkeypatch):
    """A model that never thinks mounts no thinking bubble at all (lazy mount)."""
    def text_only(messages, tools=None):
        yield TextDelta("Hi!")
    monkeypatch.setattr(client, "stream_chat", text_only)

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.press("h", "i")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        boxes = list(app.query(Chatbox))
        assert [b for b in boxes if b.has_class("chatbox--thinking")] == []
        assert len(boxes) == 2  # user + answer only
        assert boxes[-1]._content == "Hi!"


@pytest.mark.asyncio
async def test_second_turn_carries_history(monkeypatch):
    """The second request must carry the [user, assistant, user] history."""
    captured = []

    def recording_fake(messages, tools=None):
        captured.append(list(messages))  # spy: record what the client received
        yield TextDelta("answer")

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


@pytest.mark.asyncio
async def test_worker_error_becomes_a_bubble_not_a_crash(monkeypatch):
    """A failing stream must leave the app alive with an error bubble in place."""
    def broken(messages, tools=None):
        yield TextDelta("partial ")
        raise RuntimeError("boom")
    monkeypatch.setattr(client, "stream_chat", broken)

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.press("h", "i")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        answer = list(app.query(Chatbox))[-1]
        assert answer.has_class("chatbox--error")
        assert "boom" in answer._content
        assert not app._exit  # still running
    assert [m["role"] for m in app.session.messages] == ["user"]  # no phantom reply recorded


@pytest.mark.asyncio
async def test_stream_is_closed_when_cancelled(monkeypatch):
    """Sending a new message must close the previous (cancelled) stream."""
    closed = []
    def slow(messages, tools=None):
        try:
            for _ in range(50):
                time.sleep(0.02)
                yield TextDelta("x")
        finally:
            closed.append(True)  # runs on exhaustion OR on close()
    monkeypatch.setattr(client, "stream_chat", slow)

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.press("a")
        await pilot.press("enter")
        await pilot.pause(0.1)
        await pilot.press("b")   # cancels the first worker
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert len(closed) == 2  # both streams released, including the cancelled one


@pytest.mark.asyncio
async def test_ctrl_d_quits():
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.press("ctrl+d")
        await pilot.pause()
        assert app._exit

@pytest.mark.asyncio
async def test_tool_call_turn_renders_and_persists(monkeypatch):
    """A tool-calling turn mounts tool bubbles and persists assistant+tool+assistant."""
    from ahacode.events import ToolCall

    turns = iter([
        [ToolCall(id="c1", name="read", arguments={"path": "README.md"})],  # turn 1: call read
        [TextDelta("done")],                                                # turn 2: final answer
    ])

    def scripted(messages, tools=None):
        return iter(next(turns))

    monkeypatch.setattr(client, "stream_chat", scripted)

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.press("r")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        boxes = list(app.query(Chatbox))
        assert any(b.has_class("chatbox--tool-call") for b in boxes)     # 🔧 request bubble
        assert any(b.has_class("chatbox--tool-result") for b in boxes)   # 📄 output bubble
        assert boxes[-1]._content == "done"                             # final answer bubble

    # The whole turn is persisted in OpenAI message shapes.
    assert [m["role"] for m in app.session.messages] == ["user", "assistant", "tool", "assistant"]
    assert app.session.messages[1]["tool_calls"][0]["function"]["name"] == "read"
    assert "AhaCode" in app.session.messages[2]["content"]  # real README content fed back


async def _wait_for_modal(pilot, app):
    """Give the worker time to reach the tool call and push the approval modal."""
    from ahacode.widgets.approval_modal import ApprovalModal
    for _ in range(150):
        await pilot.pause(0.02)
        if isinstance(app.screen, ApprovalModal):
            return
    raise AssertionError("approval modal never appeared")


@pytest.mark.asyncio
async def test_bash_prompts_and_runs_on_approval(monkeypatch):
    """bash is gated: pressing 'y' runs the command and feeds its output back."""
    from ahacode.events import ToolCall
    turns = iter([
        [ToolCall(id="c1", name="bash", arguments={"command": "echo hi"})],
        [TextDelta("ran it")],
    ])
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter(next(turns)))

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "run echo hi"
        await pilot.press("enter")
        await _wait_for_modal(pilot, app)
        await pilot.press("y")  # approve
        await app.workers.wait_for_complete()
        await pilot.pause()
        results = [b for b in app.query(Chatbox) if b.has_class("chatbox--tool-result")]
        assert results and "hi" in results[-1]._content

    assert [m["role"] for m in app.session.messages] == ["user", "assistant", "tool", "assistant"]
    assert app.session.messages[2]["content"] == "hi"  # echo hi output, fed back


@pytest.mark.asyncio
async def test_bash_skipped_on_denial(monkeypatch):
    """Pressing 'n' skips the command; the model sees a 'denied' result."""
    from ahacode.events import ToolCall
    turns = iter([
        [ToolCall(id="c1", name="bash", arguments={"command": "echo hi"})],
        [TextDelta("ok, skipped")],
    ])
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter(next(turns)))

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "run echo hi"
        await pilot.press("enter")
        await _wait_for_modal(pilot, app)
        await pilot.press("n")  # deny
        await app.workers.wait_for_complete()
        await pilot.pause()
        errors = [b for b in app.query(Chatbox) if b.has_class("chatbox--tool-error")]
        assert errors and "denied" in errors[-1]._content

    assert app.session.messages[2]["content"] == "denied by user"  # command never ran


@pytest.mark.asyncio
async def test_plan_mode_restricts_tools_and_injects_system_prompt(monkeypatch):
    """Plan mode exposes only read-only tools and prepends the plan system prompt."""
    captured = {}

    def rec(messages, tools=None):
        captured["messages"] = list(messages)
        captured["tools"] = tools
        yield TextDelta("here is the plan")

    monkeypatch.setattr(client, "stream_chat", rec)

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        app.query_one("#mode-select", Select).value = "plan"  # toggle via the bar
        await pilot.pause()
        assert app.mode == "plan"
        app.query_one("#prompt", Input).value = "fix the bug"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert {t["function"]["name"] for t in captured["tools"]} == {"read", "todo_write"}  # no bash
    assert captured["messages"][0]["role"] == "system"
    assert "PLAN MODE" in captured["messages"][0]["content"]
    assert app.session.messages[0]["role"] == "user"  # system prompt is not stored


@pytest.mark.asyncio
async def test_act_mode_exposes_all_tools(monkeypatch):
    """The default (act) mode exposes the full tool set and injects no system prompt."""
    captured = {}

    def rec(messages, tools=None):
        captured["messages"] = list(messages)
        captured["tools"] = tools
        yield TextDelta("hi")

    monkeypatch.setattr(client, "stream_chat", rec)

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "hello"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert {t["function"]["name"] for t in captured["tools"]} == {"read", "write", "edit", "bash", "todo_write"}
    assert captured["messages"][0]["role"] == "user"  # no system prompt in act mode


@pytest.mark.asyncio
async def test_escape_stops_the_current_turn(monkeypatch):
    """Pressing escape cancels an in-flight response; no assistant reply is recorded."""
    def slow(messages, tools=None):
        for _ in range(50):
            time.sleep(0.02)
            yield TextDelta("x")

    monkeypatch.setattr(client, "stream_chat", slow)

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "go"
        await pilot.press("enter")
        await pilot.pause(0.1)       # let the stream start
        await pilot.press("escape")  # stop it
        await app.workers.wait_for_complete()
        await pilot.pause()
    # cancelled before completion -> no assistant turn persisted
    assert [m["role"] for m in app.session.messages] == ["user"]


@pytest.mark.asyncio
async def test_status_shows_token_stats_after_turn(monkeypatch):
    """After a turn, the status bar shows the token counts from the usage trailer."""
    from ahacode.events import Usage

    def fake(messages, tools=None):
        yield TextDelta("hello there")
        yield Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120)

    monkeypatch.setattr(client, "stream_chat", fake)

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "hi"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        status = app._last_status
    assert "prompt 100" in status
    assert "gen 20" in status
    assert "tok/s" in status


@pytest.mark.asyncio
async def test_auto_approve_runs_bash_without_a_modal(monkeypatch):
    """With auto-approve on, a bash call runs without the confirmation modal."""
    from ahacode.events import ToolCall
    from ahacode.widgets.approval_modal import ApprovalModal

    turns = iter([
        [ToolCall(id="1", name="bash", arguments={"command": "echo hi"})],
        [TextDelta("done")],
    ])
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter(next(turns)))

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#auto-approve", Checkbox).value = True  # toggle on
        await pilot.pause()
        assert app.auto_approve is True
        app.query_one("#prompt", Input).value = "run echo hi"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not isinstance(app.screen, ApprovalModal)  # never prompted
        results = [b for b in app.query(Chatbox) if b.has_class("chatbox--tool-result")]
        assert results and "hi" in results[-1]._content

    assert app.session.messages[2]["content"] == "hi"  # command actually ran


@pytest.mark.asyncio
async def test_todo_write_updates_pinned_panel_not_a_bubble(monkeypatch):
    """todo_write fills the pinned panel; it does not drop an inline tool bubble."""
    from ahacode.events import ToolCall
    from ahacode.widgets.todo_panel import TodoPanel

    turns = iter([
        [ToolCall(id="1", name="todo_write", arguments={"items": [
            {"content": "step one", "status": "done"},
            {"content": "step two", "status": "pending"},
        ]})],
        [TextDelta("planned")],
    ])
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter(next(turns)))

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "plan it"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        assert panel.display is True
        assert "step one" in panel._content and "step two" in panel._content
        # no inline tool-call / plan bubble was mounted for todo_write
        assert [b for b in app.query(Chatbox) if b.has_class("chatbox--tool-call")] == []


@pytest.mark.asyncio
async def test_todo_panel_marks_complete_when_all_done(monkeypatch):
    from ahacode.events import ToolCall
    from ahacode.widgets.todo_panel import TodoPanel

    turns = iter([
        [ToolCall(id="1", name="todo_write", arguments={"items": [
            {"content": "only step", "status": "done"},
        ]})],
        [TextDelta("done")],
    ])
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter(next(turns)))

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "go"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        assert panel.has_class("todo-panel--done")
        assert "complete" in panel._content


@pytest.mark.asyncio
async def test_bash_approval_via_button_click(monkeypatch):
    """The Run button (not just the y key) approves and runs the command."""
    from ahacode.events import ToolCall

    turns = iter([
        [ToolCall(id="1", name="bash", arguments={"command": "echo hi"})],
        [TextDelta("done")],
    ])
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter(next(turns)))

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "run echo hi"
        await pilot.press("enter")
        await _wait_for_modal(pilot, app)
        await pilot.click("#approve-btn")  # click, don't press a key
        await app.workers.wait_for_complete()
        await pilot.pause()
        results = [b for b in app.query(Chatbox) if b.has_class("chatbox--tool-result")]
        assert results and "hi" in results[-1]._content

    assert app.session.messages[2]["content"] == "hi"


@pytest.mark.asyncio
async def test_write_streams_into_one_bubble(monkeypatch, tmp_path):
    """Streamed ToolCallDeltas fill a single write bubble (path header + content);
    the final ToolCall doesn't add a second bubble."""
    import json
    from ahacode.events import ToolCall, ToolCallDelta

    target = tmp_path / "out.py"
    content = "x = 1\ny = 2"
    args_json = json.dumps({"path": str(target), "content": content})
    frags = [args_json[i:i + 7] for i in range(0, len(args_json), 7)]

    turn1 = [ToolCallDelta(index=0, name="write", fragment=f) for f in frags]
    turn1.append(ToolCall(id="1", name="write", arguments={"path": str(target), "content": content}))
    turns = iter([turn1, [TextDelta("done")]])
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter(next(turns)))

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.auto_approve = True  # write needs approval; skip the modal for the test
        app.query_one("#prompt", Input).value = "write it"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        bubbles = [b for b in app.query(Chatbox) if b.has_class("chatbox--tool-call")]
        assert len(bubbles) == 1                       # not duplicated by the final ToolCall
        assert "write · " in bubbles[0]._content       # clean header, not raw JSON
        assert "x = 1" in bubbles[0]._content          # streamed content, not a label dump

    assert target.read_text(encoding="utf-8") == content  # and it actually wrote


@pytest.mark.asyncio
async def test_edit_renders_colored_diff(monkeypatch, tmp_path):
    """An edit call renders one -/+ diff bubble and actually applies the change."""
    from ahacode.events import ToolCall

    f = tmp_path / "a.py"
    f.write_text("keep\nfor j in range(len(a)):\n    swap(a, j)\nkeep2\n", encoding="utf-8")
    old = "for j in range(len(a)):\n    swap(a, j)"
    new = "for j in range(len(a) - 1 - i):\n    a[j], a[j+1] = a[j+1], a[j]"
    turns = iter([
        [ToolCall(id="1", name="edit", arguments={"path": str(f), "old_string": old, "new_string": new})],
        [TextDelta("done")],
    ])
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter(next(turns)))

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.auto_approve = True
        app.query_one("#prompt", Input).value = "edit it"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        diffs = [b for b in app.query(Chatbox) if b.has_class("chatbox--tool-diff")]
        assert len(diffs) == 1
        c = diffs[0]._content
        assert "- " in c and "+ " in c          # deletions AND additions
        assert "swap(a, j)" in c                 # removed line
        assert "a[j], a[j+1]" in c               # added line

    assert "range(len(a) - 1 - i)" in f.read_text(encoding="utf-8")  # change applied


@pytest.mark.asyncio
async def test_new_session_starts_with_a_header():
    """A fresh session file's first line is a header carrying kind + model."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
    header = storage.read_header(app.session_path)
    assert header is not None
    assert header["type"] == "header"
    assert header["kind"] == "main"
    assert header["model"]  # the configured model was recorded
    assert header["parent_id"] is None
