import json
import time

import pytest
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Checkbox, Select

from ahacode import client, config, storage
from ahacode.app import AhaCodeApp
from ahacode.events import TextDelta, ToolCall
from ahacode.widgets.chatbox import Chatbox
from ahacode.widgets.prompt_input import PromptInput
from ahacode.widgets.todo_panel import TodoPanel


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
    # Auto-title runs after a turn as a background worker that would hit the network;
    # keep it offline + fast. Tests that check the trigger override generate_title.
    monkeypatch.setattr(client, "complete", lambda messages: "")
    monkeypatch.setattr(AhaCodeApp, "generate_title", lambda self, *a, **k: None)
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
async def test_thinking_block_autocollapses_after_answer(fake_llm):
    """The reasoning block stays open while streaming, then folds once the answer
    begins (auto-collapsed). Its content is preserved so a click can reopen it."""
    from ahacode.widgets.thinking import ThinkingBlock

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.press("h", "i")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        block = app.query_one(ThinkingBlock)
        assert block.collapsed is True  # folded after the answer started
        assert block._box._content.strip() == client.FAKE_THINKING


@pytest.mark.asyncio
async def test_assistant_reply_grouped_in_a_turn_rail(fake_llm):
    """A reply's blocks (thinking, answer, …) are grouped in one .turn rail; the
    user message stays outside it, so the turn's flow reads as one group."""
    from textual.containers import Vertical

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.press("h", "i")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        turns = [w for w in app.query(Vertical) if w.has_class("turn")]
        assert len(turns) == 1
        inside = list(turns[0].query(Chatbox))
        assert any(b.has_class("chatbox--assistant") for b in inside)  # answer in the rail
        assert not any(b.has_class("chatbox--user") for b in inside)   # user stays outside


@pytest.mark.asyncio
async def test_restored_bash_is_one_card_titled_with_the_command(fake_llm):
    """A reloaded session shows bash as one card titled with the command — no raw
    tool-call bubble and no bare '58 lines' with the command missing."""
    from ahacode.widgets.tool_result import ToolResultBlock

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.session.messages = [
            {"role": "user", "content": "check status"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "git status --short"}'},
            }]},
            {"role": "tool", "tool_call_id": "c1",
             "content": "\n".join(f"line {i}" for i in range(20))},
            {"role": "assistant", "content": "done"},
        ]
        await app._render_history()
        await pilot.pause()
        assert [b for b in app.query(Chatbox) if b.has_class("chatbox--tool-call")] == []
        cards = list(app.query(ToolResultBlock))
        assert any("git status --short" in c.title for c in cards)  # command shown, not "20 lines"


@pytest.mark.asyncio
async def test_bracketed_content_renders_without_markup_crash(fake_llm):
    """Regression: model/tool text with '[' — file dumps (list[dict], arr[0]),
    JSON, markdown links — must render literally, not be parsed as Rich markup.
    This crashed the app with MarkupError the moment a tool returned Python source
    (run_test re-raises app._exception on exit, so a markup crash fails here)."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        container = app.query_one("#chat-container")
        crash = 'def f(x: list[dict]): return x[0]  # [name="read"]'
        box = Chatbox(crash, role="tool-result")
        await container.mount(box)
        await pilot.pause()
        assert app._exception is None       # no render crash (was MarkupError)
        assert box._render_markup is False  # content treated literally
        assert box._content == crash


@pytest.mark.asyncio
async def test_tool_result_block_folds_by_size_and_error(fake_llm):
    """Tool results render as foldable cards: a short success stays open, a long
    result and any error start collapsed (the reference long-output auto-collapse
    + failure-collapses pattern). The inner bubble keeps its role class + content."""
    from ahacode.widgets.tool_result import ToolResultBlock

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        c = app.query_one("#chat-container")
        short = ToolResultBlock("bash", "hi", is_error=False)
        long = ToolResultBlock("read", "\n".join(f"line {i}" for i in range(40)))
        err = ToolResultBlock("bash", "boom", is_error=True)
        await c.mount(short)
        await c.mount(long)
        await c.mount(err)
        await pilot.pause()
        assert short.collapsed is False           # short success stays open
        assert long.collapsed is True             # long output folds away
        assert err.collapsed is True              # failures fold to the header
        assert err.has_class("tool-block--error")
        assert long._box.has_class("chatbox--tool-result")
        assert app._exception is None             # rendered without a crash


@pytest.mark.asyncio
async def test_assistant_answer_renders_as_markdown(monkeypatch):
    """Assistant answers render as Markdown so ```code``` / ```diff fences become
    highlighted blocks (Rich's Markdown), not literal backticks. The raw markdown
    is still kept in _content for logic/tests."""
    from rich.markdown import Markdown as RichMarkdown

    def with_code(messages, tools=None):
        yield TextDelta("Here you go:\n\n```python\nprint('hi')\n```\n")

    monkeypatch.setattr(client, "stream_chat", with_code)
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", PromptInput).text = "show code"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        answer = [b for b in app.query(Chatbox) if b.has_class("chatbox--assistant")][-1]
        assert "```python" in answer._content            # raw markdown kept
        assert isinstance(answer.render(), RichMarkdown)  # rendered as markdown
        assert app._exception is None


@pytest.mark.asyncio
async def test_prompt_enter_sends_and_shift_enter_newlines(fake_llm):
    """Enter submits the prompt; Shift+Enter inserts a newline (multi-line input).
    The multi-line text becomes one user turn and the box clears on send."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        await pilot.press("a")
        await pilot.press("shift+enter")     # newline, stays in the box
        await pilot.press("b")
        assert prompt.text == "a\nb"
        await pilot.press("enter")            # now send
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert prompt.text == ""              # cleared on submit
        users = [b for b in app.query(Chatbox) if b.has_class("chatbox--user")]
        assert users and users[-1]._content == "a\nb"


@pytest.mark.asyncio
async def test_send_button_submits_the_prompt(fake_llm):
    """Clicking the composer's Send button submits the prompt, like Enter."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.text = "hello there"
        await pilot.click("#send-btn")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert prompt.text == ""  # cleared on send
        users = [b for b in app.query(Chatbox) if b.has_class("chatbox--user")]
        assert users and users[-1]._content == "hello there"


@pytest.mark.asyncio
async def test_header_new_button_starts_new_session(fake_llm):
    """The header's New button clears the chat and starts a fresh session."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", PromptInput).text = "hi"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.session.messages  # a turn happened
        await pilot.click("#new-session-btn")
        await pilot.pause()
        assert app.session.messages == []  # fresh session


@pytest.mark.asyncio
async def test_header_sessions_button_opens_picker():
    """The header's Sessions button opens the SessionPicker modal."""
    from ahacode.widgets.session_picker import SessionPicker

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.click("#open-sessions-btn")
        await pilot.pause()
        assert isinstance(app.screen, SessionPicker)


@pytest.mark.asyncio
async def test_header_bar_shows_session_title(fake_llm):
    """The top bar reflects the session title set on it."""
    from ahacode.widgets.header_bar import HeaderBar

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app._set_header_title("Quicksort bench")
        await pilot.pause()
        assert "Quicksort bench" in app.query_one(HeaderBar)._title_text


@pytest.mark.asyncio
async def test_endpoint_shows_in_header_not_the_composer(fake_llm):
    """The endpoint moved to the header bar; the composer footer no longer has it."""
    from ahacode.widgets.header_bar import HeaderBar
    from ahacode.widgets.model_bar import ModelBar

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        shown = app.query_one(HeaderBar)._endpoint_text
        assert "127.0.0.1:" in shown       # compact host:port
        assert "http://" not in shown and not shown.endswith("/v1")  # scheme/suffix stripped
        assert not app.query_one(ModelBar).query("#endpoint")        # gone from the footer


@pytest.mark.asyncio
async def test_send_button_becomes_stop_while_streaming(monkeypatch):
    """While a turn streams the Send button reads Stop; clicking it cancels."""
    from textual.widgets import Button

    def slow(messages, tools=None):
        for _ in range(50):
            time.sleep(0.02)
            yield TextDelta("x")

    monkeypatch.setattr(client, "stream_chat", slow)
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", PromptInput).text = "go"
        await pilot.press("enter")
        btn = app.query_one("#send-btn", Button)
        for _ in range(50):  # poll instead of a fixed sleep (robust under load)
            if "Stop" in str(btn.label):
                break
            await pilot.pause(0.02)
        assert "Stop" in str(btn.label)
        await pilot.click("#send-btn")  # the button doubles as Stop
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "Send" in str(btn.label)  # reverted once cancelled


@pytest.mark.asyncio
async def test_model_select_has_no_blank_placeholder(fake_llm):
    """The model picker must not offer an empty 'model' entry (allow_blank=False)."""
    from textual.widgets import Select

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.query_one("#model-select", Select)._allow_blank is False


@pytest.mark.asyncio
async def test_dropdown_button_toggles_open_and_closed(fake_llm):
    """Clicking the dropdown button again closes the open menu (Textual's default
    would reopen it). Driven via the internal messages since Pilot can't click an
    open overlay reliably."""
    from textual.widgets import Select
    from textual.widgets._select import SelectCurrent, SelectOverlay

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        s = app.query_one("#model-select", Select)

        async def settle():
            for _ in range(3):
                await pilot.pause()

        s.post_message(SelectCurrent.Toggle())
        await settle()
        assert s.expanded is True  # opened
        # a click on the button while open = overlay blur (Dismiss) then Toggle
        s.post_message(SelectOverlay.Dismiss(lost_focus=True))
        s.post_message(SelectCurrent.Toggle())
        await settle()
        assert s.expanded is False  # closed, not reopened
        s.post_message(SelectCurrent.Toggle())
        await settle()
        assert s.expanded is True  # a later click reopens normally


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

    # act mode prepends the system prompt; the [user, assistant, user] history follows.
    assert [m["role"] for m in captured[1]] == ["system", "user", "assistant", "user"]
    assert captured[1][2]["content"] == "answer"  # turn 1's reply rides in turn 2
    assert len(app.session.messages) == 4  # system prompt is not stored in the session


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
        app.query_one("#prompt", PromptInput).text = "/help"
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
        app.query_one("#prompt", PromptInput).text = "/model my-new-model"
        await pilot.press("enter")
        await pilot.pause()
        boxes = list(app.query(Chatbox))
        assert "my-new-model" in boxes[0]._content
    assert config.load().name == "my-new-model"  # persisted to config.toml


@pytest.mark.asyncio
async def test_unknown_command_suggests_help():
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", PromptInput).text = "/nonsense"
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
        app.query_one("#prompt", PromptInput).text = "/model custom-model"
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
    """Stopping a turn must release its stream (close the generator), and the next
    message after that gets a fresh one."""
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
        await pilot.press("escape")   # stop the first turn
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(closed) == 1       # the cancelled stream was released
        await pilot.press("b")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert len(closed) == 2  # both streams released


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
        from ahacode.widgets.tool_result import ToolResultBlock

        boxes = list(app.query(Chatbox))
        cards = list(app.query(ToolResultBlock))
        assert any("read" in c.title for c in cards)                     # tool + input in the card title (IN)
        assert any(b.has_class("chatbox--tool-result") for b in boxes)   # output (OUT) bubble
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
        app.query_one("#prompt", PromptInput).text = "run echo hi"
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
        app.query_one("#prompt", PromptInput).text = "run echo hi"
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
        app.query_one("#prompt", PromptInput).text = "fix the bug"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    # read-only: the search tools are in (planning means investigating), bash is not
    assert {t["function"]["name"] for t in captured["tools"]} == {
        "read", "glob", "grep", "plan_submit"
    }
    assert captured["messages"][0]["role"] == "system"
    assert "PLAN MODE" in captured["messages"][0]["content"]
    assert app.session.messages[0]["role"] == "user"  # system prompt is not stored


@pytest.mark.asyncio
async def test_act_mode_exposes_all_tools(monkeypatch):
    """The default (act) mode exposes the full tool set and grounds the turn with the
    AhaCode system prompt."""
    captured = {}

    def rec(messages, tools=None):
        captured["messages"] = list(messages)
        captured["tools"] = tools
        yield TextDelta("hi")

    monkeypatch.setattr(client, "stream_chat", rec)

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", PromptInput).text = "hello"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    # act mode at depth 0 also offers `task` (the main agent may spawn sub-agents;
    # subagent_depth defaults to 1, so depth 0 < 1 exposes it).
    assert {t["function"]["name"] for t in captured["tools"]} == {
        "read", "glob", "grep", "write", "edit", "bash", "todo_write", "task"
    }
    # act mode now grounds the turn with the AhaCode system prompt (env-injected),
    # followed by the user's message; the system prompt is never stored in the session.
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][0]["content"].startswith("You are AhaCode")
    assert captured["messages"][1]["role"] == "user"
    assert app.session.messages[0]["role"] == "user"


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
        app.query_one("#prompt", PromptInput).text = "go"
        await pilot.press("enter")
        for _ in range(50):  # wait until the worker is actually running (robust under load)
            w = getattr(app, "_response_worker", None)
            if w is not None and w.is_running:
                break
            await pilot.pause(0.02)
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
        app.query_one("#prompt", PromptInput).text = "hi"
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
        app.query_one("#prompt", PromptInput).text = "run echo hi"
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
        app.query_one("#prompt", PromptInput).text = "plan it"
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
        app.query_one("#prompt", PromptInput).text = "go"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        assert panel.has_class("todo-panel--done")
        assert "complete" in panel._content


@pytest.mark.asyncio
async def test_new_session_clears_the_plan(monkeypatch):
    """/new empties the pinned plan — a hidden-but-stale list would misdescribe the
    fresh session."""
    from ahacode.events import ToolCall
    from ahacode.widgets.todo_panel import TodoPanel

    turns = iter([
        [ToolCall(id="1", name="todo_write", arguments={"items": [
            {"content": "old step", "status": "pending"},
        ]})],
        [TextDelta("planned")],
    ])
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter(next(turns)))

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", PromptInput).text = "plan it"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.query_one(TodoPanel).items  # the plan is there

        app.query_one("#prompt", PromptInput).text = "/new"
        await pilot.press("enter")
        await pilot.pause()

        panel = app.query_one(TodoPanel)
        assert panel.items == []      # not merely hidden — actually empty
        assert panel.display is False

        # ...and approving in the fresh session finds no plan file to hand off.
        await app._start_impl_session()
        await pilot.pause()
        assert "실행할 계획이 없어요" in list(app.query(Chatbox))[-1]._content


@pytest.mark.asyncio
async def test_switching_sessions_restores_each_ones_plan(monkeypatch):
    """The panel is a view of the OPEN session: switching away hides the plan,
    switching back replays it from the stored todo_write call."""
    from ahacode.events import ToolCall
    from ahacode.widgets.todo_panel import TodoPanel

    turns = iter([
        [ToolCall(id="1", name="todo_write", arguments={"items": [
            {"content": "session one step", "status": "pending"},
        ]})],
        [TextDelta("planned")],
    ])
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter(next(turns)))

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", PromptInput).text = "plan it"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        first = app.session_path

        app.query_one("#prompt", PromptInput).text = "/new"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one(TodoPanel).items == []

        await app._switch_session(first.stem)  # back to the planning session
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        assert [it["content"] for it in panel.items] == ["session one step"]
        assert panel.display is True


@pytest.mark.asyncio
async def test_restored_todo_write_goes_to_the_panel_not_a_card(monkeypatch):
    """A reloaded session renders todo_write exactly like the live turn did:
    the pinned panel, not a tool-result card (and no empty turn rail left behind)."""
    from textual.containers import Vertical

    from ahacode.widgets.todo_panel import TodoPanel
    from ahacode.widgets.tool_result import ToolResultBlock

    path = storage.new_session_path()
    storage.write_header(path, storage.make_header(path.stem, kind="main"))
    for msg in [
        {"role": "user", "content": "plan it"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "1", "type": "function",
            "function": {"name": "todo_write", "arguments": json.dumps({"items": [
                {"content": "restored step", "status": "in_progress"},
            ]})},
        }]},
        {"role": "tool", "tool_call_id": "1", "content": "▶ restored step"},
        {"role": "assistant", "content": "here is the plan"},
    ]:
        storage.append_message(path, msg)

    app = AhaCodeApp()  # startup resumes the newest main session
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        assert [it["content"] for it in panel.items] == ["restored step"]
        assert panel.display is True
        # no tool-result card for it, and no bare rail where its turn would have been
        assert [b for b in app.query(ToolResultBlock) if "todo_write" in b.title] == []
        assert all(rail.children for rail in app.query(".turn").results(Vertical))


@pytest.mark.asyncio
async def test_context_compaction_tells_the_user(monkeypatch):
    """Silently losing history is the bad failure mode — a compacted turn drops a
    neutral system bubble, and the answer that follows opens a fresh bubble below it."""
    from dataclasses import replace

    from ahacode import context
    from ahacode.events import Usage

    monkeypatch.setattr(
        config, "load",
        lambda *a, **k: replace(config.DEFAULTS, context_window=100, keep_recent_messages=2),
    )
    monkeypatch.setattr(context, "llm_summarize", lambda msgs: "they agreed on utf-8")
    turns = iter([
        [TextDelta("first"), Usage(10, 1, 11)],             # a small prompt: no action
        [TextDelta("second"), Usage(9_999, 1, 10_000)],     # the server reports a huge one
        [TextDelta("third")],                               # ...so THIS turn condenses first
    ])
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter(next(turns)))

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        for text in ("one", "two", "three"):
            app.query_one("#prompt", PromptInput).text = text
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

        notices = [b for b in app.query(Chatbox)
                   if b.has_class("chatbox--system") and "압축" in b._content]
        assert len(notices) == 1
        # the real messages were still persisted — the summary lives only in flight
        assert [m["role"] for m in app.session.messages] == [
            "user", "assistant", "user", "assistant", "user", "assistant"
        ]
        assert app.session.messages[0]["content"] == "one"    # not replaced by a summary
        assert app.session.messages[-1]["content"] == "third"
        assert storage.load_messages(app.session_path)[0]["content"] == "one"  # nor on disk


@pytest.mark.asyncio
async def test_allow_rule_runs_bash_without_a_modal(monkeypatch):
    """A pre-approved command never reaches the approval screen."""
    from dataclasses import replace

    from ahacode.events import ToolCall

    monkeypatch.setattr(
        config, "load",
        lambda *a, **k: replace(config.DEFAULTS, allow_rules=("bash:echo *",)),
    )
    turns = iter([
        [ToolCall(id="1", name="bash", arguments={"command": "echo hi"})],
        [TextDelta("done")],
    ])
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter(next(turns)))

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", PromptInput).text = "run it"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.screen is app.screen_stack[0]      # no modal was ever pushed
        assert any("hi" in (m.get("content") or "") for m in app.session.messages)


@pytest.mark.asyncio
async def test_a_command_outside_the_rules_still_asks(monkeypatch):
    from dataclasses import replace

    from ahacode.events import ToolCall
    from ahacode.widgets.approval_modal import ApprovalModal

    monkeypatch.setattr(
        config, "load",
        lambda *a, **k: replace(config.DEFAULTS, allow_rules=("bash:echo *",)),
    )
    turns = iter([
        [ToolCall(id="1", name="bash", arguments={"command": "rm build.log"})],
        [TextDelta("done")],
    ])
    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter(next(turns)))

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", PromptInput).text = "run it"
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, ApprovalModal)
        await pilot.press("n")
        await app.workers.wait_for_complete()
        await pilot.pause()


@pytest.mark.asyncio
async def test_parallel_subagents_need_no_modals_when_pre_approved(monkeypatch):
    """The reason the rules exist: only one modal can be on screen, so N parallel
    verifiers each needing approval would queue behind each other. Pre-approved,
    they run with zero dialogs."""
    from dataclasses import replace

    from ahacode.events import ToolCall

    monkeypatch.setattr(
        config, "load",
        lambda *a, **k: replace(config.DEFAULTS, allow_rules=("task", "bash:echo *")),
    )
    fan_out = [
        ToolCall(id=f"t{i}", name="task",
                 arguments={"description": f"verify {i}", "prompt": f"check {i}"})
        for i in range(3)
    ]
    # These bash commands REALLY run — that is the point of the test. They must be
    # trivial and must never re-enter this suite: `pytest` here made each sub-agent
    # re-run every test, this one included, three at a time (3^n).
    turns = iter([
        fan_out,
        *[[ToolCall(id="b", name="bash",
                  arguments={"command": f"echo verified {i}"})] for i in range(3)],
        *[[TextDelta(f"verified {i}")] for i in range(3)],
        [TextDelta("all verified")],
    ])
    lock = __import__("threading").Lock()

    def stream(messages, tools=None):
        with lock:
            events = next(turns)
        yield from events

    monkeypatch.setattr(client, "stream_chat", stream)

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", PromptInput).text = "검증해줘"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.screen is app.screen_stack[0]   # not a single dialog
        assert app.auto_approve is False           # and auto-approve was never needed


@pytest.mark.asyncio
async def test_allow_command_lists_and_adds(monkeypatch):
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", PromptInput).text = "/allow"
        await pilot.press("enter")
        await pilot.pause()
        assert "no allow rules" in list(app.query(Chatbox))[-1]._content

        app.query_one("#prompt", PromptInput).text = "/allow bash:git status*"
        await pilot.press("enter")
        await pilot.pause()
        assert "bash:git status*" in list(app.query(Chatbox))[-1]._content
        assert config.load().allow_rules == ("bash:git status*",)  # persisted

        app.query_one("#prompt", PromptInput).text = "/allow"
        await pilot.press("enter")
        await pilot.pause()
        assert "bash:git status*" in list(app.query(Chatbox))[-1]._content


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
        app.query_one("#prompt", PromptInput).text = "run echo hi"
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
        app.query_one("#prompt", PromptInput).text = "write it"
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
        app.query_one("#prompt", PromptInput).text = "edit it"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        diffs = [b for b in app.query(Chatbox) if b.has_class("chatbox--tool-diff")]
        assert len(diffs) == 1
        box = diffs[0]
        c = box._content
        assert "- " in c and "+ " in c          # deletions AND additions
        assert "swap(a, j)" in c                 # removed line
        assert "a[j], a[j+1]" in c               # added line
        assert "edit" in str(box.border_title)   # path header in the card title
        assert "+" in str(box.border_subtitle)   # +N −M count chip

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


@pytest.mark.asyncio
async def test_new_command_starts_a_fresh_session(fake_llm):
    """/new switches to a new session file and clears the chat."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.press("h", "i")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        first = app.session_path
        assert len(app.session.messages) == 2

        app.query_one("#prompt", PromptInput).text = "/new"
        await pilot.press("enter")
        await pilot.pause()

        assert app.session_path != first          # a different file
        assert app.session.messages == []          # fresh state
        assert storage.read_header(app.session_path)["kind"] == "main"
        boxes = list(app.query(Chatbox))
        assert len(boxes) == 1 and boxes[0].has_class("chatbox--system")  # only "new session"


@pytest.mark.asyncio
async def test_sessions_picker_switches_back(fake_llm):
    """/sessions opens the picker; choosing a session loads its history."""
    from ahacode.widgets.session_picker import SessionPicker

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.press("a")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        s1 = app.session_path

        app.query_one("#prompt", PromptInput).text = "/new"
        await pilot.press("enter")
        await pilot.pause()
        assert app.session_path != s1

        app.query_one("#prompt", PromptInput).text = "/sessions"
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, SessionPicker)
        app.screen.dismiss(s1.stem)                 # choose session 1
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert app.session_path.stem == s1.stem
    assert [m["role"] for m in app.session.messages] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_auto_title_triggered_once_per_session(monkeypatch, fake_llm):
    """The first assistant reply of an untitled session kicks off titling exactly once."""
    calls = []
    monkeypatch.setattr(AhaCodeApp, "generate_title", lambda self, msgs, path: calls.append(path))

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.press("h", "i")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(calls) == 1                 # titled after the first reply

        await pilot.press("y", "o")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(calls) == 1                 # already titled -> not triggered again


@pytest.mark.asyncio
async def test_task_tool_spawns_subagent(monkeypatch):
    """The model calling `task` spawns a child agent: its flow renders in a nested
    SubagentCard, it gets its own session file nested under the parent, and its
    result is injected back into the parent transcript."""
    from ahacode import tools
    from ahacode.events import ToolCall
    from ahacode.widgets.subagent_card import SubagentCard

    calls = {"n": 0}

    def scripted(messages, tools=None):
        calls["n"] += 1
        if calls["n"] == 1:  # parent turn 1: delegate via task
            yield ToolCall(id="t1", name="task",
                           arguments={"prompt": "investigate X", "description": "probe"})
        elif calls["n"] == 2:  # child turn 1: the sub-agent answers
            yield TextDelta("sub-agent finding")
        else:  # parent turn 2: final answer
            yield TextDelta("all done")

    monkeypatch.setattr(client, "stream_chat", scripted)

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.auto_approve = True  # skip the spawn-approval modal
        app.query_one("#prompt", PromptInput).text = "delegate please"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        # 1) a nested sub-agent card was mounted, then folded (✓ chip) on completion
        assert len(app.query(SubagentCard)) == 1
        card = app.query_one(SubagentCard)
        assert card.collapsed and "✓" in card.title

        # 2) the child ran as its own session file, nested under the parent
        headers = [storage.read_header(p) for p in storage.SESSIONS_DIR.glob("*.jsonl")]
        subs = [h for h in headers if h and h.get("kind") == "subagent"]
        assert len(subs) == 1
        assert subs[0]["depth"] == 1 and subs[0]["parent_id"] is not None
        assert subs[0]["relation"] == "delegate"   # a task fan-out — the ⑂ tree edge

        # 3) at depth 1 (== subagent_depth) the child has no task tool -> cannot recurse
        assert "task" not in tools.registry_for(subs[0]["depth"], config.load().subagent_depth)

        # 4) the child's result was injected back into the parent transcript
        assert any(
            m.get("role") == "tool" and "sub-agent finding" in m.get("content", "")
            for m in app.session.messages
        )


def test_gate_caps_concurrent_requests(monkeypatch):
    """The global gate bounds concurrent gateway requests to max_parallel_agents, no
    matter how many callers — this is what stops a parallel sub-agent fan-out from
    swamping the single-GPU backend. Held per-request, so it never deadlocks."""
    import threading as _t
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace

    config.save(replace(config.DEFAULTS, max_parallel_agents=2))
    client.reset()

    barrier = _t.Barrier(2, timeout=3)  # exactly 2 must be admitted together to pass
    counter, peak, lock, broke = [0], [0], _t.Lock(), []

    class FakeStream:
        def __enter__(self):
            with lock:
                counter[0] += 1
                peak[0] = max(peak[0], counter[0])
            try:
                barrier.wait()
            except _t.BrokenBarrierError:
                broke.append(1)
            return iter([])  # no chunks -> _iter_events yields nothing

        def __exit__(self, *a):
            with lock:
                counter[0] -= 1
            return False

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return FakeStream()

    monkeypatch.setattr(client, "_ensure_client", lambda: (FakeClient(), config.load()))

    def one(_):
        list(client.stream_chat([{"role": "user", "content": "x"}]))

    with ThreadPoolExecutor(max_workers=4) as ex:  # two full batches of 2
        list(ex.map(one, range(4)))

    assert not broke        # the gate admitted 2 at once (lower bound)
    assert peak[0] == 2     # and never more than 2 (upper bound) — the cap holds
    assert counter[0] == 0  # every permit was released


@pytest.mark.asyncio
async def test_parallel_task_fanout(monkeypatch):
    """Two task calls in one turn spawn two sub-agents concurrently: two nested
    cards, two child sessions, both results injected in call order. The fake stream
    is hit from parallel threads, so it branches on message content (thread-safe),
    not a shared counter."""
    from ahacode.events import ToolCall
    from ahacode.widgets.subagent_card import SubagentCard

    def scripted(messages, tools=None):
        # a child: its history opens with the sub-agent system prompt
        if messages and messages[0].get("role") == "system" and "sub-agent" in messages[0]["content"]:
            yield TextDelta("child result")
            return
        # the parent, after both delegations have returned -> final answer
        if any(m.get("role") == "tool" for m in messages):
            yield TextDelta("all done")
            return
        # the parent's first turn: fan out to two sub-agents at once
        yield ToolCall(id="t1", name="task", arguments={"prompt": "A", "description": "a"})
        yield ToolCall(id="t2", name="task", arguments={"prompt": "B", "description": "b"})

    monkeypatch.setattr(client, "stream_chat", scripted)

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.auto_approve = True
        app.query_one("#prompt", PromptInput).text = "fan out"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert len(app.query(SubagentCard)) == 2  # two nested cards
        subs = [h for h in (storage.read_header(p) for p in storage.SESSIONS_DIR.glob("*.jsonl"))
                if h and h.get("kind") == "subagent"]
        assert len(subs) == 2
        # both results injected in call order (t1 then t2), matching the tool_calls
        tool_ids = [m["tool_call_id"] for m in app.session.messages if m.get("role") == "tool"]
        assert tool_ids == ["t1", "t2"]


@pytest.mark.asyncio
async def test_parallel_tasks_actually_overlap_in_time(monkeypatch):
    """Not just spawned together — genuinely concurrent: each child holds a barrier
    that only releases once BOTH are inside it, so the turn can only finish if the
    two sub-agent loops ran at the same time."""
    import threading

    from ahacode.events import ToolCall
    from ahacode.widgets.subagent_card import SubagentCard

    barrier = threading.Barrier(2, timeout=3)
    both_in = []

    def scripted(messages, tools=None):
        if messages and messages[0].get("role") == "system" and "sub-agent" in messages[0]["content"]:
            try:
                barrier.wait()          # blocks until the SECOND child arrives
                both_in.append(True)
            except threading.BrokenBarrierError:
                pass
            return iter([TextDelta("child result")])
        if any(m.get("role") == "tool" for m in messages):
            return iter([TextDelta("all done")])
        return iter([
            ToolCall(id="t1", name="task", arguments={"prompt": "A", "description": "a"}),
            ToolCall(id="t2", name="task", arguments={"prompt": "B", "description": "b"}),
        ])

    monkeypatch.setattr(client, "stream_chat", scripted)
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.auto_approve = True
        app.query_one("#prompt", PromptInput).text = "fan out"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(both_in) == 2                       # the barrier released → both ran together
        assert app.session.messages[-1]["content"] == "all done"


@pytest.mark.asyncio
async def test_grandchild_nests_under_child(monkeypatch):
    """With subagent_depth=2 a sub-agent can spawn its OWN sub-agent: the grandchild
    records depth=2 and parents to the CHILD (not the main session), its card nests
    inside the child's card, and the depth gate still stops a great-grandchild."""
    from dataclasses import replace
    from ahacode.events import ToolCall
    from ahacode.widgets.subagent_card import SubagentCard

    config.save(replace(config.DEFAULTS, subagent_depth=2))
    client.reset()

    def scripted(messages, tools=None):
        is_child = bool(messages) and messages[0].get("role") == "system" \
            and "sub-agent" in messages[0].get("content", "")
        user = messages[1].get("content", "") if len(messages) > 1 else ""
        has_result = any(m.get("role") == "tool" for m in messages)
        if is_child:
            if "LEAF" in user:            # the grandchild: just answer
                yield TextDelta("leaf result")
            elif has_result:              # the child, after its grandchild returned
                yield TextDelta("child done")
            else:                         # the child's first turn: spawn a grandchild
                yield ToolCall(id="g1", name="task",
                               arguments={"prompt": "LEAF: answer", "description": "grand"})
            return
        # the main agent
        if has_result:
            yield TextDelta("main done")
        else:
            yield ToolCall(id="c1", name="task",
                           arguments={"prompt": "spawn a grandchild", "description": "child"})

    monkeypatch.setattr(client, "stream_chat", scripted)

    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.auto_approve = True
        app.query_one("#prompt", PromptInput).text = "go deep"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        main_id = app.session_path.stem
        headers = [h for h in (storage.read_header(p) for p in storage.SESSIONS_DIR.glob("*.jsonl")) if h]
        child = next(h for h in headers if h["depth"] == 1)
        grand = next(h for h in headers if h["depth"] == 2)
        # the tree threads correctly: child -> main, grandchild -> child (NOT main)
        assert child["parent_id"] == main_id
        assert grand["parent_id"] == child["id"] and grand["parent_id"] != main_id
        # the depth gate stopped a great-grandchild (depth 2 == limit -> no task tool)
        assert not any(h["depth"] >= 3 for h in headers)
        # the cards nest too: the grandchild's card lives inside the child's card
        cards = app.query(SubagentCard)
        assert len(cards) == 2
        child_card = next(c for c in cards if "child" in c.title)
        assert len(child_card.query(SubagentCard)) == 1


def _seed_main_and_subagent():
    """Write a drivable main session and a view-only sub-agent child of it.
    Returns (main_stem, child_stem)."""
    main = storage.new_session_path()
    storage.write_header(main, storage.make_header(main.stem, kind="main", model="qwen38"))
    child = storage.new_session_path()
    storage.write_header(child, storage.make_header(
        child.stem, parent_id=main.stem, kind="subagent", depth=1, model="qwen38", title="probe"))
    storage.append_message(child, {"role": "assistant", "content": "child finding"})
    return main.stem, child.stem


@pytest.mark.asyncio
async def test_subagent_session_opens_read_only(fake_llm):
    """Opening a sub-agent (depth>0) session shows it read-only: view_only is set, its
    history renders (열람), a 🔒 banner appears, and a typed message is refused — not
    recorded and no LLM turn — the fix for sitting in a depth-gated child and stalling."""
    _, child_stem = _seed_main_and_subagent()
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await app._switch_session(child_stem)
        await pilot.pause()

        assert app.view_only is True
        # 열람 works: the child's transcript is on screen
        assert any("child finding" in b._content for b in app.query(Chatbox))
        before = list(app.session.messages)

        # a plain message is refused: nothing appended, and a 🔒 banner explains why
        app.query_one("#prompt", PromptInput).text = "keep working"
        await pilot.press("enter")
        await pilot.pause()
        assert app.session.messages == before  # the guard blocked the turn
        assert any("보기 전용" in b._content for b in app.query(Chatbox))


@pytest.mark.asyncio
async def test_leaving_view_only_restores_driving(fake_llm):
    """The escape hatch works: even in a view-only session /new is a slash command, so
    the guard lets it through — it opens a fresh main session that is drivable again."""
    _, child_stem = _seed_main_and_subagent()
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await app._switch_session(child_stem)
        await pilot.pause()
        assert app.view_only is True

        app.query_one("#prompt", PromptInput).text = "/new"
        await pilot.press("enter")
        await pilot.pause()
        assert app.view_only is False and app.session_depth == 0

        # and a normal message now goes through (recorded as a user turn)
        app.query_one("#prompt", PromptInput).text = "hello"
        await pilot.press("enter")
        await pilot.pause()
        assert any(m.get("role") == "user" and m["content"] == "hello"
                   for m in app.session.messages)


@pytest.mark.asyncio
async def test_think_command_sets_budget(fake_llm):
    """/think <n> persists the per-turn thinking budget; /think off = unbounded."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", PromptInput).text = "/think 2048"
        await pilot.press("enter")
        await pilot.pause()
        assert config.load().thinking_token_budget == 2048

        app.query_one("#prompt", PromptInput).text = "/think off"
        await pilot.press("enter")
        await pilot.pause()
        assert config.load().thinking_token_budget == 0

        # bare /think reports current state (no change)
        app.query_one("#prompt", PromptInput).text = "/think"
        await pilot.press("enter")
        await pilot.pause()
        assert "unbounded" in list(app.query(Chatbox))[-1]._content


@pytest.mark.asyncio
async def test_todo_panel_does_not_cover_the_header():
    """The pinned plan must sit BELOW the header, not overlap its session buttons
    (two widgets docked to the same edge overlap in this Textual build)."""
    from ahacode.widgets.header_bar import HeaderBar
    from ahacode.widgets.todo_panel import TodoPanel

    app = AhaCodeApp()
    async with app.run_test(size=(100, 40)) as pilot:
        app.query_one(TodoPanel).update_todos([{"content": "a step", "status": "pending"}])
        await pilot.pause()
        header, panel = app.query_one(HeaderBar), app.query_one(TodoPanel)
        assert panel.region.y >= header.region.y + header.region.height  # stacked, not on top


@pytest.mark.asyncio
async def test_manual_scroll_up_during_stream_sticks():
    """Scrolling up mid-stream turns following OFF and keeps it off, so new chunks
    don't yank the view back to the bottom (the reported bug)."""
    from textual.containers import Vertical, VerticalScroll
    from ahacode.events import ThinkingDelta

    app = AhaCodeApp()
    # Window must be tall enough that the docked header + composer leave the chat a
    # real, scrollable height (a tiny window collapses it to 0 and nothing scrolls).
    async with app.run_test(size=(100, 30)) as pilot:
        for i in range(50):                       # enough content to scroll
            await app._say_system(f"filler {i}")
        await pilot.pause()                       # let the layout settle before scrolling
        sc = app.query_one("#chat-container", VerticalScroll)
        sc.scroll_end(animate=False)              # pin to the bottom
        await pilot.pause()
        assert sc.max_scroll_y > 0
        assert app._follow_output is True         # at the bottom → following

        sc.scroll_y = 0                           # user scrolls up
        await pilot.pause()
        assert sc.scroll_y == 0
        assert app._follow_output is False        # watcher disengaged following

        turn = Vertical(classes="turn")           # more streamed content arrives
        await sc.mount(turn)
        boxes = {"thinking": None, "answer": None, "tool": {}, "tool_buf": {}, "call_args": {}}
        for _ in range(5):
            await app._render_event(ThinkingDelta("more "), boxes, turn)
        await pilot.pause()
        assert sc.scroll_y == 0                    # stayed put — not yanked to the bottom

        sc.scroll_end(animate=False)              # user returns to the bottom
        await pilot.pause()
        assert app._follow_output is True         # following re-engages


@pytest.mark.asyncio
async def test_ctrl_y_copies_last_answer_to_clipboard():
    """ctrl+y copies the last assistant answer via copy_to_clipboard (OSC 52)."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.session.messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "the latest answer"},
        ]
        captured = []
        app.copy_to_clipboard = captured.append  # capture instead of writing OSC 52
        await pilot.press("ctrl+y")
        await pilot.pause()
        assert captured == ["the latest answer"]  # the LAST answer, not the first


@pytest.mark.asyncio
async def test_ctrl_y_with_no_answer_copies_nothing():
    """With no assistant answer yet, ctrl+y warns and copies nothing (no crash)."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.session.messages = [{"role": "user", "content": "just asked"}]
        captured = []
        app.copy_to_clipboard = captured.append
        await pilot.press("ctrl+y")
        await pilot.pause()
        assert captured == []


# --- the pinned plan: ticking, revision, ownership ---------------------------

@pytest.mark.asyncio
async def test_a_revised_plan_replaces_the_pinned_one(monkeypatch):
    """Discussing the plan and having the model re-send it must rewrite the panel —
    including dropping a step that was already ticked."""
    from ahacode.widgets.todo_panel import TodoPanel

    def fake_stream(messages, tools=None):
        # one revision, then a plain answer so the loop terminates
        if not any(m.get("role") == "tool" for m in messages):
            yield ToolCall(id="t1", name="todo_write", arguments={"items": [
                {"content": "Write solver.py", "status": "done"},
                {"content": "Benchmark it", "status": "pending"},
            ]})
            return
        yield TextDelta("계획을 고쳤어요")

    monkeypatch.setattr(client, "stream_chat", fake_stream)
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        panel = app.query_one(TodoPanel)
        panel.update_todos([{"content": "Old step", "status": "done"}])
        app.query_one("#prompt", PromptInput).text = "이 계획 좀 고쳐줘"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert [i["content"] for i in panel.items] == ["Write solver.py", "Benchmark it"]
        assert "☑ Write solver.py" in panel._content
        assert "☐ Benchmark it" in panel._content
        assert "Old step" not in panel._content


# --- Stop: fold the plan, then resume it -------------------------------------

# --- Stop while a tool is waiting for approval -------------------------------
# The regression: an approval modal shadows the app's escape=stop binding AND covers
# the Stop button, so while a tool (very often a sub-agent's) waited to be approved
# the run could not be stopped at all — escape denied one call and the loop asked for
# the next one.

async def _wait_for_modal(pilot, app, present=True, tries=40):
    from ahacode.widgets.approval_modal import ApprovalModal
    for _ in range(tries):
        await pilot.pause()
        if isinstance(app.screen, ApprovalModal) is present:
            return True
    return False


@pytest.mark.asyncio
async def test_escape_on_the_approval_modal_denies_but_does_not_stop(monkeypatch):
    """Documents the deliberate split: escape closes the dialog, it does not end the
    run. If this ever changes, the `s` binding below is what users were told to use."""
    from textual.worker import WorkerState

    def fake_stream(messages, tools=None):
        if not any(m.get("role") == "tool" for m in messages):
            yield ToolCall(id="t1", name="write",
                           arguments={"path": "x.txt", "content": "hi"})
            return
        yield TextDelta("carried on")

    monkeypatch.setattr(client, "stream_chat", fake_stream)
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", PromptInput).text = "write a file"
        await pilot.press("enter")
        assert await _wait_for_modal(pilot, app, True)
        worker = app._response_worker
        await pilot.press("escape")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert worker.state is not WorkerState.CANCELLED   # the run continued
        assert any("carried on" in (m.get("content") or "")
                   for m in app.session.messages if m["role"] == "assistant")


@pytest.mark.asyncio
async def test_s_on_the_approval_modal_stops_the_whole_run(monkeypatch):
    """The way out that did not exist: one keypress ends the run from inside the modal."""
    from textual.worker import WorkerState

    turns = {"n": 0}

    def fake_stream(messages, tools=None):
        turns["n"] += 1
        yield ToolCall(id=f"t{turns['n']}", name="write",
                       arguments={"path": f"x{turns['n']}.txt", "content": "hi"})

    monkeypatch.setattr(client, "stream_chat", fake_stream)
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", PromptInput).text = "write files forever"
        await pilot.press("enter")
        assert await _wait_for_modal(pilot, app, True)
        worker = app._response_worker
        await pilot.press("s")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert worker.state is WorkerState.CANCELLED
        assert turns["n"] == 1          # no further turn was taken
        assert app._stopping


@pytest.mark.asyncio
async def test_a_stopped_run_stops_asking_for_approval(monkeypatch):
    """Sub-agents queue on the approval lock; after a stop none of them may put another
    dialog on screen. Saying stop once has to be enough."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app._stopping = True
        call = ToolCall(id="t1", name="write", arguments={"path": "x.txt", "content": "hi"})
        # would block on a modal if it asked; returns straight away instead
        from ahacode.widgets.approval_modal import ApprovalModal
        assert app._approve_tool(call) is False       # returned without blocking
        assert not isinstance(app.screen, ApprovalModal)  # and put nothing on screen
        # a new turn clears the stop
        app.query_one("#prompt", PromptInput).text = "hello"
        monkeypatch.setattr(client, "stream_chat",
                            lambda m, tools=None: iter([TextDelta("hi")]))
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._stopping is False


def test_the_folded_plan_line_shows_how_to_resume():
    panel = TodoPanel()
    panel.update_todos([{"content": "a", "status": "done"}, {"content": "b"},
                        {"content": "c"}, {"content": "d"}])
    panel.set_collapsed(True)
    assert panel._content == "▸ Plan 1/4 · 클릭 펼치기"
    # nothing done yet -> "실행", not "이어서"
    fresh = TodoPanel()
    fresh.update_todos([{"content": "a"}, {"content": "b"}])
    fresh.set_collapsed(True)
    assert fresh._content.endswith("클릭 펼치기")


def test_the_folded_plan_line_stays_one_row_on_a_narrow_terminal():
    """Korean glyphs are two cells wide; a folded line that wraps is not folded.
    Guards the length rather than the rendering, so it needs no running app."""
    panel = TodoPanel()
    panel.update_todos([{"content": "a", "status": "done"}] + [{"content": c} for c in "bcd"])
    panel.set_collapsed(True)
    cells = sum(2 if ord(ch) > 0x2500 and not ch.isascii() else 1 for ch in panel._content)
    assert cells <= 40, f"{cells} cells — wraps below a 44-column terminal"




# --- sub-agents and the pinned plan / stop (task-tool paths) -----------------
# These used to be exercised through /run; the behaviours are the loop's, not the
# runner's, so they are re-stated through an ordinary act turn that delegates.

@pytest.mark.asyncio
async def test_a_subagents_plan_does_not_overwrite_the_parents(monkeypatch):
    """A child planning its own sub-task must not wipe the parent's pinned checklist."""
    from ahacode.widgets.todo_panel import TodoPanel

    def fake_stream(messages, tools=None):
        if messages and "sub-agent" in messages[0].get("content", ""):
            yield ToolCall(id="c1", name="todo_write",
                           arguments={"items": [{"content": "CHILD STEP"}]})
            yield TextDelta("child done")
            return
        if not any(m.get("role") == "tool" for m in messages):
            yield ToolCall(id="t1", name="task",
                           arguments={"description": "sub", "prompt": "do it"})
            return
        yield TextDelta("done")

    monkeypatch.setattr(client, "stream_chat", fake_stream)
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        panel = app.query_one(TodoPanel)
        panel.update_todos([{"content": "Write solver.py"}])
        app.auto_approve = True
        app.query_one("#prompt", PromptInput).text = "위임해줘"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert [i["content"] for i in panel.items] == ["Write solver.py"]
        assert "CHILD STEP" not in panel._content


@pytest.mark.asyncio
async def test_stop_halts_the_child_too_not_just_the_parent(monkeypatch):
    """Stop cancels the whole tree, not just the level you can see: a sub-agent runs
    INSIDE the parent's worker and is handed that worker's is_cancelled."""
    child_turns = []

    def fake_stream(messages, tools=None):
        if messages and "sub-agent" in messages[0].get("content", ""):
            child_turns.append(1)
            if len(child_turns) == 1:
                app._response_worker.cancel()   # user hits Stop while the child works
            yield ToolCall(id=f"c{len(child_turns)}", name="read",
                           arguments={"path": "conftest.py"})
            return                              # a tool call => the child would loop again
        yield ToolCall(id="t1", name="task", arguments={"description": "sub", "prompt": "do it"})

    monkeypatch.setattr(client, "stream_chat", fake_stream)
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.auto_approve = True
        app.query_one("#prompt", PromptInput).text = "위임해줘"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(child_turns) == 1


@pytest.mark.asyncio
async def test_stop_folds_the_pinned_plan_without_losing_it(monkeypatch):
    """A stop is when the user most wants the chat area back, and the plan is the
    widest thing on screen. Folding is presentation only — the steps survive."""
    from ahacode.widgets.todo_panel import TodoPanel

    calls = iter(range(100))

    def fake_stream(messages, tools=None):
        n = next(calls)
        if n == 0:
            return iter([ToolCall(id="t1", name="todo_write", arguments={"items": [
                {"content": "Write solver.py", "status": "done"},
                {"content": "Run the examples"}, {"content": "Report the timings"}]})])
        app.call_from_thread(app.action_stop)   # the user hits Stop on the next turn
        return iter([TextDelta("…")])

    monkeypatch.setattr(client, "stream_chat", fake_stream)
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        panel = app.query_one(TodoPanel)
        app.query_one("#prompt", PromptInput).text = "solve it"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert panel.collapsed
        assert panel._content.startswith("▸ Plan 1/3")
        assert len(panel.items) == 3          # the plan itself is intact
        panel.on_click()                       # click to unfold
        assert not panel.collapsed and "☑ Write solver.py" in panel._content


# --- one turn at a time ------------------------------------------------------

@pytest.mark.asyncio
async def test_typing_while_a_turn_streams_is_refused_and_kept(monkeypatch):
    """A message typed mid-turn is neither sent nor allowed to cancel the turn: it
    stays in the composer and the user is told how to stop."""
    from textual.worker import WorkerState

    def slow(messages, tools=None):
        for _ in range(60):
            time.sleep(0.02)
            yield TextDelta("x")

    monkeypatch.setattr(client, "stream_chat", slow)
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", PromptInput).text = "go"
        await pilot.press("enter")
        for _ in range(50):
            w = getattr(app, "_response_worker", None)
            if w is not None and w.is_running:
                break
            await pilot.pause(0.02)
        worker = app._response_worker
        app.query_one("#prompt", PromptInput).text = "그리고 이것도"
        await pilot.press("enter")
        await pilot.pause()
        assert worker.state is not WorkerState.CANCELLED          # the turn went on
        assert app.query_one("#prompt", PromptInput).text == "그리고 이것도"  # kept
        said = [b._content for b in app.query(Chatbox) if b.has_class("chatbox--system")]
        assert any("진행 중" in s for s in said)
        await pilot.press("escape")
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert [m["role"] for m in app.session.messages] == ["user"]   # only one turn was ever sent


@pytest.mark.asyncio
async def test_a_stop_keeps_the_rounds_that_finished(monkeypatch):
    """Stopping mid-turn persists the completed assistant+tool rounds, so the model
    resumes from a transcript that knows what it already did."""
    calls = iter(range(100))

    def fake_stream(messages, tools=None):
        n = next(calls)
        if n == 0:
            return iter([ToolCall(id="t1", name="read", arguments={"path": "conftest.py"})])
        app.call_from_thread(app.action_stop)   # Stop lands during the second round
        return iter([TextDelta("never kept")])

    monkeypatch.setattr(client, "stream_chat", fake_stream)
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", PromptInput).text = "read it"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
    roles = [m["role"] for m in app.session.messages]
    assert roles == ["user", "assistant", "tool"]          # round one survived the stop
    assert not any("never kept" in (m.get("content") or "") for m in app.session.messages)


@pytest.mark.asyncio
async def test_the_panel_repairs_items_that_arrive_as_a_string(monkeypatch):
    """The live regression: a stringified list drew one ☐ per character."""
    from ahacode.widgets.todo_panel import TodoPanel

    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter([
        ToolCall(id="t1", name="todo_write",
                 arguments={"items": '["Write sort.py: 버블, 삽입", "Run it"]'}),
        TextDelta("planned"),
    ]))
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", PromptInput).text = "plan it"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        panel = app.query_one(TodoPanel)
        assert [it["content"] for it in panel.items] == ["Write sort.py: 버블, 삽입", "Run it"]
        assert "☐ Write sort.py: 버블, 삽입" in panel._content


# --- resume: repairing an interrupted turn -----------------------------------

@pytest.mark.asyncio
async def test_switching_into_an_interrupted_session_fills_dangling_calls(monkeypatch):
    """A session whose last turn ended with unanswered tool_calls (Stop/Ctrl+D mid-tool)
    is repaired on open: each dangling call gets a synthetic result and a note tells the
    model it was interrupted — so its next turn is API-valid and knows to reassess."""
    from ahacode.widgets.prompt_input import PromptInput

    # a fresh drivable session with a dangling tool call persisted
    other = storage.new_session_path()
    storage.write_header(other, storage.make_header(other.stem, kind="main"))
    for m in [
        {"role": "user", "content": "read it"},
        {"role": "assistant", "tool_calls": [
            {"id": "x1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
    ]:
        storage.append_message(other, m)

    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter([TextDelta("ok")]))
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await app._switch_session(other.stem)
        await pilot.pause()
        # the dangling call now has a synthetic tool result, then an interrupt note
        roles = [m["role"] for m in app.session.messages]
        assert roles == ["user", "assistant", "tool", "user"]
        assert app.session.messages[2]["tool_call_id"] == "x1"
        # the result names WHICH tool was cut off (bare id says nothing)
        assert "Interrupted" in app.session.messages[2]["content"]
        assert "`read`" in app.session.messages[2]["content"]
        assert "interrupted" in app.session.messages[3]["content"].lower()
        # persisted, so reopening finds nothing dangling
        assert storage.dangling_tool_call_ids(storage.load_messages(other)) == []


@pytest.mark.asyncio
async def test_a_clean_session_is_not_touched_on_open(monkeypatch):
    """No dangling calls → no repair messages appended."""
    other = storage.new_session_path()
    storage.write_header(other, storage.make_header(other.stem, kind="main"))
    for m in [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]:
        storage.append_message(other, m)

    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter([TextDelta("ok")]))
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await app._switch_session(other.stem)
        await pilot.pause()
        assert [m["role"] for m in app.session.messages] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_parallel_dangling_calls_are_each_named_by_their_command(monkeypatch):
    """Three bash calls interrupted at once each get a result naming its own command,
    so a bare id is never the only way to tell them apart."""
    other = storage.new_session_path()
    storage.write_header(other, storage.make_header(other.stem, kind="impl", relation="handoff", parent_id="p"))
    for m in [
        {"role": "user", "content": "build all"},
        {"role": "assistant", "tool_calls": [
            {"id": "b1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "python build_a.py"}'}},
            {"id": "b2", "type": "function", "function": {"name": "bash", "arguments": '{"command": "python build_b.py"}'}},
        ]},
    ]:
        storage.append_message(other, m)

    monkeypatch.setattr(client, "stream_chat", lambda m, tools=None: iter([TextDelta("ok")]))
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await app._switch_session(other.stem)
        await pilot.pause()
        results = {m["tool_call_id"]: m["content"] for m in app.session.messages if m.get("role") == "tool"}
        assert "python build_a.py" in results["b1"]
        assert "python build_b.py" in results["b2"]
