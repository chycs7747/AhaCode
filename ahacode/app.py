"""The Textual App: it holds the session state and wires keys, buttons and
messages to the collaborators that do the work."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Select
from textual.worker import get_current_worker
from rich.theme import Theme

from ahacode import client, config, prompts, storage, tools
from ahacode.commands import Commands
from ahacode.plan_run import PlanRun
from ahacode.runner import TurnRunner
from ahacode.tools import spill
from ahacode.session import ChatSession
from ahacode.session_ctl import SessionControl
from ahacode.turn_view import TurnView
from ahacode.widgets.chatbox import Chatbox
from ahacode.widgets.header_bar import HeaderBar
from ahacode.widgets.settings import Settings
from ahacode.widgets.prompt_input import PromptInput
from ahacode.widgets.subagent_card import SubagentCard
from ahacode.widgets.todo_panel import TodoPanel
from ahacode.widgets.model_bar import ModelBar


# Rich's Markdown defaults (magenta headings, cyan-on-black code) are harsh on a
# dark terminal; a Markdown renderable resolves these styles by name at draw time.
MARKDOWN_THEME = Theme(
    {
        "markdown.h1": "bold #7dcfff",
        "markdown.h2": "bold #82aaff",
        "markdown.h3": "bold #c792ea",
        "markdown.h4": "#c792ea",
        "markdown.h5": "italic #c792ea",
        "markdown.h6": "dim italic",
        "markdown.code": "#a6e3a1",
        "markdown.block_quote": "#82aaff",
        "markdown.list": "#82aaff",
        "markdown.item.number": "#82aaff",
        "markdown.link": "underline #82aaff",
        "markdown.link_url": "dim #82aaff",
    }
)

# How long a quit waits for an orderly shutdown before leaving anyway: enough for
# an in-flight session append to land, short enough that a wedged worker is not
# the user's problem.
QUIT_GRACE_SECONDS = 1.5


class AhaCodeApp(App):
    """AhaCode's Textual App. It holds the session state; behaviour lives in the
    collaborators it constructs."""

    CSS_PATH = "ahacode.tcss"
    # priority=True: checked before the focused widget's own bindings, which would
    # otherwise swallow ctrl+d (delete right) and ctrl+y.
    BINDINGS = [
        Binding("ctrl+d", "quit", "Quit", priority=True),
        Binding("escape", "stop", "Stop", show=False),
        Binding("ctrl+y", "copy_answer", "Copy answer", priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.session = ChatSession()
        self.commands = Commands(self)
        self.plan = PlanRun(self)
        self.turn_view = TurnView(self)
        self.sessions = SessionControl(self)
        self.runner = TurnRunner(self)
        self.mode = "act"  # "act" (full tools) or "plan" (read-only + plan_submit)
        self._last_status = ""
        self.auto_approve = False  # session-only: skip the approval modal
        self._follow_output = True  # is the view pinned to the bottom? (_update_follow)
        self._approval_lock = threading.Lock()  # one modal at a time; children queue
        self._last_prompt_tokens: int | None = None  # the server's count, for compaction
        # This loop's own running tools: call id -> (name, started at). Sub-agent
        # tools report in their own card instead.
        self._running_tools: dict[str, tuple[str, float]] = {}
        latest = storage.latest_session()
        if latest:
            self.session_path = latest
            self.session.messages = storage.load_messages(latest)
        else:
            self.session_path = storage.new_session_path()
            storage.write_header(
                self.session_path,
                storage.make_header(
                    self.session_path.stem, kind="main", model=config.load().name
                ),
            )
        self._has_title = bool((storage.read_session_meta(self.session_path) or {}).get("title"))
        spill.set_session(self.session_path)
        header = storage.read_header(self.session_path) or {}
        self.session_depth = int(header.get("depth", 0))  # gates the `task` tool
        self.session_kind = str(header.get("kind", "main"))  # picks the turn cap and mode
        self.session_parent_id = header.get("parent_id")

    # --- what the worker posts back ------------------------------------------

    @dataclass
    class ResponseComplete(Message):
        """The agent loop finished; carries every message it appended."""

        messages: list[dict]
        stats: str = ""
        prompt_tokens: int | None = None  # the server's count for this turn's request

    @dataclass
    class ResponseFailed(Message):
        """The agent loop hit an error; the app stays alive and shows it."""

        error: str

    # --- startup -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield HeaderBar()
        yield TodoPanel()
        with VerticalScroll(id="chat-container") as container:
            container.can_focus = False  # keep initial focus on the input
        with Vertical(id="bottom"):
            yield PromptInput(id="prompt")
            yield ModelBar()

    async def on_mount(self) -> None:
        """Restore the saved history as chat bubbles."""
        self.console.push_theme(MARKDOWN_THEME)
        self.set_interval(1.0, self._tick_progress)
        meta = storage.read_session_meta(self.session_path) or {}
        self._set_header_title(meta.get("title", ""))
        self._set_header_endpoint()
        await self.sessions.render_history()
        self._reflect_view_only()
        if not self.view_only:
            await self.sessions.repair_interrupted()
        # Follow the stream only while pinned to the bottom: our own scroll_end
        # lands exactly there, a user scroll-up drops below it.
        self._chat_scroller = self.query_one("#chat-container", VerticalScroll)
        self.watch(self._chat_scroller, "scroll_y", self._update_follow, init=False)
        self.query_one("#prompt", PromptInput).focus()

    # --- buttons -------------------------------------------------------------

    @on(Button.Pressed, "#settings-btn")
    def _settings_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.push_screen(Settings(config.load()), self._save_settings)

    @on(Button.Pressed, "#new-session-btn")
    async def _new_session_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        await self.sessions.new()

    @on(Button.Pressed, "#open-sessions-btn")
    def _open_sessions_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.sessions.open_picker()

    @on(Button.Pressed, "#send-btn")
    def _send_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._anything_running():  # the button doubles as Stop while a turn runs
            self.action_stop()
        else:
            self.query_one("#prompt", PromptInput).submit()

    @on(Button.Pressed, "#plan-gate-run")
    async def _plan_run_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.plan.settle("▶ 실행")
        await self.plan.start_impl_session()

    @on(Button.Pressed, "#plan-gate-continue")
    async def _plan_revise_pressed(self, event: Button.Pressed) -> None:
        """✎ 수정: the plan stays on screen and the next message revises it."""
        event.stop()
        self.plan.settle("✎ 수정 계속")
        self.query_one("#prompt", PromptInput).focus()

    # --- settings ------------------------------------------------------------

    def _save_settings(self, chosen: config.ModelConfig | None) -> None:
        """Persist what the settings modal handed back and reset the client, so the
        next request uses the new endpoint, model, timeout and gate size."""
        if chosen is None:
            return
        before = config.load()
        config.save(chosen)
        client.reset()
        self.refresh_config_ui()
        self.run_worker(self._say_system(self._settings_summary(before, chosen)),
                        exclusive=False)

    @staticmethod
    def _settings_summary(before, after) -> str:
        """One line of what changed; the endpoint and model lead, and only when they moved."""
        def budget(v):
            return "전역" if v is None else f"{v // 1024}K"

        moved = []
        if after.base_url != before.base_url:
            moved.append(f"엔드포인트 {after.base_url}")
        if after.name != before.name:
            moved.append(f"모델 {after.name} (다음 메시지에 적용)")
        lead = (" · ".join(moved) + " · ") if moved else ""
        window = "압축 끔" if after.context_window == 0 else f"{after.context_window // 1024}K"
        return (
            f"Settings 저장 — {lead}최대 병렬 {after.max_parallel_agents} · "
            f"깊이 {after.subagent_depth} · 컨텍스트 {window} · "
            f"압축 {int(after.compact_threshold * 100)}% · 사고 "
            f"plan {budget(after.plan_thinking_budget)}/"
            f"impl {budget(after.impl_thinking_budget)}/"
            f"sub {budget(after.subagent_thinking_budget)} · 도구후사고 "
            f"{'끔' if after.no_think_after_tools else '켬'}"
        )

    # --- the composer and the model bar --------------------------------------

    @on(PromptInput.Submitted)
    async def user_submitted(self, event: PromptInput.Submitted) -> None:
        text = event.text.strip()
        if not text:
            if self.plan.pending:  # an empty Enter is the keyboard's ▶ on the gate
                self.plan.settle("▶ 실행 (Enter)")
                await self.plan.start_impl_session()
            return

        # One turn at a time, slash commands included: /new or /sessions mid-turn
        # would move session_path out from under the worker's persist.
        if self._anything_running():
            self.query_one("#prompt", PromptInput).text = text  # kept, not sent
            await self._say_system(
                "⏳ 진행 중이라 보내지 않았어요 — 멈추려면 Stop(Esc), 끝나면 그대로 Enter."
            )
            return

        # A sub-agent session is view-only; slash commands stay as escape hatches.
        if self.view_only and not text.startswith("/"):
            await self._say_system(
                "🔒 보기 전용 세션(서브에이전트 기록)이라 대화를 보낼 수 없어요. "
                "/new (또는 상단 New) 로 새 세션을 시작하세요."
            )
            return

        # Text typed while the gate is open is feedback on the plan.
        if self.plan.pending:
            self.plan.settle("✎ 수정 계속")

        if text.startswith("/"):  # never reaches the model, never recorded
            if text == "/new":
                await self.sessions.new()
                return
            if text == "/sessions":
                self.sessions.open_picker()
                return
            await self._say_system(self.commands.handle(text))
            return

        self.session.add_user(text)
        storage.append_message(self.session_path, {"role": "user", "content": text})
        self.plan.stalled = 0  # a typed instruction is a fresh start for the stall count

        self._follow_output = True
        container = self.query_one("#chat-container", VerticalScroll)
        await container.mount(Chatbox(text, role="user"))
        await self._start_turn()

    @on(ModelBar.ModelChosen)
    async def model_chosen(self, event: ModelBar.ModelChosen) -> None:
        await self._say_system(self.commands.switch_model(event.name))

    @on(ModelBar.ModeChosen)
    async def mode_chosen(self, event: ModelBar.ModeChosen) -> None:
        if event.mode == self.mode:
            return  # a programmatic re-sync, not a real switch
        self.mode = event.mode
        if self.mode == "plan":
            await self._say_system("plan mode ON — read-only tools; the model plans, not acts.")
        else:
            await self._say_system("act mode — full tools (bash asks first).")

    @on(ModelBar.AutoApproveChanged)
    async def auto_approve_changed(self, event: ModelBar.AutoApproveChanged) -> None:
        if event.value == self.auto_approve:
            return  # a programmatic re-sync, not a real toggle
        self.auto_approve = event.value
        if event.value:
            await self._say_system(
                "auto-approve ON — tools run without asking "
                "(dangerous commands are still blocked)."
            )
        else:
            await self._say_system("auto-approve OFF — tools ask first.")

    # --- key bindings --------------------------------------------------------

    def action_copy_answer(self) -> None:
        """Copy the last assistant answer to the clipboard (OSC 52, so it works over SSH)."""
        text = next(
            (m["content"] for m in reversed(self.session.messages)
             if m.get("role") == "assistant" and m.get("content")),
            "",
        )
        if not text:
            self.notify("복사할 답변이 아직 없어요.", severity="warning", timeout=2)
            return
        self.copy_to_clipboard(text)
        self.notify("답변을 클립보드에 복사했어요.", timeout=2)

    def action_stop(self) -> None:
        """Cancel the turn in flight; the loops check is_cancelled between events."""
        # Set first: sub-agents queued on the approval lock must see the stop, or
        # each pops its own modal after the user has already said stop.
        self._stopping = True
        stopped = False
        worker = getattr(self, "_response_worker", None)
        if worker is not None and worker.is_running:
            worker.cancel()
            stopped = True
        if stopped:
            self._set_status("■ stopped")
            self._set_send_running(False)
            # Fold the pinned plan to give the chat area back. query, not query_one:
            # this can be called from the approval modal's screen.
            for panel in self.query(TodoPanel):
                panel.set_collapsed(True)

    async def action_quit(self) -> None:
        """Quit without waiting on a worker blocked in a socket read: ask it to stop,
        then leave once an in-flight session append has had time to land."""
        self._stopping = True
        worker = getattr(self, "_response_worker", None)
        if worker is not None and worker.is_running:
            worker.cancel()
        if not self.is_headless:  # under run_test the process to leave is the test runner's
            leave = threading.Timer(QUIT_GRACE_SECONDS, self._force_exit)
            leave.daemon = True
            leave.start()
        self.exit()

    def _force_exit(self) -> None:  # seam: tests replace this rather than dying
        os._exit(0)

    # --- a turn, start to finish ---------------------------------------------

    async def _start_turn(self) -> None:
        """Mount a fresh turn rail and run the agent loop over the current history."""
        container = self.query_one("#chat-container", VerticalScroll)
        # The whole reply (thinking → tools → answer) mounts into one rail, so the
        # steps read as one connected flow; the user message stays outside it.
        self._turn = Vertical(classes="turn")
        await container.mount(self._turn)
        container.scroll_end(animate=False)

        # A copy goes to the worker, so it never shares a mutable list with the
        # main thread.
        base = prompts.plan_system() if self.mode == "plan" else prompts.act_system()
        history = [{"role": "system", "content": base}, *self.session.messages]
        self._set_status("● waiting…")
        self._stopping = False
        self._response_worker = self.stream_response(history, self._turn)
        self._set_send_running(True)

    # exclusive=True: a new turn cancels the previous worker.
    # exit_on_error=False: a failing worker must not take the whole app down.
    @work(thread=True, exclusive=True, exit_on_error=False)
    def stream_response(self, messages: list[dict], turn) -> None:
        """Run the agent loop in a thread, rendering its events into the chat."""
        self.runner.run(messages, turn, get_current_worker())

    def _anything_running(self) -> bool:
        worker = getattr(self, "_response_worker", None)
        return worker is not None and worker.is_running

    @on(ResponseComplete)
    async def response_complete(self, event: ResponseComplete) -> None:
        self._set_send_running(False)
        self._prune_empty_turn()
        for msg in event.messages:  # shared state is only touched on the main thread
            self.session.messages.append(msg)
            storage.append_message(self.session_path, msg)
        self._set_status(event.stats)
        if event.prompt_tokens:
            self._last_prompt_tokens = event.prompt_tokens
        if not self._has_title and any(m.get("role") == "assistant" for m in self.session.messages):
            self._has_title = True
            self.generate_title(list(self.session.messages), self.session_path)
        # An impl session exists to finish its plan: snapshot where it stands after
        # every turn, and carry on or stop.
        if self.session_kind == "impl" and event.messages:
            await self.plan.snapshot_progress()
            await self.plan.auto_continue()

    @on(ResponseFailed)
    async def response_failed(self, event: ResponseFailed) -> None:
        self._set_send_running(False)
        self._prune_empty_turn()
        container = self.query_one("#chat-container", VerticalScroll)
        await container.mount(
            Chatbox(f"⚠ {event.error}\n(check the server, then try again)", role="error")
        )
        container.scroll_end(animate=False)
        self._set_status("")

    @work(thread=True, exit_on_error=False)
    def generate_title(self, messages: list[dict], path) -> None:
        """Name an untitled session in the background. A shim: @work needs the
        App's run_worker, the work itself is the runner's."""
        self.runner.make_title(messages, path)

    # --- what the open session is --------------------------------------------

    @property
    def view_only(self) -> bool:
        """Whether the open session is a sub-agent transcript: readable, not drivable.

        Derived from session_depth, the same axis that gates the `task` tool.
        """
        return self.session_depth > 0

    def _set_mode(self, mode: str) -> None:
        """Switch modes from code. The field is set first, so the Select's handler
        sees a match and no-ops instead of re-triggering."""
        if self.mode != mode:
            self.mode = mode
            self.query_one("#mode-select", Select).value = mode

    def _registry_for_mode(self) -> dict:
        """The tools this session may use this turn: plan mode gets the read-only
        tools plus plan_submit; act mode gets everything, with `task` depth-gated."""
        if self.mode == "plan":
            return {
                "read": tools.READ,
                "glob": tools.GLOB,
                "grep": tools.GREP,
                "plan_submit": tools.PLAN_SUBMIT,
            }
        return tools.registry_for(self.session_depth, config.load().subagent_depth)

    # --- the chrome ----------------------------------------------------------

    def _set_status(self, text: str) -> None:
        """Push live turn status to the bar; "" is idle."""
        self._last_status = text
        for bar in self.query(ModelBar).results(ModelBar):  # no bar is not an error
            bar.set_status(text)

    def _set_send_running(self, running: bool) -> None:
        """Flip the composer button between Send (idle) and Stop (streaming).

        query, not query_one: a turn can end after the composer is gone (quitting
        mid-stream), and nothing to update is a valid outcome then.
        """
        if not running:
            # A cancelled turn never delivers the ToolResult that would retire its entry.
            self._running_tools.clear()
        for btn in self.query("#send-btn").results(Button):
            btn.label = "■ Stop" if running else "↑ Send"
            btn.variant = "error" if running else "primary"
        for prompt in self.query("#prompt").results(PromptInput):
            if running:
                prompt.border_subtitle = "Esc 로 중지"
            elif not self.view_only:
                prompt.border_subtitle = "Enter to send · Shift+Enter for newline"

    def _set_header_title(self, title: str) -> None:
        self.query_one(HeaderBar).set_title(title)

    def _set_header_endpoint(self) -> None:
        self.query_one(HeaderBar).set_endpoint(config.load().base_url)

    def refresh_config_ui(self, *, reload_models: bool = False) -> None:
        """Re-read the config into the chrome that displays it.

        Args:
            reload_models: Also fetch the model list; a new endpoint offers a
                different one, and fetching it is a request.
        """
        bar = self.query_one(ModelBar)
        bar.refresh_state()
        if reload_models:
            bar.load_models()
        self._set_header_endpoint()

    def _reflect_view_only(self) -> None:
        """Mirror the read-only state in the composer's hint line."""
        prompt = self.query_one("#prompt", PromptInput)
        prompt.border_subtitle = (
            "🔒 보기 전용 · /new 로 새 세션"
            if self.view_only
            else "Enter to send · Shift+Enter for newline"
        )

    async def _say_system(self, text: str) -> None:
        """Show an informational bubble; never part of the session."""
        container = self.query_one("#chat-container", VerticalScroll)
        await container.mount(Chatbox(text, role="system"))
        container.scroll_end(animate=False)

    def _prune_empty_turn(self) -> None:
        """Drop the turn's rail if the reply produced no blocks."""
        turn = getattr(self, "_turn", None)
        if turn is not None and turn.is_mounted and not turn.children:
            turn.remove()
        self._turn = None

    def _tick_progress(self) -> None:
        """Once a second, show how long the running work has been running; a
        status line that never changes looks like a frozen app."""
        for card in list(self.query(SubagentCard)):
            card.tick()
        if not self._running_tools:
            return
        now = time.monotonic()
        oldest = min(self._running_tools.values(), key=lambda v: v[1])
        seconds = int(now - oldest[1])
        if len(self._running_tools) > 1:
            self._set_status(f"● 도구 {len(self._running_tools)}개 · {seconds}초")
        else:
            self._set_status(f"● {oldest[0]} · {seconds}초")

    def _update_follow(self, scroll_y: float) -> None:
        """Pin auto-scroll while the view is at the bottom (within rounding), unpin
        it when the user scrolls up."""
        self._follow_output = scroll_y >= self._chat_scroller.max_scroll_y - 2


app = AhaCodeApp


def main() -> None:
    """The `ahacode` console script, so AhaCode can be launched from any project directory."""
    AhaCodeApp().run()


if __name__ == "__main__":
    main()
