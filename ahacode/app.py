import json
import os
import re
import threading
import time
from dataclasses import dataclass, field, replace

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Select
from textual.worker import get_current_worker
from rich.text import Text
from rich.theme import Theme

from ahacode import (
    agent, client, config, permissions, prompts, storage, subagent, tools,
)
from ahacode.commands import Commands
from ahacode.plan_run import PlanRun
from ahacode.tools import spill
from ahacode.events import (
    Notice, Phase, TextDelta, ThinkingDelta, ToolCall, ToolCallDelta, ToolResult, Usage,
)
from ahacode.render import diff_stats, edit_diff_lines, tool_summary
from ahacode.session import ChatSession
from ahacode.widgets.approval_modal import ApprovalModal
from ahacode.widgets.chatbox import Chatbox
from ahacode.widgets.header_bar import HeaderBar
from ahacode.widgets.settings import Settings, SettingsResult
from ahacode.widgets.prompt_input import PromptInput
from ahacode.widgets.session_picker import SessionPicker
from ahacode.widgets.subagent_card import SubagentCard
from ahacode.widgets.thinking import ThinkingBlock
from ahacode.widgets.tool_result import ToolResultBlock
from ahacode.widgets.todo_panel import TodoPanel
from ahacode.widgets.model_bar import ModelBar


# System prompts now live in ahacode/prompts.py (assembled per mode/model).

# Harness phases share _running_tools with the tools so they get the same ticking
# clock. This is the id they book it under: not a call id, so it cannot collide
# with one, and a single slot because phases do not nest.
_PHASE_ID = "\0phase"

# Eye-friendly Markdown palette. Rich's defaults paint headings magenta and inline
# code "bold cyan on black" — harsh on a dark terminal. Pushed onto the app console:
# a Markdown renderable resolves these styles by name at draw time.
MARKDOWN_THEME = Theme(
    {
        "markdown.h1": "bold #7dcfff",
        "markdown.h2": "bold #82aaff",
        "markdown.h3": "bold #c792ea",
        "markdown.h4": "#c792ea",
        "markdown.h5": "italic #c792ea",
        "markdown.h6": "dim italic",
        "markdown.code": "#a6e3a1",          # soft green, no black-box background
        "markdown.block_quote": "#82aaff",
        "markdown.list": "#82aaff",
        "markdown.item.number": "#82aaff",
        "markdown.link": "underline #82aaff",
        "markdown.link_url": "dim #82aaff",
    }
)

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}

# How long a quit waits for an orderly shutdown before leaving anyway. Long enough
# for Textual to restore the terminal and for an in-flight one-line append to the
# session file to land; short enough that a wedged worker is not the user's problem.
QUIT_GRACE_SECONDS = 1.5


def _tool_unescape(s: str) -> str:
    """Decode a (possibly incomplete) JSON string value, escape by escape."""
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            out.append(_ESCAPES.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _render_tool_stream(name: str, args: str) -> str:
    """Live label for a streaming tool call whose args JSON may be incomplete.

    write is shown as a path header + streamed content (pull known fields out
    early); every other tool shows its raw accumulating args.
    """
    if name == "write":
        m = re.search(r'"path"\s*:\s*"((?:[^"\\]|\\.)*)"', args)
        path = _tool_unescape(m.group(1)) if m else "…"
        body = ""
        cm = re.search(r'"content"\s*:\s*"', args)
        if cm:
            tail = re.sub(r'"\s*}?\s*$', "", args[cm.end():])
            body = _tool_unescape(tail)
        return f"🔧 write · {path}\n{body}"
    return f"🔧 {name}  {args}"


class AhaCodeApp(App):
    """AhaCode: a Textual-based TUI agent client."""

    CSS_PATH = "ahacode.tcss"
    # priority=True: checked before the focused widget's own bindings — the Input
    # binds ctrl+d to "delete character right" and would otherwise swallow it.
    BINDINGS = [
        Binding("ctrl+d", "quit", "Quit", priority=True),
        Binding("escape", "stop", "Stop", show=False),
        # priority=True: the focused prompt (a TextArea) would otherwise swallow it.
        Binding("ctrl+y", "copy_answer", "Copy answer", priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.session = ChatSession()
        self.commands = Commands(self)  # /model, /url, /allow, /think
        self.plan = PlanRun(self)  # the gate, the handoff, and the stall detector
        self.mode = "act"  # "act" (full tools) or "plan" (read-only + todo_write)
        self._last_status = ""
        self.auto_approve = False  # session-only: skip the approval modal when on
        self._follow_output = True  # is the view pinned to the bottom? (_update_follow)
        self._approval_lock = threading.Lock()  # one modal at a time; children queue
        # The server's own count for the last request — what compaction measures
        # against, rather than an estimate.
        self._last_prompt_tokens: int | None = None
        # This loop's own running tools: call id -> (name, started at). Sub-agent
        # tools are deliberately absent — they report in their own card.
        self._running_tools: dict[str, tuple[str, float]] = {}
        # The message that started the turn in flight. Cleared after the first
        # transcript write: several rounds are still one question.
        self._turn_question = ""
        latest = storage.latest_session()
        if latest:  # resume the most recent session
            self.session_path = latest
            self.session.messages = storage.load_messages(latest)
        else:  # first run: a new session, with a header
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
        # depth gates the `task` tool (0 = main); kind picks the turn cap and mode.
        self.session_depth = int(header.get("depth", 0))
        self.session_kind = str(header.get("kind", "main"))
        self.session_parent_id = header.get("parent_id")

    @dataclass
    class ResponseComplete(Message):
        """Posted by the worker once the agent loop finishes a response.

        Carries every message the loop appended (assistant, tool, assistant, ...)
        so the main-thread handler can persist the whole turn at once.
        """

        messages: list[dict]
        stats: str = ""
        prompt_tokens: int | None = None  # the server's count for this turn's request
        # The same numbers the status line renders, unformatted, so the turn can be
        # recorded as well as displayed (see storage.append_stats).
        metrics: dict = field(default_factory=dict)

    @dataclass
    class ResponseFailed(Message):
        """Posted when the loop hits an error. The app stays alive; the handler
        drops a fresh error bubble at the bottom of the chat."""

        error: str

    def compose(self) -> ComposeResult:
        # Static skeleton only — chat bubbles are mounted at runtime.
        yield HeaderBar()  # docked top: session title + New / Sessions buttons
        yield TodoPanel()  # pinned plan checklist (docked top, hidden until used)
        with VerticalScroll(id="chat-container") as container:
            container.can_focus = False  # keep initial focus on the input
        with Vertical(id="bottom"):
            yield PromptInput(id="prompt")  # multi-line: Enter sends, Shift+Enter newline
            yield ModelBar()

    async def on_mount(self) -> None:
        """Restore saved history as chat bubbles (Compose runs before Mount)."""
        self.console.push_theme(MARKDOWN_THEME)  # soften Rich Markdown colours
        # One slow tick for the whole app: it writes nothing while nothing is running,
        # so an idle session costs no repaints (see _tick_progress).
        self.set_interval(1.0, self._tick_progress)
        meta = storage.read_session_meta(self.session_path) or {}
        self._set_header_title(meta.get("title", ""))
        self._set_header_endpoint()
        await self._render_history()
        self._reflect_view_only()  # resumed session is a main one, but stay correct
        if not self.view_only:
            await self._repair_interrupted()  # startup resume of an interrupted session
        # Follow the stream only while pinned to the bottom: watch the scroller's
        # scroll_y and re-derive the flag. Our own scroll_end lands exactly at the
        # bottom (flag stays on); a user scroll-up drops below it (flag off, sticky).
        self._chat_scroller = self.query_one("#chat-container", VerticalScroll)
        self.watch(self._chat_scroller, "scroll_y", self._update_follow, init=False)
        self.query_one("#prompt", PromptInput).focus()  # not the header buttons

    def _update_follow(self, scroll_y: float) -> None:
        """Pin/unpin auto-scroll from the live scroll position: 'at the bottom' (within
        a small tolerance for rounding) follows the stream; anything above stays put.
        Fires on every scroll change, so it holds a cached scroller ref (no per-call
        DOM query) — our own scroll_end re-confirms the flag, a user scroll-up clears it."""
        self._follow_output = scroll_y >= self._chat_scroller.max_scroll_y - 2

    def _set_header_title(self, title: str) -> None:
        """Reflect the current session's title in the top bar."""
        self.query_one(HeaderBar).set_title(title)

    def _set_header_endpoint(self) -> None:
        """Reflect the current server endpoint in the top bar."""
        self.query_one(HeaderBar).set_endpoint(config.load().base_url)

    @on(Button.Pressed, "#settings-btn")
    def _on_settings_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.push_screen(Settings(config.load()), self._settings_saved)

    def _settings_saved(self, result: "SettingsResult | None") -> None:
        """Persist every field the modal owns and reset the client, so the next
        request uses the new endpoint, model, timeout and gate size (same path as
        /model and /url, which set one field each)."""
        if result is None:
            return
        from dataclasses import replace
        before = config.load()
        config.save(replace(
            before,
            base_url=result.base_url, api_key=result.api_key,
            name=result.model, timeout=result.timeout,
            max_parallel_agents=result.max_parallel, subagent_depth=result.depth,
            impl_max_turns=result.impl_max_turns,
            auto_continue_stall=result.auto_continue_stall,
            stall_rounds=result.stall_rounds,
            context_window=result.context_window, compact_threshold=result.compact_threshold,
            keep_recent_messages=result.keep_recent,
            thinking_token_budget=result.thinking_budget,
            reasoning_effort=result.reasoning_effort,
            plan_thinking_budget=result.plan_thinking, impl_thinking_budget=result.impl_thinking,
            subagent_thinking_budget=result.subagent_thinking,
            no_think_after_tools=result.no_think_after_tools,
        ))
        client.reset()
        self._set_header_endpoint()
        self.query_one(ModelBar).refresh_state()
        window = "압축 끔" if result.context_window == 0 else f"{result.context_window // 1024}K"
        def think(v):
            return "전역" if v is None else f"{v // 1024}K"
        # Endpoint and model first, and only when they moved: they are the two that
        # change what answers, and a silent switch is the one worth noticing.
        moved = []
        if result.base_url != before.base_url:
            moved.append(f"엔드포인트 {result.base_url}")
        if result.model != before.name:
            moved.append(f"모델 {result.model} (다음 메시지에 적용)")
        lead = (" · ".join(moved) + " · ") if moved else ""
        self.run_worker(
            self._say_system(
                f"Settings 저장 — {lead}최대 병렬 {result.max_parallel} · 깊이 {result.depth} · "
                f"컨텍스트 {window} · 압축 {int(result.compact_threshold * 100)}% · "
                f"사고 plan {think(result.plan_thinking)}/impl {think(result.impl_thinking)}/"
                f"sub {think(result.subagent_thinking)} · 도구후사고 "
                f"{'끔' if result.no_think_after_tools else '켬'}"
            ),
            exclusive=False,
        )

    @on(Button.Pressed, "#new-session-btn")
    async def _on_new_session_button(self, event: Button.Pressed) -> None:
        event.stop()
        await self._new_session()

    @on(Button.Pressed, "#open-sessions-btn")
    def _on_open_sessions_button(self, event: Button.Pressed) -> None:
        event.stop()
        self._open_picker()

    @on(Button.Pressed, "#send-btn")
    def _on_send_button(self, event: Button.Pressed) -> None:
        event.stop()
        # The button doubles as Stop while ANYTHING runs (see action_stop) —
        # checking only the chat worker left it labelled ■ Stop but acting as Send.
        if self._anything_running():
            self.action_stop()
        else:
            self.query_one("#prompt", PromptInput).submit()

    def _anything_running(self) -> bool:
        worker = getattr(self, "_response_worker", None)
        return worker is not None and worker.is_running

    def _set_send_running(self, running: bool) -> None:
        """Flip the composer button between Send (idle) and Stop (streaming).

        query, not query_one: a turn can end after the composer is gone (quitting
        mid-stream), and NoMatches raised from a tidy-up callback turns an orderly
        shutdown into a crash. Nothing to update is a valid outcome here.
        """
        if not running:
            # A cancelled turn never delivers the ToolResult that would retire its
            # entry, and a leftover entry means _tick_progress counts on for work
            # that stopped — the exact lie the counter exists to prevent.
            self._running_tools.clear()
        for btn in self.query("#send-btn").results(Button):
            btn.label = "■ Stop" if running else "↑ Send"
            btn.variant = "error" if running else "primary"
        # Not in the status text: the hint cost 14 of the ~7 columns the composer
        # leaves for status at 80 wide, crowding out the elapsed seconds entirely.
        for prompt in self.query("#prompt").results(PromptInput):
            if running:
                prompt.border_subtitle = "Esc 로 중지"
            elif not self.view_only:
                prompt.border_subtitle = "Enter to send · Shift+Enter for newline"

    def _prune_empty_turn(self) -> None:
        """Drop the turn's rail if the reply produced no blocks (immediate error)."""
        turn = getattr(self, "_turn", None)
        if turn is not None and turn.is_mounted and not turn.children:
            turn.remove()
        self._turn = None

    async def _render_history(self) -> None:
        """Clear the chat and remount the session's messages, matching the live
        rendering exactly — turn rails, titled tool cards, diffs, todo_write into
        the pinned panel — so a reloaded session looks like the turn that made it.

        Also the ONE owner of the pinned plan: cleared here and refilled from the
        history, so the panel is always a function of the open session. Every
        session switch comes through here, which is what stops a previous plan
        lingering behind a hidden panel.
        """
        container = self.query_one("#chat-container", VerticalScroll)
        await container.remove_children()
        todo = self.query_one(TodoPanel)
        todo.clear()
        call_args: dict[str, dict] = {}   # tool_call_id -> parsed arguments
        call_names: dict[str, str] = {}   # tool_call_id -> tool name
        turn = None
        for msg in self.session.messages:
            role = msg["role"]
            content = msg.get("content") or ""
            if role == "user":
                turn = None  # a user message closes the previous assistant turn
                await container.mount(Chatbox(content, role="user"))
                continue
            if turn is None:  # assistant / tool -> one rail
                turn = Vertical(classes="turn")
                await container.mount(turn)
            if role == "assistant":
                if content:  # the model's text answer (tool calls become cards)
                    await turn.mount(Chatbox(content, role="assistant", markdown=True))
                for c in msg.get("tool_calls") or []:
                    cid, name = c["id"], c["function"]["name"]
                    call_names[cid] = name
                    try:
                        call_args[cid] = json.loads(c["function"]["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        call_args[cid] = {}
                    if name == "edit":  # its result is skipped below
                        await turn.mount(self._edit_card(call_args[cid]))
                    elif name == "todo_write":  # to the pinned panel, as when live
                        todo.update_todos(call_args[cid].get("items", []))
                    elif name == "plan_submit":  # the submitted plan IS the checklist
                        todo.update_todos(self.plan.items(call_args[cid]))
            elif role == "tool":
                cid = msg.get("tool_call_id")
                name = call_names.get(cid, "tool")
                if name in ("edit", "todo_write"):
                    continue  # already shown as the diff card / the pinned panel
                if name == "plan_submit":
                    if not self.plan.is_rejection(content):
                        continue  # success shows as the plan panel, not a card
                    summary = tool_summary(name, call_args.get(cid, {}))
                    await turn.mount(ToolResultBlock(name, content, True, summary=summary))
                    continue
                summary = tool_summary(name, call_args.get(cid, {}))
                await turn.mount(ToolResultBlock(name, content, summary=summary))
        # A turn whose every block went somewhere else (a lone todo_write → the panel)
        # would leave a bare green rail behind; drop those, as the live path does.
        for rail in list(container.query(".turn")):
            if not rail.children:
                await rail.remove()
        container.scroll_end(animate=False)
        # After the scroll to the end: a restored gate scrolls to itself, and that
        # has to be the last word — it is the thing the session is waiting on.
        await self.plan.restore(container, call_args, call_names)

    async def _new_session(self) -> None:
        """Start a fresh session (new file + header) and clear the view."""
        self.session = ChatSession()
        self.session_path = storage.new_session_path()
        storage.write_header(
            self.session_path,
            storage.make_header(self.session_path.stem, kind="main", model=config.load().name),
        )
        self.session_depth = 0
        self.session_kind = "main"
        self.session_parent_id = None
        self._has_title = False
        self.plan.reset()
        spill.set_session(self.session_path)
        self._set_header_title("")
        await self._render_history()  # clears the pinned plan with the rest of the view
        self._reflect_view_only()  # a fresh main session is drivable again
        self._status("")
        await self._say_system("new session started")

    async def _switch_session(self, session_id: str) -> None:
        """Load another session by id and show its history."""
        self.session = ChatSession()
        self.session_path = storage.SESSIONS_DIR / f"{session_id}.jsonl"
        self.session.messages = storage.load_messages(self.session_path)
        meta = storage.read_session_meta(self.session_path) or {}
        self.session_depth = int(meta.get("depth", 0))
        self.session_kind = str(meta.get("kind", "main"))
        self.session_parent_id = meta.get("parent_id")
        self._has_title = bool(meta.get("title"))
        self.plan.reset()
        spill.set_session(self.session_path)
        self._set_header_title(meta.get("title", ""))
        await self._render_history()  # replays this session's plan into the panel
        self._reflect_view_only()
        if self.session_kind == "impl":
            self._set_mode("act")  # an impl session exists to act; planning is its parent's
        if not self.view_only:
            await self._repair_interrupted()  # a turn cut off mid-tool: fill + note
        if self.view_only:  # opened a sub-agent transcript — announce it's read-only
            await self._say_system(
                f"🔒 보기 전용 — 서브에이전트가 자동 생성한 기록(깊이 {self.session_depth})입니다. "
                "읽기만 가능해요. /new 로 새 세션을 시작하세요."
            )
        self._status("")

    async def _repair_interrupted(self) -> None:
        """Fill in the results a turn cut off mid-tool never produced.

        The API demands a result for every tool_call, so each dangling call gets a
        synthetic one, plus a note telling the model to reassess the real state
        rather than trust a half-finished summary. Appended and persisted, so a
        reopened session finds nothing left to repair.
        """
        dangling = storage.dangling_tool_calls(self.session.messages)
        if not dangling:
            return
        for call in dangling:
            # Name WHICH call was cut off: tool + subject, so three bash calls that
            # ran at once stay distinguishable. Same IN-line the result card shows.
            subject = tool_summary(call["name"], call["arguments"])
            what = f"`{call['name']}` ({subject})" if subject else f"`{call['name']}`"
            msg = {"role": "tool", "tool_call_id": call["id"],
                   "content": f"Interrupted: the {what} call did not complete."}
            self.session.messages.append(msg)
            storage.append_message(self.session_path, msg)
        note = {
            "role": "user",
            "content": (
                "[system] The previous turn was interrupted before it finished. The "
                "project may have changed on disk — re-check the actual state (files, "
                "tests) and bring the plan's checklist into line with it before "
                "continuing. If the last step did not complete, redo it."
            ),
        }
        self.session.messages.append(note)
        storage.append_message(self.session_path, note)
        await self._say_system("↻ 이전 턴이 중단됐어요 — 상태를 다시 확인하고 이어갑니다.")

    def _open_picker(self) -> None:
        """The picker needs to know which session is open (deleting it must move
        the app off the file first) and whether its turn is running (then it is
        not deletable at all)."""
        current = self.session_path.stem
        locked = current if self._anything_running() else None
        self.push_screen(SessionPicker(current=current, locked=locked), self._session_picked)

    def _session_picked(self, result: str | None) -> None:
        """SessionPicker dismissed — run the switch/new as an async worker."""
        if result == "new":
            self.run_worker(self._new_session(), exclusive=False)
        elif result:
            self.run_worker(self._switch_session(result), exclusive=False)
        else:  # closed without choosing — the open session may have been renamed there
            meta = storage.read_session_meta(self.session_path) or {}
            self._set_header_title(meta.get("title", ""))
            self._has_title = bool(meta.get("title"))

    @staticmethod
    def _edit_card(args: dict) -> Chatbox:
        """Build the green edit-diff card (path title + count chip + -/+ lines) —
        shared by the live turn and history restore."""
        path = args.get("path", "?")
        old, new = args.get("old_string", ""), args.get("new_string", "")
        text, plain = edit_diff_lines(old, new)
        added, removed = diff_stats(old, new)
        box = Chatbox("", role="tool-diff")
        box.set_rich(text, plain)
        box.border_title = f"✏ edit · {path}"
        box.border_subtitle = f"+{added} −{removed}"
        return box

    async def _say_system(self, text: str) -> None:
        """Show an informational bubble (commands, status) — never part of the session."""
        container = self.query_one("#chat-container", VerticalScroll)
        await container.mount(Chatbox(text, role="system"))
        container.scroll_end(animate=False)

    @on(PromptInput.Submitted)
    async def user_submitted(self, event: PromptInput.Submitted) -> None:
        text = event.text.strip()
        if not text:
            # An empty Enter while the gate is open is the keyboard's ▶: the card
            # is the only thing waiting, and it says so.
            if self.plan.pending:
                self.plan.settle("▶ 실행 (Enter)")
                await self.plan.start_impl_session()
            return
        # PromptInput clears itself on submit.

        # One turn at a time. A message typed mid-turn used to CANCEL it (exclusive
        # worker) and drop the work in flight; now it is refused and kept in the
        # composer. Slash commands included — /new or /sessions mid-turn would move
        # session_path out from under the worker's persist.
        if self._anything_running():
            self.query_one("#prompt", PromptInput).text = text  # not lost, not sent
            await self._say_system(
                "⏳ 진행 중이라 보내지 않았어요 — 멈추려면 Stop(Esc), 끝나면 그대로 Enter."
            )
            return

        # A sub-agent session is view-only: refuse new turns, but let slash commands
        # through so /new and /sessions stay as keyboard escape hatches.
        if self.view_only and not text.startswith("/"):
            await self._say_system(
                "🔒 보기 전용 세션(서브에이전트 기록)이라 대화를 보낼 수 없어요. "
                "/new (또는 상단 New) 로 새 세션을 시작하세요."
            )
            return

        # Text typed while the gate is open is feedback on the plan: the card settles
        # as "revise", the plan is not executed, and the message goes to the model,
        # which revises and resubmits.
        if self.plan.pending:
            self.plan.settle("✎ 수정 계속")

        if text.startswith("/"):
            # Slash commands configure the app; they never reach the LLM
            # and are not recorded in the session.
            if text == "/new":
                await self._new_session()
                return
            if text == "/sessions":
                self._open_picker()
                return
            await self._say_system(self.commands.handle(text))
            return

        self.session.add_user(text)
        storage.append_message(self.session_path, {"role": "user", "content": text})
        self._turn_question = text
        # A typed instruction is a fresh start: whatever the run was stuck on, the
        # user has now said something about it, so the previous turns of no progress
        # should not count against the turns that follow.
        self.plan.stalled = 0

        self._follow_output = True  # a new turn re-pins to the bottom to show the reply
        container = self.query_one("#chat-container", VerticalScroll)
        await container.mount(Chatbox(text, role="user"))
        await self._start_turn()

    async def _start_turn(self) -> None:
        """Mount a fresh turn rail and run the agent loop over the current history.

        Every user message comes through here; the history already holds whatever
        the previous turn ended on (a plan_submit result, a paused gate), so the
        loop simply carries on from there.
        """
        container = self.query_one("#chat-container", VerticalScroll)
        # The assistant's whole reply (thinking → tools → answer) is mounted into one
        # .turn container with a green left rail, so the steps read as one connected
        # flow rather than a flat stack. The user message stays outside it.
        self._turn = Vertical(classes="turn")
        await container.mount(self._turn)
        container.scroll_end(animate=False)

        # A snapshot copy goes to the worker, so it never shares a mutable list with
        # the main thread; reply bubbles mount lazily as events arrive. Every turn is
        # grounded by a system prompt: act gets the agent prompt, plan the planner.
        base = prompts.plan_system() if self.mode == "plan" else prompts.act_system()
        history = [{"role": "system", "content": base}, *self.session.messages]
        self._status("● waiting…")
        self._stopping = False  # a new turn clears a previous stop
        self._response_worker = self.stream_response(history, self._turn)
        self._set_send_running(True)  # the Send button becomes Stop

    def _set_mode(self, mode: str) -> None:
        """Switch modes from code: set the field FIRST — the Select's handler no-ops
        when its value already matches, so this never re-triggers itself."""
        if self.mode != mode:
            self.mode = mode
            self.query_one("#mode-select", Select).value = mode

    def _write_transcript_turn(self, event: "AhaCodeApp.ResponseComplete") -> None:
        """Append this turn to the session's readable transcript.

        The JSONL beside it holds the same thing as the MODEL sees it — one line per
        message, arguments as escaped JSON. This is the conversation as it appeared
        on screen, each turn stamped with what it cost, readable without the app.
        """
        answer = "\n\n".join(
            m["content"] for m in event.messages
            if m.get("role") == "assistant" and m.get("content")
        )
        tools = [
            f"🔧 {c['function']['name']} · "
            f"{tool_summary(c['function']['name'], self._safe_args(c)) or ''}".strip(" ·")
            for m in event.messages for c in (m.get("tool_calls") or [])
        ]
        if not (self._turn_question or answer or tools):
            return
        storage.append_turn(
            storage.transcript_path(self.session_path),
            user=self._turn_question, answer=answer, tools=tools,
            metrics=event.metrics,
        )
        self._turn_question = ""  # written once; a resumed loop is not a new question

    @staticmethod
    def _safe_args(call: dict) -> dict:
        try:
            return json.loads(call["function"]["arguments"])
        except (json.JSONDecodeError, TypeError, KeyError):
            return {}

    @staticmethod
    def _turn_metrics(stats: dict) -> dict:
        """The turn's numbers, ready to record: the same quantities the status line
        shows, kept as numbers so they can be summed later instead of re-parsed."""
        gen = stats["completion"]
        if not gen:
            return {}
        first = stats["t_first"] or stats["t_start"]
        return {
            "prompt": stats["prompt"],
            "gen": gen,
            "gen_seconds": round(max(time.monotonic() - first, 1e-9), 3),
            "ttft": round((stats["t_first"] - stats["t_start"]) if stats["t_first"] else 0.0, 3),
            "model": config.load().name,
        }

    @staticmethod
    def _format_stats(stats: dict) -> str:
        """One-line token/speed summary for the status bar (empty if no output)."""
        gen = stats["completion"]
        if not gen:
            return ""
        first = stats["t_first"] or stats["t_start"]
        gen_elapsed = max(time.monotonic() - first, 1e-9)
        ttft = (stats["t_first"] - stats["t_start"]) if stats["t_first"] else 0.0
        return f"prompt {stats['prompt']} · gen {gen} · {gen / gen_elapsed:.0f} tok/s · ttft {ttft:.1f}s"

    def _status(self, text: str) -> None:
        """Push live turn status to the bar (empty = idle)."""
        self._last_status = text
        # query, not query_one — same reason as _set_send_running. _last_status above
        # is the state; the bar is only its display, so no bar is not an error.
        for bar in self.query(ModelBar).results(ModelBar):
            bar.set_status(text)

    def refresh_config_ui(self, *, reload_models: bool = False) -> None:
        """Re-read the config into the chrome that displays it.

        The seam commands.py calls instead of importing widgets: a command edits
        the config file, then says so here. reload_models is for /url alone — a
        new endpoint offers a different model list, and fetching it is a request.
        """
        bar = self.query_one(ModelBar)
        bar.refresh_state()
        if reload_models:
            bar.load_models()
        self._set_header_endpoint()

    @on(ModelBar.ModelChosen)
    async def model_chosen(self, event: ModelBar.ModelChosen) -> None:
        await self._say_system(self.commands.switch_model(event.name))

    @on(ModelBar.ModeChosen)
    async def mode_chosen(self, event: ModelBar.ModeChosen) -> None:
        if event.mode == self.mode:
            return  # programmatic re-sync from the Select, not a real switch
        self.mode = event.mode
        if self.mode == "plan":
            await self._say_system("plan mode ON — read-only tools; the model plans, not acts.")
        else:
            await self._say_system("act mode — full tools (bash asks first).")

    @on(ModelBar.AutoApproveChanged)
    async def auto_approve_changed(self, event: ModelBar.AutoApproveChanged) -> None:
        if event.value == self.auto_approve:
            return  # programmatic re-sync, not a real toggle
        self.auto_approve = event.value
        if event.value:
            await self._say_system(
                "auto-approve ON — tools run without asking "
                "(dangerous commands are still blocked)."
            )
        else:
            await self._say_system("auto-approve OFF — tools ask first.")

    @on(Button.Pressed, "#plan-gate-run")
    async def _on_plan_gate_run(self, event: Button.Pressed) -> None:
        event.stop()
        self.plan.settle("▶ 실행")
        await self.plan.start_impl_session()

    @on(Button.Pressed, "#plan-gate-continue")
    async def _on_plan_gate_continue(self, event: Button.Pressed) -> None:
        """✎ 수정 — the plan stays on screen; the user says what to change and the
        next turn revises it (plan_submit again replaces the plan file)."""
        event.stop()
        self.plan.settle("✎ 수정 계속")
        self.query_one("#prompt", PromptInput).focus()

    @property
    def view_only(self) -> bool:
        """Is the open session browsable-but-not-drivable? A sub-agent session
        (depth > 0) is a machine-authored child transcript, and its depth gates the
        `task` tool off (see registry_for), so typing into it would dead-end. We let
        the user OPEN one to read it, but refuse new turns; /new leaves. Derived from
        session_depth — the same axis as the task gate — so the lock can't drift."""
        return self.session_depth > 0

    def _reflect_view_only(self) -> None:
        """Mirror the read-only state in the composer's hint line (the up-front
        signal, before a blocked keypress teaches it the hard way)."""
        prompt = self.query_one("#prompt", PromptInput)
        prompt.border_subtitle = (
            "🔒 보기 전용 · /new 로 새 세션"
            if self.view_only
            else "Enter to send · Shift+Enter for newline"
        )

    def _registry_for_mode(self) -> dict:
        """The tools this session may use this turn. Plan mode stays read-only (no
        side effects — and no `task`, since a sub-agent could act), but it does get
        the search tools: planning means investigating first, and without them the
        model would have to already know every path. Act mode gets the base tools
        plus `task`, but only while depth < subagent_depth so a sub-agent at the
        limit cannot recurse (see tools.registry_for)."""
        if self.mode == "plan":
            return {
                "read": tools.READ,
                "glob": tools.GLOB,
                "grep": tools.GREP,
                "plan_submit": tools.PLAN_SUBMIT,  # the way OUT of plan mode
            }
        return tools.registry_for(self.session_depth, config.load().subagent_depth)

    @on(ResponseComplete)
    async def response_complete(self, event: ResponseComplete) -> None:
        self._set_send_running(False)  # turn done — Stop reverts to Send
        self._prune_empty_turn()
        # Shared state is only ever touched on the main thread. Persist the whole
        # turn — assistant text, tool calls, and tool results alike.
        for msg in event.messages:
            self.session.messages.append(msg)
            storage.append_message(self.session_path, msg)
        if event.metrics:
            # Recorded, not just shown: the status line is gone the moment the next
            # turn starts, and "how fast has this been?" had no answer afterwards.
            storage.append_stats(self.session_path, event.metrics)
        self._write_transcript_turn(event)
        self._status(event.stats)
        if event.prompt_tokens:
            self._last_prompt_tokens = event.prompt_tokens
        # First real reply of an untitled session -> generate a title in the background.
        if not self._has_title and any(m.get("role") == "assistant" for m in self.session.messages):
            self._has_title = True
            self.generate_title(list(self.session.messages), self.session_path)
        # An impl session exists to finish its plan, and behind a folded panel
        # "finished talking" looks the same as "finished the plan". So: snapshot
        # where it stands after every turn, and say so.
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
        self._status("")

    def _fold_thinking(self, boxes: dict) -> None:
        """Collapse this turn's thinking block once the reasoning is done.

        The block stays open while it streams, then folds when the answer or a
        tool call begins (auto-collapsing finished reasoning).
        """
        block = boxes["thinking"]
        if block is not None:
            block.done()
            boxes["thinking"] = None

    async def _render_event(
        self, event, boxes: dict, container: VerticalScroll
    ) -> None:
        """Main-thread renderer: mount/append bubbles as the loop emits events.

        Runs via call_from_thread (which awaits this coroutine), so each mount is
        complete before the next append — no races with the worker thread. `boxes`
        holds the current turn's live thinking/answer bubbles; a tool call ends a
        turn, so the next thinking/text opens fresh bubbles.
        """
        if isinstance(event, Notice):
            # Harness-authored, not the model's answer — the same neutral bubble a
            # slash command gets. It ends the current answer, so the next TextDelta
            # opens a fresh one below it.
            self._fold_thinking(boxes)
            boxes["answer"] = None
            await container.mount(Chatbox(event.text, role="system"))
        elif isinstance(event, Phase):
            # Harness work with a duration, on the same clock the tools use: the
            # point is a number that MOVES, because a static line is the picture a
            # deadlock makes. A sub-agent's phases are skipped like its tools — its
            # card carries its own clock, and the gate check tells the two apart.
            if not boxes.get("gate"):
                return
            if event.done:
                self._running_tools.pop(_PHASE_ID, None)
            else:
                self._running_tools[_PHASE_ID] = (event.name, time.monotonic())
                self._status(f"● {event.name} · 0초")
        elif isinstance(event, ThinkingDelta):
            if boxes["thinking"] is None:
                boxes["thinking"] = ThinkingBlock()  # foldable; starts expanded
                await container.mount(boxes["thinking"])
            boxes["thinking"].append_chunk(event.text)
            self._status("● thinking…")
        elif isinstance(event, TextDelta):
            self._fold_thinking(boxes)  # answer starting → auto-collapse the reasoning
            if boxes["answer"] is None:
                # markdown=True: render the answer as Markdown so ```code``` and
                # ```diff fences become highlighted blocks.
                boxes["answer"] = Chatbox("", role="assistant", markdown=True)
                await container.mount(boxes["answer"])
            boxes["answer"].append_chunk(event.text)
            self._status("● generating…")
        elif isinstance(event, ToolCallDelta):
            if event.name == "todo_write":
                return  # todo_write shows in the panel on its final call, not a bubble
            if event.name == "edit":
                self._fold_thinking(boxes)
                boxes["answer"] = None
                self._status("● editing…")
                return  # edit's coloured diff is rendered on the final ToolCall
            self._fold_thinking(boxes)
            boxes["answer"] = None
            if event.name == "write":
                # write streams its content live into one bubble
                buf = boxes["tool_buf"].get(event.index, "") + event.fragment
                boxes["tool_buf"][event.index] = buf
                box = boxes["tool"].get(event.index)
                if box is None:
                    box = Chatbox("", role="tool-call")
                    await container.mount(box)
                    boxes["tool"][event.index] = box
                box._content = _render_tool_stream(event.name, buf)
                box.update(box._content)
                self._status("● writing…")
            else:
                # bash / read / grep …: no call bubble — the result card shows the
                # command in its title (IN) and the output in its body (OUT), the
                # Claude Code shape. The status bar reports progress until it lands.
                self._status(f"● running {event.name}…")
        elif isinstance(event, ToolCall):
            self._fold_thinking(boxes)  # fold reasoning; next turn opens fresh bubbles
            boxes["answer"] = None
            # Remember the call's input so the result card can title itself with it.
            boxes.setdefault("call_args", {})[event.id] = event.arguments
            # Only this loop's own tools drive the status line. A sub-agent reports
            # inside its card, and with several running in parallel their tools would
            # otherwise take turns overwriting each other in the one status line.
            if boxes.get("gate"):
                self._running_tools[event.id] = (event.name, time.monotonic())
            if event.name == "todo_write":  # goes to the pinned panel, not a bubble
                items = event.arguments.get("items", [])
                # Only THIS loop owns the pinned plan. A sub-agent runs with its own
                # boxes (no "gate" key); planning its private sub-task used to
                # overwrite the parent's checklist wholesale.
                if boxes.get("gate"):
                    self.plan.note_todo_update(self.query_one(TodoPanel), items)
                boxes["tool"].clear()
                boxes["tool_buf"].clear()
                self._status("● planning…")
                return
            if event.name == "plan_submit":  # the plan goes to the panel; the gate
                if boxes.get("gate"):        # opens when its result confirms the save
                    self.query_one(TodoPanel).update_todos(self.plan.items(event.arguments))
                boxes["tool"].clear()
                boxes["tool_buf"].clear()
                self._status("● submitting plan…")
                return
            if event.name == "edit":  # green diff card (shared with history restore)
                await container.mount(self._edit_card(event.arguments))
                boxes["tool"].clear()
                boxes["tool_buf"].clear()
                self._status("● editing…")
                return
            # write streamed a live content bubble (kept); other tools show only the
            # result card, so there's no call bubble to keep here.
            boxes["tool"].clear()
            boxes["tool_buf"].clear()
            self._status(f"● running {event.name}…")
        elif isinstance(event, ToolResult):
            self._running_tools.pop(event.id, None)
            if event.name == "todo_write":
                return  # already reflected in the pinned panel
            if event.name == "plan_submit" and not event.is_error and boxes.get("gate"):
                # A rejected submission falls through to the error card below, so the
                # user sees why; the model already has the reason and resubmits.
                args = boxes.get("call_args", {}).get(event.id, {})
                path = event.output.split(" (", 1)[0].removeprefix("Plan saved to ")
                await self.plan.open(
                    [it["content"] for it in self.plan.items(args)],
                    str(args.get("summary", "")).strip(), path, container,
                )
                return
            if event.name == "edit" and not event.is_error:
                return  # a successful edit is already shown as the diff card
            if event.name == "task":
                return  # the sub-agent's own 🤖 card already shows its flow + result
            # One foldable card: the command/path in the title (IN), the output in
            # the body (OUT). Long output / errors fold away.
            summary = tool_summary(event.name, boxes.get("call_args", {}).get(event.id, {}))
            await container.mount(
                ToolResultBlock(event.name, event.output, event.is_error, summary=summary)
            )
        # Follow the stream ONLY while _follow_output is set (a watcher on scroll_y
        # drives it — see _update_follow), so a manual scroll-up during streaming
        # STAYS off. The old per-chunk "near the bottom?" snap re-pinned every chunk,
        # making it impossible to scroll away mid-answer.
        if self._follow_output:
            self.query_one("#chat-container", VerticalScroll).scroll_end(animate=False)

    def action_copy_answer(self) -> None:
        """Copy the last assistant answer to the system clipboard (OSC 52 — works over
        SSH). The TUI captures the mouse, so terminal drag-select is unreliable; this
        gives a one-key copy of the reply. The full transcript also lives in the
        session JSONL for anything more."""
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

    def _tick_progress(self) -> None:
        """Once a second, say how long the running work has been running.

        A status line that never changes is the same picture as a frozen app — the
        number is the whole difference between a slow test run and a deadlock.
        Sub-agent cards tick their own: with several running in parallel, one status
        line cannot speak for all of them.
        """
        for card in list(self.query(SubagentCard)):
            card.tick()
        if not self._running_tools:
            return  # nothing to time; leave whatever status is up alone
        now = time.monotonic()
        oldest = min(self._running_tools.values(), key=lambda v: v[1])
        seconds = int(now - oldest[1])
        if len(self._running_tools) > 1:
            self._status(f"● 도구 {len(self._running_tools)}개 · {seconds}초")
        else:
            self._status(f"● {oldest[0]} · {seconds}초")

    def _force_exit(self) -> None:  # seam: tests replace this rather than dying
        os._exit(0)

    async def action_quit(self) -> None:
        """Quit means quit, whatever a worker is stuck on.

        Textual restores the terminal promptly, then the process waits on its
        threads — and a worker blocked on a socket read holds that for the whole
        request timeout (15 minutes by default). The cooperative stop is checked
        only between events, so nothing the user presses reaches a blocked read.

        Closing the response would unblock this one read and leave the next kind of
        stuck call to bring the bug back. So: ask the workers to stop, then go
        anyway once an in-flight session append has had time to land. Nearly always
        the process is gone before the timer fires.
        """
        self._stopping = True
        worker = getattr(self, "_response_worker", None)
        if worker is not None and worker.is_running:
            worker.cancel()
        # Headless means run_test: there is no terminal being held hostage, and the
        # process to leave would be the test runner's. The backstop is for a real
        # session, where a wedged worker is the user's problem.
        if not self.is_headless:
            leave = threading.Timer(QUIT_GRACE_SECONDS, self._force_exit)
            leave.daemon = True  # never the reason the process stays up
            leave.start()
        self.exit()

    def action_stop(self) -> None:
        """Cancel whatever is in flight (cooperative — the loops check is_cancelled).

        Two workers can now run at once (a chat turn and a plan run live in different
        exclusive groups), and "stop" means stop what is running — so both are checked.
        """
        # Raised before anything else: sub-agents queue on the approval lock, so a stop
        # must be visible to the ones still waiting or each pops its own modal after
        # the user has already said stop.
        self._stopping = True
        stopped = False
        worker = getattr(self, "_response_worker", None)
        if worker is not None and worker.is_running:
            worker.cancel()
            stopped = True
        if stopped:
            self._status("■ stopped")
            self._set_send_running(False)
            # Fold the pinned plan: a stopped run is when the user wants the chat
            # area back, and the plan is the widest thing on screen. Presentation
            # only — the steps survive. query, not query_one: this can be called
            # from the approval modal, whose screen is not the one being queried.
            for panel in self.query(TodoPanel):
                panel.set_collapsed(True)

    @work(thread=True, exit_on_error=False)
    def generate_title(self, messages: list[dict], path) -> None:
        """Ask the model for a short session title (background, non-streaming)."""
        convo = "\n".join(
            f"{m['role']}: {m.get('content', '')}"
            for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        )[:1500]
        try:
            title = client.complete(
                [{"role": "system", "content": prompts.title_system()}, {"role": "user", "content": convo}]
            )
        except Exception:
            return  # a failed title is not worth surfacing; leave it untitled
        title = title.strip().strip('"').strip()[:60]
        if title:
            self.call_from_thread(storage.set_title, path, title)
            if path == self.session_path:  # not if the user already switched away
                self.call_from_thread(self._set_header_title, title)

    def _approve_tool(self, call) -> bool:
        """Approval handshake for one tool call, shared by the agent loop and every
        sub-agent. Auto-approve short-circuits; otherwise push a modal on the main
        thread and block THIS worker thread on an Event until the user answers.
        Parallel sub-agents each need approval at once but only one dialog can be on
        screen, so they queue on the lock."""
        # Rule-based pre-approval comes FIRST — that is what keeps a parallel fan-out
        # from serialising behind a queue of dialogs (only one fits on screen). It
        # cannot widen anything: _gate_tool ran the denylist before calling us, so a
        # rule skips the question, never the safety gate.
        if permissions.allowed(call.name, call.arguments):
            return True
        if self.auto_approve:  # denylist already hard-blocked the dangerous ones
            return True
        if getattr(self, "_stopping", False):
            # The user has already stopped the run. Whatever is still queued behind the
            # approval lock must not put another dialog on screen — answering "stop"
            # once has to be enough, however many sub-agents were mid-flight.
            return False
        with self._approval_lock:
            if getattr(self, "_stopping", False):  # stopped while we waited our turn
                return False
            # Cross-thread handshake: the loop runs on a worker thread but the modal
            # lives on the main thread. Push it (non-blocking) via call_from_thread,
            # then block here until the modal's dismiss callback sets the Event.
            answered = threading.Event()
            verdict: dict[str, bool] = {}

            def ask() -> None:
                def on_dismiss(approved: bool | None) -> None:
                    verdict["ok"] = bool(approved)
                    answered.set()

                # Raw arguments → the modal renders a per-tool preview (write → code,
                # edit → diff) inside a scrollable box.
                self.push_screen(ApprovalModal(call.name, call.arguments), on_dismiss)

            self.call_from_thread(ask)
            answered.wait()
            return verdict.get("ok", False)

    def _make_subagent_ctx(self, parent_path, parent_depth, container, worker, approve):
        """Build the AgentContext whose run_subagent spawns a child agent into a
        nested card and runs it to completion.

        run_subagent blocks, but agent.run puts parallelizable `task` calls on a
        thread pool, so two delegations in one turn really do run CONCURRENTLY,
        bounded only by the gateway's gate. A per-level factory: each child parents
        to THIS level and mounts inside this card, so the tree nests at any depth —
        and stops itself, since a child at the depth limit gets no task tool.
        """
        def run_subagent(prompt: str, description: str) -> str:
            cfg = config.load()
            child_depth = parent_depth + 1
            child_path = storage.new_session_path()
            storage.write_header(child_path, storage.make_header(
                child_path.stem, parent_id=parent_path.stem, kind="subagent",
                relation="delegate",  # a task fan-out — the ⑂ edge in the session tree
                depth=child_depth, model=cfg.name, title=(description or prompt)[:40],
            ))
            card = SubagentCard(description or "task", cfg.name)
            self.call_from_thread(container.mount, card)
            child_boxes = {"thinking": None, "answer": None, "tool": {}, "tool_buf": {}, "call_args": {}}

            def child_emit(event) -> None:
                if isinstance(event, Usage):
                    return  # child token accounting isn't surfaced in this slice
                self.call_from_thread(self._render_event, event, child_boxes, card.body)

            result = subagent.run(
                prompt,
                emit=child_emit,
                approve=approve,  # the child's own bash/write are confirmed too
                registry=tools.registry_for(child_depth, cfg.subagent_depth),
                # a grandchild parents to THIS child, one level deeper, in its card
                ctx=self._make_subagent_ctx(child_path, child_depth, card.body, worker, approve),
                is_cancelled=lambda: worker.is_cancelled,
            )
            for msg in result.messages:
                storage.append_message(child_path, msg)
            # Fold the card now the child is done (its answer stays one click away).
            tool_count = sum(1 for m in result.messages if m.get("role") == "tool")
            self.call_from_thread(card.done, tool_count)
            return result.result

        return subagent.AgentContext(run_subagent=run_subagent, session_path=parent_path)

    # exclusive=True: a new message cancels the previous worker.
    # exit_on_error=False: a failing worker must not take the whole app down.
    @work(thread=True, exclusive=True, exit_on_error=False)
    def stream_response(self, messages: list[dict], turn) -> None:
        """Run the agent loop in a thread, rendering its events into the chat."""
        worker = get_current_worker()
        container = turn  # the reply's blocks mount into this turn's rail container
        # Current turn's live bubbles; _render_event fills these in on the main thread.
        # "gate": only this loop may open the plan gate — a sub-agent's boxes carry
        # no such key, so a child's todo_write can never pause its parent.
        boxes: dict = {"thinking": None, "answer": None, "tool": {}, "tool_buf": {},
                       "call_args": {}, "gate": True}

        stats = {"prompt": 0, "completion": 0, "t_start": time.monotonic(), "t_first": None,
                 "last_prompt": None}

        def emit(event) -> None:
            if isinstance(event, Usage):  # accounting only — never a bubble
                # One usage trailer per model call, so this is also the round
                # counter the stall backstop needs — counted here rather than off
                # tool calls, which arrive several to a round.
                self.plan.rounds_since_step += 1
                stats["prompt"] += event.prompt_tokens
                stats["completion"] += event.completion_tokens
                # The LAST turn's prompt size is what the next turn has to fit under;
                # the running total above is only for the throughput readout.
                stats["last_prompt"] = event.prompt_tokens
                return
            if stats["t_first"] is None and isinstance(event, (ThinkingDelta, TextDelta)):
                stats["t_first"] = time.monotonic()
            # Hop to the main thread to touch widgets. call_from_thread blocks the
            # worker until the UI has rendered — built-in backpressure.
            self.call_from_thread(self._render_event, event, boxes, container)

        approve = self._approve_tool
        ctx = self._make_subagent_ctx(
            self.session_path, self.session_depth, container, worker, approve
        )

        stall_rounds = self.plan.begin_turn()  # arms the round backstop for this turn

        # The turn's mode picks its thinking budget (config.thinking_budget_for):
        # plan thinks deep, impl shallow, a plain act turn uses the global.
        turn_mode = "plan" if self.mode == "plan" else ("impl" if self.session_kind == "impl" else None)
        try:
            with client.mode(turn_mode):
                new_messages = agent.run(
                    messages,
                    emit=emit,
                    is_cancelled=lambda: worker.is_cancelled,
                    approve=approve,                    # bash and friends are confirmed first
                    registry=self._registry_for_mode(),  # plan mode = read-only subset
                    ctx=ctx,                            # task delegates through this
                    max_turns=self.plan.max_turns(),    # larger for an impl session
                    prompt_tokens=self._last_prompt_tokens,  # measured, for compaction
                    # the plan gate, and the round-level stall backstop
                    should_pause=lambda: self.plan.should_pause(stall_rounds),
                )
        except Exception as exc:
            # Server 500, timeout, connection refused... all become a bubble, not a crash.
            summary = f"{type(exc).__name__}: {exc}"[:300]
            self.post_message(self.ResponseFailed(summary))
            return
        if worker.is_cancelled:
            # Stopped: keep every message that FINISHED (the loop returns whole
            # assistant+tool rounds; the one in flight is dropped), so the transcript
            # the model resumes from knows what it already did. Continue, never restart.
            self.post_message(self.ResponseComplete(new_messages, "■ stopped", stats["last_prompt"]))
            return
        self.post_message(self.ResponseComplete(
            new_messages, self._format_stats(stats), stats["last_prompt"],
            self._turn_metrics(stats),
        ))


app = AhaCodeApp


def main() -> None:
    """The `ahacode` console script (see [project.scripts]). Its job is to exist:
    an installed entry point is what lets AhaCode be launched from *another*
    project's directory, which is the whole point of workspace.PROJECT_ROOT being
    the launch directory rather than wherever this file was installed."""
    AhaCodeApp().run()


if __name__ == "__main__":
    main()
