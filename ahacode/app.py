import json
import re
import threading
import time
from dataclasses import dataclass, replace

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
    agent, client, config, orchestrator, permissions, prompts, storage, subagent, tools,
)
from ahacode.tools import spill
from ahacode.events import (
    Notice, TextDelta, ThinkingDelta, ToolCall, ToolCallDelta, ToolResult, Usage,
)
from ahacode.render import diff_stats, edit_diff_lines, tool_summary
from ahacode.session import ChatSession
from ahacode.widgets.approval_modal import ApprovalModal
from ahacode.widgets.chatbox import Chatbox
from ahacode.widgets.header_bar import HeaderBar
from ahacode.widgets.plan_gate import PlanGate
from ahacode.widgets.prompt_input import PromptInput
from ahacode.widgets.session_picker import SessionPicker
from ahacode.widgets.subagent_card import SubagentCard
from ahacode.widgets.thinking import ThinkingBlock
from ahacode.widgets.tool_result import ToolResultBlock
from ahacode.widgets.todo_panel import TodoPanel
from ahacode.widgets.model_bar import ModelBar


# System prompts now live in ahacode/prompts.py (assembled per mode/model).

# Eye-friendly Markdown palette. Rich's defaults paint headings magenta and inline
# code "bold cyan on black" — harsh reds/boxes on a dark terminal. We push softer
# styles onto the app console; a Markdown renderable resolves these by name at draw
# time (verified: pushing a theme recolours Static-rendered Markdown in Textual).
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

    write is shown as a path header + streamed content (the roo trick: pull known
    fields out early); every other tool shows its raw accumulating args.
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
        # Session state lives in a plain Python object, decoupled from widgets.
        self.session = ChatSession()
        self.mode = "act"  # "act" (full tools) or "plan" (read-only + todo_write)
        self._last_status = ""
        self.auto_approve = False  # session-only: skip the approval modal when on
        # Auto-scroll follows the stream only while the view is pinned to the bottom.
        # A watcher on the chat scroller flips this off the moment the user scrolls up
        # (so a manual scroll during streaming sticks) and back on when they return.
        self._follow_output = True
        self._approval_lock = threading.Lock()  # one approval modal at a time (parallel children queue)
        # The server's prompt-token count for the last request, carried across turns
        # so context compaction measures the real thing instead of an estimate.
        self._last_prompt_tokens: int | None = None
        # Plan gate: the loop pauses between turns while this is set, so the user can
        # decide whether a fresh multi-step plan runs step-by-step or straight through.
        # Written on the main thread, read by the worker's should_pause — a single
        # bool, so no lock is needed.
        self._plan_gate_pending = False
        self._plan_gate: PlanGate | None = None
        self._gated_plan: tuple[str, ...] | None = None  # the plan we already asked about
        latest = storage.latest_session()
        if latest:  # resume the most recent session
            self.session_path = latest
            self.session.messages = storage.load_messages(latest)
        else:  # first run: start a new session with a header (records model + kind)
            self.session_path = storage.new_session_path()
            storage.write_header(
                self.session_path,
                storage.make_header(
                    self.session_path.stem, kind="main", model=config.load().name
                ),
            )
        # Skip auto-titling if this (resumed) session already has a title.
        self._has_title = bool((storage.read_session_meta(self.session_path) or {}).get("title"))
        spill.set_session(self.session_path)
        # Depth in the session tree (0 = a main session); gates whether this session
        # is offered the `task` tool — a sub-agent at the limit cannot recurse.
        self.session_depth = int((storage.read_header(self.session_path) or {}).get("depth", 0))

    @dataclass
    class ResponseComplete(Message):
        """Posted by the worker once the agent loop finishes a response.

        Carries every message the loop appended (assistant, tool, assistant, ...)
        so the main-thread handler can persist the whole turn at once.
        """

        messages: list[dict]
        stats: str = ""
        prompt_tokens: int | None = None  # the server's count for this turn's request

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
        meta = storage.read_session_meta(self.session_path) or {}
        self._set_header_title(meta.get("title", ""))
        self._set_header_endpoint()
        await self._render_history()
        self._reflect_view_only()  # resumed session is a main one, but stay correct
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

    @on(Button.Pressed, "#new-session-btn")
    async def _on_new_session_button(self, event: Button.Pressed) -> None:
        event.stop()
        await self._new_session()

    @on(Button.Pressed, "#open-sessions-btn")
    def _on_open_sessions_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.push_screen(SessionPicker(), self._session_picked)

    @on(Button.Pressed, "#send-btn")
    def _on_send_button(self, event: Button.Pressed) -> None:
        event.stop()
        worker = getattr(self, "_response_worker", None)
        if worker is not None and worker.is_running:
            self.action_stop()  # the button doubles as Stop while a turn streams
        else:
            self.query_one("#prompt", PromptInput).submit()

    def _set_send_running(self, running: bool) -> None:
        """Flip the composer button between Send (idle) and Stop (streaming)."""
        btn = self.query_one("#send-btn", Button)
        btn.label = "■ Stop" if running else "↑ Send"
        btn.variant = "error" if running else "primary"

    def _prune_empty_turn(self) -> None:
        """Drop the turn's rail if the reply produced no blocks (immediate error)."""
        turn = getattr(self, "_turn", None)
        if turn is not None and turn.is_mounted and not turn.children:
            turn.remove()
        self._turn = None

    async def _render_history(self) -> None:
        """Clear the chat and remount the session's messages, matching the live
        rendering: each assistant turn under a .turn rail, bash/read as one titled
        card (command/path in the title), edit as a diff card, todo_write into the
        pinned panel — no raw tool-call bubbles, so a reloaded session looks exactly
        like the turn that made it.

        This is also the ONE owner of the pinned plan: the panel is cleared here and
        refilled from the history, so its contents are always a function of the open
        session's messages. Session switches go through here, which is what keeps a
        previous session's plan from lingering behind a hidden panel (`/run` reads it).
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
            elif role == "tool":
                cid = msg.get("tool_call_id")
                name = call_names.get(cid, "tool")
                if name in ("edit", "todo_write"):
                    continue  # already shown as the diff card / the pinned panel
                summary = tool_summary(name, call_args.get(cid, {}))
                await turn.mount(ToolResultBlock(name, content, summary=summary))
        # A turn whose every block went somewhere else (a lone todo_write → the panel)
        # would leave a bare green rail behind; drop those, as the live path does.
        for rail in list(container.query(".turn")):
            if not rail.children:
                await rail.remove()
        container.scroll_end(animate=False)

    async def _new_session(self) -> None:
        """Start a fresh session (new file + header) and clear the view."""
        self.session = ChatSession()
        self.session_path = storage.new_session_path()
        storage.write_header(
            self.session_path,
            storage.make_header(self.session_path.stem, kind="main", model=config.load().name),
        )
        self.session_depth = 0
        self._has_title = False
        self._reset_plan_gate()
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
        self._has_title = bool(meta.get("title"))
        self._reset_plan_gate()
        spill.set_session(self.session_path)
        self._set_header_title(meta.get("title", ""))
        await self._render_history()  # replays this session's plan into the panel
        self._reflect_view_only()
        if self.view_only:  # opened a sub-agent transcript — announce it's read-only
            await self._say_system(
                f"🔒 보기 전용 — 서브에이전트가 자동 생성한 기록(깊이 {self.session_depth})입니다. "
                "읽기만 가능해요. /new 로 새 세션을 시작하세요."
            )
        self._status("")

    def _session_picked(self, result: str | None) -> None:
        """SessionPicker dismissed — run the switch/new as an async worker."""
        if result == "new":
            self.run_worker(self._new_session(), exclusive=False)
        elif result:
            self.run_worker(self._switch_session(result), exclusive=False)

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
            return
        # PromptInput clears itself on submit.

        # A sub-agent session is view-only: refuse new turns, but let slash commands
        # through so /new and /sessions stay as keyboard escape hatches.
        if self.view_only and not text.startswith("/"):
            await self._say_system(
                "🔒 보기 전용 세션(서브에이전트 기록)이라 대화를 보낼 수 없어요. "
                "/new (또는 상단 New) 로 새 세션을 시작하세요."
            )
            return

        # A message typed while the gate is open answers it: the user moved on, so
        # the plan is not executed and the paused loop is not resumed.
        if self._plan_gate_pending:
            self._settle_plan_gate("다른 지시로 진행")

        if text.startswith("/"):
            # Slash commands configure the app; they never reach the LLM
            # and are not recorded in the session.
            if text == "/new":
                await self._new_session()
                return
            if text == "/sessions":
                self.push_screen(SessionPicker(), self._session_picked)
                return
            if text == "/run":  # execute the current plan structurally (a long worker)
                await self._start_plan_run()
                return
            await self._say_system(self._handle_command(text))
            return

        self.session.add_user(text)
        storage.append_message(self.session_path, {"role": "user", "content": text})

        self._follow_output = True  # a new turn re-pins to the bottom to show the reply
        container = self.query_one("#chat-container", VerticalScroll)
        await container.mount(Chatbox(text, role="user"))
        await self._start_turn()

    async def _start_turn(self) -> None:
        """Mount a fresh turn rail and run the agent loop over the current history.

        Shared by a new user message and by the plan gate's "continue" path: the
        history there already ends with the todo_write result, so re-entering the
        loop simply carries on from where the pause stopped it.
        """
        container = self.query_one("#chat-container", VerticalScroll)
        # The assistant's whole reply (thinking → tools → answer) is mounted into one
        # .turn container with a green left rail, so the steps read as one connected
        # flow rather than a flat stack. The user message stays outside it.
        self._turn = Vertical(classes="turn")
        await container.mount(self._turn)
        container.scroll_end(animate=False)

        # Run the agent loop in a worker. A snapshot copy is passed so the worker
        # never shares a mutable list with the main thread; bubbles for the reply
        # are mounted lazily as loop events arrive (turn count is not known ahead).
        # Every turn is grounded by a system prompt: act gets the full agent prompt
        # (base + live environment), plan the planner.
        base = prompts.plan_system() if self.mode == "plan" else prompts.act_system()
        history = [{"role": "system", "content": base}, *self.session.messages]
        self._status("● waiting…  (esc to stop)")
        self._response_worker = self.stream_response(history, self._turn)
        self._set_send_running(True)  # the Send button becomes Stop

    async def _start_plan_run(self) -> None:
        """/run — hand the current plan (TodoPanel steps) to the structural runner:
        each step delegated to a fresh sub-agent, in order. The plan was reviewed by
        the user first (Claude Code's plan → approve → execute), so this is explicit."""
        if self.view_only:  # a sub-agent transcript can't drive new work
            await self._say_system("🔒 보기 전용 세션에서는 계획을 실행할 수 없어요. /new 로 시작하세요.")
            return
        steps = [it["content"] for it in self.query_one(TodoPanel).items if it.get("content")]
        if not steps:
            await self._say_system("실행할 계획이 없어요 — plan 모드에서 todo_write 로 계획을 먼저 세우세요.")
            return
        # The plan's origin is the last thing the user asked — the overall task.
        task = next(
            (m["content"] for m in reversed(self.session.messages) if m.get("role") == "user"), ""
        )
        if not task:
            await self._say_system("계획의 원본 작업(직전 사용자 메시지)을 찾지 못했어요.")
            return
        container = self.query_one("#chat-container", VerticalScroll)
        # /run IS execution, so it belongs in act mode. The sub-agents it spawns were
        # always given the full tool set (run_subagent builds their registry from
        # tools.registry_for, never from self.mode), so plan mode had been quietly
        # watching writes happen while the bar still said "plan". Switching here
        # changes nothing about what runs — it stops the bar from lying, and leaves
        # the session in the mode the user is actually in afterwards.
        if self.mode != "act":
            self.mode = "act"  # set first: the Select's handler no-ops when it matches
            self.query_one("#mode-select", Select).value = "act"
            await self._say_system("계획 실행이라 act 모드로 전환했어요.")
        await self._say_system(
            f"▶ 계획 실행 — {len(steps)}단계를 각각 새 컨텍스트의 서브에이전트에 순차 위임합니다."
        )
        self._turn = Vertical(classes="turn")
        await container.mount(self._turn)
        container.scroll_end(animate=False)
        self._status("● running plan…  (esc to stop)")
        self._response_worker = self.run_plan_response(task, steps, self._turn)
        self._set_send_running(True)

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
        self.query_one(ModelBar).set_status(text)

    def _switch_model(self, name: str) -> str:
        """Persist a model choice. The server actually loads it on the next request."""
        config.save(replace(config.load(), name=name))
        client.reset()  # next request picks up the new config
        self.query_one(ModelBar).refresh_state()
        return f"model switched to: {name} (loads on the next message — may take a while)"

    def _handle_command(self, text: str) -> str:
        """Handle a /command typed into the chat input; returns the reply text."""
        parts = text.split()
        cmd, args = parts[0].lower(), parts[1:]
        if cmd == "/help":
            return (
                "Commands:\n"
                "  /model           show the current model and endpoint\n"
                "  /model <name>    switch to a different model\n"
                "  /url <base_url>  switch to a different endpoint\n"
                "  /think <n>       per-turn thinking budget in tokens (off = unbounded)\n"
                "  /allow           list the rules that skip the approval prompt\n"
                "  /allow <rule>    add one, e.g. /allow bash:uv run pytest*\n"
                "  /run             execute the current plan — each step a fresh sub-agent\n"
                "  /new             start a new session\n"
                "  /sessions        switch between sessions\n"
                "  /help            this message"
            )
        if cmd == "/model":
            cfg = config.load()
            if not args:
                return f"model: {cfg.name}\nendpoint: {cfg.base_url}"
            return self._switch_model(args[0])
        if cmd == "/url":
            cfg = config.load()
            if not args:
                return f"endpoint: {cfg.base_url}"
            config.save(replace(cfg, base_url=args[0]))
            client.reset()
            bar = self.query_one(ModelBar)
            bar.refresh_state()
            bar.load_models()  # a new endpoint offers a new model list
            self._set_header_endpoint()
            return f"endpoint switched to: {args[0]}"
        if cmd == "/allow":
            cfg = config.load()
            if not args:
                if not cfg.allow_rules:
                    return (
                        "no allow rules — every side-effecting tool asks first.\n"
                        "add one with e.g.  /allow bash:uv run pytest*"
                    )
                return "allow rules:\n" + "\n".join(f"  {r}" for r in cfg.allow_rules)
            # Re-split on the raw text, not `parts`: a rule keeps its spaces
            # ("bash:git status*" is one rule, not two words).
            rule = text.split(maxsplit=1)[1].strip()
            if ":" not in rule and rule not in tools.REGISTRY:
                return f"unknown tool: {rule} — use  tool:pattern  (e.g. bash:ls*)"
            if rule in cfg.allow_rules:
                return f"already allowed: {rule}"
            config.save(replace(cfg, allow_rules=(*cfg.allow_rules, rule)))
            client.reset()
            return f"allowed without asking: {rule}\n(the denylist still blocks dangerous commands)"
        if cmd == "/think":
            cfg = config.load()
            if not args:
                shown = f"{cfg.thinking_token_budget} tokens/turn" if cfg.thinking_token_budget else "off (unbounded)"
                return f"thinking budget: {shown}  ·  reasoning_effort: {cfg.reasoning_effort}"
            val = args[0].lower()
            if val in ("off", "0", "none"):
                budget = 0
            elif val.isdigit():
                budget = int(val)
            else:
                return "usage: /think <tokens>  |  /think off"
            config.save(replace(cfg, thinking_token_budget=budget))
            client.reset()
            if not budget:
                return "thinking budget: off (unbounded)"
            return f"thinking budget: {budget} tokens/turn  (needs the server's reasoning-config to take effect)"
        return f"unknown command: {cmd} — try /help"

    @on(ModelBar.ModelChosen)
    async def model_chosen(self, event: ModelBar.ModelChosen) -> None:
        await self._say_system(self._switch_model(event.name))

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

    async def _maybe_open_plan_gate(self, items: list[dict], boxes: dict, container) -> None:
        """Hold execution when the model lays out a fresh multi-step plan.

        Each condition closes a specific hole:
        - `boxes["gate"]`: the MAIN loop only. A sub-agent has no user to ask, and
          pausing one would stall the parent that is blocked waiting for it.
        - act mode: in plan mode nothing can execute, so there is nothing to hold.
        - a step threshold: splitting a two-step plan into fresh sub-agents costs a
          request per phase and buys nothing — the loop does it cheaper in one go.
        - every item still `pending`: todo_write is called repeatedly to update
          status (the tool takes the whole list each time), and a progress update is
          not a plan awaiting approval.
        - a fingerprint of the plan we already asked about, so re-sending the same
          list after the user chose "continue" does not ask again.
        """
        steps = [it["content"] for it in items if it.get("content")]
        min_steps = config.load().plan_gate_min_steps
        fingerprint = tuple(steps)
        if not (
            boxes.get("gate")
            and self.mode == "act"
            and min_steps
            and len(steps) >= min_steps
            and all(it.get("status", "pending") == "pending" for it in items)
            and not self._plan_gate_pending
            and fingerprint != self._gated_plan
        ):
            return
        # Setting this stops the loop between turns (agent.run's should_pause), so
        # nothing in the plan runs until a button is pressed.
        self._plan_gate_pending = True
        self._gated_plan = fingerprint
        self._plan_gate = PlanGate(steps)
        await container.mount(self._plan_gate)
        self._status("⏸ 계획 승인 대기")

    def _settle_plan_gate(self, choice: str) -> None:
        """Answer the open gate (whatever the user chose) and release the loop."""
        self._plan_gate_pending = False
        if self._plan_gate is not None and self._plan_gate.is_mounted:
            self._plan_gate.settle(choice)
        self._plan_gate = None

    def _reset_plan_gate(self) -> None:
        """Forget the gate entirely — a different session asks about its own plans."""
        self._plan_gate_pending = False
        self._plan_gate = None
        self._gated_plan = None

    @on(Button.Pressed, "#plan-gate-run")
    async def _on_plan_gate_run(self, event: Button.Pressed) -> None:
        event.stop()
        self._settle_plan_gate("▶ 단계별 실행")
        await self._start_plan_run()

    @on(Button.Pressed, "#plan-gate-continue")
    async def _on_plan_gate_continue(self, event: Button.Pressed) -> None:
        event.stop()
        self._settle_plan_gate("계속 — 한 세션에서 진행")
        await self._start_turn()  # the paused loop picks up where it stopped

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
                "todo_write": tools.TODO_WRITE,
            }
        return tools.registry_for(self.session_depth, config.load().subagent_depth)

    @on(ResponseComplete)
    def response_complete(self, event: ResponseComplete) -> None:
        self._set_send_running(False)  # turn done — Stop reverts to Send
        self._prune_empty_turn()
        # Shared state is only ever touched on the main thread. Persist the whole
        # turn — assistant text, tool calls, and tool results alike.
        for msg in event.messages:
            self.session.messages.append(msg)
            storage.append_message(self.session_path, msg)
        self._status(event.stats)
        if event.prompt_tokens:
            self._last_prompt_tokens = event.prompt_tokens
        # First real reply of an untitled session -> generate a title in the background.
        if not self._has_title and any(m.get("role") == "assistant" for m in self.session.messages):
            self._has_title = True
            self.generate_title(list(self.session.messages), self.session_path)

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
        elif isinstance(event, ThinkingDelta):
            if boxes["thinking"] is None:
                boxes["thinking"] = ThinkingBlock()  # foldable; starts expanded
                await container.mount(boxes["thinking"])
            boxes["thinking"].append_chunk(event.text)
            self._status("● thinking…  (esc to stop)")
        elif isinstance(event, TextDelta):
            self._fold_thinking(boxes)  # answer starting → auto-collapse the reasoning
            if boxes["answer"] is None:
                # markdown=True: render the answer as Markdown so ```code``` and
                # ```diff fences become highlighted blocks.
                boxes["answer"] = Chatbox("", role="assistant", markdown=True)
                await container.mount(boxes["answer"])
            boxes["answer"].append_chunk(event.text)
            self._status("● generating…  (esc to stop)")
        elif isinstance(event, ToolCallDelta):
            if event.name == "todo_write":
                return  # todo_write shows in the panel on its final call, not a bubble
            if event.name == "edit":
                self._fold_thinking(boxes)
                boxes["answer"] = None
                self._status("● editing…  (esc to stop)")
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
                self._status("● writing…  (esc to stop)")
            else:
                # bash / read / grep …: no call bubble — the result card shows the
                # command in its title (IN) and the output in its body (OUT), the
                # Claude Code shape. The status bar reports progress until it lands.
                self._status(f"● running {event.name}…  (esc to stop)")
        elif isinstance(event, ToolCall):
            self._fold_thinking(boxes)  # fold reasoning; next turn opens fresh bubbles
            boxes["answer"] = None
            # Remember the call's input so the result card can title itself with it.
            boxes.setdefault("call_args", {})[event.id] = event.arguments
            if event.name == "todo_write":  # goes to the pinned panel, not a bubble
                items = event.arguments.get("items", [])
                self.query_one(TodoPanel).update_todos(items)
                boxes["tool"].clear()
                boxes["tool_buf"].clear()
                self._status("● planning…  (esc to stop)")
                # A fresh multi-step plan stops here for the user's go-ahead. This
                # runs on the main thread while the worker blocks in call_from_thread,
                # so the flag is set before the loop checks it again.
                await self._maybe_open_plan_gate(items, boxes, container)
                return
            if event.name == "edit":  # green diff card (shared with history restore)
                await container.mount(self._edit_card(event.arguments))
                boxes["tool"].clear()
                boxes["tool_buf"].clear()
                self._status("● editing…  (esc to stop)")
                return
            # write streamed a live content bubble (kept); other tools show only the
            # result card, so there's no call bubble to keep here.
            boxes["tool"].clear()
            boxes["tool_buf"].clear()
            self._status(f"● running {event.name}…  (esc to stop)")
        elif isinstance(event, ToolResult):
            if event.name == "todo_write":
                return  # already reflected in the pinned panel
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
        # Smart auto-scroll: follow the stream ONLY while _follow_output is set. The
        # flag is driven by a watcher on the scroller's scroll_y (see _update_follow),
        # so a manual scroll-up during streaming turns following OFF and STAYS off —
        # the old per-event "near the bottom?" snap re-pinned to the bottom on every
        # chunk, making it impossible to scroll away mid-answer.
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

    def action_stop(self) -> None:
        """Cancel the in-flight response (cooperative — the loop checks is_cancelled)."""
        worker = getattr(self, "_response_worker", None)
        if worker is not None and worker.is_running:
            worker.cancel()
            self._status("■ stopped")
            self._set_send_running(False)

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
        # Rule-based pre-approval comes FIRST. A call the user already authorised by
        # rule never reaches the modal, which is what keeps a parallel fan-out from
        # serialising behind a queue of dialogs (only one can be on screen). This
        # cannot widen anything: _gate_tool ran the dangerous-command denylist before
        # calling us, so a rule can only skip the *question*, never the safety gate.
        if permissions.allowed(call.name, call.arguments):
            return True
        if self.auto_approve:  # denylist already hard-blocked the dangerous ones
            return True
        with self._approval_lock:
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
        """Build the AgentContext whose run_subagent spawns a fresh child agent into a
        nested card, run to completion HERE on the worker thread (so the parent pauses
        until it returns — a sequential delegate → resume). A per-level factory: each
        child is parented to THIS level (parent_path / parent_depth + 1) and mounts one
        deeper inside this card's body, so the tree nests at ANY depth; recursion stops
        itself — a child at the depth limit is given no task tool. Shared by the agent
        loop's `task` tool and the structural plan runner (orchestrator.run_plan)."""
        def run_subagent(prompt: str, description: str) -> str:
            cfg = config.load()
            child_depth = parent_depth + 1
            child_path = storage.new_session_path()
            storage.write_header(child_path, storage.make_header(
                child_path.stem, parent_id=parent_path.stem, kind="subagent",
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

        return subagent.AgentContext(run_subagent=run_subagent)

    @work(thread=True, exclusive=True, exit_on_error=False)
    def run_plan_response(self, task: str, steps: list[str], turn) -> None:
        """Structurally execute a plan: delegate each step to a fresh sub-agent, in
        order, threading each concise result forward (orchestrator.run_plan). The
        decision to split is harness code, not a model tool call — the fix for the
        single-session thinking spiral (the model never self-delegates; measured)."""
        worker = get_current_worker()
        container = turn
        approve = self._approve_tool
        ctx = self._make_subagent_ctx(
            self.session_path, self.session_depth, container, worker, approve
        )
        # The reduce step: after the phases run, the MAIN session combines their concise
        # results into one final answer, streamed into this turn (not a sub-agent card),
        # so it reads as "the main organizing the results". Its context is only the task
        # + the short results, so it stays small (no spiral). Only worth it for a real
        # multi-step plan — a single phase is already its own answer.
        synth_boxes = {"thinking": None, "answer": None, "tool": {}, "tool_buf": {}, "call_args": {}}

        def synthesize(task_text: str, phases) -> str:
            results = "\n\n".join(f"## {p.description}\n{p.result}" for p in phases)
            messages = [
                {"role": "system", "content": prompts.synthesis_system()},
                {"role": "user", "content": f"# Task\n{task_text}\n\n# Phase results\n{results}"},
            ]
            text = ""
            for event in client.stream_chat(messages):  # text-only reduce, no tools
                if worker.is_cancelled:
                    break
                if isinstance(event, Usage):
                    continue
                if isinstance(event, TextDelta):
                    text += event.text
                self.call_from_thread(self._render_event, event, synth_boxes, container)
            return text

        try:
            result = orchestrator.run_plan(
                task, steps, ctx.run_subagent,
                synthesize=synthesize if len(steps) >= 2 else None,
                is_cancelled=lambda: worker.is_cancelled,
            )
        except Exception as exc:
            self.post_message(self.ResponseFailed(f"{type(exc).__name__}: {exc}"[:300]))
            return
        if worker.is_cancelled:  # cancelled mid-plan: don't persist a partial outcome
            return
        # The plan's outcome becomes an assistant turn in the parent session (a reload
        # shows it; each sub-agent's own transcript lives in its linked child file). With
        # ≥2 steps the synthesis already streamed into the turn; a single phase did not,
        # so mount its result as the answer bubble.
        final = result.result or "(plan produced no result)"
        if len(steps) < 2:
            self.call_from_thread(
                container.mount, Chatbox(final, role="assistant", markdown=True)
            )
        self.post_message(self.ResponseComplete([{"role": "assistant", "content": final}], ""))

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

        try:
            new_messages = agent.run(
                messages,
                emit=emit,
                is_cancelled=lambda: worker.is_cancelled,
                approve=approve,                    # bash and friends are confirmed first
                registry=self._registry_for_mode(),  # plan mode = read-only subset
                ctx=ctx,                            # task delegates through this
                prompt_tokens=self._last_prompt_tokens,  # measured, for compaction
                should_pause=lambda: self._plan_gate_pending,  # the plan gate
            )
        except Exception as exc:
            # Server 500, timeout, connection refused... all become a bubble, not a crash.
            summary = f"{type(exc).__name__}: {exc}"[:300]
            self.post_message(self.ResponseFailed(summary))
            return
        if worker.is_cancelled:  # cancelled mid-run: don't persist a partial turn
            return
        self.post_message(self.ResponseComplete(
            new_messages, self._format_stats(stats), stats["last_prompt"]
        ))


app = AhaCodeApp

if __name__ == "__main__":
    AhaCodeApp().run()
