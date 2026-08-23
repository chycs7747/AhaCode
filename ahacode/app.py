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
from textual.widgets import Button
from textual.worker import get_current_worker
from rich.text import Text
from rich.theme import Theme

from ahacode import agent, client, config, storage, subagent, tools
from ahacode.events import TextDelta, ThinkingDelta, ToolCall, ToolCallDelta, ToolResult, Usage
from ahacode.render import diff_stats, edit_diff_lines, tool_summary
from ahacode.session import ChatSession
from ahacode.widgets.approval_modal import ApprovalModal
from ahacode.widgets.chatbox import Chatbox
from ahacode.widgets.header_bar import HeaderBar
from ahacode.widgets.prompt_input import PromptInput
from ahacode.widgets.session_picker import SessionPicker
from ahacode.widgets.subagent_card import SubagentCard
from ahacode.widgets.thinking import ThinkingBlock
from ahacode.widgets.tool_result import ToolResultBlock
from ahacode.widgets.todo_panel import TodoPanel
from ahacode.widgets.model_bar import ModelBar


# Injected only in plan mode. Without a system-prompt layer yet, this is what
# makes the mode behave: fewer tools *and* an instruction to plan, not act.
PLAN_SYSTEM_PROMPT = (
    "You are in PLAN MODE. Do not change anything or run commands. If needed, "
    "investigate with the read tool, then call todo_write to lay out a clear, "
    "step-by-step plan for the user to review. Do not carry out the plan."
)

TITLE_SYSTEM = (
    "You write a very short title (2-5 words) for a conversation. "
    "Reply with ONLY the title — no quotes, no trailing punctuation."
)

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
    ]

    def __init__(self) -> None:
        super().__init__()
        # Session state lives in a plain Python object, decoupled from widgets.
        self.session = ChatSession()
        self.mode = "act"  # "act" (full tools) or "plan" (read-only + todo_write)
        self._last_status = ""
        self.auto_approve = False  # session-only: skip the approval modal when on
        self._approval_lock = threading.Lock()  # one approval modal at a time (parallel children queue)
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
        self.query_one("#prompt", PromptInput).focus()  # not the header buttons

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
        card (command/path in the title), edit as a diff card — no raw tool-call
        bubbles, so a reloaded session looks exactly like the turn that made it."""
        container = self.query_one("#chat-container", VerticalScroll)
        await container.remove_children()
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
            elif role == "tool":
                cid = msg.get("tool_call_id")
                name = call_names.get(cid, "tool")
                if name == "edit":
                    continue  # already shown as the diff card
                summary = tool_summary(name, call_args.get(cid, {}))
                await turn.mount(ToolResultBlock(name, content, summary=summary))
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
        self._set_header_title("")
        self.query_one(TodoPanel).display = False
        await self._render_history()
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
        self._set_header_title(meta.get("title", ""))
        self.query_one(TodoPanel).display = False
        await self._render_history()
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

        if text.startswith("/"):
            # Slash commands configure the app; they never reach the LLM
            # and are not recorded in the session.
            if text == "/new":
                await self._new_session()
                return
            if text == "/sessions":
                self.push_screen(SessionPicker(), self._session_picked)
                return
            await self._say_system(self._handle_command(text))
            return

        self.session.add_user(text)
        storage.append_message(self.session_path, {"role": "user", "content": text})

        container = self.query_one("#chat-container", VerticalScroll)
        await container.mount(Chatbox(text, role="user"))
        # The assistant's whole reply (thinking → tools → answer) is mounted into one
        # .turn container with a green left rail, so the steps read as one connected
        # flow rather than a flat stack. The user message stays outside it.
        self._turn = Vertical(classes="turn")
        await container.mount(self._turn)
        container.scroll_end(animate=False)

        # Run the agent loop in a worker. A snapshot copy is passed so the worker
        # never shares a mutable list with the main thread; bubbles for the reply
        # are mounted lazily as loop events arrive (turn count is not known ahead).
        history = list(self.session.messages)
        if self.mode == "plan":
            history = [{"role": "system", "content": PLAN_SYSTEM_PROMPT}, *history]
        self._status("● waiting…  (esc to stop)")
        self._response_worker = self.stream_response(history, self._turn)
        self._set_send_running(True)  # the Send button becomes Stop

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

    def _registry_for_mode(self) -> dict:
        """The tools this session may use this turn. Plan mode stays read-only (no
        side effects — and no `task`, since a sub-agent could act). Act mode gets the
        base tools plus `task`, but only while depth < subagent_depth so a sub-agent
        at the limit cannot recurse (see tools.registry_for)."""
        if self.mode == "plan":
            return {"read": tools.READ, "todo_write": tools.TODO_WRITE}
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

        Mirrors kilocode's `auto_collapse_reasoning` (config.ts:109 — "collapse
        reasoning blocks after the agent finishes writing them"): the block stays
        open while it streams, then folds when the answer or a tool call begins.
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
        if isinstance(event, ThinkingDelta):
            if boxes["thinking"] is None:
                boxes["thinking"] = ThinkingBlock()  # foldable; starts expanded
                await container.mount(boxes["thinking"])
            boxes["thinking"].append_chunk(event.text)
            self._status("● thinking…  (esc to stop)")
        elif isinstance(event, TextDelta):
            self._fold_thinking(boxes)  # answer starting → auto-collapse the reasoning
            if boxes["answer"] is None:
                # markdown=True: render the answer as Markdown so ```code``` and
                # ```diff fences become highlighted blocks (elia's Chatbox model).
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
                # write streams its content live into one bubble (kilocode-style)
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
                self.query_one(TodoPanel).update_todos(event.arguments.get("items", []))
                boxes["tool"].clear()
                boxes["tool_buf"].clear()
                self._status("● planning…  (esc to stop)")
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
        # Smart auto-scroll on the chat scroller (container is the turn, not the
        # scroller): follow only while the user is near the bottom.
        scroller = self.query_one("#chat-container", VerticalScroll)
        if scroller.scroll_y in range(
            scroller.max_scroll_y - 3, scroller.max_scroll_y + 1
        ):
            scroller.scroll_end(animate=False)

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
                [{"role": "system", "content": TITLE_SYSTEM}, {"role": "user", "content": convo}]
            )
        except Exception:
            return  # a failed title is not worth surfacing; leave it untitled
        title = title.strip().strip('"').strip()[:60]
        if title:
            self.call_from_thread(storage.set_title, path, title)
            if path == self.session_path:  # not if the user already switched away
                self.call_from_thread(self._set_header_title, title)

    # exclusive=True: a new message cancels the previous worker.
    # exit_on_error=False: a failing worker must not take the whole app down.
    @work(thread=True, exclusive=True, exit_on_error=False)
    def stream_response(self, messages: list[dict], turn) -> None:
        """Run the agent loop in a thread, rendering its events into the chat."""
        worker = get_current_worker()
        container = turn  # the reply's blocks mount into this turn's rail container
        # Current turn's live bubbles; _render_event fills these in on the main thread.
        boxes: dict = {"thinking": None, "answer": None, "tool": {}, "tool_buf": {}, "call_args": {}}

        stats = {"prompt": 0, "completion": 0, "t_start": time.monotonic(), "t_first": None}

        def emit(event) -> None:
            if isinstance(event, Usage):  # accounting only — never a bubble
                stats["prompt"] += event.prompt_tokens
                stats["completion"] += event.completion_tokens
                return
            if stats["t_first"] is None and isinstance(event, (ThinkingDelta, TextDelta)):
                stats["t_first"] = time.monotonic()
            # Hop to the main thread to touch widgets. call_from_thread blocks the
            # worker until the UI has rendered — built-in backpressure.
            self.call_from_thread(self._render_event, event, boxes, container)

        def approve(call) -> bool:
            if self.auto_approve:  # denylist already hard-blocked the dangerous ones
                return True
            # Serialize modals: parallel sub-agents may each need approval at once,
            # but only one dialog can be on screen — the rest queue on this lock.
            with self._approval_lock:
                # Cross-thread handshake: the loop runs here (worker thread) but the
                # modal lives on the main thread. We push it (non-blocking) via
                # call_from_thread, then block this thread on an Event until the user
                # answers — the modal's dismiss callback sets the Event.
                answered = threading.Event()
                verdict: dict[str, bool] = {}

                def ask() -> None:
                    def on_dismiss(approved: bool | None) -> None:
                        verdict["ok"] = bool(approved)
                        answered.set()

                    # Pass the raw arguments; the modal renders a proper per-tool
                    # preview (write → code, edit → diff) inside a scrollable box.
                    self.push_screen(ApprovalModal(call.name, call.arguments), on_dismiss)

                self.call_from_thread(ask)
                answered.wait()
                return verdict.get("ok", False)

        def run_subagent(prompt: str, description: str) -> str:
            # The `task` tool calls this (through ctx) to delegate a subtask. A
            # sub-agent is just another agent loop, run to completion HERE on this
            # worker thread — so the parent naturally pauses until it returns (Roo's
            # sequential delegate → resume). It gets its own session file (parent_id
            # set, so the picker nests it) and a nested 🤖 card its events render into.
            cfg = config.load()
            child_depth = self.session_depth + 1
            child_path = storage.new_session_path()
            storage.write_header(child_path, storage.make_header(
                child_path.stem, parent_id=self.session_path.stem, kind="subagent",
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
                ctx=ctx,          # lets a grandchild spawn when subagent_depth > 1
                is_cancelled=lambda: worker.is_cancelled,
            )
            for msg in result.messages:
                storage.append_message(child_path, msg)
            # Fold the card now the child is done (its answer stays one click away);
            # the parent's synthesis is what reads inline. Title shows the tool count.
            tool_count = sum(1 for m in result.messages if m.get("role") == "tool")
            self.call_from_thread(card.done, tool_count)
            return result.result

        ctx = subagent.AgentContext(run_subagent=run_subagent)

        try:
            new_messages = agent.run(
                messages,
                emit=emit,
                is_cancelled=lambda: worker.is_cancelled,
                approve=approve,                    # bash and friends are confirmed first
                registry=self._registry_for_mode(),  # plan mode = read-only subset
                ctx=ctx,                            # task delegates through this
            )
        except Exception as exc:
            # Server 500, timeout, connection refused... all become a bubble, not a crash.
            summary = f"{type(exc).__name__}: {exc}"[:300]
            self.post_message(self.ResponseFailed(summary))
            return
        if worker.is_cancelled:  # cancelled mid-run: don't persist a partial turn
            return
        self.post_message(self.ResponseComplete(new_messages, self._format_stats(stats)))


app = AhaCodeApp

if __name__ == "__main__":
    AhaCodeApp().run()
