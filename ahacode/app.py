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

from ahacode import agent, client, config, storage, tools
from ahacode.events import TextDelta, ThinkingDelta, ToolCall, ToolCallDelta, ToolResult, Usage
from ahacode.render import diff_stats, edit_diff_lines, tool_summary
from ahacode.session import ChatSession
from ahacode.widgets.approval_modal import ApprovalModal
from ahacode.widgets.chatbox import Chatbox
from ahacode.widgets.header_bar import HeaderBar
from ahacode.widgets.prompt_input import PromptInput
from ahacode.widgets.session_picker import SessionPicker
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

    async def _render_history(self) -> None:
        """Clear the chat and remount the current session's messages as bubbles."""
        container = self.query_one("#chat-container", VerticalScroll)
        await container.remove_children()
        # Map tool_call_id -> tool name so restored tool results can render as the
        # same foldable ToolResultBlock the live turn produced.
        names: dict[str, str] = {}
        for msg in self.session.messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for c in msg["tool_calls"]:
                    names[c["id"]] = c["function"]["name"]
            await container.mount(self._bubble_for(msg, names))
        container.scroll_end(animate=False)

    async def _new_session(self) -> None:
        """Start a fresh session (new file + header) and clear the view."""
        self.session = ChatSession()
        self.session_path = storage.new_session_path()
        storage.write_header(
            self.session_path,
            storage.make_header(self.session_path.stem, kind="main", model=config.load().name),
        )
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
    def _bubble_for(msg: dict, names: dict[str, str] | None = None):
        """Turn a stored message (OpenAI roles) into a display widget.

        Handles the tool-calling shapes: an assistant message may carry
        tool_calls with null content, and tool results use the "tool" role.
        Tool results become the same foldable ToolResultBlock as a live turn
        (`names` maps tool_call_id -> tool name so the header is right).
        """
        role = msg["role"]
        content = msg.get("content") or ""
        if role == "assistant" and msg.get("tool_calls"):
            tool_names = ", ".join(c["function"]["name"] for c in msg["tool_calls"])
            content = f"🔧 {tool_names}\n{content}".rstrip()
            return Chatbox(content, role="tool-call")
        if role == "tool":
            name = (names or {}).get(msg.get("tool_call_id"), "tool")
            return ToolResultBlock(name, content)
        # Restored assistant answers render as Markdown too (see _render_event).
        return Chatbox(content, role=role, markdown=(role == "assistant"))

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
        container.scroll_end(animate=False)

        # Run the agent loop in a worker. A snapshot copy is passed so the worker
        # never shares a mutable list with the main thread; bubbles for the reply
        # are mounted lazily as loop events arrive (turn count is not known ahead).
        history = list(self.session.messages)
        if self.mode == "plan":
            history = [{"role": "system", "content": PLAN_SYSTEM_PROMPT}, *history]
        self._status("● waiting…  (esc to stop)")
        self._response_worker = self.stream_response(history)
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
        """Plan mode hides side-effecting tools so the model can only investigate + plan."""
        if self.mode == "plan":
            return {"read": tools.READ, "todo_write": tools.TODO_WRITE}
        return tools.REGISTRY

    @on(ResponseComplete)
    def response_complete(self, event: ResponseComplete) -> None:
        self._set_send_running(False)  # turn done — Stop reverts to Send
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
            if event.name == "edit":  # green diff card: header + count chip + -/+ lines
                a = event.arguments
                path = a.get("path", "?")
                old, new = a.get("old_string", ""), a.get("new_string", "")
                text, plain = edit_diff_lines(old, new)
                added, removed = diff_stats(old, new)
                box = Chatbox("", role="tool-diff")
                box.set_rich(text, plain)
                box.border_title = f"✏ edit · {path}"
                box.border_subtitle = f"+{added} −{removed}"
                await container.mount(box)
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
            # One foldable card: the command/path in the title (IN), the output in
            # the body (OUT). Long output / errors fold away.
            summary = tool_summary(event.name, boxes.get("call_args", {}).get(event.id, {}))
            await container.mount(
                ToolResultBlock(event.name, event.output, event.is_error, summary=summary)
            )
        # Smart auto-scroll: follow only while the user is near the bottom.
        if container.scroll_y in range(
            container.max_scroll_y - 3, container.max_scroll_y + 1
        ):
            container.scroll_end(animate=False)

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
    def stream_response(self, messages: list[dict]) -> None:
        """Run the agent loop in a thread, rendering its events into the chat."""
        worker = get_current_worker()
        container = self.query_one("#chat-container", VerticalScroll)
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

        try:
            new_messages = agent.run(
                messages,
                emit=emit,
                is_cancelled=lambda: worker.is_cancelled,
                approve=approve,                    # bash and friends are confirmed first
                registry=self._registry_for_mode(),  # plan mode = read-only subset
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
