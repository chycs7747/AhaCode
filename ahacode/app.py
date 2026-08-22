import threading
import time
from dataclasses import dataclass, replace

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Input
from textual.worker import get_current_worker

from ahacode import agent, client, config, storage, tools
from ahacode.events import TextDelta, ThinkingDelta, ToolCall, ToolResult, Usage
from ahacode.session import ChatSession
from ahacode.widgets.approval_modal import ApprovalModal
from ahacode.widgets.chatbox import Chatbox
from ahacode.widgets.model_bar import ModelBar


# Injected only in plan mode. Without a system-prompt layer yet, this is what
# makes the mode behave: fewer tools *and* an instruction to plan, not act.
PLAN_SYSTEM_PROMPT = (
    "You are in PLAN MODE. Do not change anything or run commands. If needed, "
    "investigate with the read tool, then call todo_write to lay out a clear, "
    "step-by-step plan for the user to review. Do not carry out the plan."
)


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
        latest = storage.latest_session()
        if latest:  # resume the most recent session
            self.session_path = latest
            self.session.messages = storage.load_messages(latest)
        else:  # first run: start a new session file
            self.session_path = storage.new_session_path()

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
        with VerticalScroll(id="chat-container") as container:
            container.can_focus = False  # keep initial focus on the input
        with Vertical(id="bottom"):
            yield Input(placeholder="Type a message and press Enter...", id="prompt")
            yield ModelBar()

    async def on_mount(self) -> None:
        """Restore saved history as chat bubbles.

        Compose is guaranteed to run before Mount, so #chat-container exists here.
        """
        container = self.query_one("#chat-container", VerticalScroll)
        for msg in self.session.messages:
            await container.mount(self._bubble_for(msg))
        container.scroll_end(animate=False)

    @staticmethod
    def _bubble_for(msg: dict) -> Chatbox:
        """Turn a stored message (OpenAI roles) into a display bubble.

        Handles the tool-calling shapes: an assistant message may carry
        tool_calls with null content, and tool results use the "tool" role.
        """
        role = msg["role"]
        content = msg.get("content") or ""
        if role == "assistant" and msg.get("tool_calls"):
            names = ", ".join(c["function"]["name"] for c in msg["tool_calls"])
            content = f"🔧 {names}\n{content}".rstrip()
            return Chatbox(content, role="tool-call")
        if role == "tool":
            return Chatbox(content, role="tool-result")
        return Chatbox(content, role=role)

    async def _say_system(self, text: str) -> None:
        """Show an informational bubble (commands, status) — never part of the session."""
        container = self.query_one("#chat-container", VerticalScroll)
        await container.mount(Chatbox(text, role="system"))
        container.scroll_end(animate=False)

    @on(Input.Submitted)
    async def user_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()

        if text.startswith("/"):
            # Slash commands configure the app; they never reach the LLM
            # and are not recorded in the session.
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

    def _registry_for_mode(self) -> dict:
        """Plan mode hides side-effecting tools so the model can only investigate + plan."""
        if self.mode == "plan":
            return {"read": tools.READ, "todo_write": tools.TODO_WRITE}
        return tools.REGISTRY

    @on(ResponseComplete)
    def response_complete(self, event: ResponseComplete) -> None:
        # Shared state is only ever touched on the main thread. Persist the whole
        # turn — assistant text, tool calls, and tool results alike.
        for msg in event.messages:
            self.session.messages.append(msg)
            storage.append_message(self.session_path, msg)
        self._status(event.stats)

    @on(ResponseFailed)
    async def response_failed(self, event: ResponseFailed) -> None:
        container = self.query_one("#chat-container", VerticalScroll)
        await container.mount(
            Chatbox(f"⚠ {event.error}\n(check the server, then try again)", role="error")
        )
        container.scroll_end(animate=False)
        self._status("")

    async def _render_event(
        self, event, boxes: dict[str, Chatbox | None], container: VerticalScroll
    ) -> None:
        """Main-thread renderer: mount/append bubbles as the loop emits events.

        Runs via call_from_thread (which awaits this coroutine), so each mount is
        complete before the next append — no races with the worker thread. `boxes`
        holds the current turn's live thinking/answer bubbles; a tool call ends a
        turn, so the next thinking/text opens fresh bubbles.
        """
        if isinstance(event, ThinkingDelta):
            if boxes["thinking"] is None:
                boxes["thinking"] = Chatbox("", role="thinking")
                await container.mount(boxes["thinking"])
            boxes["thinking"].append_chunk(event.text)
            self._status("● thinking…  (esc to stop)")
        elif isinstance(event, TextDelta):
            if boxes["answer"] is None:
                boxes["answer"] = Chatbox("", role="assistant")
                await container.mount(boxes["answer"])
            boxes["answer"].append_chunk(event.text)
            self._status("● generating…  (esc to stop)")
        elif isinstance(event, ToolCall):
            args = ", ".join(f"{k}={v!r}" for k, v in event.arguments.items())
            await container.mount(Chatbox(f"🔧 {event.name}({args})", role="tool-call"))
            boxes["thinking"] = boxes["answer"] = None  # next turn opens fresh bubbles
            self._status(f"● running {event.name}…  (esc to stop)")
        elif isinstance(event, ToolResult):
            if event.is_error:
                role = "tool-error"
            elif event.name == "todo_write":
                role = "plan"  # render the plan checklist distinctly
            else:
                role = "tool-result"
            await container.mount(Chatbox(event.output, role=role))
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

    # exclusive=True: a new message cancels the previous worker.
    # exit_on_error=False: a failing worker must not take the whole app down.
    @work(thread=True, exclusive=True, exit_on_error=False)
    def stream_response(self, messages: list[dict]) -> None:
        """Run the agent loop in a thread, rendering its events into the chat."""
        worker = get_current_worker()
        container = self.query_one("#chat-container", VerticalScroll)
        # Current turn's live bubbles; _render_event fills these in on the main thread.
        boxes: dict[str, Chatbox | None] = {"thinking": None, "answer": None}

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
            # Cross-thread handshake: the loop runs here (worker thread) but the
            # modal lives on the main thread. We push it (non-blocking) via
            # call_from_thread, then block this thread on an Event until the user
            # answers — the modal's dismiss callback sets the Event.
            answered = threading.Event()
            verdict: dict[str, bool] = {}

            def ask() -> None:
                summary = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())

                def on_dismiss(approved: bool | None) -> None:
                    verdict["ok"] = bool(approved)
                    answered.set()

                self.push_screen(ApprovalModal(call.name, summary), on_dismiss)

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
