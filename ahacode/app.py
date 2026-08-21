from dataclasses import dataclass, replace

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Input
from textual.worker import get_current_worker

from ahacode import client, config, storage
from ahacode.session import ChatSession
from ahacode.widgets.chatbox import Chatbox
from ahacode.widgets.model_bar import ModelBar


class AhaCodeApp(App):
    """AhaCode: a Textual-based TUI agent client."""

    CSS_PATH = "ahacode.tcss"
    # priority=True: checked before the focused widget's own bindings — the Input
    # binds ctrl+d to "delete character right" and would otherwise swallow it.
    BINDINGS = [Binding("ctrl+d", "quit", "Quit", priority=True)]

    def __init__(self) -> None:
        super().__init__()
        # Session state lives in a plain Python object, decoupled from widgets.
        self.session = ChatSession()
        latest = storage.latest_session()
        if latest:  # resume the most recent session
            self.session_path = latest
            self.session.messages = storage.load_messages(latest)
        else:  # first run: start a new session file
            self.session_path = storage.new_session_path()

    @dataclass
    class ResponseComplete(Message):
        """Posted by the streaming worker once a response is fully received.

        The main-thread handler records it into the session.
        """

        text: str
    
    @dataclass
    class ResponseFailed(Message):
        """Posted when the streaming worker hits an error. The app stays alive;
        the handler turns the pre-mounted answer bubble into an error bubble."""

        error: str
        response_box: Chatbox

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
            await container.mount(Chatbox(msg["content"], role=msg["role"]))
        container.scroll_end(animate=False)

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

        # Pre-mount empty bubbles so streamed deltas have a place to land.
        # The thinking box starts hidden: whether the model will think is
        # unknown until the first delta arrives (it reveals itself then).
        thinking_box = Chatbox("", role="thinking")
        thinking_box.display = False
        response_box = Chatbox("", role="assistant")
        await container.mount(thinking_box, response_box)
        container.scroll_end(animate=False)

        # Schedule the streaming worker; this call returns immediately.
        # A snapshot copy is passed so the worker never shares a mutable list
        # with the main thread.
        self.stream_response(list(self.session.messages), thinking_box, response_box)

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

    @on(ResponseComplete)
    def response_complete(self, event: ResponseComplete) -> None:
        # Shared state is only ever touched on the main thread.
        self.session.add_assistant(event.text)
        storage.append_message(
            self.session_path, {"role": "assistant", "content": event.text}
        )
    
    @on(ResponseFailed)
    def response_failed(self, event: ResponseFailed) -> None:
        # Reuse the pre-mounted answer bubble so the error sits where the reply would.
        event.response_box.add_class("chatbox--error")
        event.response_box.append_chunk(f"⚠ {event.error}\n(check the server, then try again)")

    # exclusive=True: a new message cancels the previous worker.
    # exit_on_error=False: a failing worker must not take the whole app down.
    @work(thread=True, exclusive=True, exit_on_error=False)
    def stream_response(
        self, messages: list[dict], thinking_box: Chatbox, response_box: Chatbox
    ) -> None:
        """Stream LLM deltas from a thread worker into the pre-mounted bubbles."""
        worker = get_current_worker()
        container = self.query_one("#chat-container", VerticalScroll)
        full_text = ""
        stream = client.stream_chat(messages)
        try:
            for kind, chunk in stream:
                if worker.is_cancelled:  # cooperative cancellation — threads can't be killed
                    return
                target = thinking_box if kind == "thinking" else response_box
                if kind == "text":
                    full_text += chunk
                self.call_from_thread(target.append_chunk, chunk)
                # Smart auto-scroll: follow only while the user is near the bottom.
                if container.scroll_y in range(
                    container.max_scroll_y - 3, container.max_scroll_y + 1
                ):
                    self.call_from_thread(container.scroll_end, animate=False)
        except Exception as exc:
            # Server 500, timeout, connection refused... all become a bubble, not a crash.
            summary = f"{type(exc).__name__}: {exc}"[:300]
            self.post_message(self.ResponseFailed(summary, response_box))
            return
        finally:
            stream.close()  # release the connection on every exit path (done, cancelled, failed)
        # Normal loop exit means the response is complete.
        self.post_message(self.ResponseComplete(full_text))


app = AhaCodeApp

if __name__ == "__main__":
    AhaCodeApp().run()
