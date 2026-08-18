from dataclasses import dataclass

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Input
from textual.worker import get_current_worker

from ahacode import client, storage
from ahacode.session import ChatSession
from ahacode.widgets.chatbox import Chatbox


class AhaCodeApp(App):
    """AhaCode: a Textual-based TUI agent client."""

    CSS_PATH = "ahacode.tcss"

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

    def compose(self) -> ComposeResult:
        # Static skeleton only — chat bubbles are mounted at runtime.
        with VerticalScroll(id="chat-container") as container:
            container.can_focus = False  # keep initial focus on the input
        yield Input(placeholder="Type a message and press Enter...", id="prompt")

    async def on_mount(self) -> None:
        """Restore saved history as chat bubbles.

        Compose is guaranteed to run before Mount, so #chat-container exists here.
        """
        container = self.query_one("#chat-container", VerticalScroll)
        for msg in self.session.messages:
            await container.mount(Chatbox(msg["content"], role=msg["role"]))
        container.scroll_end(animate=False)

    @on(Input.Submitted)
    async def user_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()
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

    @on(ResponseComplete)
    def response_complete(self, event: ResponseComplete) -> None:
        # Shared state is only ever touched on the main thread.
        self.session.add_assistant(event.text)
        storage.append_message(
            self.session_path, {"role": "assistant", "content": event.text}
        )

    # exclusive=True: submitting a new message cancels the previous worker.
    @work(thread=True, exclusive=True)
    def stream_response(
        self, messages: list[dict], thinking_box: Chatbox, response_box: Chatbox
    ) -> None:
        """Stream LLM deltas from a thread worker into the pre-mounted bubbles."""
        worker = get_current_worker()
        container = self.query_one("#chat-container", VerticalScroll)
        full_text = ""
        for kind, chunk in client.stream_chat(messages):
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
        # Normal loop exit means the response is complete.
        # post_message is thread-safe, so the worker may call it directly.
        self.post_message(self.ResponseComplete(full_text))


app = AhaCodeApp

if __name__ == "__main__":
    AhaCodeApp().run()
