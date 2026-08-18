from textual.widgets import Static


class Chatbox(Static):
    """A single chat bubble, styled by role (user / assistant / thinking)."""

    def __init__(self, content: str = "", role: str = "user") -> None:
        super().__init__(content)
        self._content = content
        self.add_class(f"chatbox--{role}")

    def append_chunk(self, chunk: str) -> None:
        """Append a streamed delta. Called from the worker via call_from_thread."""
        if not self.display:  # a bubble born hidden reveals itself on the first delta
            self.display = True
        self._content += chunk
        self.update(self._content)
