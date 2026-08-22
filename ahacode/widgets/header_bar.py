from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Static


class HeaderBar(Horizontal):
    """Top bar: the current session's title on the left, session actions on the
    right ([+ New], [≡ Sessions]).

    Real Button widgets (not clickable text) so the hit target is unambiguous —
    this mirrors Claude Code's header (title + new + history) and elia's clickable
    header (reference/elia/.../app_header.py), while reusing our existing
    SessionPicker modal and background auto-title. Button.Pressed bubbles up to the
    app, which owns session state.
    """

    def compose(self) -> ComposeResult:
        yield Static("AhaCode", id="session-title")
        yield Static("", id="endpoint")  # connection identity lives up here, not in the composer
        yield Button("+ New", id="new-session-btn", classes="header-btn")
        yield Button("≡ Sessions", id="open-sessions-btn", classes="header-btn")

    def set_title(self, title: str) -> None:
        """Show the session's (auto-generated) title next to the app name."""
        self._title_text = f"AhaCode · {title}" if title else "AhaCode"
        self.query_one("#session-title", Static).update(self._title_text)

    def set_endpoint(self, url: str) -> None:
        """Show a compact host:port (drop the scheme and trailing /v1)."""
        short = url.split("://", 1)[-1].rstrip("/")
        if short.endswith("/v1"):
            short = short[:-3]
        self._endpoint_text = short
        self.query_one("#endpoint", Static).update(short)
