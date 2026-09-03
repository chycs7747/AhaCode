"""The top bar: session title, endpoint, and the Settings / New / Sessions buttons."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Static


class HeaderBar(Horizontal):
    """The top bar; its Button.Pressed events bubble up to the app."""

    def compose(self) -> ComposeResult:
        yield Static("AhaCode", id="session-title")
        yield Static("", id="endpoint")
        yield Button("⚙ Settings", id="settings-btn", classes="header-btn")
        yield Button("+ New", id="new-session-btn", classes="header-btn")
        yield Button("≡ Sessions", id="open-sessions-btn", classes="header-btn")

    def set_title(self, title: str) -> None:
        """Show the session's title next to the app name.

        Args:
            title: The title, or "" for an untitled session.
        """
        self._title_text = f"AhaCode · {title}" if title else "AhaCode"
        self.query_one("#session-title", Static).update(self._title_text)

    def set_endpoint(self, url: str) -> None:
        """Show the endpoint as a compact host:port.

        Args:
            url: The base URL.
        """
        short = url.split("://", 1)[-1].rstrip("/")
        if short.endswith("/v1"):
            short = short[:-3]
        self._endpoint_text = short
        self.query_one("#endpoint", Static).update(short)
