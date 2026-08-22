"""A modal to open another session or start a new one.

Lists sessions/*.jsonl as a tree (storage.build_tree over their headers) and
dismisses with the chosen session id, "new", or None (cancel).
"""

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, Static

from ahacode import storage

_ICON = {"main": "🧠", "subagent": "🤖", "fork": "🌿"}


def _flatten(nodes: list[dict], level: int = 0):
    """Pre-order walk of the session tree, yielding (indent level, node)."""
    for n in nodes:
        yield level, n
        yield from _flatten(n["children"], level + 1)


class SessionPicker(ModalScreen[str | None]):
    """dismiss("new") = start new · dismiss(id) = open that session · dismiss(None) = cancel."""

    BINDINGS = [("escape", "cancel", "Close")]

    def compose(self) -> ComposeResult:
        tree = storage.build_tree(storage.list_sessions())
        items = [self._new_item()]
        for level, node in _flatten(tree):
            items.append(self._session_item(level, node))
        with Vertical(id="picker-box"):
            yield Static("Sessions   (↑↓ 이동 · Enter 열기 · Esc 닫기)", id="picker-title")
            yield ListView(*items, id="picker-list")

    @staticmethod
    def _new_item() -> ListItem:
        item = ListItem(Label("＋  new session"))
        item.session_id = "new"
        return item

    @staticmethod
    def _session_item(level: int, node: dict) -> ListItem:
        indent = "  " * level
        title = node["title"] or node["id"]
        model = node.get("model") or "?"
        icon = _ICON.get(node["kind"], "•")
        item = ListItem(Label(f"{indent}{icon} {title}   · {model}"))
        item.session_id = node["id"]
        return item

    @on(ListView.Selected)
    def _picked(self, event: ListView.Selected) -> None:
        self.dismiss(getattr(event.item, "session_id", None))

    def action_cancel(self) -> None:
        self.dismiss(None)
