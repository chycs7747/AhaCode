"""A modal to open another session, start a new one, or delete one.

Lists sessions/*.jsonl as a tree (storage.build_tree over their headers) and
dismisses with the chosen session id, "new", or None (cancel).

Delete is `d` (or the button) on the highlighted row, pressed TWICE: the first
press turns the title into the question and names what would go — the row and
every session under it — the second answers it. Esc while the question is up
withdraws it instead of closing the picker. Deleting the session that is open
dismisses with "new" so the app moves off the file before it is gone; the one
whose turn is still running cannot be deleted at all.
"""

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, ListItem, ListView, Static

from ahacode import storage

_ICON = {"main": "🧠", "impl": "🛠", "subagent": "🤖", "fork": "🌿"}
_HINT = "Sessions   (↑↓ 이동 · Enter 열기 · d 삭제 · Esc 닫기)   ·   🤖 서브에이전트는 보기 전용"


def _flatten(nodes: list[dict], level: int = 0):
    """Pre-order walk of the session tree, yielding (indent level, node)."""
    for n in nodes:
        yield level, n
        yield from _flatten(n["children"], level + 1)


class SessionPicker(ModalScreen[str | None]):
    """dismiss("new") = start new · dismiss(id) = open that session · dismiss(None) = cancel."""

    BINDINGS = [("escape", "cancel", "Close"), ("d", "delete", "Delete")]

    def __init__(self, current: str | None = None, locked: str | None = None) -> None:
        super().__init__()
        self.current = current   # the session open behind the picker
        self.locked = locked     # a session whose turn is running: not deletable
        self._pending: str | None = None  # the id awaiting the second press
        self._sessions: list[dict] = []

    def compose(self) -> ComposeResult:
        self._sessions = storage.list_sessions()
        tree = storage.build_tree(self._sessions)
        items = [self._new_item()]
        for level, node in _flatten(tree):
            items.append(self._session_item(level, node))
        with Vertical(id="picker-box"):
            yield Static(_HINT, id="picker-title")
            yield ListView(*items, id="picker-list")
            with Horizontal(id="picker-actions"):
                yield Button("🗑 삭제 (d)", variant="error", id="picker-delete")

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

    @on(ListView.Highlighted)
    def _moved(self, event: ListView.Highlighted) -> None:
        if self._pending and getattr(event.item, "session_id", None) != self._pending:
            self._withdraw()  # moving off the row answers "no"

    @on(Button.Pressed, "#picker-delete")
    def _delete_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_delete()

    def _highlighted_id(self) -> str | None:
        item = self.query_one("#picker-list", ListView).highlighted_child
        return getattr(item, "session_id", None)

    def action_delete(self) -> None:
        sid = self._highlighted_id()
        if not sid or sid == "new":
            return
        title = self.query_one("#picker-title", Static)
        if sid == self.locked:
            title.update("⏳ 진행 중인 세션은 삭제할 수 없어요 — Stop 한 뒤에.")
            return
        if self._pending != sid:  # first press: ask, and say how much would go
            self._pending = sid
            below = len(storage.descendants(sid, self._sessions)) - 1
            tail = f" (아래 세션 {below}개 포함)" if below else ""
            title.update(f"🗑 삭제할까요? {self._title_of(sid)}{tail} — d 또는 버튼 다시 · Esc 취소")
            return
        # second press: do it
        gone = set(storage.delete_session(sid))
        self._pending = None
        self._sessions = [s for s in self._sessions if s["id"] not in gone]
        lv = self.query_one("#picker-list", ListView)
        for item in list(lv.children):
            if getattr(item, "session_id", None) in gone:
                item.remove()
        title.update(_HINT)
        if self.current in gone:  # the app must leave the file before it is gone
            self.dismiss("new")

    def _title_of(self, sid: str) -> str:
        node = next((s for s in self._sessions if s["id"] == sid), {})
        return node.get("title") or sid

    def _withdraw(self) -> None:
        self._pending = None
        self.query_one("#picker-title", Static).update(_HINT)

    def action_cancel(self) -> None:
        if self._pending:
            self._withdraw()
            return
        self.dismiss(None)
