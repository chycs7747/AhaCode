"""A modal to open another session, start a new one, rename or delete one.

Lists sessions/*.jsonl as a tree (storage.build_tree over their headers) and
dismisses with the chosen session id, "new", or None (cancel).

Every row carries its own ✎ / 🗑 buttons (a click on the row itself opens the
session, so an action needs a target that is not "the row"): ✎ swaps the title
for an input — Enter saves, Esc restores; 🗑 must be pressed TWICE — the first
turns it into "확인?" and puts the question in the title, naming what would go
(the row and every session under it), the second answers. Any other action,
or Esc, withdraws the question. The keyboard mirrors the buttons on the
highlighted row: r rename, d delete.

Deleting the session that is open dismisses with "new" so the app moves off the
file before it is gone; the one whose turn is still running cannot be deleted.
"""

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from ahacode import storage

_ICON = {"main": "🧠", "impl": "🛠", "subagent": "🤖", "fork": "🌿"}
# Two authored lines (own full-width Static, below the header row) so the guide
# never wraps mid-phrase against the close button.
_HINT = (
    "↑↓ 이동 · Enter/클릭 열기 · Esc 닫기\n"
    "✎ 또는 r 이름 변경 · 🗑 또는 d 삭제 · 🤖 서브에이전트는 보기 전용"
)


def _flatten(nodes: list[dict], level: int = 0):
    """Pre-order walk of the session tree, yielding (indent level, node)."""
    for n in nodes:
        yield level, n
        yield from _flatten(n["children"], level + 1)


class SessionRow(ListItem):
    """One session: indented title, then its own rename / delete buttons."""

    def __init__(self, level: int, node: dict) -> None:
        super().__init__()
        self.session_id = node["id"]
        self.level = level
        self.node = node
        self.armed = False  # 🗑 pressed once; the next press deletes

    def compose(self) -> ComposeResult:
        with Horizontal(classes="picker-row"):
            yield Label(self._text(), classes="picker-row-title")
            yield Button("✎", compact=True, classes="picker-rename")
            yield Button("🗑", compact=True, variant="error", classes="picker-delete")

    def _text(self) -> str:
        indent = "  " * self.level
        title = self.node.get("title") or self.node["id"]
        model = self.node.get("model") or "?"
        icon = _ICON.get(self.node.get("kind", "main"), "•")
        return f"{indent}{icon} {title}   · {model}"

    def refresh_title(self) -> None:
        self.query_one(".picker-row-title", Label).update(self._text())

    def arm(self, armed: bool) -> None:
        self.armed = armed
        btn = self.query_one(".picker-delete", Button)
        btn.label = "확인?" if armed else "🗑"


class SessionPicker(ModalScreen[str | None]):
    """dismiss("new") = start new · dismiss(id) = open that session · dismiss(None) = cancel."""

    BINDINGS = [
        ("escape", "cancel", "Close"),
        ("d", "delete", "Delete"),
        ("r", "rename", "Rename"),
    ]

    def __init__(self, current: str | None = None, locked: str | None = None) -> None:
        super().__init__()
        self.current = current   # the session open behind the picker
        self.locked = locked     # a session whose turn is running: not deletable
        self._sessions: list[dict] = []
        self._renaming: SessionRow | None = None

    def compose(self) -> ComposeResult:
        self._sessions = storage.list_sessions()
        tree = storage.build_tree(self._sessions)
        items = [self._new_item()]
        for level, node in _flatten(tree):
            items.append(SessionRow(level, node))
        with Vertical(id="picker-box"):
            with Horizontal(id="picker-head"):
                yield Static("Sessions", id="picker-heading")
                yield Button("✕ 닫기", compact=True, id="picker-close")
            yield Static(_HINT, id="picker-title")
            yield ListView(*items, id="picker-list")

    @staticmethod
    def _new_item() -> ListItem:
        item = ListItem(Label("＋  new session"))
        item.session_id = "new"
        return item

    # --- open ------------------------------------------------------------------

    @on(ListView.Selected)
    def _picked(self, event: ListView.Selected) -> None:
        if self._renaming is not None:
            return  # Enter inside the rename input is handled by the input
        self.dismiss(getattr(event.item, "session_id", None))

    @on(ListView.Highlighted)
    def _moved(self, event: ListView.Highlighted) -> None:
        # Moving the highlight to a DIFFERENT session row answers "no". Only a real
        # move disarms: a mount-time or duplicate Highlighted (e.g. the list settling
        # on its first row) that is not a SessionRow, or is the armed row itself, is
        # ignored — otherwise it would cancel the question the user just opened.
        item = event.item
        if not isinstance(item, SessionRow):
            return
        for row in self.query(SessionRow):
            if row.armed and row is not item:
                self._disarm()
                return

    # --- row buttons -------------------------------------------------------------

    def _row_of(self, widget) -> SessionRow | None:
        node = widget
        while node is not None and not isinstance(node, SessionRow):
            node = node.parent
        return node

    def _highlighted_row(self) -> SessionRow | None:
        item = self.query_one("#picker-list", ListView).highlighted_child
        return item if isinstance(item, SessionRow) else None

    @on(Button.Pressed, ".picker-delete")
    def _delete_button(self, event: Button.Pressed) -> None:
        event.stop()
        self._delete(self._row_of(event.button))

    @on(Button.Pressed, ".picker-rename")
    def _rename_button(self, event: Button.Pressed) -> None:
        event.stop()
        self._rename(self._row_of(event.button))

    @on(Button.Pressed, "#picker-close")
    def _close_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_cancel()  # same as Esc: withdraw a pending rename/arm, else close

    def action_delete(self) -> None:
        self._delete(self._highlighted_row())

    def action_rename(self) -> None:
        self._rename(self._highlighted_row())

    # --- delete: two presses ------------------------------------------------------

    def _delete(self, row: SessionRow | None) -> None:
        if row is None:
            return
        title = self.query_one("#picker-title", Static)
        if row.session_id == self.locked:
            title.update("⏳ 진행 중인 세션은 삭제할 수 없어요 — Stop 한 뒤에.")
            return
        if not row.armed:  # first press: ask, and say how much would go
            self._disarm()
            row.arm(True)
            below = len(storage.descendants(row.session_id, self._sessions)) - 1
            tail = f" (아래 세션 {below}개 포함)" if below else ""
            name = row.node.get("title") or row.session_id
            title.update(f"🗑 삭제할까요? {name}{tail} — 🗑/d 다시 · Esc 취소")
            return
        # second press: do it
        gone = set(storage.delete_session(row.session_id))
        self._sessions = [s for s in self._sessions if s["id"] not in gone]
        lv = self.query_one("#picker-list", ListView)
        for item in list(lv.children):
            if getattr(item, "session_id", None) in gone:
                item.remove()
        title.update(_HINT)
        if self.current in gone:  # the app must leave the file before it is gone
            self.dismiss("new")

    def _disarm(self) -> None:
        for row in self.query(SessionRow):
            if row.armed:
                row.arm(False)
        self.query_one("#picker-title", Static).update(_HINT)

    # --- rename: inline input -----------------------------------------------------

    def _rename(self, row: SessionRow | None) -> None:
        if row is None or self._renaming is not None:
            return
        self._disarm()
        self._renaming = row
        label = row.query_one(".picker-row-title", Label)
        label.display = False
        box = Input(value=row.node.get("title") or "", placeholder="새 이름", classes="picker-rename-input")
        row.query_one(".picker-row").mount(box, before=label)
        box.focus()

    @on(Input.Submitted, ".picker-rename-input")
    def _rename_done(self, event: Input.Submitted) -> None:
        event.stop()
        row = self._renaming
        new = event.value.strip()
        if row is not None and new and new != (row.node.get("title") or ""):
            storage.set_title(storage.SESSIONS_DIR / f"{row.session_id}.jsonl", new)
            row.node["title"] = new
            for s in self._sessions:
                if s["id"] == row.session_id:
                    s["title"] = new
        self._end_rename()

    def _end_rename(self) -> None:
        row = self._renaming
        self._renaming = None
        if row is None:
            return
        for box in row.query(".picker-rename-input"):
            box.remove()
        row.query_one(".picker-row-title", Label).display = True
        row.refresh_title()
        self.query_one("#picker-list", ListView).focus()

    # --- escape ------------------------------------------------------------------

    def action_cancel(self) -> None:
        if self._renaming is not None:
            self._end_rename()  # restore the label, keep the old name
            return
        if any(row.armed for row in self.query(SessionRow)):
            self._disarm()
            return
        self.dismiss(None)
