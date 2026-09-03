"""The pinned checklist of the current plan. Stateless like todo_write: each call
replaces the whole list. The status vocabulary lives in ahacode.tools.plan."""

from __future__ import annotations

from textual.widgets import Static

from ahacode.tools.plan import FINISHED, coerce_items, mark, unfinished


class TodoPanel(Static):
    """The docked-top plan checklist; hidden until the first todo_write."""

    def __init__(self) -> None:
        super().__init__("", markup=False)  # todo text comes from the model and may contain '['
        self._content = ""
        self.items: list[dict] = []  # the plan as the model last sent it
        self.collapsed = False  # folded to one line; the plan itself is untouched
        self.display = False

    def clear(self) -> None:
        """Drop the plan and hide the panel; a hidden-but-stale list would describe
        another session's plan the moment it was shown again."""
        self.items = []
        self.collapsed = False
        self._content = ""
        self.update("")
        self.display = False
        self.set_class(False, "todo-panel--done")

    def update_todos(self, items: list[dict]) -> None:
        """Replace the whole plan. Statuses come only from the model's list, so ☑ on
        screen always means the model declared the step done.

        Args:
            items: The model's full list, in any of the shapes coerce_items repairs.
        """
        self.items, _ = coerce_items(items)
        self.collapsed = False  # a new or revised plan is worth seeing in full
        self._redraw()

    def unfinished(self) -> list[dict]:
        """The steps still owed: neither done nor cancelled."""
        return unfinished(self.items)

    def set_collapsed(self, collapsed: bool) -> None:
        """Fold the plan to one summary line, or unfold it; presentation only.

        Args:
            collapsed: Whether to fold.
        """
        if self.items and self.collapsed != collapsed:
            self.collapsed = collapsed
            self._redraw()

    def on_click(self) -> None:
        self.set_collapsed(not self.collapsed)

    def _summary(self) -> str:
        """The folded line, kept short: Korean glyphs are two cells wide."""
        finished = sum(1 for it in self.items if it.get("status") in FINISHED)
        return f"▸ Plan {finished}/{len(self.items)} · 클릭 펼치기"

    def _redraw(self) -> None:
        """Draw the checklist from `items`. Not named `_render`: that is Textual's
        own hook, and shadowing it makes the widget render as None."""
        done = bool(self.items) and not self.unfinished()
        if self.collapsed:
            self._content = "✓ Plan complete" if done else self._summary()
        else:
            lines = [f"{mark(it.get('status'))} {it['content']}" for it in self.items]
            header = "✓ Plan complete" if done else "▾ Plan"
            self._content = "\n".join([header, *lines])
        self.update(self._content)
        self.display = True
        self.set_class(done, "todo-panel--done")
