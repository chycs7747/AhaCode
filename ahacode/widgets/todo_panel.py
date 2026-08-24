"""A pinned, live-updating checklist of the current plan.

When the model calls todo_write, the app feeds the items here (from the tool
call's arguments) instead of dropping a bubble that scrolls away — the plan stays
pinned to the top. Stateless like the tool: each call replaces the whole list.
"""

from textual.widgets import Static


class TodoPanel(Static):
    """Docked-top plan checklist; hidden until the first todo_write."""

    _MARKS = {"done": "☑", "in_progress": "▶", "pending": "☐"}

    def __init__(self) -> None:
        # markup=False: todo text comes from the model and may contain '[' (code,
        # paths) — see Chatbox for why unparsed content avoids a MarkupError crash.
        super().__init__("", markup=False)
        self._content = ""
        self.items: list[dict] = []  # the raw plan, so /run can execute its steps
        self.display = False  # nothing to show until a plan exists

    def clear(self) -> None:
        """Drop the plan and hide the panel.

        Not just `display = False`: `/run` executes whatever sits in `items`, so a
        hidden-but-stale list would silently run another session's plan. The panel is
        a view of the open session, so switching sessions must empty it too.
        """
        self.items = []
        self._content = ""
        self.update("")
        self.display = False
        self.set_class(False, "todo-panel--done")

    def update_todos(self, items: list[dict]) -> None:
        self.items = list(items)
        lines = [
            f"{self._MARKS.get(it.get('status', 'pending'), '☐')} {it['content']}"
            for it in items
        ]
        done = bool(items) and all(it.get("status") == "done" for it in items)
        header = "✓ Plan complete" if done else "Plan"
        self._content = "\n".join([header, *lines])
        self.update(self._content)
        self.display = True
        self.set_class(done, "todo-panel--done")
