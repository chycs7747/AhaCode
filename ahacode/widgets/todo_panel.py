"""A pinned, live-updating checklist of the current plan.

When the model calls todo_write, the app feeds the items here (from the tool
call's arguments) instead of dropping a bubble that scrolls away — the plan stays
pinned to the top. Stateless like the tool: each call replaces the whole list.

The status vocabulary (glyphs, key names, which one is terminal) is NOT defined
here — it belongs to the plan itself and lives in ahacode.tools.plan, so the tool,
its JSON schema, and this view can never disagree about what a status is.
"""

from textual.widgets import Static

from ahacode.tools.plan import FINISHED, coerce_items, mark, unfinished


class TodoPanel(Static):
    """Docked-top plan checklist; hidden until the first todo_write."""

    def __init__(self) -> None:
        # markup=False: todo text comes from the model and may contain '[' (code,
        # paths) — see Chatbox for why unparsed content avoids a MarkupError crash.
        super().__init__("", markup=False)
        self._content = ""
        self.items: list[dict] = []  # the raw plan, as the model last sent it
        self.collapsed = False       # folded to one line; the plan itself is untouched
        self.display = False  # nothing to show until a plan exists

    def clear(self) -> None:
        """Drop the plan and hide the panel.

        Not just `display = False`: the panel is a view of the open session, so
        switching sessions must empty it — a hidden-but-stale list would describe
        another session's plan the moment it was shown again.
        """
        self.items = []
        self.collapsed = False
        self._content = ""
        self.update("")
        self.display = False
        self.set_class(False, "todo-panel--done")

    def update_todos(self, items: list[dict]) -> None:
        """Replace the whole plan — the model re-sends the full list every time.

        This is also the path a plan REVISED with the user takes: they discuss, the
        model calls todo_write again with the amended list, and the pinned panel is
        rewritten from it. Statuses come ONLY from the model's list: nothing in the
        harness ticks a step, so ☑ on screen always means the model declared it done.
        """
        # The one choke point for shape: every caller (live call, history replay,
        # plan_submit's steps) passes through here, so a list that arrived as a
        # string or as bare strings is repaired once, not per caller.
        self.items, _ = coerce_items(items)
        self.collapsed = False  # a new or revised plan is worth seeing in full
        self._redraw()

    def unfinished(self) -> list[dict]:
        """Steps still owed (neither done nor cancelled) — what a turn that ends
        with the plan incomplete leaves behind."""
        return unfinished(self.items)

    def set_collapsed(self, collapsed: bool) -> None:
        """Fold the plan to a single summary line, or unfold it.

        Folding is presentation only — `items` is untouched, so /run can still resume
        from a folded plan. A long plan pinned at full height eats the chat area, and
        it is most in the way exactly when the run has stopped and there is output to
        read; app.action_stop folds it for that reason.
        """
        if self.items and self.collapsed != collapsed:
            self.collapsed = collapsed
            self._redraw()

    def on_click(self) -> None:
        """Click the panel to fold/unfold it — the only affordance it needs."""
        self.set_collapsed(not self.collapsed)

    def _summary(self) -> str:
        """The folded line: enough to know where the plan stands.

        Carrying on after a stop is just typing into the impl session (the model
        holds its own context), so there is no command to name here any more.
        """
        finished = sum(1 for it in self.items if it.get("status") in FINISHED)
        # Kept short on purpose: Korean glyphs are two cells wide, and a folded line
        # that wraps to two rows is not folded. Measured to hold one row down to a
        # 44-column terminal.
        return f"▸ Plan {finished}/{len(self.items)} · 클릭 펼치기"

    def _redraw(self) -> None:
        """Draw the checklist from `items` — the one place that reads the statuses.

        NOT named `_render`: Textual's Widget._render() is the internal hook that
        returns this widget's visual, and shadowing it makes the widget render as
        None (it crashes during layout, not at import — so the name looks fine until
        the app draws).
        """
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
