"""The nested 🤖 card a spawned sub-agent renders into: expanded while the child
works, folded with a ✓ chip when it is done."""

from __future__ import annotations

import time

from textual.containers import Vertical
from textual.widgets import Collapsible


class SubagentCard(Collapsible):
    # No helper named `_title`: Collapsible uses self._title for its title widget.
    def __init__(self, description: str, model: str) -> None:
        self._desc = description
        self._model = model
        self._t0 = time.monotonic()
        self._elapsed = 0
        self._done = False
        self._body = Vertical(classes="subagent-body")  # the child's events mount here
        super().__init__(self._body, title=self._label(), collapsed=False)
        self.add_class("subagent-card")

    def _label(self, *, done: bool = False, tools: int = 0) -> str:
        base = f"🤖 task · {self._desc} · {self._model}"
        if done:
            chip = f"✓ {tools}개 도구" if tools else "✓ 완료"
            return f"{base} · {chip} · {self._elapsed}초"
        return f"{base} · {self._elapsed}초" if self._elapsed else base

    def tick(self) -> None:
        """Count up while the child works; each card has its own clock, since one
        status line cannot speak for several parallel children."""
        if self._done:
            return
        seconds = int(time.monotonic() - self._t0)
        if seconds != self._elapsed:  # only touch the reactive when the number moves
            self._elapsed = seconds
            self.title = self._label()

    @property
    def body(self) -> Vertical:
        return self._body

    def done(self, tool_count: int = 0) -> None:
        """Show the ✓ chip and fold the card.

        Args:
            tool_count: How many tool calls the child made.
        """
        self._done = True
        self._elapsed = int(time.monotonic() - self._t0)
        self.title = self._label(done=True, tools=tool_count)
        self.collapsed = True
