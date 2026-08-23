"""SubagentCard: the nested 🤖 task card a spawned sub-agent renders into.

A foldable Collapsible (same family as ThinkingBlock / ToolResultBlock) whose body
is a live container: the child agent's events (thinking → tools → answer) are
rendered into `.body` through the app's normal _render_event, one nesting level in,
so a delegated task reads as one connected sub-flow inside the parent's turn. The
purple left rail on the body distinguishes a spawned sub-agent from the parent's
own green turn rail.

It stays expanded while the child works (you watch the flow), then folds itself on
done() with a ✓ tool-count chip — the parent's synthesis is what reads inline, and
the child's full transcript is one click away (Claude Code's collapsed-subtask look).
"""

from __future__ import annotations

from textual.containers import Vertical
from textual.widgets import Collapsible


class SubagentCard(Collapsible):
    # NOTE: don't name a helper `_title` — Collapsible uses self._title internally
    # for its CollapsibleTitle widget, so a method of that name gets shadowed on init.
    def __init__(self, description: str, model: str) -> None:
        self._desc = description
        self._model = model
        # The child's events mount into this container (via the app's _render_event).
        self._body = Vertical(classes="subagent-body")
        super().__init__(self._body, title=self._label(), collapsed=False)
        self.add_class("subagent-card")

    def _label(self, *, done: bool = False, tools: int = 0) -> str:
        base = f"🤖 task · {self._desc} · {self._model}"
        if not done:
            return base
        chip = f"✓ {tools}개 도구" if tools else "✓ 완료"
        return f"{base} · {chip}"

    @property
    def body(self) -> Vertical:
        return self._body

    def done(self, tool_count: int = 0) -> None:
        """Child finished — show a ✓ chip and fold the card. Collapsible.title is a
        reactive, so assigning it re-renders the header (_watch_title)."""
        self.title = self._label(done=True, tools=tool_count)
        self.collapsed = True
