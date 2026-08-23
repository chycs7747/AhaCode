"""SubagentCard: the nested 🤖 task card a spawned sub-agent renders into.

A foldable Collapsible (same family as ThinkingBlock / ToolResultBlock) whose body
is a live container: the child agent's events (thinking → tools → answer) are
rendered into `.body` through the app's normal _render_event, one nesting level in,
so a delegated task reads as one connected sub-flow inside the parent's turn. The
purple left rail on the body distinguishes a spawned sub-agent from the parent's
own green turn rail.
"""

from __future__ import annotations

from textual.containers import Vertical
from textual.widgets import Collapsible


class SubagentCard(Collapsible):
    def __init__(self, description: str, model: str) -> None:
        # The child's events mount into this container (via the app's _render_event).
        self._body = Vertical(classes="subagent-body")
        super().__init__(
            self._body,
            title=f"🤖 task · {description} · {model}",
            collapsed=False,  # open while the child works so its flow is visible
        )
        self.add_class("subagent-card")

    @property
    def body(self) -> Vertical:
        return self._body
