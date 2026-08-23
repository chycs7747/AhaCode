from textual.widgets import Collapsible

from ahacode.widgets.chatbox import Chatbox


class ThinkingBlock(Collapsible):
    """A foldable reasoning ("thinking") block.

    Textual's native Collapsible supplies the click-to-toggle title bar and the
    ▼/▶ affordance, so we don't hand-roll any folding. Reasoning deltas stream
    into an inner Chatbox.

    The block auto-collapses once the agent finishes writing the reasoning: it
    starts expanded so the user watches it think, then folds away on done().
    Clicking the title reopens it — handled entirely by Collapsible.
    """

    def __init__(self) -> None:
        # Keep a direct handle to the inner bubble so streamed deltas can append
        # to it after mount; Collapsible mounts it inside its Contents container.
        self._box = Chatbox("", role="thinking")
        super().__init__(self._box, title="🤔 thinking", collapsed=False)

    def append_chunk(self, chunk: str) -> None:
        """Forward a streamed reasoning delta to the inner bubble."""
        self._box.append_chunk(chunk)

    def done(self) -> None:
        """Reasoning for this turn ended — fold the block away (auto-collapse)."""
        self.collapsed = True
