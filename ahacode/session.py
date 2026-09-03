"""Conversation state, decoupled from any widget."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChatSession:
    """The messages of one conversation, in the order they are sent to the model."""

    messages: list[dict] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
