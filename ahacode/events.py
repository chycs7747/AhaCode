"""Canonical streaming events: the one vocabulary every layer speaks.

client.py emits what the model streams, agent.py what the tool loop does, and the
UI dispatches on the class.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThinkingDelta:
    """A fragment of the model's reasoning stream."""

    text: str


@dataclass
class TextDelta:
    """A fragment of the model's answer stream."""

    text: str


@dataclass
class ToolCallDelta:
    """A streamed fragment of a tool call, for live display; the parsed ToolCall
    still follows once the stream ends."""

    index: int
    name: str
    fragment: str


@dataclass
class ToolCall:
    """A completed tool call the model wants run; `arguments` is already parsed.

    `parse_error` is set when the argument JSON could not be parsed: the call is
    emitted anyway so the loop can feed an error back and the model can resend it.
    """

    id: str
    name: str
    arguments: dict
    parse_error: str | None = None


@dataclass
class ToolResult:
    """The outcome of running a ToolCall, produced by agent.py."""

    id: str
    name: str
    output: str
    is_error: bool = False


@dataclass
class Notice:
    """Something the harness tells the user, apart from the model's answer."""

    text: str


@dataclass
class Phase:
    """A stretch of harness work that streams nothing while it runs (compaction),
    so the status line can keep a clock on it. Sent in pairs: done=False opens
    the stretch, done=True closes it."""

    name: str
    done: bool = False


@dataclass
class Usage:
    """Token accounting for one model call, from the stream's usage trailer."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# The union every consumer switches on.
Event = (
    ThinkingDelta | TextDelta | ToolCallDelta | ToolCall | ToolResult | Notice
    | Phase | Usage
)
