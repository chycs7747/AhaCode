"""Canonical streaming events — the one vocabulary the UI consumes.

Two producers emit these: client.py (what the model streams) and agent.py
(what the tool loop does). This is a tagged-union event model — a set of small
typed records distinguished by their class and dispatched via isinstance, which
is Python's stand-in for matching on a `type` discriminant.
"""

from dataclasses import dataclass


@dataclass
class ThinkingDelta:
    """A fragment of the model's reasoning stream (shown in the dimmed bubble)."""

    text: str


@dataclass
class TextDelta:
    """A fragment of the model's answer stream."""

    text: str


@dataclass
class ToolCallDelta:
    """A streamed fragment of a tool call, for live display (append model,
    like TextDelta). The final parsed ToolCall still follows for execution."""

    index: int
    name: str
    fragment: str


@dataclass
class ToolCall:
    """A completed tool call the model wants run.

    Reassembled inside client.py from streamed fragments (the arguments arrive
    as JSON pieces spread across many chunks). `arguments` is already parsed
    into a dict by the time the UI/agent sees it.

    `parse_error` is set when the model's argument JSON could not be parsed: the
    call is emitted anyway (with empty arguments) so the loop can feed an error
    result back and let the model resend it, instead of silently dropping the call
    — a dropped call can leave a turn with no tool call, which reads as "done".
    """

    id: str
    name: str
    arguments: dict
    parse_error: str | None = None


@dataclass
class ToolResult:
    """The outcome of running a ToolCall — produced by agent.py, not the model."""

    id: str
    name: str
    output: str
    is_error: bool = False


@dataclass
class Notice:
    """Something the HARNESS needs to tell the user — not the model's answer.

    Context compaction is the first: silently dropping history would leave the user
    wondering why the agent forgot something, so the loop says so.
    """

    text: str


@dataclass
class Usage:
    """Token accounting for one model call, from the stream's usage trailer
    (choices=[] chunk sent when stream_options.include_usage is on)."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# The union every consumer switches on. isinstance(event, TextDelta) is the
# Python equivalent of matching on a discriminated-union `type` tag.
Event = (
    ThinkingDelta | TextDelta | ToolCallDelta | ToolCall | ToolResult | Notice | Usage
)
