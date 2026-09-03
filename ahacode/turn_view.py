"""The live turn on screen: canonical events in, mounted widgets out. The only
place that knows which event becomes a bubble, a card, or the pinned panel.
Runs on the main thread via call_from_thread."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from textual.containers import VerticalScroll

from ahacode.events import (
    Notice, Phase, TextDelta, ThinkingDelta, ToolCall, ToolCallDelta, ToolResult,
)
from ahacode.render import diff_stats, edit_diff_lines, tool_summary
from ahacode.widgets.chatbox import Chatbox
from ahacode.widgets.thinking import ThinkingBlock
from ahacode.widgets.todo_panel import TodoPanel
from ahacode.widgets.tool_result import ToolResultBlock

# Harness phases share the running-tool clock; this is the id they book it under.
# Not a call id, so it cannot collide with one; one slot, since phases do not nest.
_PHASE_ID = "\0phase"

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


@dataclass
class TurnBoxes:
    """One turn's live bubbles, plus whether it owns the session-level UI.

    Only the main loop's boxes set `owns_session`, so a sub-agent cannot touch the
    pinned checklist, the status line, or the plan gate.
    """

    thinking: ThinkingBlock | None = None
    answer: Chatbox | None = None
    tool: dict = field(default_factory=dict)  # stream index -> live bubble
    tool_buf: dict = field(default_factory=dict)  # stream index -> accumulated args
    call_args: dict = field(default_factory=dict)  # call id -> parsed arguments
    owns_session: bool = False

    def clear_tools(self) -> None:
        """A finished tool call ends the live write bubble."""
        self.tool.clear()
        self.tool_buf.clear()

    def fold_thinking(self) -> None:
        """Collapse the reasoning block once the answer or a tool call begins."""
        if self.thinking is not None:
            self.thinking.done()
            self.thinking = None

    def end_answer(self) -> None:
        """Close the current answer bubble; the next TextDelta opens a fresh one."""
        self.fold_thinking()
        self.answer = None


def tool_unescape(s: str) -> str:
    """Decode a possibly incomplete JSON string value, escape by escape.

    Args:
        s: The raw string body, without its quotes.

    Returns:
        The decoded text.
    """
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            out.append(_ESCAPES.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def render_tool_stream(name: str, args: str) -> str:
    """The live label for a streaming tool call whose argument JSON may be incomplete.

    Args:
        name: The tool name.
        args: The arguments received so far.

    Returns:
        write: a path header plus the streamed content; other tools: the raw args.
    """
    if name == "write":
        m = re.search(r'"path"\s*:\s*"((?:[^"\\]|\\.)*)"', args)
        path = tool_unescape(m.group(1)) if m else "…"
        body = ""
        cm = re.search(r'"content"\s*:\s*"', args)
        if cm:
            tail = re.sub(r'"\s*}?\s*$', "", args[cm.end():])
            body = tool_unescape(tail)
        return f"🔧 write · {path}\n{body}"
    return f"🔧 {name}  {args}"


def edit_card(args: dict) -> Chatbox:
    """The edit-diff card (path title, count chip, -/+ lines), shared by the live
    turn and the history replay.

    Args:
        args: The edit call's arguments.

    Returns:
        The mounted-ready bubble.
    """
    path = args.get("path", "?")
    old, new = args.get("old_string", ""), args.get("new_string", "")
    text, plain = edit_diff_lines(old, new)
    added, removed = diff_stats(old, new)
    box = Chatbox("", role="tool-diff")
    box.set_rich(text, plain)
    box.border_title = f"✏ edit · {path}"
    box.border_subtitle = f"+{added} −{removed}"
    return box


class TurnView:
    """Mounts the events of one turn into its container. Holds the app for the
    session-wide things (status line, running-tool clock, plan gate), each guarded
    by boxes.owns_session."""

    def __init__(self, app) -> None:
        self.app = app

    async def render(self, event, boxes: TurnBoxes, container: VerticalScroll) -> None:
        """Mount one event.

        Args:
            event: Any canonical event.
            boxes: The turn's live bubbles.
            container: Where new widgets mount.
        """
        handler = {
            Notice: self._notice,
            Phase: self._phase,
            ThinkingDelta: self._thinking,
            TextDelta: self._text,
            ToolCallDelta: self._tool_delta,
            ToolCall: self._tool_call,
            ToolResult: self._tool_result,
        }.get(type(event))
        if handler is not None:
            await handler(event, boxes, container)
        # Follow the stream only while pinned to the bottom, so a scroll-up stays off.
        if self.app._follow_output:
            self.app.query_one("#chat-container", VerticalScroll).scroll_end(animate=False)

    # --- one method per event ------------------------------------------------

    async def _notice(self, event, boxes, container) -> None:
        boxes.end_answer()  # the next TextDelta opens a fresh bubble below the notice
        await container.mount(Chatbox(event.text, role="system"))

    async def _phase(self, event, boxes, container) -> None:
        if not boxes.owns_session:  # a sub-agent's card carries its own clock
            return
        if event.done:
            self.app._running_tools.pop(_PHASE_ID, None)
        else:
            self.app._running_tools[_PHASE_ID] = (event.name, time.monotonic())
            self.app._set_status(f"● {event.name} · 0초")

    async def _thinking(self, event, boxes, container) -> None:
        if boxes.thinking is None:
            boxes.thinking = ThinkingBlock()
            await container.mount(boxes.thinking)
        boxes.thinking.append_chunk(event.text)
        self.app._set_status("● thinking…")

    async def _text(self, event, boxes, container) -> None:
        boxes.fold_thinking()
        if boxes.answer is None:
            boxes.answer = Chatbox("", role="assistant", markdown=True)
            await container.mount(boxes.answer)
        boxes.answer.append_chunk(event.text)
        self.app._set_status("● generating…")

    async def _tool_delta(self, event, boxes, container) -> None:
        if event.name == "todo_write":
            return  # shows in the panel on the final call
        boxes.end_answer()
        if event.name == "edit":
            self.app._set_status("● editing…")
            return  # the diff card is rendered on the final ToolCall
        if event.name != "write":
            # No call bubble: the result card carries the input in its title.
            self.app._set_status(f"● running {event.name}…")
            return
        # write streams its content live into one bubble
        buf = boxes.tool_buf.get(event.index, "") + event.fragment
        boxes.tool_buf[event.index] = buf
        box = boxes.tool.get(event.index)
        if box is None:
            box = Chatbox("", role="tool-call")
            await container.mount(box)
            boxes.tool[event.index] = box
        box._content = render_tool_stream(event.name, buf)
        box.update(box._content)
        self.app._set_status("● writing…")

    async def _tool_call(self, event, boxes, container) -> None:
        app = self.app
        boxes.end_answer()
        boxes.call_args[event.id] = event.arguments  # the result card titles itself with it
        if boxes.owns_session:  # only this loop's tools drive the status line
            app._running_tools[event.id] = (event.name, time.monotonic())
        boxes.clear_tools()
        if event.name == "todo_write":
            if boxes.owns_session:
                app.plan.note_todo_update(
                    app.query_one(TodoPanel), event.arguments.get("items", [])
                )
            app._set_status("● planning…")
        elif event.name == "plan_submit":  # the gate opens when the result confirms the save
            if boxes.owns_session:
                app.query_one(TodoPanel).update_todos(app.plan.items(event.arguments))
            app._set_status("● submitting plan…")
        elif event.name == "edit":
            await container.mount(edit_card(event.arguments))
            app._set_status("● editing…")
        else:
            app._set_status(f"● running {event.name}…")

    async def _tool_result(self, event, boxes, container) -> None:
        app = self.app
        app._running_tools.pop(event.id, None)
        if event.name == "todo_write":
            return  # already in the pinned panel
        if event.name == "plan_submit" and not event.is_error and boxes.owns_session:
            # A rejected submission falls through to the error card, so the user sees why.
            args = boxes.call_args.get(event.id, {})
            path = event.output.split(" (", 1)[0].removeprefix("Plan saved to ")
            await app.plan.open(
                [it["content"] for it in app.plan.items(args)],
                str(args.get("summary", "")).strip(), path, container,
            )
            return
        if event.name == "edit" and not event.is_error:
            return  # already shown as the diff card
        if event.name == "task":
            return  # the sub-agent's own card shows its flow and result
        summary = tool_summary(event.name, boxes.call_args.get(event.id, {}))
        await container.mount(
            ToolResultBlock(event.name, event.output, event.is_error, summary=summary)
        )
