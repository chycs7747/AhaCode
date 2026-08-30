"""The live turn on screen: canonical events in, mounted widgets out.

The counterpart to render.py, which builds widget-free previews. This module
does the mounting, and it is the ONLY place that knows an event's shape maps to
a bubble, a card, or the pinned panel.

It runs on the main thread via call_from_thread, which awaits each call — so a
mount completes before the next append and nothing races the worker.
"""

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

# Harness phases share _running_tools with the tools so they get the same ticking
# clock. This is the id they book it under: not a call id, so it cannot collide
# with one, and a single slot because phases do not nest.
_PHASE_ID = "\0phase"

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


@dataclass
class TurnBoxes:
    """One turn's live bubbles, plus who owns the session-level UI.

    `owns_session` is the ownership test the renderer asks before touching
    anything outside its own container: the pinned checklist, the status line,
    the plan gate. Only the main loop's boxes set it, so a sub-agent planning
    its private sub-task cannot overwrite the parent's checklist, and its
    plan_submit cannot pause the parent's loop.

    It used to be a "gate" key that had to be ABSENT — a permission bit encoded
    as a missing dict entry, which is how the sub-agent overwrite got in.
    """

    thinking: ThinkingBlock | None = None
    answer: Chatbox | None = None
    tool: dict = field(default_factory=dict)       # stream index -> live bubble
    tool_buf: dict = field(default_factory=dict)   # stream index -> accumulated args
    call_args: dict = field(default_factory=dict)  # call id -> parsed arguments
    owns_session: bool = False

    def clear_tools(self) -> None:
        """A finished tool call ends the live write-bubble, whatever it was."""
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
    """Decode a (possibly incomplete) JSON string value, escape by escape."""
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
    """Live label for a streaming tool call whose args JSON may be incomplete.

    write is shown as a path header + streamed content (pull known fields out
    early); every other tool shows its raw accumulating args.
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
    """The green edit-diff card (path title + count chip + -/+ lines) — shared by
    the live turn and history restore."""
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
    """Mounts the events of one turn into its container.

    Holds the app for the three things that are session-wide rather than
    turn-wide: the status line, the running-tool clock, and the plan gate. Each
    is guarded by boxes.owns_session, so a sub-agent's view touches none of them.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def render(self, event, boxes: TurnBoxes, container: VerticalScroll) -> None:
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
        # Follow the stream ONLY while _follow_output is set (a watcher on scroll_y
        # drives it — see _update_follow), so a manual scroll-up during streaming
        # STAYS off. The old per-chunk "near the bottom?" snap re-pinned every chunk,
        # making it impossible to scroll away mid-answer.
        if self.app._follow_output:
            self.app.query_one("#chat-container", VerticalScroll).scroll_end(animate=False)

    # --- one method per event ------------------------------------------------

    async def _notice(self, event, boxes, container) -> None:
        # Harness-authored, not the model's answer — the same neutral bubble a
        # slash command gets. It ends the current answer, so the next TextDelta
        # opens a fresh one below it.
        boxes.end_answer()
        await container.mount(Chatbox(event.text, role="system"))

    async def _phase(self, event, boxes, container) -> None:
        # Harness work with a duration, on the same clock the tools use: the point
        # is a number that MOVES, because a static line is the picture a deadlock
        # makes. A sub-agent's phases are skipped like its tools — its card carries
        # its own clock.
        if not boxes.owns_session:
            return
        if event.done:
            self.app._running_tools.pop(_PHASE_ID, None)
        else:
            self.app._running_tools[_PHASE_ID] = (event.name, time.monotonic())
            self.app._status(f"● {event.name} · 0초")

    async def _thinking(self, event, boxes, container) -> None:
        if boxes.thinking is None:
            boxes.thinking = ThinkingBlock()  # foldable; starts expanded
            await container.mount(boxes.thinking)
        boxes.thinking.append_chunk(event.text)
        self.app._status("● thinking…")

    async def _text(self, event, boxes, container) -> None:
        boxes.fold_thinking()  # answer starting → auto-collapse the reasoning
        if boxes.answer is None:
            # markdown=True: render the answer as Markdown so ```code``` and
            # ```diff fences become highlighted blocks.
            boxes.answer = Chatbox("", role="assistant", markdown=True)
            await container.mount(boxes.answer)
        boxes.answer.append_chunk(event.text)
        self.app._status("● generating…")

    async def _tool_delta(self, event, boxes, container) -> None:
        if event.name == "todo_write":
            return  # shows in the panel on its final call, not a bubble
        boxes.end_answer()
        if event.name == "edit":
            self.app._status("● editing…")
            return  # edit's coloured diff is rendered on the final ToolCall
        if event.name != "write":
            # bash / read / grep …: no call bubble — the result card shows the
            # command in its title (IN) and the output in its body (OUT), the
            # Claude Code shape. The status bar reports progress until it lands.
            self.app._status(f"● running {event.name}…")
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
        self.app._status("● writing…")

    async def _tool_call(self, event, boxes, container) -> None:
        app = self.app
        boxes.end_answer()  # fold reasoning; the next turn opens fresh bubbles
        # Remember the call's input so the result card can title itself with it.
        boxes.call_args[event.id] = event.arguments
        # Only this loop's own tools drive the status line. A sub-agent reports
        # inside its card, and with several running in parallel their tools would
        # otherwise take turns overwriting each other in the one status line.
        if boxes.owns_session:
            app._running_tools[event.id] = (event.name, time.monotonic())
        boxes.clear_tools()
        if event.name == "todo_write":  # goes to the pinned panel, not a bubble
            if boxes.owns_session:
                app.plan.note_todo_update(
                    app.query_one(TodoPanel), event.arguments.get("items", [])
                )
            app._status("● planning…")
        elif event.name == "plan_submit":  # the plan goes to the panel; the gate
            if boxes.owns_session:        # opens when its result confirms the save
                app.query_one(TodoPanel).update_todos(app.plan.items(event.arguments))
            app._status("● submitting plan…")
        elif event.name == "edit":  # green diff card (shared with history restore)
            await container.mount(edit_card(event.arguments))
            app._status("● editing…")
        else:
            # write streamed a live content bubble; every other tool shows only its
            # result card, so there is no call bubble to keep.
            app._status(f"● running {event.name}…")

    async def _tool_result(self, event, boxes, container) -> None:
        app = self.app
        app._running_tools.pop(event.id, None)
        if event.name == "todo_write":
            return  # already reflected in the pinned panel
        if event.name == "plan_submit" and not event.is_error and boxes.owns_session:
            # A rejected submission falls through to the error card below, so the
            # user sees why; the model already has the reason and resubmits.
            args = boxes.call_args.get(event.id, {})
            path = event.output.split(" (", 1)[0].removeprefix("Plan saved to ")
            await app.plan.open(
                [it["content"] for it in app.plan.items(args)],
                str(args.get("summary", "")).strip(), path, container,
            )
            return
        if event.name == "edit" and not event.is_error:
            return  # a successful edit is already shown as the diff card
        if event.name == "task":
            return  # the sub-agent's own 🤖 card already shows its flow + result
        # One foldable card: the command/path in the title (IN), the output in the
        # body (OUT). Long output / errors fold away.
        summary = tool_summary(event.name, boxes.call_args.get(event.id, {}))
        await container.mount(
            ToolResultBlock(event.name, event.output, event.is_error, summary=summary)
        )
