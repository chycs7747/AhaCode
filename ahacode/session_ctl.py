"""Opening, replacing and repairing the session on screen.

The App still HOLDS the session — session_path, session_kind and friends are
read from all over, PlanRun and TurnView included, and moving them would buy
indirection rather than clarity. What lives here is every TRANSITION between
one session and the next, which is what was scattered: three methods that each
re-derived the same eight steps, plus the history replay they all end in.
"""

from __future__ import annotations

import json

from textual.containers import Vertical, VerticalScroll

from ahacode import config, storage
from ahacode.render import tool_summary
from ahacode.session import ChatSession
from ahacode.tools import spill
from ahacode.turn_view import edit_card
from ahacode.widgets.chatbox import Chatbox
from ahacode.widgets.session_picker import SessionPicker
from ahacode.widgets.todo_panel import TodoPanel
from ahacode.widgets.tool_result import ToolResultBlock

INTERRUPT_NOTE = (
    "[system] The previous turn was interrupted before it finished. The project "
    "may have changed on disk — re-check the actual state (files, tests) and bring "
    "the plan's checklist into line with it before continuing. If the last step did "
    "not complete, redo it."
)


class SessionControl:
    """Every way the open session is replaced, and the replay that follows."""

    def __init__(self, app) -> None:
        self.app = app

    # --- transitions ---------------------------------------------------------

    async def adopt(self, path, meta: dict) -> None:
        """Make `path` the session on screen: state, then chrome, then history.

        The shared tail of new() and switch(). Order matters — render_history
        reads session.messages and refills the pinned panel, and the plan gate it
        may restore has to be the last thing scrolled to.
        """
        app = self.app
        app.session_path = path
        app.session_depth = int(meta.get("depth", 0))
        app.session_kind = str(meta.get("kind", "main"))
        app.session_parent_id = meta.get("parent_id")
        app._has_title = bool(meta.get("title"))
        app.plan.reset()  # a different session asks about its own plans
        spill.set_session(path)
        app._set_header_title(meta.get("title", ""))
        await self.render_history()
        app._reflect_view_only()
        app._set_status("")

    async def new(self) -> None:
        """Start a fresh session (new file + header) and clear the view."""
        app = self.app
        app.session = ChatSession()
        path = storage.new_session_path()
        storage.write_header(
            path, storage.make_header(path.stem, kind="main", model=config.load().name)
        )
        await self.adopt(path, {})  # no meta: a fresh main session at depth 0
        await app._say_system("new session started")

    async def switch(self, session_id: str) -> None:
        """Load another session by id and show its history."""
        app = self.app
        app.session = ChatSession()
        path = storage.SESSIONS_DIR / f"{session_id}.jsonl"
        app.session.messages = storage.load_messages(path)
        await self.adopt(path, storage.read_session_meta(path) or {})
        if app.session_kind == "impl":
            app._set_mode("act")  # an impl session acts; planning is its parent's
        if app.view_only:  # a sub-agent transcript — say up front that it is read-only
            await app._say_system(
                f"🔒 보기 전용 — 서브에이전트가 자동 생성한 기록(깊이 {app.session_depth})입니다. "
                "읽기만 가능해요. /new 로 새 세션을 시작하세요."
            )
        else:
            await self.repair_interrupted()  # a turn cut off mid-tool: fill + note

    async def repair_interrupted(self) -> None:
        """Fill in the results a turn cut off mid-tool never produced.

        The API demands a result for every tool_call, so each dangling call gets a
        synthetic one, plus a note telling the model to reassess the real state
        rather than trust a half-finished summary. Appended and persisted, so a
        reopened session finds nothing left to repair.
        """
        app = self.app
        dangling = storage.dangling_tool_calls(app.session.messages)
        if not dangling:
            return
        for call in dangling:
            # Name WHICH call was cut off: tool + subject, so three bash calls that
            # ran at once stay distinguishable. Same IN-line the result card shows.
            subject = tool_summary(call["name"], call["arguments"])
            what = f"`{call['name']}` ({subject})" if subject else f"`{call['name']}`"
            self._append({"role": "tool", "tool_call_id": call["id"],
                          "content": f"Interrupted: the {what} call did not complete."})
        self._append({"role": "user", "content": INTERRUPT_NOTE})
        await app._say_system("↻ 이전 턴이 중단됐어요 — 상태를 다시 확인하고 이어갑니다.")

    def _append(self, msg: dict) -> None:
        self.app.session.messages.append(msg)
        storage.append_message(self.app.session_path, msg)

    # --- the picker ----------------------------------------------------------

    def open_picker(self) -> None:
        """The picker needs to know which session is open (deleting it must move
        the app off the file first) and whether its turn is running (then it is
        not deletable at all)."""
        app = self.app
        current = app.session_path.stem
        locked = current if app._anything_running() else None
        app.push_screen(SessionPicker(current=current, locked=locked), self.picked)

    def picked(self, result: str | None) -> None:
        """SessionPicker dismissed — run the switch/new as an async worker."""
        app = self.app
        if result == "new":
            app.run_worker(self.new(), exclusive=False)
        elif result:
            app.run_worker(self.switch(result), exclusive=False)
        else:  # closed without choosing — the open session may have been renamed
            meta = storage.read_session_meta(app.session_path) or {}
            app._set_header_title(meta.get("title", ""))
            app._has_title = bool(meta.get("title"))

    # --- the replay ----------------------------------------------------------

    async def render_history(self) -> None:
        """Clear the chat and remount the session's messages, matching the live
        rendering exactly — turn rails, titled tool cards, diffs, todo_write into
        the pinned panel — so a reloaded session looks like the turn that made it.

        Also the ONE owner of the pinned plan: cleared here and refilled from the
        history, so the panel is always a function of the open session. Every
        session switch comes through here, which is what stops a previous plan
        lingering behind a hidden panel.
        """
        app = self.app
        container = app.query_one("#chat-container", VerticalScroll)
        await container.remove_children()
        todo = app.query_one(TodoPanel)
        todo.clear()
        call_args: dict[str, dict] = {}   # tool_call_id -> parsed arguments
        call_names: dict[str, str] = {}   # tool_call_id -> tool name
        turn = None
        for msg in app.session.messages:
            role = msg["role"]
            content = msg.get("content") or ""
            if role == "user":
                turn = None  # a user message closes the previous assistant turn
                await container.mount(Chatbox(content, role="user"))
                continue
            if turn is None:  # assistant / tool -> one rail
                turn = Vertical(classes="turn")
                await container.mount(turn)
            if role == "assistant":
                await self._replay_assistant(msg, turn, todo, call_args, call_names)
            elif role == "tool":
                await self._replay_tool(msg, content, turn, call_args, call_names)
        # A turn whose every block went somewhere else (a lone todo_write → the panel)
        # would leave a bare green rail behind; drop those, as the live path does.
        for rail in list(container.query(".turn")):
            if not rail.children:
                await rail.remove()
        container.scroll_end(animate=False)
        # After the scroll to the end: a restored gate scrolls to itself, and that
        # has to be the last word — it is the thing the session is waiting on.
        await app.plan.restore(container, call_args, call_names)

    async def _replay_assistant(self, msg, turn, todo, call_args, call_names) -> None:
        if msg.get("content"):  # the model's text answer (tool calls become cards)
            await turn.mount(Chatbox(msg["content"], role="assistant", markdown=True))
        for c in msg.get("tool_calls") or []:
            cid, name = c["id"], c["function"]["name"]
            call_names[cid] = name
            try:
                call_args[cid] = json.loads(c["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                call_args[cid] = {}
            if name == "edit":  # its result is skipped in _replay_tool
                await turn.mount(edit_card(call_args[cid]))
            elif name == "todo_write":  # to the pinned panel, as when live
                todo.update_todos(call_args[cid].get("items", []))
            elif name == "plan_submit":  # the submitted plan IS the checklist
                todo.update_todos(self.app.plan.items(call_args[cid]))

    async def _replay_tool(self, msg, content, turn, call_args, call_names) -> None:
        cid = msg.get("tool_call_id")
        name = call_names.get(cid, "tool")
        if name in ("edit", "todo_write"):
            return  # already shown as the diff card / the pinned panel
        is_error = False
        if name == "plan_submit":
            if not self.app.plan.is_rejection(content):
                return  # success shows as the plan panel, not a card
            is_error = True  # a refusal: show the reason the model has to answer
        summary = tool_summary(name, call_args.get(cid, {}))
        await turn.mount(ToolResultBlock(name, content, is_error, summary=summary))
