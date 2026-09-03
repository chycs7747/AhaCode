"""Every way the session on screen is replaced (new, switch, repair), and the
history replay that follows. The App holds the session state; this changes it."""

from __future__ import annotations

import json

from textual.containers import Vertical, VerticalScroll

from ahacode import config, storage, workspace
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

        Args:
            path: The session file.
            meta: Its header, or {} for a fresh main session.
        """
        app = self.app
        app.session_path = path
        app.session_depth = int(meta.get("depth", 0))
        app.session_kind = str(meta.get("kind", "main"))
        app.session_parent_id = meta.get("parent_id")
        app._has_title = bool(meta.get("title"))
        app.plan.reset()
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
        await self.adopt(path, {})
        await app._say_system("new session started")

    async def switch(self, session_id: str) -> None:
        """Load another session by id and show its history.

        Args:
            session_id: The session file's stem.
        """
        app = self.app
        app.session = ChatSession()
        path = workspace.SESSIONS_DIR / f"{session_id}.jsonl"
        app.session.messages = storage.load_messages(path)
        await self.adopt(path, storage.read_session_meta(path) or {})
        if app.session_kind == "impl":
            app._set_mode("act")  # an impl session acts; planning is its parent's
        if app.view_only:
            await app._say_system(
                f"🔒 보기 전용 — 서브에이전트가 자동 생성한 기록(깊이 {app.session_depth})입니다. "
                "읽기만 가능해요. /new 로 새 세션을 시작하세요."
            )
        else:
            await self.repair_interrupted()

    async def repair_interrupted(self) -> None:
        """Fill in the results a turn cut off mid-tool never produced, plus a note
        telling the model to reassess the real state. Persisted, so a reopened
        session finds nothing left to repair."""
        app = self.app
        dangling = storage.dangling_tool_calls(app.session.messages)
        if not dangling:
            return
        for call in dangling:
            subject = tool_summary(call["name"], call["arguments"])  # names which call
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
        """Open the session picker, telling it which session is open and whether
        its turn is running (then it cannot be deleted)."""
        app = self.app
        current = app.session_path.stem
        locked = current if app._anything_running() else None
        app.push_screen(SessionPicker(current=current, locked=locked), self.picked)

    def picked(self, result: str | None) -> None:
        """Act on the picker's choice.

        Args:
            result: "new", a session id, or None when closed without choosing.
        """
        app = self.app
        if result == "new":
            app.run_worker(self.new(), exclusive=False)
        elif result:
            app.run_worker(self.switch(result), exclusive=False)
        else:  # the open session may have been renamed
            meta = storage.read_session_meta(app.session_path) or {}
            app._set_header_title(meta.get("title", ""))
            app._has_title = bool(meta.get("title"))

    # --- the replay ----------------------------------------------------------

    async def render_history(self) -> None:
        """Clear the chat and remount the session's messages the way the live turn
        rendered them: turn rails, titled tool cards, diffs, and the checklist
        into the pinned panel, which is refilled from the history here."""
        app = self.app
        container = app.query_one("#chat-container", VerticalScroll)
        await container.remove_children()
        todo = app.query_one(TodoPanel)
        todo.clear()
        call_args: dict[str, dict] = {}  # tool_call_id -> parsed arguments
        call_names: dict[str, str] = {}  # tool_call_id -> tool name
        turn = None
        for msg in app.session.messages:
            role = msg["role"]
            content = msg.get("content") or ""
            if role == "user":
                turn = None  # a user message closes the previous assistant turn
                await container.mount(Chatbox(content, role="user"))
                continue
            if turn is None:
                turn = Vertical(classes="turn")
                await container.mount(turn)
            if role == "assistant":
                await self._replay_assistant(msg, turn, todo, call_args, call_names)
            elif role == "tool":
                await self._replay_tool(msg, content, turn, call_args, call_names)
        for rail in list(container.query(".turn")):  # a lone todo_write leaves a bare rail
            if not rail.children:
                await rail.remove()
        container.scroll_end(animate=False)
        await app.plan.restore(container, call_args, call_names)  # scrolls to itself last

    async def _replay_assistant(self, msg, turn, todo, call_args, call_names) -> None:
        """Mount an assistant message: its text, and its tool calls' cards or panel updates."""
        if msg.get("content"):
            await turn.mount(Chatbox(msg["content"], role="assistant", markdown=True))
        for c in msg.get("tool_calls") or []:
            cid, name = c["id"], c["function"]["name"]
            call_names[cid] = name
            try:
                call_args[cid] = json.loads(c["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                call_args[cid] = {}
            if name == "edit":
                await turn.mount(edit_card(call_args[cid]))
            elif name == "todo_write":
                todo.update_todos(call_args[cid].get("items", []))
            elif name == "plan_submit":
                todo.update_todos(self.app.plan.items(call_args[cid]))

    async def _replay_tool(self, msg, content, turn, call_args, call_names) -> None:
        """Mount a tool result as a card, unless its call already showed it."""
        cid = msg.get("tool_call_id")
        name = call_names.get(cid, "tool")
        if name in ("edit", "todo_write"):
            return  # shown as the diff card / the pinned panel
        is_error = False
        if name == "plan_submit":
            if not self.app.plan.is_rejection(content):
                return  # shown as the pinned panel
            is_error = True
        summary = tool_summary(name, call_args.get(cid, {}))
        await turn.mount(ToolResultBlock(name, content, is_error, summary=summary))
