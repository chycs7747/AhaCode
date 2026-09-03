"""The plan gate and what follows it: the handoff to an impl session, its
progress snapshots, and the stall detection that ends an unattended run.

Every step is a UI act, so this holds the app; the App still answers the button
presses and delegates here.
"""

from __future__ import annotations

from textual.containers import VerticalScroll

from ahacode import agent, config, prompts, storage, workspace
from ahacode.tools import plan as plan_tool
from ahacode.widgets.chatbox import Chatbox
from ahacode.widgets.plan_gate import PlanGate
from ahacode.widgets.todo_panel import TodoPanel


class PlanRun:
    """Plan-gate state plus the stall detection that ends an unattended run."""

    def __init__(self, app) -> None:
        self.app = app
        self.pending = False  # the loop pauses between turns while set; main thread only
        self.gate: PlanGate | None = None
        # Stall detection at two granularities; plain ints, a lost reset costs one round.
        self.done_steps = 0  # steps finished as of the last turn
        self.stalled = 0  # turns in a row that finished none
        self.rounds_since_step = 0  # rounds since a step last landed
        self.round_stalled = False  # did the round backstop end the turn in flight?

    # --- the plan as data ----------------------------------------------------

    @staticmethod
    def items(args: dict) -> list[dict]:
        """plan_submit's steps in the shape the pinned panel takes (all pending)."""
        return [{"content": st, "status": plan_tool.PENDING}
                for st in args.get("steps", []) if isinstance(st, str) and st.strip()]

    @staticmethod
    def is_rejection(text: str) -> bool:
        """Whether a plan_submit result is a refusal; success starts with "Plan saved to"."""
        return not text.startswith("Plan saved to")

    @staticmethod
    def finished_steps(panel: TodoPanel) -> int:
        return len(panel.items) - len(panel.unfinished())

    # --- the gate ------------------------------------------------------------

    async def open(self, steps: list[str], summary: str, path: str, container) -> None:
        """Hold the loop and put the submitted plan's decision on screen.

        Args:
            steps: The plan's steps.
            summary: Its one-line goal.
            path: The plan file, as displayed.
            container: The turn rail the gate card mounts into.
        """
        app = self.app
        self.pending = True
        self.gate = PlanGate(steps, summary=summary, path=path)
        await container.mount(self.gate)
        # The loop is blocked on these buttons, so they are not optional content.
        # `container` is the rail, not the scroller; reveal() scrolls the real one.
        app._follow_output = True
        app.call_after_refresh(self.reveal)
        app._set_status("⏸ 계획 승인 대기")

    def reveal(self) -> None:
        """Scroll the gate's buttons on screen after layout has caught up; in the
        mount's own frame the scroll measures the pre-mount height and lands short."""
        self.app.query_one("#chat-container", VerticalScroll).scroll_end(
            animate=False, immediate=True)

    def settle(self, choice: str) -> None:
        """Answer the open gate and release the loop.

        Args:
            choice: The label the settled card shows.
        """
        self.pending = False
        if self.gate is not None and self.gate.is_mounted:
            self.gate.settle(choice)
        self.gate = None

    def reset(self) -> None:
        """Forget the gate entirely; a different session asks about its own plans."""
        self.pending = False
        self.gate = None

    async def restore(self, container, call_args: dict, call_names: dict) -> None:
        """Re-open the gate if the session was left waiting on one.

        That is the case when the session's last entry is a successful plan_submit
        result with no turn after it. An already-run plan reopens too: impl work
        goes to a child session, so approving again makes a sibling.

        Args:
            container: The turn rail the gate card mounts into.
            call_args: tool_call_id -> parsed arguments, from the replay.
            call_names: tool_call_id -> tool name, from the replay.
        """
        app = self.app
        self.reset()
        if app.view_only or app.session_kind == "impl":
            return
        if not app.session.messages:
            return
        last = app.session.messages[-1]
        if last.get("role") != "tool":
            return
        cid = last.get("tool_call_id")
        if call_names.get(cid) != "plan_submit":
            return
        content = last.get("content") or ""
        if self.is_rejection(content):
            return
        if not storage.plan_path(app.session_path).exists():
            return
        args = call_args.get(cid, {})
        await self.open(
            [it["content"] for it in self.items(args)],
            args.get("summary", ""),
            content.split(" (", 1)[0].removeprefix("Plan saved to "),
            container,
        )

    # --- carrying the plan out -----------------------------------------------

    async def start_impl_session(self) -> None:
        """Hand the plan to a child session that carries it out.

        The child is a handoff: same depth, act mode, seeded with one message naming
        the plan file. Approving a revised plan makes a new sibling.
        """
        app = self.app
        if app.view_only:
            await app._say_system("🔒 보기 전용 세션에서는 계획을 실행할 수 없어요. /new 로 시작하세요.")
            return
        plan = storage.plan_path(app.session_path)
        if not plan.exists():
            await app._say_system("실행할 계획이 없어요 — plan 모드에서 계획을 제출하면 파일이 생깁니다.")
            return
        parent_id, depth = app.session_path.stem, app.session_depth
        child = storage.new_session_path()
        storage.write_header(child, storage.make_header(
            child.stem, parent_id=parent_id, kind="impl", relation="handoff", depth=depth,
            model=config.load().name, title=storage.plan_title(plan),
        ))
        await app.sessions.switch(child.stem)
        await app._say_system(
            f"↳ 계획 실행 세션 — {workspace.display_path(plan)} 을 읽고 진행합니다 "
            f"(계획 세션 {parent_id} 의 자식)"
        )
        await self._seed_turn(prompts.handoff_prompt(workspace.display_path(plan)), show=True)

    async def _seed_turn(self, text: str, *, show: bool) -> None:
        """Put one user message into the session and run a turn on it; the handoff
        shows the message, auto-continue does not."""
        app = self.app
        app.session.add_user(text)
        storage.append_message(app.session_path, {"role": "user", "content": text})
        app._follow_output = True
        if show:
            container = app.query_one("#chat-container", VerticalScroll)
            await container.mount(Chatbox(text, role="user"))
        await app._start_turn()

    def max_turns(self) -> int:
        """The turn cap: an impl session carries a whole plan, so it gets the larger one."""
        if self.app.session_kind == "impl":
            return config.load().impl_max_turns
        return agent.DEFAULT_MAX_TURNS

    # --- progress and stalling -----------------------------------------------

    def note_todo_update(self, panel: TodoPanel, items: list[dict]) -> None:
        """Apply the model's checklist, resetting the round counter if a step landed.

        A completed step is the only reset, which makes the counter a stall detector
        rather than a round cap.

        Args:
            panel: The pinned checklist.
            items: The model's full list.
        """
        before = self.finished_steps(panel)
        panel.update_todos(items)
        if self.finished_steps(panel) > before:
            self.rounds_since_step = 0

    def begin_turn(self) -> int:
        """Arm the round backstop for a new turn.

        Returns:
            The round limit in force: the configured stall_rounds in an impl
            session, 0 (no backstop) elsewhere.
        """
        self.rounds_since_step = 0
        self.round_stalled = False
        return config.load().stall_rounds if self.app.session_kind == "impl" else 0

    def should_pause(self, stall_rounds: int) -> bool:
        """Whether the loop should stop between rounds: the gate is waiting on the
        user, or the turn has run `stall_rounds` rounds without completing a step.

        Args:
            stall_rounds: The limit from begin_turn; 0 disables the backstop.

        Returns:
            True to stop cleanly, the way a finished answer does.
        """
        if self.pending:
            return True
        if stall_rounds and self.rounds_since_step >= stall_rounds:
            self.round_stalled = True
            return True
        return False

    async def snapshot_progress(self) -> None:
        """Write plans/{plan}.result.md from the panel and say where the plan stands."""
        app = self.app
        panel = app.query_one(TodoPanel)
        items = list(panel.items)
        if not items:
            return
        left = panel.unfinished()
        parent = app.session_parent_id or app.session_path.stem
        plan = storage.plan_path(workspace.SESSIONS_DIR / f"{parent}.jsonl")
        summary = next(
            (m["content"] for m in reversed(app.session.messages)
             if m.get("role") == "assistant" and m.get("content")), "",
        )
        out = storage.result_path(plan)
        storage.write_result(
            out, plan=plan, session_id=app.session_path.stem,
            steps=[f"{plan_tool.mark(it.get('status'))} {it.get('content', '')}" for it in items],
            done=self.finished_steps(panel), summary=summary,
        )
        if left:
            await app._say_system(
                f"⏸ 미완 항목 {len(left)}개 — 이어서 하려면 입력하세요 "
                f"(다음: {left[0].get('content', '')[:60]}) · 진행 기록 {workspace.display_path(out)}"
            )
        else:
            await app._say_system(
                f"✓ 계획 완료 — {len(items)}단계 모두 처리 · 결과 {workspace.display_path(out)}"
            )

    async def auto_continue(self) -> None:
        """Carry an impl session on to its next turn, or stop after enough turns in
        a row that completed no step; only the checklist measures progress."""
        app = self.app
        cfg = config.load()
        if not cfg.auto_continue_stall:
            return
        if self.pending or getattr(app, "_stopping", False):
            return  # something is already waiting on the user
        panel = app.query_one(TodoPanel)
        left = panel.unfinished()
        if not panel.items or not left:
            return
        done = self.finished_steps(panel)
        # Strictly greater: a turn that completes nothing is a stall, however much
        # text and tool output it produced.
        self.stalled = 0 if done > self.done_steps else self.stalled + 1
        self.done_steps = done
        why = f" (단계 완료 없이 {self.rounds_since_step}라운드)" if self.round_stalled else ""
        nxt = left[0].get("content", "")[:60]
        if self.stalled >= cfg.auto_continue_stall:
            await app._say_system(
                f"⏹ {self.stalled}턴 연속으로 완료된 단계가 없어 자동 진행을 멈췄습니다"
                f"{why} — {done}/{len(panel.items)} 완료. 막힌 곳: {nxt}"
            )
            return
        await app._say_system(
            f"▶ 자동 진행 {done}/{len(panel.items)} 완료{why} · 다음: {nxt} (Esc 로 중지)"
        )
        await self._seed_turn(prompts.CONTINUE_PROMPT, show=False)
