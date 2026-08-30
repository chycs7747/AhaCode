"""The plan gate, and what happens after it is answered.

One collaborator owns the whole arc a plan travels: submitted → held on screen
→ approved into a child session that carries it out → watched for progress →
stopped when it stalls. It was spread across nine App methods and six pieces of
App state, which is why "when does a run stop?" had no single place to read.

It holds the app rather than being held at arm's length, because every step of
that arc is a UI act: mount a card, switch sessions, start a turn. What it does
NOT do is answer button presses — Textual dispatches those to the App, which
delegates here.
"""

from __future__ import annotations

from textual.containers import VerticalScroll

from ahacode import agent, config, prompts, storage
from ahacode.tools import plan as plan_tool
from ahacode.widgets.chatbox import Chatbox
from ahacode.widgets.plan_gate import PlanGate
from ahacode.widgets.todo_panel import TodoPanel


class PlanRun:
    """Plan-gate state plus the stall detection that ends an unattended run."""

    def __init__(self, app) -> None:
        self.app = app
        # The loop pauses between turns while `pending` is set (agent.run's
        # should_pause reads it). Written only on the main thread, so a plain
        # bool is enough.
        self.pending = False
        self.gate: PlanGate | None = None
        # Stall detection at two granularities. Both are plain ints: a lost reset
        # costs one extra round, which is not worth a lock.
        self.done_steps = 0      # steps finished as of the last turn
        self.stalled = 0         # turns in a row that finished none
        self.rounds_since_step = 0   # rounds since a step last landed
        self.round_stalled = False   # did the round backstop end the turn in flight?

    # --- the plan as data ----------------------------------------------------

    @staticmethod
    def items(args: dict) -> list[dict]:
        """plan_submit's steps in the shape the pinned panel takes (all pending)."""
        return [{"content": st, "status": plan_tool.PENDING}
                for st in args.get("steps", []) if isinstance(st, str) and st.strip()]

    @staticmethod
    def is_rejection(text: str) -> bool:
        """A plan_submit result that came back refused (the model must resubmit).
        Success starts with "Plan saved to …"; a rejection is the reason itself."""
        return not text.startswith("Plan saved to")

    @staticmethod
    def finished_steps(panel: TodoPanel) -> int:
        return len(panel.items) - len(panel.unfinished())

    # --- the gate ------------------------------------------------------------

    async def open(self, steps: list[str], summary: str, path: str, container) -> None:
        """The model submitted a plan: hold the loop and put the decision on screen.

        Opened only from the MAIN loop's plan_submit result (a sub-agent is never
        offered the tool — it has no user to ask, and its parent is blocked on it).
        No heuristics: the model said it is done, so this is the moment.
        """
        app = self.app
        self.pending = True  # stops the loop between turns
        self.gate = PlanGate(steps, summary=summary, path=path)
        await container.mount(self.gate)
        # Override follow-output: the loop is BLOCKED on these buttons, so they are
        # not optional content. NOTE `container` is the turn rail, a plain Vertical —
        # scrolling that does nothing; the scroller is #chat-container, one level up.
        app._follow_output = True
        app.call_after_refresh(self.reveal)
        app._status("⏸ 계획 승인 대기")

    def reveal(self) -> None:
        """Bring the gate's buttons on screen, after layout has caught up.

        Scrolling in the mount's own frame measures the pre-mount height, and the
        pinned panel reflows in that same frame — so on a short terminal the scroll
        lands short and the loop waits on buttons below the fold.
        """
        self.app.query_one("#chat-container", VerticalScroll).scroll_end(
            animate=False, immediate=True)

    def settle(self, choice: str) -> None:
        """Answer the open gate (whatever the user chose) and release the loop."""
        self.pending = False
        if self.gate is not None and self.gate.is_mounted:
            self.gate.settle(choice)
        self.gate = None

    def reset(self) -> None:
        """Forget the gate entirely — a different session asks about its own plans."""
        self.pending = False
        self.gate = None

    async def restore(self, container, call_args: dict, call_names: dict) -> None:
        """Re-open the gate if this session was left waiting on one.

        The gate is runtime state (one widget, one bool), so quitting with a plan on
        screen stranded it: the checklist and the plan file both came back, but
        nothing on screen could run them. The transcript says when that happened —
        the session's last entry is a SUCCESSFUL plan_submit result with no turn
        after it. Anything else leaves the gate closed.

        An already-run plan reopens too, deliberately: impl work goes to a CHILD
        session, so approving again makes a sibling, never a deeper child.
        """
        app = self.app
        self.reset()
        if app.view_only or app.session_kind == "impl":
            return  # an impl session carries a plan out; it never approves one
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
            return  # the plan file was deleted; there is nothing left to run
        args = call_args.get(cid, {})
        await self.open(
            [it["content"] for it in self.items(args)],
            args.get("summary", ""),
            content.split(" (", 1)[0].removeprefix("Plan saved to "),
            container,
        )

    # --- carrying the plan out -----------------------------------------------

    async def start_impl_session(self) -> None:
        """▶ on the gate: hand the plan to a child session that carries it out.

        The child is a HANDOFF — same depth (so it keeps every tool, task
        included), act mode, seeded with one message naming the plan file. One
        continuous context does the whole plan, so a stop leaves a transcript the
        same model picks up. Approving a revised plan makes a new sibling rather
        than appending to an old child that was written for a different plan.
        """
        app = self.app
        if app.view_only:  # a sub-agent transcript can't drive new work
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
        await app._switch_session(child.stem)  # empty child; also flips the bar to act
        await app._say_system(
            f"↳ 계획 실행 세션 — {storage.display_path(plan)} 을 읽고 진행합니다 "
            f"(계획 세션 {parent_id} 의 자식)"
        )
        await self._seed_turn(
            prompts.handoff_prompt(storage.display_path(plan)),
            question=f"(계획 실행 시작) {storage.display_path(plan)}",
            show=True,
        )

    async def _seed_turn(self, text: str, *, question: str, show: bool) -> None:
        """Put one user message into the session and run a turn on it.

        Shared by the handoff (which shows the message, since the user asked for
        it) and auto-continue (which does not — the notice above it already says
        what is happening, and the prompt itself is boilerplate).
        """
        app = self.app
        app.session.add_user(text)
        storage.append_message(app.session_path, {"role": "user", "content": text})
        app._turn_question = question
        app._follow_output = True
        if show:
            container = app.query_one("#chat-container", VerticalScroll)
            await container.mount(Chatbox(text, role="user"))
        await app._start_turn()

    def max_turns(self) -> int:
        """An impl session carries a whole plan in one context, so it gets the
        larger cap; an ordinary turn answers one message."""
        if self.app.session_kind == "impl":
            return config.load().impl_max_turns
        return agent.DEFAULT_MAX_TURNS

    # --- progress and stalling -----------------------------------------------

    def note_todo_update(self, panel: TodoPanel, items: list[dict]) -> None:
        """Apply the model's checklist, and reset the round counter if a step landed.

        A completed step is the ONLY reset, which is what makes the counter a stall
        detector and not a round cap renamed: a run that keeps finishing steps never
        trips it, however many rounds each step takes.
        """
        before = self.finished_steps(panel)
        panel.update_todos(items)
        if self.finished_steps(panel) > before:
            self.rounds_since_step = 0

    def begin_turn(self) -> int:
        """Arm the round backstop for a new turn; returns the limit in force.

        Read once per turn rather than per round: config.load() reads a file, and
        the limit cannot usefully change mid-turn. Zero outside an impl session —
        an ordinary turn answers one message and needs no backstop.
        """
        self.rounds_since_step = 0
        self.round_stalled = False
        return config.load().stall_rounds if self.app.session_kind == "impl" else 0

    def should_pause(self, stall_rounds: int) -> bool:
        """Stop the loop BETWEEN rounds — cleanly, the way a finished answer does,
        so everything produced so far is kept and the caller decides what next.

        Two reasons: the gate is waiting on the user, or the turn has run
        `stall_rounds` rounds without completing a step. Without the second, "no
        turn cap" means no backstop at all — auto_continue can only judge a turn
        that ENDED, and an uncapped turn that never stops calling tools never gives
        it the chance.
        """
        if self.pending:
            return True
        if stall_rounds and self.rounds_since_step >= stall_rounds:
            self.round_stalled = True
            return True
        return False

    async def snapshot_progress(self) -> None:
        """Write plans/{plan}.result.md from the panel and announce the state."""
        app = self.app
        panel = app.query_one(TodoPanel)
        items = list(panel.items)
        if not items:
            return  # nothing declared yet — the model has not mirrored the plan
        left = panel.unfinished()
        parent = app.session_parent_id or app.session_path.stem
        plan = storage.plan_path(storage.SESSIONS_DIR / f"{parent}.jsonl")
        summary = next(
            (m["content"] for m in reversed(app.session.messages)
             if m.get("role") == "assistant" and m.get("content")), "",
        )
        out = storage.result_path(plan)
        storage.write_result(
            out, plan=plan, session_id=app.session_path.stem, items=items,
            summary=summary, complete=not left,
            throughput=storage.summarize_stats(storage.read_stats(app.session_path)),
        )
        if left:
            await app._say_system(
                f"⏸ 미완 항목 {len(left)}개 — 이어서 하려면 입력하세요 "
                f"(다음: {left[0].get('content', '')[:60]}) · 진행 기록 {storage.display_path(out)}"
            )
        else:
            await app._say_system(
                f"✓ 계획 완료 — {len(items)}단계 모두 처리 · 결과 {storage.display_path(out)}"
            )

    async def auto_continue(self) -> None:
        """Carry an impl session on to its next turn without being asked.

        What ends a run is a stall, not a count. Rounds and turns measure how hard
        the model is working; only the checklist measures whether the work is
        getting anywhere. So steps still completing means carry on however long it
        takes, and N turns in a row completing nothing means stop and say where.
        """
        app = self.app
        cfg = config.load()
        if not cfg.auto_continue_stall:
            return  # switched off: the old ask-every-turn behaviour
        if self.pending or getattr(app, "_stopping", False):
            return  # something is already waiting on the user; do not talk over it
        panel = app.query_one(TodoPanel)
        left = panel.unfinished()
        if not panel.items or not left:
            return  # no checklist to judge progress by, or the plan is finished
        done = self.finished_steps(panel)
        # Strictly greater: a turn that completes nothing is a stall even if it
        # produced text, tool calls, and a confident summary — all of which the
        # stalling turns did produce.
        self.stalled = 0 if done > self.done_steps else self.stalled + 1
        self.done_steps = done
        # Naming the round backstop matters: "the turn ended" and "the turn was cut
        # off after 40 rounds of getting nowhere" look identical from the outside,
        # and they call for opposite responses from whoever reads this in the morning.
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
        await self._seed_turn(
            prompts.continue_prompt(), question=f"(자동 진행) 다음: {nxt}", show=False,
        )
