"""The worker side of a turn: the agent loop, the approval handshake, and
sub-agents. Everything here runs on a worker thread and reaches the UI only
through call_from_thread."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from ahacode import agent, client, config, permissions, prompts, storage, subagent, tools
from ahacode.events import TextDelta, ThinkingDelta, Usage
from ahacode.turn_view import TurnBoxes
from ahacode.widgets.approval_modal import ApprovalModal
from ahacode.widgets.subagent_card import SubagentCard


@dataclass
class TurnStats:
    """What one turn cost, accumulated as it streams."""

    prompt: int = 0
    completion: int = 0
    last_prompt: int | None = None  # the last request's size: what the next turn must fit under
    t_start: float = field(default_factory=time.monotonic)
    t_first: float | None = None

    def record_usage(self, event: Usage) -> None:
        self.prompt += event.prompt_tokens
        self.completion += event.completion_tokens
        self.last_prompt = event.prompt_tokens

    def note_first_token(self) -> None:
        if self.t_first is None:
            self.t_first = time.monotonic()

    @property
    def gen_seconds(self) -> float:
        """Seconds spent generating, measured from the first token."""
        return max(time.monotonic() - (self.t_first or self.t_start), 1e-9)

    @property
    def ttft(self) -> float:
        return (self.t_first - self.t_start) if self.t_first else 0.0

    def line(self) -> str:
        """The status-bar summary; "" when nothing was generated."""
        if not self.completion:
            return ""
        return (f"prompt {self.prompt} · gen {self.completion} · "
                f"{self.completion / self.gen_seconds:.0f} tok/s · ttft {self.ttft:.1f}s")


class TurnRunner:
    """Runs the loop, approves its tools, spawns its sub-agents, and reports back
    to the main thread by message."""

    def __init__(self, app) -> None:
        self.app = app

    # --- approval ------------------------------------------------------------

    def approve_tool(self, call) -> bool:
        """The approval handshake for one tool call: a rule, then auto-approve, then
        a modal on the main thread that this worker thread blocks on.

        Parallel sub-agents queue on the lock, since only one dialog fits on screen.

        Args:
            call: The ToolCall awaiting approval.

        Returns:
            Whether the call may run.
        """
        app = self.app
        # A rule skips the question, never the safety gate: _gate_tool ran the denylist first.
        if permissions.allowed(call.name, call.arguments):
            return True
        if app.auto_approve:
            return True
        if getattr(app, "_stopping", False):  # one stop answers every queued dialog
            return False
        with app._approval_lock:
            if getattr(app, "_stopping", False):
                return False
            answered = threading.Event()
            verdict: dict[str, bool] = {}

            def ask() -> None:
                def on_dismiss(approved: bool | None) -> None:
                    verdict["ok"] = bool(approved)
                    answered.set()

                app.push_screen(ApprovalModal(call.name, call.arguments), on_dismiss)

            app.call_from_thread(ask)
            answered.wait()
            return verdict.get("ok", False)

    # --- sub-agents ----------------------------------------------------------

    def subagent_ctx(self, parent_path, parent_depth, container, worker, approve):
        """Build the AgentContext whose run_subagent spawns a child into a nested
        card and runs it to completion.

        A per-level factory: each child parents to this level and mounts inside this
        container, so the tree nests at any depth and stops itself when a child at
        the depth limit gets no task tool.

        Args:
            parent_path: The session file the child hangs off.
            parent_depth: That session's depth.
            container: Where the child's card mounts.
            worker: The Textual worker, for cancellation.
            approve: The approval handshake the child's tools use.

        Returns:
            The context to hand to agent.run.
        """
        app = self.app

        def run_subagent(prompt: str, description: str) -> str:
            cfg = config.load()
            child_depth = parent_depth + 1
            child_path = storage.new_session_path()
            storage.write_header(child_path, storage.make_header(
                child_path.stem, parent_id=parent_path.stem, kind="subagent",
                relation="delegate",
                depth=child_depth, model=cfg.name, title=(description or prompt)[:40],
            ))
            card = SubagentCard(description or "task", cfg.name)
            app.call_from_thread(container.mount, card)
            child_boxes = TurnBoxes()  # no owns_session: a child touches no session UI

            def child_emit(event) -> None:
                if isinstance(event, Usage):
                    return
                app.call_from_thread(app.turn_view.render, event, child_boxes, card.body)

            result = subagent.run(
                prompt,
                emit=child_emit,
                approve=approve,
                registry=tools.registry_for(child_depth, cfg.subagent_depth),
                ctx=self.subagent_ctx(child_path, child_depth, card.body, worker, approve),
                is_cancelled=lambda: worker.is_cancelled,
            )
            for msg in result.messages:
                storage.append_message(child_path, msg)
            tool_count = sum(1 for m in result.messages if m.get("role") == "tool")
            app.call_from_thread(card.done, tool_count)
            return result.result

        return subagent.AgentContext(run_subagent=run_subagent, session_path=parent_path)

    # --- the turn ------------------------------------------------------------

    def run(self, messages: list[dict], turn, worker) -> None:
        """Run the agent loop and post the outcome back to the main thread.

        Args:
            messages: The history to send, beginning with the system prompt.
            turn: The rail container the reply's blocks mount into.
            worker: The Textual worker, for cancellation.
        """
        app = self.app
        container = turn
        boxes = TurnBoxes(owns_session=True)  # only the main loop drives session-level UI
        stats = TurnStats()

        def emit(event) -> None:
            if isinstance(event, Usage):
                # One usage trailer per model call: also the round counter for the
                # stall backstop.
                app.plan.rounds_since_step += 1
                stats.record_usage(event)
                return
            if isinstance(event, (ThinkingDelta, TextDelta)):
                stats.note_first_token()
            # call_from_thread blocks until the UI has rendered: built-in backpressure.
            app.call_from_thread(app.turn_view.render, event, boxes, container)

        approve = self.approve_tool
        ctx = self.subagent_ctx(
            app.session_path, app.session_depth, container, worker, approve
        )
        stall_rounds = app.plan.begin_turn()
        turn_mode = ("plan" if app.mode == "plan"
                     else "impl" if app.session_kind == "impl" else None)
        try:
            with client.mode(turn_mode):  # picks the thinking budget
                new_messages = agent.run(
                    messages,
                    emit=emit,
                    is_cancelled=lambda: worker.is_cancelled,
                    approve=approve,
                    registry=app._registry_for_mode(),
                    ctx=ctx,
                    max_turns=app.plan.max_turns(),
                    prompt_tokens=app._last_prompt_tokens,
                    should_pause=lambda: app.plan.should_pause(stall_rounds),
                )
        except Exception as exc:
            app.post_message(app.ResponseFailed(f"{type(exc).__name__}: {exc}"[:300]))
            return
        if worker.is_cancelled:
            # Keep every round that finished, so the transcript the model resumes
            # from knows what it already did.
            app.post_message(app.ResponseComplete(new_messages, "■ stopped", stats.last_prompt))
            return
        app.post_message(app.ResponseComplete(new_messages, stats.line(), stats.last_prompt))

    # --- the session title ---------------------------------------------------

    def make_title(self, messages: list[dict], path) -> None:
        """Ask the model for a short session title and record it.

        Args:
            messages: The conversation so far.
            path: The session file to title.
        """
        app = self.app
        convo = "\n".join(
            f"{m['role']}: {m.get('content', '')}"
            for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        )[:1500]
        try:
            title = client.complete([
                {"role": "system", "content": prompts.side.TITLE},
                {"role": "user", "content": convo},
            ])
        except Exception:
            return  # a failed title is not worth surfacing
        title = title.strip().strip('"').strip()[:60]
        if title:
            app.call_from_thread(storage.set_title, path, title)
            if path == app.session_path:  # not if the user already switched away
                app.call_from_thread(app._set_header_title, title)
