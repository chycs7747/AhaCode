"""Driving one turn: the worker body, the approval handshake, and sub-agents.

Everything here runs on a worker thread and reaches the UI only through
call_from_thread (which blocks until the frame is drawn — the backpressure that
keeps a fast stream from outrunning the terminal).

The @work decorators stay on the App, because Textual's decorator needs the
App's run_worker to start the thread. What starts there ends up here.
"""

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
    """What one turn cost, accumulated as it streams.

    Was a five-key dict passed to two module-level formatters. The numbers and
    the two ways they are read now sit together: line() for the status bar,
    metrics() for the session file — the same quantities, kept as numbers so
    they can be summed later rather than re-parsed out of the string.
    """

    prompt: int = 0
    completion: int = 0
    # The LAST request's prompt size is what the next turn has to fit under; the
    # running total above is only for the throughput readout.
    last_prompt: int | None = None
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
        """Seconds spent generating — measured from the first token, not the
        request, so a slow time-to-first-token is not counted as slow output."""
        return max(time.monotonic() - (self.t_first or self.t_start), 1e-9)

    @property
    def ttft(self) -> float:
        return (self.t_first - self.t_start) if self.t_first else 0.0

    def line(self) -> str:
        """One-line token/speed summary for the status bar (empty if no output)."""
        if not self.completion:
            return ""
        return (f"prompt {self.prompt} · gen {self.completion} · "
                f"{self.completion / self.gen_seconds:.0f} tok/s · ttft {self.ttft:.1f}s")

    def metrics(self) -> dict:
        """The same numbers, ready to record (see storage.append_stats)."""
        if not self.completion:
            return {}
        return {
            "prompt": self.prompt,
            "gen": self.completion,
            "gen_seconds": round(self.gen_seconds, 3),
            "ttft": round(self.ttft, 3),
            "model": config.load().name,
        }


class TurnRunner:
    """The worker side of a turn: run the loop, approve its tools, spawn its
    sub-agents, and report back to the main thread by message."""

    def __init__(self, app) -> None:
        self.app = app

    # --- approval ------------------------------------------------------------

    def approve_tool(self, call) -> bool:
        """Approval handshake for one tool call, shared by the agent loop and every
        sub-agent. Auto-approve short-circuits; otherwise push a modal on the main
        thread and block THIS worker thread on an Event until the user answers.
        Parallel sub-agents each need approval at once but only one dialog can be
        on screen, so they queue on the lock."""
        app = self.app
        # Rule-based pre-approval comes FIRST — that is what keeps a parallel fan-out
        # from serialising behind a queue of dialogs (only one fits on screen). It
        # cannot widen anything: _gate_tool ran the denylist before calling us, so a
        # rule skips the question, never the safety gate.
        if permissions.allowed(call.name, call.arguments):
            return True
        if app.auto_approve:  # the denylist already hard-blocked the dangerous ones
            return True
        if getattr(app, "_stopping", False):
            # Answering "stop" once has to be enough, however many sub-agents were
            # mid-flight: whatever is still queued behind the lock must not put
            # another dialog on screen.
            return False
        with app._approval_lock:
            if getattr(app, "_stopping", False):  # stopped while we waited our turn
                return False
            # Cross-thread handshake: the loop runs on a worker thread but the modal
            # lives on the main thread. Push it (non-blocking) via call_from_thread,
            # then block here until the modal's dismiss callback sets the Event.
            answered = threading.Event()
            verdict: dict[str, bool] = {}

            def ask() -> None:
                def on_dismiss(approved: bool | None) -> None:
                    verdict["ok"] = bool(approved)
                    answered.set()

                # Raw arguments → the modal renders a per-tool preview (write → code,
                # edit → diff) inside a scrollable box.
                app.push_screen(ApprovalModal(call.name, call.arguments), on_dismiss)

            app.call_from_thread(ask)
            answered.wait()
            return verdict.get("ok", False)

    # --- sub-agents ----------------------------------------------------------

    def subagent_ctx(self, parent_path, parent_depth, container, worker, approve):
        """Build the AgentContext whose run_subagent spawns a child agent into a
        nested card and runs it to completion.

        run_subagent blocks, but agent.run puts parallelizable `task` calls on a
        thread pool, so two delegations in one turn really do run CONCURRENTLY,
        bounded only by the gateway's gate. A per-level factory: each child parents
        to THIS level and mounts inside this card, so the tree nests at any depth —
        and stops itself, since a child at the depth limit gets no task tool.
        """
        app = self.app

        def run_subagent(prompt: str, description: str) -> str:
            cfg = config.load()
            child_depth = parent_depth + 1
            child_path = storage.new_session_path()
            storage.write_header(child_path, storage.make_header(
                child_path.stem, parent_id=parent_path.stem, kind="subagent",
                relation="delegate",  # a task fan-out — the ⑂ edge in the session tree
                depth=child_depth, model=cfg.name, title=(description or prompt)[:40],
            ))
            card = SubagentCard(description or "task", cfg.name)
            app.call_from_thread(container.mount, card)
            child_boxes = TurnBoxes()  # no owns_session: a child touches no session UI

            def child_emit(event) -> None:
                if isinstance(event, Usage):
                    return  # child token accounting isn't surfaced in this slice
                app.call_from_thread(app.turn_view.render, event, child_boxes, card.body)

            result = subagent.run(
                prompt,
                emit=child_emit,
                approve=approve,  # the child's own bash/write are confirmed too
                registry=tools.registry_for(child_depth, cfg.subagent_depth),
                # a grandchild parents to THIS child, one level deeper, in its card
                ctx=self.subagent_ctx(child_path, child_depth, card.body, worker, approve),
                is_cancelled=lambda: worker.is_cancelled,
            )
            for msg in result.messages:
                storage.append_message(child_path, msg)
            # Fold the card now the child is done (its answer stays one click away).
            tool_count = sum(1 for m in result.messages if m.get("role") == "tool")
            app.call_from_thread(card.done, tool_count)
            return result.result

        return subagent.AgentContext(run_subagent=run_subagent, session_path=parent_path)

    # --- the turn ------------------------------------------------------------

    def run(self, messages: list[dict], turn, worker) -> None:
        """Run the agent loop, rendering its events into `turn`'s rail."""
        app = self.app
        container = turn  # the reply's blocks mount into this turn's rail container
        # This turn's live bubbles. owns_session: only the main loop may drive the
        # status line, the pinned checklist and the plan gate.
        boxes = TurnBoxes(owns_session=True)
        stats = TurnStats()

        def emit(event) -> None:
            if isinstance(event, Usage):  # accounting only — never a bubble
                # One usage trailer per model call, so this is also the round counter
                # the stall backstop needs — counted here rather than off tool calls,
                # which arrive several to a round.
                app.plan.rounds_since_step += 1
                stats.record_usage(event)
                return
            if isinstance(event, (ThinkingDelta, TextDelta)):
                stats.note_first_token()
            # Hop to the main thread to touch widgets. call_from_thread blocks the
            # worker until the UI has rendered — built-in backpressure.
            app.call_from_thread(app.turn_view.render, event, boxes, container)

        approve = self.approve_tool
        ctx = self.subagent_ctx(
            app.session_path, app.session_depth, container, worker, approve
        )
        stall_rounds = app.plan.begin_turn()  # arms the round backstop for this turn
        # The turn's mode picks its thinking budget (config.thinking_budget_for):
        # plan thinks deep, impl shallow, a plain act turn uses the global.
        turn_mode = ("plan" if app.mode == "plan"
                     else "impl" if app.session_kind == "impl" else None)
        try:
            with client.mode(turn_mode):
                new_messages = agent.run(
                    messages,
                    emit=emit,
                    is_cancelled=lambda: worker.is_cancelled,
                    approve=approve,                   # bash and friends are confirmed first
                    registry=app._registry_for_mode(),  # plan mode = read-only subset
                    ctx=ctx,                           # task delegates through this
                    max_turns=app.plan.max_turns(),    # larger for an impl session
                    prompt_tokens=app._last_prompt_tokens,  # measured, for compaction
                    # the plan gate, and the round-level stall backstop
                    should_pause=lambda: app.plan.should_pause(stall_rounds),
                )
        except Exception as exc:
            # Server 500, timeout, connection refused... a bubble, not a crash.
            app.post_message(app.ResponseFailed(f"{type(exc).__name__}: {exc}"[:300]))
            return
        if worker.is_cancelled:
            # Stopped: keep every message that FINISHED (the loop returns whole
            # assistant+tool rounds; the one in flight is dropped), so the transcript
            # the model resumes from knows what it already did. Continue, never restart.
            app.post_message(app.ResponseComplete(new_messages, "■ stopped", stats.last_prompt))
            return
        app.post_message(app.ResponseComplete(
            new_messages, stats.line(), stats.last_prompt, stats.metrics(),
        ))

    # --- the session title ---------------------------------------------------

    def make_title(self, messages: list[dict], path) -> None:
        """Ask the model for a short session title (background, non-streaming)."""
        app = self.app
        convo = "\n".join(
            f"{m['role']}: {m.get('content', '')}"
            for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        )[:1500]
        try:
            title = client.complete([
                {"role": "system", "content": prompts.title_system()},
                {"role": "user", "content": convo},
            ])
        except Exception:
            return  # a failed title is not worth surfacing; leave it untitled
        title = title.strip().strip('"').strip()[:60]
        if title:
            app.call_from_thread(storage.set_title, path, title)
            if path == app.session_path:  # not if the user already switched away
                app.call_from_thread(app._set_header_title, title)
