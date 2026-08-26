"""PlanGate: the approve-before-executing card.

The model ends a planning turn by calling plan_submit; the harness writes the plan
file and mounts this card into the turn. The loop is held (agent.run's
should_pause) until one of two buttons answers:

- ▶ 실행  — carry the plan out (`/run` is the keyboard path to the same thing).
- ✎ 수정  — keep planning: the card settles and the user types what to change;
           the next turn revises the plan and submits again.

The card answers once: `settle()` freezes it into a record of what was chosen, so a
scrolled-back turn cannot be re-triggered.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static


class PlanGate(Vertical):
    """A submitted plan awaiting the user's go-ahead. Buttons bubble up by id."""

    def __init__(self, steps: list[str], summary: str = "", path: str = "") -> None:
        super().__init__()
        self.steps = list(steps)
        self.summary = summary
        self.path = path
        self.add_class("plan-gate")

    def compose(self) -> ComposeResult:
        # markup=False: the text comes from the model and may contain '[' (paths,
        # code), which console markup would try to parse.
        head = f"📋 {self.summary} — " if self.summary else "📋 "
        yield Static(
            f"{head}{len(self.steps)}단계 계획이 준비됐어요 — 실행할까요?",
            classes="plan-gate-title",
            markup=False,
        )
        body = "\n".join(f"  {i}. {s}" for i, s in enumerate(self.steps, 1))
        if self.path:
            body += f"\n  📄 {self.path}"
        body += "\n  승인: ▶ 또는 빈 입력에 Enter · 수정: 바꿀 점을 그냥 입력"
        yield Static(body, classes="plan-gate-steps", markup=False)
        with Horizontal(classes="plan-gate-buttons"):
            yield Button("▶ 실행 (/run)", variant="success", id="plan-gate-run")
            yield Button("✎ 수정 계속", variant="default", id="plan-gate-continue")

    def settle(self, choice: str) -> None:
        """Replace the buttons with what was chosen, so the card reads as history."""
        self.query(".plan-gate-buttons").remove()
        self.query_one(".plan-gate-title", Static).update(
            f"📋 {len(self.steps)}단계 계획 · {choice}"
        )
        self.add_class("plan-gate--settled")
