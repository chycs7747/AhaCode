"""PlanGate: the approve-before-executing card.

When the model lays out a multi-step plan in act mode, the harness stops the loop
before any of it runs and mounts this card into the turn. Two buttons, two paths:

- ▶ 실행  — hand the plan to the structural runner (`/run`): each step in its own
  fresh sub-agent. This is the reason the gate exists — the split into phases is a
  decision the harness makes, and it needs the user's go-ahead first.
- 계속    — resume the ordinary agent loop, which carries on in one session.

The card answers once: `settle()` freezes it into a record of what was chosen, so a
scrolled-back turn cannot be re-triggered.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static


class PlanGate(Vertical):
    """A plan awaiting the user's go-ahead. Buttons bubble up to the app by id."""

    def __init__(self, steps: list[str]) -> None:
        super().__init__()
        self.steps = list(steps)
        self.add_class("plan-gate")

    def compose(self) -> ComposeResult:
        # markup=False: step text comes from the model and may contain '[' (paths,
        # code), which console markup would try to parse.
        yield Static(
            f"📋 {len(self.steps)}단계 계획이 준비됐어요 — 실행할까요?",
            classes="plan-gate-title",
            markup=False,
        )
        yield Static(
            "\n".join(f"  {i}. {s}" for i, s in enumerate(self.steps, 1)),
            classes="plan-gate-steps",
            markup=False,
        )
        with Horizontal(classes="plan-gate-buttons"):
            yield Button("▶ 실행 (단계별 서브에이전트)", variant="success", id="plan-gate-run")
            yield Button("계속 (한 세션에서)", variant="default", id="plan-gate-continue")

    def settle(self, choice: str) -> None:
        """Replace the buttons with what was chosen, so the card reads as history."""
        self.query(".plan-gate-buttons").remove()
        self.query_one(".plan-gate-title", Static).update(
            f"📋 {len(self.steps)}단계 계획 · {choice}"
        )
        self.add_class("plan-gate--settled")
