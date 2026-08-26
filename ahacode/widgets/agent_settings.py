"""AgentSettings: a small modal to set how sub-agents run — without touching
config.toml (which is awkward once the app is packaged).

Two knobs, both already config fields:
- max parallel: how many gateway requests may run at once (the client semaphore).
  1 serialises sub-agent execution — the model still delegates and results still
  combine, but children run one at a time, so a single GPU is never double-loaded.
- depth: how many generations of sub-agents may nest (0 = delegation off).

Save persists via config.save + client.reset (which resizes the semaphore); the
modal returns the chosen (max_parallel, depth) or None on cancel.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Select

# Bounds match the measured knee (≤8 concurrent) and the tree design (depth 0–3).
_PARALLEL = [(f"{n}  ({'직렬' if n == 1 else '병렬 ' + str(n)})", n) for n in range(1, 9)]
_DEPTH = [("0  (위임 끔)", 0), ("1  (기본)", 1), ("2", 2), ("3", 3)]


class AgentSettings(ModalScreen[tuple[int, int] | None]):
    """dismiss((max_parallel, depth)) = save · dismiss(None) = cancel."""

    BINDINGS = [("escape", "cancel", "Close")]

    def __init__(self, max_parallel: int, depth: int) -> None:
        super().__init__()
        self._max_parallel = max_parallel
        self._depth = depth

    def compose(self) -> ComposeResult:
        with Vertical(id="agent-settings-box"):
            yield Label("에이전트 실행 설정", id="agent-settings-title")
            yield Label(
                "병렬 수를 1로 두면 서브에이전트가 한 번에 하나씩 실행돼요 "
                "(단일 GPU 부하 방지). 깊이 0은 위임 자체를 끕니다.",
                id="agent-settings-hint",
            )
            yield Label("최대 병렬 (동시 요청)")
            yield Select(_PARALLEL, value=self._max_parallel, allow_blank=False,
                         id="agent-parallel")
            yield Label("서브에이전트 깊이")
            yield Select(_DEPTH, value=self._depth, allow_blank=False, id="agent-depth")
            with Horizontal(id="agent-settings-buttons"):
                yield Button("저장", variant="success", id="agent-settings-save")
                yield Button("취소", id="agent-settings-cancel")

    @on(Button.Pressed, "#agent-settings-save")
    def _save(self, event: Button.Pressed) -> None:
        event.stop()
        parallel = int(self.query_one("#agent-parallel", Select).value)
        depth = int(self.query_one("#agent-depth", Select).value)
        self.dismiss((parallel, depth))

    @on(Button.Pressed, "#agent-settings-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
