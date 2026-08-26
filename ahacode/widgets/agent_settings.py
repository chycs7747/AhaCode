"""AgentSettings: a small modal to tune how agents run and how their context is
managed — without editing config.toml (which is awkward once the app is packaged).

Four knobs, all already config fields:
- max parallel: how many gateway requests may run at once (the client semaphore).
  1 serialises sub-agent execution — the model still delegates and results still
  combine, but children run one at a time, so a single GPU is never double-loaded.
- depth: how many generations of sub-agents may nest (0 = delegation off).
- context window: the model's token budget; compaction keys off it (0 = off).
- compact at: the fraction of the window that triggers condensing the oldest
  messages, so a long session never overflows the window.

Save persists via config.save + client.reset (which resizes the semaphore); the
modal returns the chosen values or None on cancel.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Select


class AgentSettingsResult:
    """What the modal hands back on save (plain attributes, no dependency on config)."""

    __slots__ = ("max_parallel", "depth", "context_window", "compact_threshold")

    def __init__(self, max_parallel: int, depth: int, context_window: int, compact_threshold: float):
        self.max_parallel = max_parallel
        self.depth = depth
        self.context_window = context_window
        self.compact_threshold = compact_threshold


# Bounds match the measured knee (≤8 concurrent) and the tree design (depth 0–3).
_PARALLEL = [(f"{n}  ({'직렬' if n == 1 else '병렬 ' + str(n)})", n) for n in range(1, 9)]
_DEPTH = [("0  (위임 끔)", 0), ("1  (기본)", 1), ("2", 2), ("3", 3)]
# Common local context windows; 0 turns compaction off entirely.
_WINDOW = [("0  (압축 끔)", 0), ("8K", 8192), ("16K", 16384), ("32K", 32768),
           ("64K", 65536), ("128K", 131072)]
# The fraction of the window at which the oldest stretch is condensed.
_THRESHOLD = [("70%", 0.7), ("80%", 0.8), ("90%", 0.9), ("95%", 0.95)]


def _nearest(options, value):
    """The listed value closest to `value`, so a config set by hand still lands on
    a real option instead of leaving the Select blank."""
    return min((v for _, v in options), key=lambda v: abs(v - value))


class AgentSettings(ModalScreen[AgentSettingsResult | None]):
    """dismiss(AgentSettingsResult) = save · dismiss(None) = cancel."""

    BINDINGS = [("escape", "cancel", "Close")]

    def __init__(self, max_parallel: int, depth: int, context_window: int, compact_threshold: float) -> None:
        super().__init__()
        self._max_parallel = max_parallel
        self._depth = depth
        self._window = _nearest(_WINDOW, context_window)
        self._threshold = _nearest(_THRESHOLD, compact_threshold)

    def compose(self) -> ComposeResult:
        with Vertical(id="agent-settings-box"):
            yield Label("에이전트 · 컨텍스트 설정", id="agent-settings-title")
            yield Label(
                "병렬 1 = 서브에이전트를 한 번에 하나씩 (단일 GPU 부하 방지). "
                "깊이 0 = 위임 끔. 압축은 컨텍스트가 창의 지정 비율에 닿으면 "
                "오래된 대화를 요약으로 줄여 초과를 막습니다.",
                id="agent-settings-hint",
            )
            yield Label("최대 병렬 (동시 요청)")
            yield Select(_PARALLEL, value=self._max_parallel, allow_blank=False, id="agent-parallel")
            yield Label("서브에이전트 깊이")
            yield Select(_DEPTH, value=self._depth, allow_blank=False, id="agent-depth")
            yield Label("컨텍스트 창 (토큰)")
            yield Select(_WINDOW, value=self._window, allow_blank=False, id="agent-window")
            yield Label("압축 시작 (창 대비)")
            yield Select(_THRESHOLD, value=self._threshold, allow_blank=False, id="agent-threshold")
            with Horizontal(id="agent-settings-buttons"):
                yield Button("저장", variant="success", id="agent-settings-save")
                yield Button("취소", id="agent-settings-cancel")

    @on(Button.Pressed, "#agent-settings-save")
    def _save(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(AgentSettingsResult(
            max_parallel=int(self.query_one("#agent-parallel", Select).value),
            depth=int(self.query_one("#agent-depth", Select).value),
            context_window=int(self.query_one("#agent-window", Select).value),
            compact_threshold=float(self.query_one("#agent-threshold", Select).value),
        ))

    @on(Button.Pressed, "#agent-settings-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
