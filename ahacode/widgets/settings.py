"""The settings modal: every config.toml field worth changing, on four tabs down
the left edge (연결, 에이전트, 컨텍스트, 사고).

The 연결 tab asks the address in the box for its /v1/models, so the endpoint and
the model name cannot drift apart. Every pane stays mounted, so a save reads all
four. The modal returns the edited ModelConfig, or None on cancel.
"""

from __future__ import annotations

from dataclasses import replace

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, ContentSwitcher, Input, Label, Select

from ahacode import client, config


TABS = [("connection", "연결"), ("agent", "에이전트"),
        ("context", "컨텍스트"), ("thinking", "사고")]

# The options each Select offers. A config value set by hand lands on the nearest.
_TIMEOUT = [("30s", 30.0), ("60s", 60.0), ("120s", 120.0), ("300s", 300.0),
            ("600s", 600.0), ("900s", 900.0)]
_PARALLEL = [(f"{n}  ({'직렬' if n == 1 else '병렬 ' + str(n)})", n) for n in range(1, 9)]
_DEPTH = [("0  (위임 끔)", 0), ("1  (기본)", 1), ("2", 2), ("3", 3)]
_IMPL_TURNS = [("0  (무제한)", 0), ("10", 10), ("20", 20), ("30  (기본)", 30),
               ("50", 50), ("100", 100)]
_STALL = [("0  (자동 진행 끔)", 0), ("2", 2), ("3  (기본)", 3), ("5", 5), ("10", 10)]
_STALL_ROUNDS = [("0  (끔)", 0), ("20", 20), ("40  (기본)", 40), ("80", 80), ("150", 150)]
_WINDOW = [("0  (압축 끔)", 0), ("8K", 8192), ("16K", 16384), ("32K", 32768),
           ("64K", 65536), ("128K", 131072), ("192K", 196608), ("256K", 262144)]
_THRESHOLD = [("70%", 0.7), ("80%", 0.8), ("90%", 0.9), ("95%", 0.95)]
_KEEP_RECENT = [("2", 2), ("4", 4), ("6  (기본)", 6), ("8", 8), ("12", 12)]
# A per-mode budget of None means "follow the global one"; a Select needs a real
# value for it, so -1 stands in and is mapped back on save.
_GLOBAL = -1
_THINK = [("전역 따름", _GLOBAL), ("1K", 1024), ("2K", 2048), ("4K", 4096),
          ("8K", 8192), ("16K", 16384)]
_THINK_GLOBAL = [("0  (무제한)", 0), ("1K", 1024), ("2K", 2048), ("4K", 4096),
                 ("8K", 8192), ("16K", 16384)]
_EFFORT = [("low", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "xhigh")]
# The stored flag is no_think_after_tools, the inverse of what the label says.
_AFTER_TOOLS = [("켬 (사고함)", False), ("끔 (사고 안 함)", True)]


def _nearest(options, value):
    """The listed value closest to `value`, so a hand-set config lands on a real option."""
    return min((v for _, v in options), key=lambda v: abs(v - value))


def _think_value(override):
    """A per-mode budget (int | None) as its Select value."""
    return _GLOBAL if override is None else _nearest([o for o in _THINK if o[1] != _GLOBAL], override)


class Settings(ModalScreen["config.ModelConfig | None"]):
    """dismiss(ModelConfig) saves, dismiss(None) cancels."""

    BINDINGS = [("escape", "cancel", "Close")]

    def __init__(self, cfg) -> None:
        super().__init__()
        self._cfg = cfg
        self._models = [cfg.name]  # until the server reports, the configured name is all we know

    # --- layout ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        cfg = self._cfg
        with Horizontal(id="settings-box"):
            with Vertical(id="settings-rail"):
                yield Label("Settings", id="settings-title")
                for key, label in TABS:
                    yield Button(label, id=f"settings-tab-{key}",
                                 classes="settings-tab" + (" -active" if key == "connection" else ""))
            with Vertical(id="settings-right"):
                with ContentSwitcher(initial="settings-pane-connection", id="settings-panes"):
                    yield from self._connection_pane(cfg)
                    yield from self._agent_pane(cfg)
                    yield from self._context_pane(cfg)
                    yield from self._thinking_pane(cfg)
                with Horizontal(id="settings-buttons"):
                    yield Button("저장", variant="success", id="settings-save")
                    yield Button("취소", id="settings-cancel")

    def _connection_pane(self, cfg) -> ComposeResult:
        with VerticalScroll(id="settings-pane-connection"):
            yield Label(
                "모델 목록은 아래 주소에서 직접 불러옵니다 — 목록 조회는 GET이라 "
                "모델을 올리지 않습니다. 고른 모델도 다음 메시지에서야 서버에 전달됩니다.",
                classes="settings-hint")
            yield Label("게이트웨이 주소 (base_url)")
            yield Input(value=cfg.base_url, placeholder=config.DEFAULTS.base_url,
                        id="settings-base-url")
            yield Label("API 키")
            yield Input(value=cfg.api_key, placeholder=config.DEFAULTS.api_key,
                        id="settings-api-key")
            with Horizontal(id="settings-fetch-row"):
                yield Button("모델 불러오기", id="settings-fetch-models")
                yield Label("", id="settings-fetch-status")
            yield Label("모델")
            yield Select([(m, m) for m in self._models], value=cfg.name,
                         allow_blank=False, id="settings-model")
            yield Label("요청 타임아웃")
            yield Select(_TIMEOUT, value=_nearest(_TIMEOUT, cfg.timeout),
                         allow_blank=False, id="settings-timeout")

    def _agent_pane(self, cfg) -> ComposeResult:
        with VerticalScroll(id="settings-pane-agent"):
            yield Label(
                "병렬 1 = 서브에이전트를 한 번에 하나씩 (단일 GPU 부하 방지). "
                "깊이 0 = 위임 끔.", classes="settings-hint")
            yield Label("최대 병렬 (동시 요청)")
            yield Select(_PARALLEL, value=cfg.max_parallel_agents,
                         allow_blank=False, id="settings-parallel")
            yield Label("서브에이전트 깊이")
            yield Select(_DEPTH, value=cfg.subagent_depth,
                         allow_blank=False, id="settings-depth")
            yield Label("계획 실행 세션 턴 상한")
            yield Select(_IMPL_TURNS, value=_nearest(_IMPL_TURNS, cfg.impl_max_turns),
                         allow_blank=False, id="settings-impl-turns")
            yield Label("자동 진행 중단 (완료 없는 턴)")
            yield Select(_STALL, value=_nearest(_STALL, cfg.auto_continue_stall),
                         allow_blank=False, id="settings-stall")
            yield Label("자동 진행 중단 (완료 없는 라운드)")
            yield Select(_STALL_ROUNDS,
                         value=_nearest(_STALL_ROUNDS, cfg.stall_rounds),
                         allow_blank=False, id="settings-stall-rounds")
            yield Label(
                "계획 실행 세션은 미완 단계가 남아 있으면 스스로 다음 턴을 시작합니다. "
                "단계가 하나도 완료되지 않은 턴이 이만큼 연속되면 멈추고 알려줍니다. "
                "라운드 쪽은 한 턴 안에서 같은 판단을 하며, 턴 상한을 무제한으로 두었을 때 "
                "실행을 끝낼 수 있는 유일한 장치입니다. 단계가 하나라도 완료되면 둘 다 "
                "초기화되므로, 느리지만 진행 중인 작업은 잘리지 않습니다.",
                classes="settings-hint")

    def _context_pane(self, cfg) -> ComposeResult:
        with VerticalScroll(id="settings-pane-context"):
            yield Label(
                "컨텍스트가 창의 비율에 닿으면 오래된 대화를 요약으로 줄입니다. "
                "최근 N개는 항상 원문으로 남습니다.", classes="settings-hint")
            yield Label("컨텍스트 창 (토큰)")
            yield Select(_WINDOW, value=_nearest(_WINDOW, cfg.context_window),
                         allow_blank=False, id="settings-window")
            yield Label("압축 시작 (창 대비)")
            yield Select(_THRESHOLD, value=_nearest(_THRESHOLD, cfg.compact_threshold),
                         allow_blank=False, id="settings-threshold")
            yield Label("원문 유지 개수")
            yield Select(_KEEP_RECENT, value=_nearest(_KEEP_RECENT, cfg.keep_recent_messages),
                         allow_blank=False, id="settings-keep-recent")

    def _thinking_pane(self, cfg) -> ComposeResult:
        with VerticalScroll(id="settings-pane-thinking"):
            yield Label(
                "한 턴이 추론에 쓸 수 있는 토큰 상한입니다. 모드별 값을 '전역 따름'으로 "
                "두면 아래 세 모드가 맨 위의 전역 예산을 그대로 씁니다. plan은 깊게, "
                "impl과 서브에이전트는 얕게 두는 것이 보통입니다.", classes="settings-hint")
            yield Label("전역 사고 예산 (모드별 값이 없을 때 쓰임)")
            yield Select(_THINK_GLOBAL, value=_nearest(_THINK_GLOBAL, cfg.thinking_token_budget),
                         allow_blank=False, id="settings-think-global")
            yield Label("reasoning effort (힌트)")
            yield Select(_EFFORT, value=cfg.reasoning_effort if any(
                cfg.reasoning_effort == v for _, v in _EFFORT) else config.DEFAULTS.reasoning_effort,
                allow_blank=False, id="settings-effort")
            yield Label("사고 예산 · plan (깊게)")
            yield Select(_THINK, value=_think_value(cfg.plan_thinking_budget),
                         allow_blank=False, id="settings-think-plan")
            yield Label("사고 예산 · impl (실행)")
            yield Select(_THINK, value=_think_value(cfg.impl_thinking_budget),
                         allow_blank=False, id="settings-think-impl")
            yield Label("사고 예산 · subagent")
            yield Select(_THINK, value=_think_value(cfg.subagent_thinking_budget),
                         allow_blank=False, id="settings-think-subagent")
            yield Label("도구 결과 후 사고")
            yield Select(_AFTER_TOOLS, value=cfg.no_think_after_tools,
                         allow_blank=False, id="settings-after-tools")

    # --- tabs ------------------------------------------------------------

    @on(Button.Pressed, ".settings-tab")
    def _switch_tab(self, event: Button.Pressed) -> None:
        event.stop()
        key = str(event.button.id).removeprefix("settings-tab-")
        self.query_one("#settings-panes", ContentSwitcher).current = f"settings-pane-{key}"
        for button in self.query(".settings-tab").results(Button):
            button.set_class(button is event.button, "-active")

    # --- model list ------------------------------------------------------

    @on(Button.Pressed, "#settings-fetch-models")
    def _fetch(self, event: Button.Pressed) -> None:
        event.stop()
        self.query_one("#settings-fetch-status", Label).update("불러오는 중…")
        self._load_models(self.query_one("#settings-base-url", Input).value.strip(),
                          self.query_one("#settings-api-key", Input).value.strip())

    @work(exclusive=True, thread=True)
    def _load_models(self, base_url: str, api_key: str) -> None:
        """Ask the typed endpoint for its models off the UI thread."""
        try:
            names = client.list_models(base_url=base_url, api_key=api_key)
            self.app.call_from_thread(self._models_arrived, names, None)
        except Exception as exc:
            self.app.call_from_thread(self._models_arrived, [], f"{type(exc).__name__}")

    def _models_arrived(self, names: list[str], error: str | None) -> None:
        status = self.query_one("#settings-fetch-status", Label)
        if error or not names:
            status.update(f"실패: {error}" if error else "모델이 없습니다")
            return
        select = self.query_one("#settings-model", Select)
        keep = select.value  # survives the reload when the server still offers it
        self._models = names
        select.set_options([(n, n) for n in names])
        select.value = keep if keep in names else names[0]
        status.update(f"{len(names)}개")

    # --- save / cancel ---------------------------------------------------

    def _value(self, suffix: str):
        return self.query_one(f"#settings-{suffix}", Select).value

    def _think(self, which: str) -> int | None:
        v = int(self._value(f"think-{which}"))
        return None if v == _GLOBAL else v

    @on(Button.Pressed, "#settings-save")
    def _save(self, event: Button.Pressed) -> None:
        event.stop()
        # replace(): the fields this modal does not own carry through untouched.
        self.dismiss(replace(
            self._cfg,
            base_url=self.query_one("#settings-base-url", Input).value.strip() or self._cfg.base_url,
            api_key=self.query_one("#settings-api-key", Input).value.strip() or self._cfg.api_key,
            name=str(self._value("model")),
            timeout=float(self._value("timeout")),
            max_parallel_agents=int(self._value("parallel")),
            subagent_depth=int(self._value("depth")),
            impl_max_turns=int(self._value("impl-turns")),
            auto_continue_stall=int(self._value("stall")),
            stall_rounds=int(self._value("stall-rounds")),
            context_window=int(self._value("window")),
            compact_threshold=float(self._value("threshold")),
            keep_recent_messages=int(self._value("keep-recent")),
            thinking_token_budget=int(self._value("think-global")),
            reasoning_effort=str(self._value("effort")),
            plan_thinking_budget=self._think("plan"),
            impl_thinking_budget=self._think("impl"),
            subagent_thinking_budget=self._think("subagent"),
            no_think_after_tools=bool(self._value("after-tools")),
        ))

    @on(Button.Pressed, "#settings-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
