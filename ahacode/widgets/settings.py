"""Settings: every config.toml field that is worth changing without opening the
file, grouped into tabs down the left edge.

Four groups, because they answer four different questions:
- 연결   — which server, as what, and which model on it.
- 에이전트 — how many agents run at once and how deep they may nest.
- 컨텍스트 — how much history a request may carry before it is condensed.
- 사고   — how many reasoning tokens each kind of turn may spend.

The tabs are Buttons over a ContentSwitcher rather than TabbedContent, because
TabbedContent puts its tabs on top and the list here is a left rail. Every pane
stays mounted (the switcher toggles display), so a save reads all four whichever
one is showing.

The 연결 tab is the reason this exists. Endpoint, key and model used to be
reachable only through /url and /model, which set them *independently* — and a
model name is only meaningful for the server that serves it. Here 모델 불러오기
asks the address in the box (not the saved one) for its /v1/models and offers only
what came back, so the pair cannot drift apart. Listing is a GET and loads nothing;
picking a model here does not send a request either — the server sees it on the
next message.

Save persists via config.save + client.reset (which rebuilds the client and
resizes the semaphore); the modal returns the chosen values or None on cancel.
"""

from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, ContentSwitcher, Input, Label, Select

from ahacode import client


class SettingsResult:
    """What the modal hands back on save (plain attributes, no dependency on
    config). The three per-mode thinking budgets are int | None — None means
    "follow the global budget"."""

    __slots__ = ("base_url", "api_key", "model", "timeout",
                 "max_parallel", "depth", "impl_max_turns",
                 "context_window", "compact_threshold", "keep_recent",
                 "thinking_budget", "reasoning_effort",
                 "plan_thinking", "impl_thinking", "subagent_thinking",
                 "no_think_after_tools")

    def __init__(self, base_url, api_key, model, timeout,
                 max_parallel, depth, impl_max_turns,
                 context_window, compact_threshold, keep_recent,
                 thinking_budget, reasoning_effort,
                 plan_thinking, impl_thinking, subagent_thinking,
                 no_think_after_tools):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_parallel = max_parallel
        self.depth = depth
        self.impl_max_turns = impl_max_turns
        self.context_window = context_window
        self.compact_threshold = compact_threshold
        self.keep_recent = keep_recent
        self.thinking_budget = thinking_budget
        self.reasoning_effort = reasoning_effort
        self.plan_thinking = plan_thinking
        self.impl_thinking = impl_thinking
        self.subagent_thinking = subagent_thinking
        self.no_think_after_tools = no_think_after_tools


TABS = [("connection", "연결"), ("agent", "에이전트"),
        ("context", "컨텍스트"), ("thinking", "사고")]

# Seconds a request may block between chunks. Long options exist because a cold
# model load on the gateway can take minutes before the first token arrives.
_TIMEOUT = [("30s", 30.0), ("60s", 60.0), ("120s", 120.0), ("300s", 300.0),
            ("600s", 600.0), ("900s", 900.0)]
# Bounds match the measured knee (≤8 concurrent) and the tree design (depth 0–3).
_PARALLEL = [(f"{n}  ({'직렬' if n == 1 else '병렬 ' + str(n)})", n) for n in range(1, 9)]
_DEPTH = [("0  (위임 끔)", 0), ("1  (기본)", 1), ("2", 2), ("3", 3)]
# Turn cap for a session carrying out a whole plan — larger than an ordinary turn's.
_IMPL_TURNS = [("10", 10), ("20", 20), ("30  (기본)", 30), ("50", 50)]
# Common local context windows; 0 turns compaction off entirely.
_WINDOW = [("0  (압축 끔)", 0), ("8K", 8192), ("16K", 16384), ("32K", 32768),
           ("64K", 65536), ("128K", 131072)]
# The fraction of the window at which the oldest stretch is condensed.
_THRESHOLD = [("70%", 0.7), ("80%", 0.8), ("90%", 0.9), ("95%", 0.95)]
# Newest messages always kept verbatim when the oldest are summarised.
_KEEP_RECENT = [("2", 2), ("4", 4), ("6  (기본)", 6), ("8", 8), ("12", 12)]
# Per-mode thinking budget. -1 is the "전역" sentinel (Select needs a real value,
# not None); mapped back to None on save so the mode follows the global budget.
_GLOBAL = -1
# "전역 따름", not "전역": the pane also has a field literally called 전역 사고 예산,
# so the bare word read as "this mode IS the global one" rather than "this mode has
# no value of its own and takes that one".
_THINK = [("전역 따름", _GLOBAL), ("1K", 1024), ("2K", 2048), ("4K", 4096),
          ("8K", 8192), ("16K", 16384)]
# The global budget itself has no "전역" to fall back to; 0 means unbounded.
_THINK_GLOBAL = [("0  (무제한)", 0), ("1K", 1024), ("2K", 2048), ("4K", 4096),
                 ("8K", 8192), ("16K", 16384)]
# An OpenAI-style hint; servers that don't map it ignore it.
_EFFORT = [("low", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "xhigh")]
# Whether a turn that just got a tool result also thinks. 켬 = it thinks (analyses
# the result, capped by the budget); 끔 = it skips thinking and just acts, which
# stops a local model re-deliberating every turn. The stored flag is the inverse
# (no_think_after_tools), so 켬 maps to False and 끔 to True.
_AFTER_TOOLS = [("켬 (사고함)", False), ("끔 (사고 안 함)", True)]


def _nearest(options, value):
    """The listed value closest to `value`, so a config set by hand still lands on
    a real option instead of leaving the Select blank."""
    return min((v for _, v in options), key=lambda v: abs(v - value))


def _think_value(override):
    """A per-mode budget (int | None) → the Select value: the global sentinel when
    None, else the nearest listed budget."""
    return _GLOBAL if override is None else _nearest([o for o in _THINK if o[1] != _GLOBAL], override)


class Settings(ModalScreen["SettingsResult | None"]):
    """dismiss(SettingsResult) = save · dismiss(None) = cancel."""

    BINDINGS = [("escape", "cancel", "Close")]

    def __init__(self, cfg) -> None:
        super().__init__()
        self._cfg = cfg
        # Whatever the server reports replaces this; until then the configured name
        # is the only one we know exists, and it must stay selectable.
        self._models = [cfg.name]

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
            yield Input(value=cfg.base_url, placeholder="http://localhost:8888/v1",
                        id="settings-base-url")
            yield Label("API 키")
            yield Input(value=cfg.api_key, placeholder="EMPTY", id="settings-api-key")
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
                cfg.reasoning_effort == v for _, v in _EFFORT) else "medium",
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
        """Ask the typed endpoint for its models, off the UI thread — it is a network
        call and a wrong address takes the full probe timeout to fail."""
        try:
            names = client.list_models(base_url=base_url, api_key=api_key)
            self.app.call_from_thread(self._models_arrived, names, None)
        except Exception as exc:  # any transport/HTTP failure is the user's answer
            self.app.call_from_thread(self._models_arrived, [], f"{type(exc).__name__}")

    def _models_arrived(self, names: list[str], error: str | None) -> None:
        status = self.query_one("#settings-fetch-status", Label)
        if error or not names:
            status.update(f"실패: {error}" if error else "모델이 없습니다")
            return
        select = self.query_one("#settings-model", Select)
        keep = select.value  # survive the reload when the server still offers it
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
        self.dismiss(SettingsResult(
            base_url=self.query_one("#settings-base-url", Input).value.strip() or self._cfg.base_url,
            api_key=self.query_one("#settings-api-key", Input).value.strip() or self._cfg.api_key,
            model=str(self._value("model")),
            timeout=float(self._value("timeout")),
            max_parallel=int(self._value("parallel")),
            depth=int(self._value("depth")),
            impl_max_turns=int(self._value("impl-turns")),
            context_window=int(self._value("window")),
            compact_threshold=float(self._value("threshold")),
            keep_recent=int(self._value("keep-recent")),
            thinking_budget=int(self._value("think-global")),
            reasoning_effort=str(self._value("effort")),
            plan_thinking=self._think("plan"),
            impl_thinking=self._think("impl"),
            subagent_thinking=self._think("subagent"),
            no_think_after_tools=bool(self._value("after-tools")),
        ))

    @on(Button.Pressed, "#settings-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
