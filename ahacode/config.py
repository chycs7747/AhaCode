"""User configuration — config.toml in the project root, created on first load."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

# <project root>/config.toml — next to pyproject.toml, kept out of git.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.toml"

DEFAULT_BASE_URL = "http://127.0.0.1:8078/v1"
DEFAULT_MODEL = "qwen38-nvfp4"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_TIMEOUT = 60.0
DEFAULT_SUBAGENT_DEPTH = 1  # generations of sub-agents that may nest (0 = none)
DEFAULT_IMPL_MAX_TURNS = 30  # turn cap for a session carrying out a whole plan
DEFAULT_MAX_PARALLEL_AGENTS = 8  # cap on concurrent gateway requests (measured knee)
DEFAULT_THINKING_TOKEN_BUDGET = 4096  # per-turn reasoning-token cap; 0 = unbounded
DEFAULT_REASONING_EFFORT = "medium"   # OpenAI-style hint (low|medium|high|xhigh)
# After a tool result, the next turn's job is to act on it, not re-deliberate — so
# skip <think> on those turns. Kills the per-turn re-thinking that otherwise stacks
# across a multi-turn loop into a spiral (thinking budget only caps ONE turn).
DEFAULT_NO_THINK_AFTER_TOOLS = True
# Context management. The window is the model's; the threshold is the fraction of it
# at which the oldest stretch is condensed into a summary. 0 disables compaction.
DEFAULT_CONTEXT_WINDOW = 32768
DEFAULT_COMPACT_THRESHOLD = 0.8
DEFAULT_KEEP_RECENT_MESSAGES = 6  # newest messages always kept verbatim
# Plan gate: in act mode, a fresh plan of at least this many steps pauses the loop
# and asks before any of it runs. 0 disables the gate.
# Tool calls matching one of these run without the approval modal. Empty by
# default: pre-approval is the user's call, never a shipped assumption.
DEFAULT_ALLOW_RULES: tuple[str, ...] = ()
# How long a bash command may run. 30s could not finish this project's own test
# suite (~60s), and running the project's tests is the main thing bash is for.
DEFAULT_BASH_TIMEOUT = 120


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    name: str
    api_key: str
    timeout: float
    # Agent behaviour (not model connection): how deep sub-agent nesting may go.
    subagent_depth: int = DEFAULT_SUBAGENT_DEPTH
    # Max concurrent gateway requests across ALL agents (main + sub-agents at every
    # depth). One process-wide cap protecting the single-GPU backend (~8 measured).
    max_parallel_agents: int = DEFAULT_MAX_PARALLEL_AGENTS
    # Per-turn thinking control. budget hard-caps reasoning tokens (0 = unbounded) —
    # the server must have its reasoning-config set for the cap to take effect, else
    # the request is refused and client.py retries without it. reasoning_effort is an
    # OpenAI-style hint, ignored by servers that don't map it.
    thinking_token_budget: int = DEFAULT_THINKING_TOKEN_BUDGET
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    # Per-mode thinking budget overrides. None (the default) = use the global
    # thinking_token_budget above; a value lets plan think deep while impl and
    # sub-agents think shallow. Only the budget varies by mode — NOT the model
    # (switching models reloads the gateway's vLLM container, minutes each time).
    plan_thinking_budget: int | None = None
    impl_thinking_budget: int | None = None
    subagent_thinking_budget: int | None = None
    # When True, turns whose last message is a tool result are sent with thinking
    # disabled (enable_thinking=False), so the model executes instead of re-thinking.
    no_think_after_tools: bool = DEFAULT_NO_THINK_AFTER_TOOLS
    # Context management: once a request's prompt reaches context_window *
    # compact_threshold tokens, the oldest messages are replaced by one summary,
    # keeping the newest keep_recent_messages verbatim. context_window = 0 is off.
    context_window: int = DEFAULT_CONTEXT_WINDOW
    compact_threshold: float = DEFAULT_COMPACT_THRESHOLD
    keep_recent_messages: int = DEFAULT_KEEP_RECENT_MESSAGES
    # Turn cap for an impl session (one continuous context carrying out a whole
    # plan) — larger than an ordinary turn's, which answers one message.
    impl_max_turns: int = DEFAULT_IMPL_MAX_TURNS
    # Pre-approval rules, "tool:pattern" (see permissions.py). A tuple, not a
    # list, because this dataclass is frozen and must stay hashable.
    allow_rules: tuple[str, ...] = DEFAULT_ALLOW_RULES
    # Seconds a bash command may run before it is killed. A single call can ask
    # for more (up to bash.MAX_TIMEOUT) when it knows it will be slow.
    bash_timeout: int = DEFAULT_BASH_TIMEOUT

    def thinking_budget_for(self, mode: str | None) -> int:
        """The reasoning-token cap for this kind of turn: the mode's override if it
        set one, otherwise the global budget. mode is "plan" | "impl" | "subagent"
        | None (a plain act turn uses the global)."""
        override = {
            "plan": self.plan_thinking_budget,
            "impl": self.impl_thinking_budget,
            "subagent": self.subagent_thinking_budget,
        }.get(mode)
        return self.thinking_token_budget if override is None else override


DEFAULTS = ModelConfig(
    base_url=DEFAULT_BASE_URL,
    name=DEFAULT_MODEL,
    api_key=DEFAULT_API_KEY,
    timeout=DEFAULT_TIMEOUT,
)


def _opt_int(value) -> int | None:
    """A per-mode override: an int, or None when the key is absent/blank (= global)."""
    return None if value in (None, "") else int(value)


def _mode_budget_lines(cfg: ModelConfig) -> str:
    """Render only the per-mode budgets the user actually set, so the default file
    stays clean. Absent = the mode follows the global thinking_token_budget."""
    rows = [("plan", cfg.plan_thinking_budget), ("impl", cfg.impl_thinking_budget),
            ("subagent", cfg.subagent_thinking_budget)]
    out = [f"{name}_thinking_budget = {val}" for name, val in rows if val is not None]
    return ("\n" + "\n".join(out)) if out else ""


def _render(cfg: ModelConfig) -> str:
    allow = list(cfg.allow_rules)  # rendered as a TOML array
    return f"""\
# AhaCode configuration
# Any OpenAI-compatible endpoint works (vLLM, Ollama, gateways, ...).
# Editable here, or from inside the chat with /model and /url.

[model]
base_url = "{cfg.base_url}"
name = "{cfg.name}"
api_key = "{cfg.api_key}"  # many local servers ignore this, but the SDK requires one
timeout = {cfg.timeout}    # seconds; caps how long a read may block between chunks
thinking_token_budget = {cfg.thinking_token_budget}  # per-turn reasoning cap; 0 = unbounded (server reasoning-config required)
reasoning_effort = "{cfg.reasoning_effort}"          # low|medium|high|xhigh — a hint; effect is server-dependent
no_think_after_tools = {str(cfg.no_think_after_tools).lower()}  # skip <think> on turns that just got a tool result (stops the multi-turn spiral)
context_window = {cfg.context_window}  # the model's context window in tokens; 0 disables compaction

[agent]
# How many generations of sub-agents may nest. 1 = the main agent may spawn
# sub-agents, but those sub-agents cannot spawn their own (no grandchildren).
subagent_depth = {cfg.subagent_depth}
# Turn cap for a session carrying out an approved plan (an ordinary turn has 10).
impl_max_turns = {cfg.impl_max_turns}
# Per-mode reasoning-token cap (optional). Absent = the mode uses the global
# thinking_token_budget above. Lets plan think deep, impl / sub-agents shallow.{_mode_budget_lines(cfg)}
# Max concurrent requests to the gateway across all agents (the single-GPU backend
# saturates around here; higher just queues and adds latency).
max_parallel_agents = {cfg.max_parallel_agents}
# Condense the oldest messages once a request reaches this fraction of the window,
# always keeping the newest keep_recent_messages untouched.
compact_threshold = {cfg.compact_threshold}
keep_recent_messages = {cfg.keep_recent_messages}
# Seconds a bash command may run before it is killed (a call may ask for more).
bash_timeout = {cfg.bash_timeout}

[permissions]
# Tool calls matching a rule run WITHOUT asking — "tool:pattern", where pattern is
# fnmatch (* = anything) against the command (bash), the pattern (grep/glob), or the
# path. A bare "tool" allows that tool outright.
#   allow = ["bash:uv run pytest*", "bash:git status", "bash:ls*", "edit:ahacode/*"]
# For bash EVERY chained sub-command must match, so "git status*" cannot smuggle in
# "&& rm -rf ~". The dangerous-command denylist still runs first and always wins.
allow = {allow!r}
"""


def save(cfg: ModelConfig, path: Path | None = None) -> None:
    """Write the config file (used by first-run defaults and /commands)."""
    path = path or CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render(cfg), encoding="utf-8")


def load(path: Path | None = None) -> ModelConfig:
    """Load the model config, writing a commented default file on first run."""
    path = path or CONFIG_PATH
    if not path.exists():
        save(DEFAULTS, path)
    with path.open("rb") as f:  # tomllib requires binary mode
        data = tomllib.load(f)
    model = data.get("model", {})
    agent = data.get("agent", {})
    perms = data.get("permissions", {})
    # Missing keys fall back to defaults, so partial configs stay valid.
    return ModelConfig(
        base_url=model.get("base_url", DEFAULT_BASE_URL),
        name=model.get("name", DEFAULT_MODEL),
        api_key=model.get("api_key", DEFAULT_API_KEY),
        timeout=float(model.get("timeout", DEFAULT_TIMEOUT)),
        subagent_depth=int(agent.get("subagent_depth", DEFAULT_SUBAGENT_DEPTH)),
        impl_max_turns=int(agent.get("impl_max_turns", DEFAULT_IMPL_MAX_TURNS)),
        plan_thinking_budget=_opt_int(agent.get("plan_thinking_budget")),
        impl_thinking_budget=_opt_int(agent.get("impl_thinking_budget")),
        subagent_thinking_budget=_opt_int(agent.get("subagent_thinking_budget")),
        max_parallel_agents=int(agent.get("max_parallel_agents", DEFAULT_MAX_PARALLEL_AGENTS)),
        thinking_token_budget=int(model.get("thinking_token_budget", DEFAULT_THINKING_TOKEN_BUDGET)),
        reasoning_effort=str(model.get("reasoning_effort", DEFAULT_REASONING_EFFORT)),
        no_think_after_tools=bool(model.get("no_think_after_tools", DEFAULT_NO_THINK_AFTER_TOOLS)),
        context_window=int(model.get("context_window", DEFAULT_CONTEXT_WINDOW)),
        compact_threshold=float(agent.get("compact_threshold", DEFAULT_COMPACT_THRESHOLD)),
        keep_recent_messages=int(agent.get("keep_recent_messages", DEFAULT_KEEP_RECENT_MESSAGES)),
        allow_rules=tuple(str(r) for r in perms.get("allow", DEFAULT_ALLOW_RULES)),
        bash_timeout=int(agent.get("bash_timeout", DEFAULT_BASH_TIMEOUT)),
    )
