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
DEFAULT_PLAN_GATE_MIN_STEPS = 3
# Tool calls matching one of these run without the approval modal. Empty by
# default: pre-approval is the user's call, never a shipped assumption.
DEFAULT_ALLOW_RULES: tuple[str, ...] = ()


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
    # When True, turns whose last message is a tool result are sent with thinking
    # disabled (enable_thinking=False), so the model executes instead of re-thinking.
    no_think_after_tools: bool = DEFAULT_NO_THINK_AFTER_TOOLS
    # Context management: once a request's prompt reaches context_window *
    # compact_threshold tokens, the oldest messages are replaced by one summary,
    # keeping the newest keep_recent_messages verbatim. context_window = 0 is off.
    context_window: int = DEFAULT_CONTEXT_WINDOW
    compact_threshold: float = DEFAULT_COMPACT_THRESHOLD
    keep_recent_messages: int = DEFAULT_KEEP_RECENT_MESSAGES
    # In act mode, a fresh plan (todo_write) with at least this many steps pauses
    # the agent loop and asks the user before anything runs. 0 = never ask.
    plan_gate_min_steps: int = DEFAULT_PLAN_GATE_MIN_STEPS
    # Pre-approval rules, "tool:pattern" (see permissions.py). A tuple, not a
    # list, because this dataclass is frozen and must stay hashable.
    allow_rules: tuple[str, ...] = DEFAULT_ALLOW_RULES


DEFAULTS = ModelConfig(
    base_url=DEFAULT_BASE_URL,
    name=DEFAULT_MODEL,
    api_key=DEFAULT_API_KEY,
    timeout=DEFAULT_TIMEOUT,
)


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
# Max concurrent requests to the gateway across all agents (the single-GPU backend
# saturates around here; higher just queues and adds latency).
max_parallel_agents = {cfg.max_parallel_agents}
# Condense the oldest messages once a request reaches this fraction of the window,
# always keeping the newest keep_recent_messages untouched.
compact_threshold = {cfg.compact_threshold}
keep_recent_messages = {cfg.keep_recent_messages}
# In act mode, pause and ask before running a fresh plan of this many steps or
# more (the model lays it out with todo_write). 0 never asks.
plan_gate_min_steps = {cfg.plan_gate_min_steps}

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
        max_parallel_agents=int(agent.get("max_parallel_agents", DEFAULT_MAX_PARALLEL_AGENTS)),
        thinking_token_budget=int(model.get("thinking_token_budget", DEFAULT_THINKING_TOKEN_BUDGET)),
        reasoning_effort=str(model.get("reasoning_effort", DEFAULT_REASONING_EFFORT)),
        no_think_after_tools=bool(model.get("no_think_after_tools", DEFAULT_NO_THINK_AFTER_TOOLS)),
        context_window=int(model.get("context_window", DEFAULT_CONTEXT_WINDOW)),
        compact_threshold=float(agent.get("compact_threshold", DEFAULT_COMPACT_THRESHOLD)),
        keep_recent_messages=int(agent.get("keep_recent_messages", DEFAULT_KEEP_RECENT_MESSAGES)),
        plan_gate_min_steps=int(agent.get("plan_gate_min_steps", DEFAULT_PLAN_GATE_MIN_STEPS)),
        allow_rules=tuple(str(r) for r in perms.get("allow", DEFAULT_ALLOW_RULES)),
    )
