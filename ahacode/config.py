"""User configuration: ~/.ahacode/config.toml with an optional per-project override.

Which endpoint to talk to is a fact about the machine, so it lives once in the
global file; a project's .ahacode/config.toml is merged over it key by key.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ahacode import workspace


@dataclass(frozen=True)
class ModelConfig:
    """Everything read from config.toml, with the defaults a first run gets."""

    base_url: str = "http://localhost:8888/v1"  # Ollama serves :11434, vLLM :8000
    name: str = "qwen3.8-flash-next"
    api_key: str = "EMPTY"  # many local servers ignore it, but the SDK requires one
    timeout: float = 900.0  # seconds a read may block between chunks
    thinking_token_budget: int = 4096  # per-turn reasoning cap; 0 = unbounded
    reasoning_effort: str = "medium"  # low|medium|high|xhigh; a hint
    no_think_after_tools: bool = True  # skip reasoning on a turn that acts on a tool result
    context_window: int = 32768  # tokens; 0 disables compaction
    subagent_depth: int = 1  # generations of sub-agents that may nest; 0 = none
    max_parallel_agents: int = 8  # cap on concurrent requests across all agents
    impl_max_turns: int = 30  # turn cap for a session carrying out a plan; 0 = uncapped
    auto_continue_stall: int = 3  # turns in a row finishing no step before an impl run stops; 0 = off
    stall_rounds: int = 40  # rounds in one turn finishing no step before the turn ends; 0 = off
    plan_thinking_budget: int | None = None  # per-mode reasoning caps; None = the global one
    impl_thinking_budget: int | None = None
    subagent_thinking_budget: int | None = None
    compact_threshold: float = 0.8  # condense once a request reaches this fraction of the window
    keep_recent_messages: int = 6  # newest messages always kept verbatim
    bash_timeout: int = 120  # seconds a bash command may run; a call may ask for more
    allow_rules: tuple[str, ...] = ()  # "tool:pattern" rules that skip approval (permissions.py)

    def thinking_budget_for(self, mode: str | None) -> int:
        """The reasoning-token cap for a turn: the mode's override, else the global one.

        Args:
            mode: "plan", "impl", "subagent", or None for a plain act turn.

        Returns:
            The budget in tokens; 0 means unbounded.
        """
        override = {
            "plan": self.plan_thinking_budget,
            "impl": self.impl_thinking_budget,
            "subagent": self.subagent_thinking_budget,
        }.get(mode)
        return self.thinking_token_budget if override is None else override


DEFAULTS = ModelConfig()

# The TOML section each field is written to and read from.
_SECTIONS: dict[str, tuple[str, ...]] = {
    "model": ("base_url", "name", "api_key", "timeout", "thinking_token_budget",
              "reasoning_effort", "no_think_after_tools", "context_window"),
    "agent": ("subagent_depth", "max_parallel_agents", "impl_max_turns",
              "auto_continue_stall", "stall_rounds", "plan_thinking_budget",
              "impl_thinking_budget", "subagent_thinking_budget", "compact_threshold",
              "keep_recent_messages", "bash_timeout"),
    "permissions": ("allow_rules",),
}


def _toml(value) -> str:
    """One TOML value literal: bool, int, float, str, or a sequence of str."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)  # a JSON string is a valid TOML string
    return "[" + ", ".join(json.dumps(v, ensure_ascii=False) for v in value) + "]"


def _dump(cfg: ModelConfig) -> str:
    """Render a config as TOML, values only; None fields are omitted."""
    lines = ["# AhaCode configuration — see README.md#configuration", ""]
    for section, names in _SECTIONS.items():
        lines.append(f"[{section}]")
        for name in names:
            value = getattr(cfg, name)
            if value is not None:
                lines.append(f"{name} = {_toml(value)}")
        lines.append("")
    return "\n".join(lines)


def _read(path: Path) -> dict:
    """The raw TOML of one config file, or {} when there is none."""
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _layer(base: dict, over: dict) -> dict:
    """Merge one config file over another per section and per key."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def _build(data: dict) -> ModelConfig:
    """Turn merged TOML into the config object; missing keys keep their defaults."""
    kw = {name: data[section][name]
          for section, names in _SECTIONS.items()
          for name in names if name in data.get(section, {})}
    if "allow_rules" in kw:
        kw["allow_rules"] = tuple(str(r) for r in kw["allow_rules"])
    return ModelConfig(**kw)


def load() -> ModelConfig:
    """Load the global config with this project's override layered on top.

    The global file is written with the defaults on first run.

    Returns:
        The merged config.
    """
    if not workspace.GLOBAL_CONFIG_PATH.exists():
        _write(workspace.GLOBAL_CONFIG_PATH, DEFAULTS)
    return _build(_layer(_read(workspace.GLOBAL_CONFIG_PATH), _read(workspace.CONFIG_PATH)))


def save(cfg: ModelConfig) -> None:
    """Write the config to the project override if one exists, else to the global file.

    Args:
        cfg: The config to persist.
    """
    _write(workspace.CONFIG_PATH if workspace.CONFIG_PATH.exists()
           else workspace.GLOBAL_CONFIG_PATH, cfg)


def _write(path: Path, cfg: ModelConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump(cfg), encoding="utf-8")
