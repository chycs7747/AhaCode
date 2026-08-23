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


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    name: str
    api_key: str
    timeout: float
    # Agent behaviour (not model connection): how deep sub-agent nesting may go.
    subagent_depth: int = DEFAULT_SUBAGENT_DEPTH


DEFAULTS = ModelConfig(
    base_url=DEFAULT_BASE_URL,
    name=DEFAULT_MODEL,
    api_key=DEFAULT_API_KEY,
    timeout=DEFAULT_TIMEOUT,
)


def _render(cfg: ModelConfig) -> str:
    return f"""\
# AhaCode configuration
# Any OpenAI-compatible endpoint works (vLLM, Ollama, gateways, ...).
# Editable here, or from inside the chat with /model and /url.

[model]
base_url = "{cfg.base_url}"
name = "{cfg.name}"
api_key = "{cfg.api_key}"  # many local servers ignore this, but the SDK requires one
timeout = {cfg.timeout}    # seconds; caps how long a read may block between chunks

[agent]
# How many generations of sub-agents may nest. 1 = the main agent may spawn
# sub-agents, but those sub-agents cannot spawn their own (no grandchildren).
subagent_depth = {cfg.subagent_depth}
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
    # Missing keys fall back to defaults, so partial configs stay valid.
    return ModelConfig(
        base_url=model.get("base_url", DEFAULT_BASE_URL),
        name=model.get("name", DEFAULT_MODEL),
        api_key=model.get("api_key", DEFAULT_API_KEY),
        timeout=float(model.get("timeout", DEFAULT_TIMEOUT)),
        subagent_depth=int(agent.get("subagent_depth", DEFAULT_SUBAGENT_DEPTH)),
    )
