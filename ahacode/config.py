"""User configuration — ~/.ahacode/config.toml, created with defaults on first load."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path.home() / ".ahacode" / "config.toml"

DEFAULT_BASE_URL = "http://127.0.0.1:8078/v1"
DEFAULT_MODEL = "qwen38-nvfp4"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_TIMEOUT = 60.0

DEFAULT_CONFIG = f"""\
# AhaCode configuration
# Any OpenAI-compatible endpoint works (vLLM, Ollama, gateways, ...).

[model]
base_url = "{DEFAULT_BASE_URL}"
name = "{DEFAULT_MODEL}"
api_key = "{DEFAULT_API_KEY}"  # many local servers ignore this, but the SDK requires one
timeout = {DEFAULT_TIMEOUT}    # seconds; caps how long a read may block between chunks
"""


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    name: str
    api_key: str
    timeout: float


def load(path: Path | None = None) -> ModelConfig:
    """Load the model config, writing a commented default file on first run."""
    path = path or CONFIG_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CONFIG, encoding="utf-8")
    with path.open("rb") as f:  # tomllib requires binary mode
        data = tomllib.load(f)
    model = data.get("model", {})
    # Missing keys fall back to defaults, so partial configs stay valid.
    return ModelConfig(
        base_url=model.get("base_url", DEFAULT_BASE_URL),
        name=model.get("name", DEFAULT_MODEL),
        api_key=model.get("api_key", DEFAULT_API_KEY),
        timeout=float(model.get("timeout", DEFAULT_TIMEOUT)),
    )
