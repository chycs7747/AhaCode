import time
from collections.abc import Iterator

from openai import OpenAI

from ahacode import config

# The unified delta format the UI consumes.
# Provider-specific differences are absorbed entirely inside this module.
Delta = tuple[str, str]  # ("thinking" | "text", text fragment)

# Client and config are created lazily on the first real request, so merely
# importing this module has no side effects (no config file written, no
# network client built) — tests that fake stream_chat never touch either.
_client: OpenAI | None = None
_cfg: config.ModelConfig | None = None


def reset() -> None:
    """Forget the cached client and config; the next request reloads from disk.

    Called after /commands change the config file.
    """
    global _client, _cfg
    _client = None
    _cfg = None


def _ensure_client() -> tuple[OpenAI, config.ModelConfig]:
    global _client, _cfg
    if _client is None:
        _cfg = config.load()
        # Local servers often ignore the key, but the SDK requires one.
        # The timeout caps how long a blocking read may wait between chunks.
        _client = OpenAI(
            base_url=_cfg.base_url, api_key=_cfg.api_key, timeout=_cfg.timeout
        )
    return _client, _cfg


def stream_chat(messages: list[dict]) -> Iterator[Delta]:
    """Send the full conversation history and yield (kind, fragment) deltas."""
    client, cfg = _ensure_client()
    response = client.chat.completions.create(
        model=cfg.name,
        messages=messages,
        stream=True,
    )
    for chunk in response:
        if not chunk.choices:  # guard: the final usage-only chunk has empty choices
            continue
        delta = chunk.choices[0].delta
        # Some servers stream thinking under "reasoning" (verified on the wire
        # against vLLM). The field is not part of the SDK's typed model, hence
        # the defensive getattr.
        reasoning = getattr(delta, "reasoning", None)
        if isinstance(reasoning, str) and reasoning:
            yield ("thinking", reasoning)
        if isinstance(delta.content, str) and delta.content:
            yield ("text", delta.content)


def list_models() -> list[str]:
    """Model ids offered by the endpoint (GET /v1/models)."""
    client, _ = _ensure_client()
    return [m.id for m in client.models.list()]


FAKE_THINKING = "The user greeted me. Keep the reply short."
FAKE_RESPONSE = "Hello! How can I help you today?"


def stream_chat_fake(messages: list[dict]) -> Iterator[Delta]:
    """Offline fake stream with the same (kind, fragment) protocol — for tests."""
    for word in FAKE_THINKING.split(" "):
        time.sleep(0.01)
        yield ("thinking", word + " ")
    for word in FAKE_RESPONSE.split(" "):
        time.sleep(0.01)
        yield ("text", word + " ")
