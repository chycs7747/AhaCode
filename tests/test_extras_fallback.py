"""Vendor extensions are not portable. vLLM, Ollama and llama.cpp disagree about
which of enable_thinking / thinking_token_budget / top_k / min_p exist at all, so a
server that rejects them must degrade to a plain request instead of failing every
turn — and must not keep paying for the discovery."""

import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from ahacode import client, config
from ahacode.events import Notice, TextDelta


class Rejected(Exception):
    def __init__(self, message="unrecognized field", status_code=400):
        super().__init__(message)
        self.status_code = status_code


class _Response:
    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return iter(self._chunks)

    def __exit__(self, *a):
        return False


@pytest.fixture(autouse=True)
def endpoint(monkeypatch):
    tmp = Path(tempfile.mkdtemp())
    monkeypatch.setattr(config, "CONFIG_PATH", tmp / "c.toml")
    monkeypatch.setattr(config, "GLOBAL_CONFIG_PATH", tmp / "g.toml")
    cfg = replace(config.DEFAULTS, base_url="http://elsewhere:1234/v1", name="qwen3")
    monkeypatch.setattr(client, "_ensure_client", lambda: (_Picky(), cfg))
    monkeypatch.setattr(client, "_wait_for_permit", _ungated)
    client.reset()
    yield
    client.reset()


def _ungated(timeout, limit):
    import threading
    yield from ()
    return threading.Semaphore(1), False


class _Picky:
    """A server that refuses anything outside the plain request shape."""

    seen: list = []

    def __init__(self):
        self.chat = self                      # chat.completions.create(...)
        self.completions = self

    def create(self, **kwargs):
        _Picky.seen.append(kwargs)
        if kwargs.get("extra_body"):
            raise Rejected()
        if kwargs.get("stream"):
            return _Response([_chunk("ok")])
        return _Plain("titled")


class _Plain:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]


def _chunk(text):
    delta = type("D", (), {"content": text, "tool_calls": None})()
    choice = type("Ch", (), {"delta": delta, "finish_reason": None})()
    return type("K", (), {"choices": [choice], "usage": None})()


def test_a_server_that_refuses_extras_still_answers():
    _Picky.seen.clear()
    events = list(client.stream_chat([{"role": "user", "content": "hi"}]))
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["ok"]
    assert [e for e in events if isinstance(e, Notice)], \
        "degraded in silence — the ignored thinking budget needs saying"
    assert _Picky.seen[0].get("extra_body")          # tried the extensions
    assert not _Picky.seen[1].get("extra_body")      # retried plain


def test_the_rejection_is_paid_once():
    """A failed round trip per turn would make a working server feel broken."""
    _Picky.seen.clear()
    list(client.stream_chat([{"role": "user", "content": "a"}]))
    first = len(_Picky.seen)
    list(client.stream_chat([{"role": "user", "content": "b"}]))
    assert len(_Picky.seen) == first + 1, "asked again after being told no"
    assert not _Picky.seen[-1].get("extra_body")


def test_complete_degrades_too():
    """Titling and context compaction run through complete(); an endpoint that
    rejects extras there would fail every compaction, which is how a long session
    stops working."""
    _Picky.seen.clear()
    assert client.complete([{"role": "user", "content": "title this"}]) == "titled"
    assert not _Picky.seen[-1].get("extra_body")


def test_reset_re_probes():
    """Changing the endpoint or its config is the moment to try the extras again."""
    list(client.stream_chat([{"role": "user", "content": "a"}]))
    assert client._NO_EXTRAS
    client.reset()
    assert not client._NO_EXTRAS
