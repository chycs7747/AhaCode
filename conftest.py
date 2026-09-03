"""Test-wide fixtures: a private .ahacode/, offline app doubles, and a guard against a
test launching the suite from inside itself.
"""

import re
import subprocess
import time
from dataclasses import dataclass

import pytest

from ahacode import client, workspace
from ahacode.events import TextDelta, ThinkingDelta

# Anything that would start another test run of this project, matched as a command
# rather than a substring: pytest's own tmp_path contains "pytest-of-<user>".
_RECURSIVE = tuple(
    re.compile(rf"(?<![\w./\\-]){pattern}(?![\w./\\-])")
    for pattern in (r"pytest", r"uv\s+run", r"tox")
)


@pytest.fixture(autouse=True)
def isolated_workspace(monkeypatch, tmp_path):
    """Point every path under .ahacode/ at tmp. The session directory is tmp_path itself,
    so a test can drop a .jsonl there; the client's cached config is reset around each test.
    """
    monkeypatch.setattr(workspace, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(workspace, "PLANS_DIR", tmp_path / "plans")
    monkeypatch.setattr(workspace, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(workspace, "GLOBAL_CONFIG_PATH", tmp_path / "global-config.toml")
    client.reset()
    yield
    client.reset()


@dataclass
class FakeLLM:
    """An offline model: streams a fixed reasoning line, then a fixed answer."""

    thinking: str = "The user greeted me. Keep the reply short."
    response: str = "Hello! How can I help you today?"

    def stream(self, messages, tools=None):
        for word in self.thinking.split(" "):
            time.sleep(0.01)
            yield ThinkingDelta(word + " ")
        for word in self.response.split(" "):
            time.sleep(0.01)
            yield TextDelta(word + " ")


@pytest.fixture
def offline_app(monkeypatch):
    """Keep an AhaCodeApp off the network: a fixed model list and no auto-title.

    Not autouse: tests of client.complete itself must see the real function.
    """
    from ahacode.app import AhaCodeApp

    monkeypatch.setattr(client, "list_models", lambda *a, **k: ["qwen38", "qwen3-4b"])
    monkeypatch.setattr(client, "complete", lambda messages: "")
    monkeypatch.setattr(AhaCodeApp, "generate_title", lambda self, *a, **k: None)


@pytest.fixture
def fake_llm(monkeypatch) -> FakeLLM:
    """Replace client.stream_chat with a FakeLLM for the duration of a test."""
    fake = FakeLLM()
    monkeypatch.setattr(client, "stream_chat", fake.stream)
    return fake


@pytest.fixture(autouse=True)
def no_recursive_test_runs(monkeypatch):
    """Fail loudly if a test tries to launch the test suite from inside itself."""
    real_popen = subprocess.Popen

    def guard(command, *args, **kwargs):
        text = command if isinstance(command, str) else " ".join(map(str, command))
        for needle in _RECURSIVE:
            if needle.search(text):
                raise AssertionError(
                    f"a test tried to run the test suite as a subprocess: {text!r}\n"
                    "That re-enters this file and forks exponentially. Use a trivial "
                    "command (echo/true) when the tool under test really executes."
                )
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", guard)
    yield
