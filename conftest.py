"""Test-wide safety net.

The bash tool really executes what it is given, so a test that hands it `pytest`
re-runs this entire suite — which contains that test, which does it again. It is a
fork bomb: one such test reached ~460 processes and a load average of 198 before it
was noticed, because a timeout kills only the shell and leaves the tree orphaned.

A comment would not have prevented it. This does.
"""

import re
import subprocess

import pytest

from ahacode import client, workspace

# Anything that would start another test run of this project. Matched as COMMANDS,
# not as substrings: pytest's own tmp_path is C:\...\pytest-of-<user>\pytest-91\...,
# so a plain `in` test flags every command that merely touches a temp file. The
# lookarounds reject a hit that is glued to a path separator, a word character, or a
# hyphen — which is every path-shaped occurrence, and none of the real invocations.
_RECURSIVE = tuple(
    re.compile(rf"(?<![\w./\\-]){pattern}(?![\w./\\-])")
    for pattern in (r"pytest", r"uv\s+run", r"tox")
)


@pytest.fixture(autouse=True)
def isolated_workspace(monkeypatch, tmp_path):
    """Point every path under .ahacode/ at tmp, so no test touches the real one.

    The session directory is tmp_path itself, so a test can drop a .jsonl there
    directly. The client caches its config, so it is reset around each test.
    """
    monkeypatch.setattr(workspace, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(workspace, "PLANS_DIR", tmp_path / "plans")
    monkeypatch.setattr(workspace, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(workspace, "GLOBAL_CONFIG_PATH", tmp_path / "global-config.toml")
    client.reset()
    yield
    client.reset()


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
