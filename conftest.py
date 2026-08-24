"""Test-wide safety net.

The bash tool really executes what it is given, so a test that hands it `pytest`
re-runs this entire suite — which contains that test, which does it again. It is a
fork bomb: one such test reached ~460 processes and a load average of 198 before it
was noticed, because a timeout kills only the shell and leaves the tree orphaned.

A comment would not have prevented it. This does.
"""

import subprocess

import pytest

# Anything that would start another test run of this project.
_RECURSIVE = ("pytest", "uv run", "tox")


@pytest.fixture(autouse=True)
def no_recursive_test_runs(monkeypatch):
    """Fail loudly if a test tries to launch the test suite from inside itself."""
    real_popen = subprocess.Popen

    def guard(command, *args, **kwargs):
        text = command if isinstance(command, str) else " ".join(map(str, command))
        for needle in _RECURSIVE:
            if needle in text:
                raise AssertionError(
                    f"a test tried to run the test suite as a subprocess: {text!r}\n"
                    "That re-enters this file and forks exponentially. Use a trivial "
                    "command (echo/true) when the tool under test really executes."
                )
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", guard)
    yield
