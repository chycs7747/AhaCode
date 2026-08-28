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

from ahacode import config, storage

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
def isolated_config(monkeypatch, tmp_path):
    """Point BOTH config layers at tmp, for every test.

    config.load() reads the global file on every call and creates it on first run,
    and config.save() writes to whichever layer is in play — so without this a test
    run writes settings into the developer's home and into the working copy's own
    .ahacode/, then reads them back, making results depend on the machine it ran on.
    A project config left behind that way also outranks the global one at runtime,
    which is a confusing thing to debug long after the test that wrote it.

    Tests needing a specific project override still patch config.CONFIG_PATH in their
    own fixture; a module-level autouse fixture runs after this one, so theirs wins.
    """
    monkeypatch.setattr(config, "GLOBAL_CONFIG_PATH", tmp_path / "global-config.toml")
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "project-config.toml")


@pytest.fixture(autouse=True)
def isolated_output_dirs(monkeypatch, tmp_path):
    """Point everything the app GENERATES at tmp, for every test.

    Test files patch the directory they assert on and leave the rest pointing at the
    working copy, so any output a test does not care about lands in the developer's
    own .ahacode/ — a run of the suite left 48 transcripts of the fake model saying
    "Hello! How can I help you today?" among the real ones. A test that patches its
    own directory still wins: a module-level autouse fixture runs after this one.
    """
    for name in ("SESSIONS_DIR", "PLANS_DIR", "TRANSCRIPTS_DIR", "SCRATCH_DIR"):
        monkeypatch.setattr(storage, name, tmp_path / name.lower())


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
