"""grep / glob — the read-only search tools."""

import pytest

from ahacode.tools import glob as glob_mod
from ahacode.tools import grep as grep_mod
from ahacode.tools import walk


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A miniature project: two source files, a vendored clone, a cache dir."""
    (tmp_path / "ahacode").mkdir()
    (tmp_path / "ahacode" / "agent.py").write_text(
        "def run():\n    return 'agent'\n", encoding="utf-8"
    )
    (tmp_path / "ahacode" / "client.py").write_text(
        "def run():\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "notes.md").write_text("run the agent\n", encoding="utf-8")
    (tmp_path / "reference" / "elia").mkdir(parents=True)
    (tmp_path / "reference" / "elia" / "chat.py").write_text("def run():\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "agent.pyc").write_text("def run():\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.py").write_text("def run():\n", encoding="utf-8")
    # Both tools resolve relative paths and report results against the project root.
    for mod in (glob_mod, grep_mod):
        monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("ahacode.tools.base.PROJECT_ROOT", tmp_path)
    return tmp_path


# --- glob -----------------------------------------------------------------

def test_glob_finds_files_by_pattern(project):
    out = glob_mod.GLOB.execute({"pattern": "**/*.py"})
    assert "ahacode/agent.py" in out
    assert "ahacode/client.py" in out


def test_glob_skips_vendored_and_cache_dirs(project):
    out = glob_mod.GLOB.execute({"pattern": "**/*.py"})
    assert "reference" not in out   # vendored clone
    assert "__pycache__" not in out  # build cache
    assert ".git" not in out         # any dot-directory


def test_glob_orders_newest_first(project):
    import os
    import time

    later = time.time() + 10
    os.utime(project / "ahacode" / "client.py", (later, later))
    lines = glob_mod.GLOB.execute({"pattern": "**/*.py"}).splitlines()
    assert lines[0] == "ahacode/client.py"


def test_glob_explicit_root_overrides_the_skip_list(project):
    """A denylisted directory is skipped while WALKING, but searchable when asked for
    by name — the escape hatch that keeps reference/ readable on purpose."""
    out = glob_mod.GLOB.execute({"pattern": "**/*.py", "path": "reference/elia"})
    assert "reference/elia/chat.py" in out


def test_glob_no_match_is_not_an_error(project):
    assert glob_mod.GLOB.execute({"pattern": "**/*.rs"}) == "(no files matched)"


# --- grep -----------------------------------------------------------------

def test_grep_reports_path_line_text(project):
    out = grep_mod.GREP.execute({"pattern": r"def run"})
    assert "ahacode/agent.py:1:def run():" in out
    assert "ahacode/client.py:1:def run():" in out


def test_grep_skips_vendored_and_cache_dirs(project):
    out = grep_mod.GREP.execute({"pattern": r"def run"})
    assert "reference" not in out
    assert "__pycache__" not in out
    assert ".git" not in out


def test_grep_narrows_by_glob(project):
    out = grep_mod.GREP.execute({"pattern": "run", "glob": "**/*.md"})
    assert "notes.md:1:run the agent" in out
    assert "agent.py" not in out


def test_grep_is_a_regex(project):
    out = grep_mod.GREP.execute({"pattern": r"^def \w+\(\):$"})
    assert "ahacode/agent.py:1" in out
    assert "notes.md" not in out


def test_grep_bad_regex_raises_for_the_loop_to_report(project):
    """A broken pattern must surface as a tool error the model can fix, not a
    traceback from the middle of the walk."""
    import re

    with pytest.raises(re.error):
        grep_mod.GREP.execute({"pattern": "("})


def test_grep_no_match_is_not_an_error(project):
    assert grep_mod.GREP.execute({"pattern": "zzzz-nothing"}) == "(no matches)"


def test_grep_caps_its_output(project, monkeypatch):
    monkeypatch.setattr(grep_mod, "_MAX_MATCHES", 2)
    out = grep_mod.GREP.execute({"pattern": "run"})
    assert "stopped at 2 matches" in out


def test_grep_truncates_a_very_long_line(project, monkeypatch):
    monkeypatch.setattr(grep_mod, "_MAX_LINE_CHARS", 10)
    (project / "long.txt").write_text("needle " + "x" * 500 + "\n", encoding="utf-8")
    out = grep_mod.GREP.execute({"pattern": "needle", "glob": "*.txt"})
    assert "…" in out
    assert len(out.splitlines()[0]) < 60


# --- traversal ------------------------------------------------------------

def test_binary_and_oversized_files_are_skipped(project, monkeypatch):
    (project / "blob.bin").write_bytes(b"\xff\xfe\x00needle")
    assert walk.read_text_or_none(project / "blob.bin") is None
    monkeypatch.setattr(walk, "MAX_FILE_BYTES", 4)
    assert walk.read_text_or_none(project / "notes.md") is None


def test_unreadable_file_does_not_crash_a_search(project):
    assert walk.read_text_or_none(project / "does-not-exist.py") is None
