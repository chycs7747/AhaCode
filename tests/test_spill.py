"""Spilled tool output — the file is a real file, so the existing tools can go back to it."""

import pytest

from ahacode import storage
from ahacode.tools import base, spill
from ahacode.tools import glob as glob_mod
from ahacode.tools import grep as grep_mod
from ahacode.tools.glob import GLOB
from ahacode.tools.grep import GREP
from ahacode.tools.read import READ


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "SESSIONS_DIR", tmp_path / "sessions")
    # One project root everywhere: spill reports paths against it and the tools
    # resolve them against it. glob/grep hold their own imported copies.
    monkeypatch.setattr(base, "PROJECT_ROOT", tmp_path)
    for mod in (glob_mod, grep_mod):
        monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(spill, "_session_dir", None)
    yield


def test_files_land_beside_the_session_that_made_them():
    """Shared lifetime: deleting a session takes its spilled output with it."""
    session = tmp = storage.SESSIONS_DIR / "2026-08-24_120000.jsonl"
    spill.set_session(session)
    assert spill.target_dir().name == "2026-08-24_120000-out"
    assert spill.target_dir().parent == storage.SESSIONS_DIR
    assert tmp  # the session file itself is untouched


def test_without_a_session_it_still_has_somewhere_to_go():
    spill.set_session(None)
    assert spill.target_dir() == storage.SESSIONS_DIR / "tool-output"


def test_concurrent_spills_never_collide():
    """Sub-agents spill in parallel; mkstemp claims each name atomically."""
    paths = {spill.write(f"body {i}") for i in range(20)}
    assert len(paths) == 20
    assert all(p is not None for p in paths)


def test_the_file_is_capped_so_a_runaway_command_cannot_fill_the_disk(monkeypatch):
    monkeypatch.setattr(spill, "MAX_FILE_CHARS", 100)
    path = spill.write("y" * 10_000)
    assert len(path.read_text(encoding="utf-8")) == 100


def test_an_unwritable_target_returns_none_rather_than_raising(monkeypatch):
    monkeypatch.setattr(spill, "target_dir", lambda: (_ for _ in ()).throw(OSError("nope")))
    assert spill.write("x") is None


def test_paths_come_back_in_the_form_the_tools_take(tmp_path):
    path = spill.write("x")
    assert not spill.relative(path).startswith("/")   # relative to the project root


# --- the point of spilling: the model can go back to it ---------------------

def test_read_pages_through_a_spilled_file():
    path = spill.write("\n".join(f"line {i}" for i in range(500)))
    out = READ.execute({"path": spill.relative(path), "offset": 100, "limit": 3})
    assert out.splitlines()[:3] == ["line 99", "line 100", "line 101"]


def test_grep_searches_a_spilled_file_directly():
    """A file as grep's root has to work, or a spilled log is unsearchable —
    Path.glob on a non-directory quietly yields nothing."""
    path = spill.write("ok\nFAILED tests/test_x.py::test_boom\nok\n")
    out = GREP.execute({"pattern": "FAILED", "path": spill.relative(path)})
    assert "test_boom" in out
    assert ":2:" in out  # reported with its line number


def test_glob_accepts_a_file_root_too():
    path = spill.write("x")
    out = GLOB.execute({"pattern": "*", "path": spill.relative(path)})
    assert path.name in out
