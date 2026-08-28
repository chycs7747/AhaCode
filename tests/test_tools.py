import sys
import time
from pathlib import Path

import pytest

from ahacode import config, shell, tools

# The interpreter running these tests, as a path the shell will not mangle: on
# Windows sys.executable is C:\...\python.exe, and bash reads every backslash as an
# escape. Forward slashes work in both. "python3" would not work at all — Windows
# names it python.exe, and the bare name hits the Store alias stub.
PY = Path(sys.executable).as_posix()
# A Windows box with no bash runs cmd (see ahacode/shell.py), which shares none of
# this syntax. The tool is honest about that; these tests skip rather than pretend.
needs_posix_shell = pytest.mark.skipif(
    shell.NAME != "bash", reason="needs a POSIX shell — install Git for Windows"
)


def test_read_returns_file_contents(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("line1\nline2\nline3", encoding="utf-8")
    out = tools.READ.execute({"path": str(f)})
    assert out == "line1\nline2\nline3"


def test_read_windows_with_offset_and_limit(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("\n".join(f"L{i}" for i in range(1, 11)), encoding="utf-8")
    out = tools.READ.execute({"path": str(f), "offset": 3, "limit": 2})
    assert out.startswith("L3\nL4")
    assert "more lines" in out  # signals the file continues


def test_bash_runs_and_captures_output():
    out = tools.BASH.execute({"command": "echo aha"})
    assert out == "aha"


def test_bash_reports_nonzero_exit():
    out = tools.BASH.execute({"command": "exit 3"})
    assert "exit code 3" in out


def test_bash_spills_big_output_instead_of_discarding_it(tmp_path, monkeypatch):
    """Truncation throws the middle away for good. Spilling keeps everything: a
    header points at the file, the preview keeps both ends, and nothing is lost."""
    from ahacode.tools import bash as bash_mod
    from ahacode.tools import spill

    monkeypatch.setattr(spill, "_session_dir", tmp_path / "s-out")
    out = bash_mod.BASH.execute(
        {"command": f"{PY} -c \"print('HEAD'); print('x'*80000); print('TAIL')\""}
    )
    assert out.startswith("[output was 80,0")     # the header stands in for the bulk
    assert "HEAD" in out and "TAIL" in out         # both ends survive in the preview
    assert len(out) < bash_mod._PREVIEW_CHARS + 600

    spilled = list((tmp_path / "s-out").glob("bash-*.txt"))
    assert len(spilled) == 1
    full = spilled[0].read_text(encoding="utf-8")
    assert full.startswith("HEAD") and full.rstrip().endswith("TAIL")
    assert len(full) > 80_000                      # nothing was thrown away


def test_small_bash_output_is_untouched(tmp_path, monkeypatch):
    from ahacode.tools import bash as bash_mod
    from ahacode.tools import spill

    monkeypatch.setattr(spill, "_session_dir", tmp_path / "s-out")
    assert bash_mod.BASH.execute({"command": "echo hello"}) == "hello"
    assert not (tmp_path / "s-out").exists()       # no file for output that fits


def test_bash_falls_back_to_truncation_when_it_cannot_spill(tmp_path, monkeypatch):
    """A spill that fails must not break the tool — it degrades to the old behaviour."""
    from ahacode.tools import bash as bash_mod
    from ahacode.tools import spill

    monkeypatch.setattr(spill, "write", lambda text, prefix="out": None)
    out = bash_mod.BASH.execute(
        {"command": f"{PY} -c \"print('HEAD'); print('x'*80000); print('TAIL')\""}
    )
    assert out.startswith("HEAD") and out.endswith("TAIL")
    assert "elided" in out


@needs_posix_shell
def test_non_ascii_output_comes_back_intact(tmp_path):
    """Output is decoded as UTF-8, not as the machine's locale.

    text=True decodes with the locale encoding, which on a Korean Windows box is
    cp949: the first UTF-8 byte from `cat` on a Korean file killed the reader
    thread, communicate() returned None, and the tool raised "object of type
    'NoneType' has no len()". Every command whose output held a non-ASCII character
    failed that way — which, in a project whose plans and comments are Korean, is
    most of them.
    """
    from ahacode.tools import bash as bash_mod

    f = tmp_path / "korean.txt"
    f.write_text("계획 요약: 이진 탐색 + 후위 그리디\n", encoding="utf-8")
    out = bash_mod.BASH.execute({"command": f'cat "{f.as_posix()}"'})
    assert "계획 요약: 이진 탐색 + 후위 그리디" in out


def test_the_recursion_guard_matches_commands_not_paths():
    """The guard was tightened to stop flagging pytest's own tmp_path (which is
    literally .../pytest-of-<user>/pytest-91/...). It must still catch the thing it
    exists for: a test that shells out to this suite and forks exponentially."""
    import conftest

    def flagged(text):
        return any(p.search(text) for p in conftest._RECURSIVE)

    assert flagged("uv run pytest -q")
    assert flagged("pytest tests/")
    assert flagged("cd sub && tox")
    assert not flagged('cat "C:/Users/x/Temp/pytest-of-cyh/pytest-91/korean.txt"')
    assert not flagged("echo pytest-report.xml")


def test_undecodable_output_is_replaced_not_raised():
    """Bytes that are not valid UTF-8 still come back as text — a mangled line tells
    the model more than an exception does."""
    from ahacode.tools import bash as bash_mod

    out = bash_mod.BASH.execute(
        {"command": f"{PY} -c \"import sys; sys.stdout.buffer.write(b'ok-\\xff\\xfe-end')\""}
    )
    assert "ok-" in out and "-end" in out


@needs_posix_shell
def test_timeout_keeps_the_partial_output(tmp_path, monkeypatch):
    """A killed command still printed something. Discarding it leaves the model with
    nothing — not even how far the command got. (The partial output arrives as BYTES
    off TimeoutExpired even with text=True, so it has to be decoded by hand.)"""
    from ahacode.tools import bash as bash_mod

    out = bash_mod.BASH.execute({
        "command": "for i in 1 2 3 4 5; do echo line $i; sleep 1; done",
        "timeout": 2,
    })
    assert "line 1" in out
    assert "timed out after 2s" in out
    assert "timeout=4" in out  # tells the model how to retry


def test_timeout_kills_the_whole_process_tree():
    """A timeout must not leave the command's children running.

    subprocess.run's own timeout kills only the shell, so anything it launched is
    orphaned and keeps burning CPU — a few timed-out `uv run pytest` calls once left
    ~140 stray processes behind. The command gets its own process group and the
    GROUP is killed.
    """
    import os

    if not os.path.isdir("/proc"):
        pytest.skip("needs /proc to see the process table")

    from ahacode.tools import bash as bash_mod

    def sleep_pids():
        found = set()
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                if open(f"/proc/{d}/comm").read().strip() == "sleep":
                    found.add(int(d))
            except OSError:
                pass
        return found

    before = sleep_pids()
    bash_mod.BASH.execute({"command": "sleep 60 & sleep 60 & sleep 60", "timeout": 1})
    time.sleep(1.5)
    leaked = sleep_pids() - before
    for pid in leaked:            # never leave strays behind, even on failure
        try:
            os.kill(pid, 9)
        except OSError:
            pass
    assert leaked == set(), f"{len(leaked)} orphaned processes survived the timeout"


def test_timeout_comes_from_config_and_a_call_may_raise_it(monkeypatch):
    from dataclasses import replace

    from ahacode.tools import bash as bash_mod

    monkeypatch.setattr(config, "load", lambda *a, **k: replace(config.DEFAULTS, bash_timeout=45))
    assert bash_mod._resolve_timeout(None) == 45          # the configured default
    assert bash_mod._resolve_timeout(90) == 90            # a call may ask for more
    assert bash_mod._resolve_timeout("nonsense") == 45    # junk falls back
    assert bash_mod._resolve_timeout(10_000) == bash_mod.MAX_TIMEOUT  # but is capped


def test_default_timeout_clears_this_project_s_own_test_suite():
    """The system prompt tells the model to verify with the project's tests, and this
    suite takes ~60s — 30s could never finish it."""
    assert config.DEFAULTS.bash_timeout >= 120


def test_registry_and_approval_flags():
    assert set(tools.REGISTRY) == {
        "read", "glob", "grep", "write", "edit", "bash", "webfetch", "todo_write"
    }
    assert tools.REGISTRY["grep"].requires_approval is False  # search is read-only
    assert tools.REGISTRY["glob"].requires_approval is False
    assert tools.REGISTRY["bash"].requires_approval is True
    assert tools.REGISTRY["read"].requires_approval is False
    assert tools.REGISTRY["webfetch"].requires_approval is True  # reaches the network


def test_only_pure_reads_are_parallelizable():
    """The safety envelope for one-turn tool batching: the pure reads may run
    concurrently, everything that touches the filesystem (or whose effects can't be
    proven absent, like bash) stays serial. The agent loop only takes the parallel
    path when ALL calls in a batch are parallelizable, so a lone False here forces
    the whole batch back to serial."""
    for name in ("read", "glob", "grep", "webfetch"):  # webfetch is a network read
        assert tools.REGISTRY[name].parallelizable is True, name
    for name in ("write", "edit", "bash", "todo_write"):
        assert tools.REGISTRY[name].parallelizable is False, name


def test_specs_are_openai_function_schema():
    specs = tools.specs()
    assert {s["function"]["name"] for s in specs} == {
        "read", "glob", "grep", "write", "edit", "bash", "webfetch", "todo_write"
    }
    read_spec = next(s for s in specs if s["function"]["name"] == "read")
    assert read_spec["type"] == "function"
    assert read_spec["function"]["parameters"]["required"] == ["path"]


def test_bash_denylist_blocks_catastrophic_commands():
    dangerous = [
        "rm -rf /",
        "rm -rf ~",
        "rm -fr /*",
        ":(){ :|:& };:",
        "mkfs.ext4 /dev/sda",
        "dd if=/dev/zero of=/dev/sda",
        "ls && rm -rf /",          # dangerous half of a chain is caught
    ]
    for cmd in dangerous:
        assert tools.BASH.validate({"command": cmd}) is not None, cmd


def test_bash_denylist_allows_ordinary_commands():
    safe = [
        "ls -la",
        "rm -rf ./build",          # a targeted delete, not root/home
        "rm -rf /tmp/scratch",
        "python sort/compare.py",
        "mkdir -p sort/results",
        "echo hello",
    ]
    for cmd in safe:
        assert tools.BASH.validate({"command": cmd}) is None, cmd
