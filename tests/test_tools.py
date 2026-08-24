from ahacode import tools


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
        {"command": "python3 -c \"print('HEAD'); print('x'*80000); print('TAIL')\""}
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
        {"command": "python3 -c \"print('HEAD'); print('x'*80000); print('TAIL')\""}
    )
    assert out.startswith("HEAD") and out.endswith("TAIL")
    assert "elided" in out


def test_registry_and_approval_flags():
    assert set(tools.REGISTRY) == {
        "read", "glob", "grep", "write", "edit", "bash", "todo_write"
    }
    assert tools.REGISTRY["grep"].requires_approval is False  # search is read-only
    assert tools.REGISTRY["glob"].requires_approval is False
    assert tools.REGISTRY["bash"].requires_approval is True
    assert tools.REGISTRY["read"].requires_approval is False


def test_specs_are_openai_function_schema():
    specs = tools.specs()
    assert {s["function"]["name"] for s in specs} == {
        "read", "glob", "grep", "write", "edit", "bash", "todo_write"
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
