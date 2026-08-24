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


def test_bash_output_is_capped_keeping_both_ends():
    """One `cat` of a big file must not put the whole thing in the context. Both ends
    survive — a build log's error is at the end, a listing's header at the start."""
    from ahacode.tools import bash as bash_mod

    out = bash_mod.BASH.execute(
        {"command": "python3 -c \"print('HEAD'); print('x'*80000); print('TAIL')\""}
    )
    assert out.startswith("HEAD")
    assert out.endswith("TAIL")
    assert "elided" in out
    assert len(out) < bash_mod._MAX_OUTPUT_CHARS + 200


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
