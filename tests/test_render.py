"""Unit tests for the shared tool-preview rendering (ahacode/render.py)."""

from rich.console import Console

from ahacode.render import diff_rows, edit_diff, lexer_for, tool_preview


def _text(renderable) -> str:
    """Render a Rich renderable to plain text (no colour) for assertions."""
    console = Console(width=80, color_system=None)
    with console.capture() as cap:
        console.print(renderable)
    return cap.get()


def test_write_preview_shows_real_newlines_not_a_repr():
    out = _text(tool_preview("write", {"path": "b.py", "content": "def f():\n    return 1"}))
    assert "def f():" in out
    assert "return 1" in out
    assert "\\n" not in out       # real line breaks, not an escaped repr
    assert "b.py" in out          # path header


def test_edit_preview_shows_minus_plus_diff():
    out = _text(tool_preview("edit", {"path": "b.py", "old_string": "a - b", "new_string": "a + b"}))
    assert "- a - b" in out
    assert "+ a + b" in out


def test_bash_preview_shows_the_command():
    assert "$ echo hi" in _text(tool_preview("bash", {"command": "echo hi"}))


def test_diff_rows_marks_replacements():
    rows = diff_rows("x = 1", "x = 2")
    assert ("-", "x = 1") in rows
    assert ("+", "x = 2") in rows


def test_lexer_for_extensions():
    assert lexer_for("a/b.py") == "python"
    assert lexer_for("readme.md") == "markdown"
    assert lexer_for("mystery.xyz") == "text"
