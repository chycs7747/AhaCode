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


# --- markdown line breaks ----------------------------------------------------
# CommonMark folds a single newline into a space, so the model's three lines became
# one re-wrapped block — visibly denser than the plain-Text bubbles beside it.

def test_prose_line_breaks_survive_markdown():
    from ahacode.text import keep_line_breaks
    out = keep_line_breaks("첫 줄입니다\n둘째 줄입니다\n셋째 줄입니다")
    assert out == "첫 줄입니다  \n둘째 줄입니다  \n셋째 줄입니다"


def test_a_blank_line_still_separates_paragraphs():
    from ahacode.text import keep_line_breaks
    out = keep_line_breaks("문단 하나\n\n문단 둘")
    assert out == "문단 하나\n\n문단 둘"       # nothing appended before a blank line


def test_code_fences_are_untouched():
    """A hard break inside code would show up as trailing whitespace in the code."""
    from ahacode.text import keep_line_breaks
    src = "설명입니다\n\n```python\na = 1\nb = 2\n```"
    assert keep_line_breaks(src) == src


def test_block_constructs_are_left_alone():
    """Tables, headings, lists and quotes are their own blocks already; a trailing
    hard break there is either meaningless or changes how they parse."""
    from ahacode.text import keep_line_breaks
    for src in ("| a | b |\n|---|---|\n| 1 | 2 |",
                "# 제목\n## 부제목",
                "- 하나\n- 둘",
                "1. 하나\n2. 둘",
                "> 인용\n> 계속",
                "    indented code\n    more code"):
        assert keep_line_breaks(src) == src, src


def test_an_existing_hard_break_is_not_doubled():
    from ahacode.text import keep_line_breaks
    assert keep_line_breaks("줄 하나  \n줄 둘") == "줄 하나  \n줄 둘"
    assert keep_line_breaks("줄 하나\\\n줄 둘") == "줄 하나\\\n줄 둘"


def test_the_answer_bubble_renders_the_authors_breaks():
    """End to end through the widget: the rendered Markdown keeps three lines."""
    from rich.console import Console

    from ahacode.widgets.chatbox import Chatbox
    box = Chatbox("첫 줄입니다\n둘째 줄입니다\n셋째 줄입니다", role="assistant", markdown=True)
    console = Console(width=60)
    with console.capture() as cap:
        console.print(box.render())
    body = [ln.rstrip() for ln in cap.get().rstrip("\n").split("\n") if ln.strip()]
    assert body == ["첫 줄입니다", "둘째 줄입니다", "셋째 줄입니다"]
