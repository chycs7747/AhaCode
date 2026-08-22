import pytest

from ahacode.tools import edit


def test_edit_replaces_unique_snippet(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\ny = 2\n", encoding="utf-8")
    msg = edit.EDIT.execute({"path": str(f), "old_string": "y = 2", "new_string": "y = 3"})
    assert f.read_text(encoding="utf-8") == "x = 1\ny = 3\n"
    assert "edited" in msg


def test_edit_errors_when_not_found(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        edit.EDIT.execute({"path": str(f), "old_string": "nope", "new_string": "z"})


def test_edit_errors_when_ambiguous(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("dup\ndup\n", encoding="utf-8")
    with pytest.raises(ValueError):
        edit.EDIT.execute({"path": str(f), "old_string": "dup", "new_string": "x"})


def test_edit_requires_approval():
    assert edit.EDIT.requires_approval is True
