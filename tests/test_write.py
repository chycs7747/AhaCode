from ahacode.tools import write


def test_write_creates_file_and_parent_dirs(tmp_path):
    target = tmp_path / "sort" / "results" / "out.txt"
    msg = write.WRITE.execute({"path": str(target), "content": "hello\nworld"})
    assert target.read_text(encoding="utf-8") == "hello\nworld"
    assert "wrote" in msg


def test_write_requires_approval():
    assert write.WRITE.requires_approval is True
