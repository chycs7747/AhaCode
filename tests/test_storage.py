from ahacode import storage


def test_append_and_load_roundtrip(tmp_path):
    """What we save must come back identical (roundtrip check)."""
    path = storage.new_session_path(base_dir=tmp_path)
    storage.append_message(path, {"role": "user", "content": "hello"})
    storage.append_message(path, {"role": "assistant", "content": "hi there"})

    loaded = storage.load_messages(path)
    assert loaded == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_load_missing_file_returns_empty(tmp_path):
    assert storage.load_messages(tmp_path / "missing.jsonl") == []


def test_latest_session_picks_newest(tmp_path):
    (tmp_path / "2026-08-17_100000.jsonl").write_text("{}\n")
    (tmp_path / "2026-08-18_090000.jsonl").write_text("{}\n")
    assert storage.latest_session(base_dir=tmp_path).name == "2026-08-18_090000.jsonl"


def test_latest_session_empty_dir(tmp_path):
    assert storage.latest_session(base_dir=tmp_path) is None
