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


def test_header_roundtrip(tmp_path):
    p = storage.new_session_path(base_dir=tmp_path)
    h = storage.make_header(p.stem, kind="subagent", parent_id="root1",
                            depth=2, model="qwen3-4b", title="bench")
    storage.write_header(p, h)
    assert storage.read_header(p) == h


def test_load_messages_skips_header_line(tmp_path):
    p = storage.new_session_path(base_dir=tmp_path)
    storage.write_header(p, storage.make_header(p.stem, title="t"))
    storage.append_message(p, {"role": "user", "content": "hi"})
    assert storage.load_messages(p) == [{"role": "user", "content": "hi"}]


def test_read_header_none_for_legacy_file(tmp_path):
    p = storage.new_session_path(base_dir=tmp_path)
    storage.append_message(p, {"role": "user", "content": "hi"})  # no header
    assert storage.read_header(p) is None


def test_list_sessions_reads_headers_and_synthesizes_legacy(tmp_path):
    p1 = tmp_path / "2026-01-01_000000.jsonl"
    storage.write_header(p1, storage.make_header(p1.stem, title="withheader"))
    p2 = tmp_path / "2026-01-02_000000.jsonl"
    storage.append_message(p2, {"role": "user", "content": "legacy hi"})  # no header

    sessions = storage.list_sessions(base_dir=tmp_path)
    assert [s["id"] for s in sessions] == ["2026-01-01_000000", "2026-01-02_000000"]
    assert sessions[0]["title"] == "withheader"
    assert sessions[1]["title"] == "legacy hi"  # synthesized from the first user message


def test_build_tree_nests_by_parent_id(tmp_path):
    sessions = [
        storage.make_header("root", kind="main"),
        storage.make_header("child1", parent_id="root", kind="subagent"),
        storage.make_header("child2", parent_id="root", kind="subagent"),
        storage.make_header("grand", parent_id="child1", kind="subagent"),
    ]
    roots = storage.build_tree(sessions)
    assert [r["id"] for r in roots] == ["root"]
    assert [c["id"] for c in roots[0]["children"]] == ["child1", "child2"]
    assert [c["id"] for c in roots[0]["children"][0]["children"]] == ["grand"]


def test_build_tree_orphan_becomes_root(tmp_path):
    sessions = [storage.make_header("x", parent_id="ghost")]  # parent not present
    assert [r["id"] for r in storage.build_tree(sessions)] == ["x"]


def test_set_title_and_read_session_meta_last_wins(tmp_path):
    p = storage.new_session_path(base_dir=tmp_path)
    storage.write_header(p, storage.make_header(p.stem, title=""))
    storage.append_message(p, {"role": "user", "content": "hi"})
    storage.set_title(p, "first title")
    storage.set_title(p, "final title")  # a later title overrides

    meta = storage.read_session_meta(p)
    assert meta["title"] == "final title"
    assert storage.load_messages(p) == [{"role": "user", "content": "hi"}]  # title lines skipped


def test_read_session_meta_none_for_legacy(tmp_path):
    p = storage.new_session_path(base_dir=tmp_path)
    storage.append_message(p, {"role": "user", "content": "hi"})  # no header
    assert storage.read_session_meta(p) is None


def test_new_session_path_unique_under_threads(tmp_path):
    """Concurrent callers (parallel sub-agent spawns in the same second) each get a
    distinct path — the atomic claim stops two children clobbering one file."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as ex:
        paths = list(ex.map(lambda _: storage.new_session_path(base_dir=tmp_path), range(8)))
    assert len(set(paths)) == 8
