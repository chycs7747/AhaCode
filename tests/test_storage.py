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


def test_latest_session_skips_subagent(tmp_path):
    """Startup resumes the newest MAIN session, never a (newer) sub-agent child —
    landing in a depth-gated sub-agent session makes the task tool look missing."""
    main = tmp_path / "2026-08-23_100000.jsonl"
    storage.write_header(main, storage.make_header(main.stem, kind="main"))
    sub = tmp_path / "2026-08-23_110000.jsonl"  # newer, but a spawned child
    storage.write_header(sub, storage.make_header(
        sub.stem, kind="subagent", parent_id=main.stem, depth=1))
    assert storage.latest_session(base_dir=tmp_path).name == main.name


def test_header_relation_roundtrip(tmp_path):
    p = storage.new_session_path(base_dir=tmp_path)
    h = storage.make_header(p.stem, kind="impl", parent_id="plan1", relation="handoff")
    storage.write_header(p, h)
    assert storage.read_session_meta(p)["relation"] == "handoff"


def test_header_relation_defaults_to_none():
    assert storage.make_header("root")["relation"] is None


def test_header_without_relation_field_still_loads(tmp_path):
    # A header written before the field existed: relation is simply absent.
    p = storage.new_session_path(base_dir=tmp_path)
    old = {k: v for k, v in storage.make_header(p.stem, kind="subagent").items()
           if k != "relation"}
    storage.write_header(p, old)
    meta = storage.read_session_meta(p)
    assert meta is not None and meta.get("relation") is None
    assert storage.build_tree([meta])[0]["id"] == p.stem


# --- plan files --------------------------------------------------------------

def test_plan_path_is_named_after_the_session(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "PLANS_DIR", tmp_path / "plans")
    assert storage.plan_path(tmp_path / "2026-08-26_1200.jsonl") == tmp_path / "plans" / "2026-08-26_1200.md"


def test_write_plan_creates_the_directory_and_renders_markdown(tmp_path):
    path = tmp_path / "plans" / "s.md"
    storage.write_plan(path, summary="Goal", steps=["Write a.py with f()"],
                       validation=["uv run pytest"], body="notes")
    assert path.read_text(encoding="utf-8") == (
        "# Goal\n\n## Steps\n\n1. Write a.py with f()\n\n"
        "## Validation\n\n- uv run pytest\n\n## Notes\n\nnotes\n"
    )


def test_display_path_is_project_relative_inside_the_root():
    assert storage.display_path(storage.PROJECT_ROOT / "plans" / "x.md") == "plans/x.md"


def test_result_file_sits_beside_the_plan_and_snapshots_the_checklist(tmp_path):
    plan = tmp_path / "plans" / "s1.md"
    out = storage.result_path(plan)
    assert out == tmp_path / "plans" / "s1.result.md"
    storage.write_result(out, plan=plan, session_id="s2", complete=False, summary="halfway",
                         items=[{"content": "a", "status": "done"}, {"content": "b", "status": "in_progress"},
                                {"content": "c", "status": "cancelled"}, {"content": "d"}])
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# 진행 중 2/4 — s1.md\n")
    assert "- session: s2" in text
    assert "☑ a\n▶ b\n✗ c\n☐ d" in text
    assert "## Latest summary\n\nhalfway" in text
    storage.write_result(out, plan=plan, session_id="s2", complete=True, summary="",
                         items=[{"content": "a", "status": "done"}])
    assert out.read_text(encoding="utf-8").startswith("# 완료 — s1.md\n")
