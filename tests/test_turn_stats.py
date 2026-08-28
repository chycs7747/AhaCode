"""Throughput was computed for the status bar and then thrown away, so "how fast
has this been?" had no answer after the fact. It is recorded per turn now, and the
progress report carries the aggregate."""

import pytest

from ahacode import client, config, storage
from ahacode.app import AhaCodeApp
from ahacode.events import TextDelta


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(storage, "PLANS_DIR", tmp_path / "plans")
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(client, "list_models", lambda *a, **k: ["m"])
    monkeypatch.setattr(client, "complete", lambda m: "")
    monkeypatch.setattr(client, "stream_chat",
                        lambda m, tools=None: iter([TextDelta("hi")]))
    monkeypatch.setattr(AhaCodeApp, "generate_title", lambda self, *a, **k: None)


def test_stats_are_recorded_but_never_replayed(tmp_path):
    """The numbers ride in the session file, and load_messages must not hand them
    back to the model as conversation."""
    path = tmp_path / "s.jsonl"
    storage.append_message(path, {"role": "user", "content": "hi"})
    storage.append_stats(path, {"prompt": 1200, "gen": 300, "gen_seconds": 10.0,
                                "ttft": 0.5, "model": "m"})
    assert storage.load_messages(path) == [{"role": "user", "content": "hi"}]
    assert storage.read_stats(path)[0]["gen"] == 300


def test_the_aggregate_is_weighted_by_tokens_not_averaged_per_turn():
    """A 900-token turn and a 10-token turn are not equal evidence about speed."""
    rows = [
        {"gen": 900, "gen_seconds": 30.0, "ttft": 1.0},   # 30 tok/s
        {"gen": 10, "gen_seconds": 10.0, "ttft": 3.0},    # 1 tok/s
    ]
    out = storage.summarize_stats(rows)
    assert "910" in out and "40초" in out
    assert "22.8 tok/s" in out          # 910/40, not the mean of 30 and 1
    assert "3.0초" in out               # median ttft of the two


def test_no_stats_no_section():
    assert storage.summarize_stats([]) == ""
    assert storage.summarize_stats([{"gen": 0, "gen_seconds": 0}]) == ""


def test_the_report_carries_throughput(tmp_path):
    out = tmp_path / "r.md"
    storage.write_result(
        out, plan=tmp_path / "p.md", session_id="s", complete=False,
        items=[{"content": "step", "status": "done"}], summary="done it",
        throughput="2턴 · 생성 910 토큰 / 40초 = 평균 22.8 tok/s · TTFT 중앙값 3.0초",
    )
    text = out.read_text(encoding="utf-8")
    assert "## Throughput" in text and "22.8 tok/s" in text


def test_a_report_without_stats_omits_the_section(tmp_path):
    out = tmp_path / "r.md"
    storage.write_result(out, plan=tmp_path / "p.md", session_id="s", complete=True,
                         items=[{"content": "step", "status": "done"}], summary="")
    assert "## Throughput" not in out.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_finished_turn_writes_its_numbers():
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.post_message(AhaCodeApp.ResponseComplete(
            [{"role": "assistant", "content": "hi"}], "prompt 5 · gen 2", 5,
            {"prompt": 5, "gen": 2, "gen_seconds": 0.5, "ttft": 0.1, "model": "m"},
        ))
        await pilot.pause()
        await pilot.pause()
        rows = storage.read_stats(app.session_path)
    assert rows and rows[0]["gen"] == 2 and rows[0]["model"] == "m"
