"""A readable transcript beside the JSONL: the conversation as it appeared on
screen, each turn stamped with what it cost."""

import pytest

from ahacode import client, config, storage
from ahacode.app import AhaCodeApp
from ahacode.events import TextDelta
from ahacode.widgets.prompt_input import PromptInput

METRICS = {"prompt": 21374, "gen": 712, "gen_seconds": 25.0, "ttft": 2.0, "model": "m"}


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(storage, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(client, "list_models", lambda *a, **k: ["m"])
    monkeypatch.setattr(client, "complete", lambda m: "")
    monkeypatch.setattr(client, "stream_chat",
                        lambda m, tools=None: iter([TextDelta("답변입니다")]))
    monkeypatch.setattr(AhaCodeApp, "generate_title", lambda self, *a, **k: None)


def test_a_turn_reads_back_with_its_cost(tmp_path):
    out = tmp_path / "t.md"
    storage.append_turn(out, user="이거 고쳐줘", answer="고쳤습니다",
                        tools=["🔧 read · app.py", "🔧 bash · pytest"],
                        metrics=METRICS, stamp="03:41:12")
    text = out.read_text(encoding="utf-8")
    assert "03:41:12 · 사용자" in text and "이거 고쳐줘" in text
    assert "고쳤습니다" in text
    assert "🔧 read · app.py" in text and "🔧 bash · pytest" in text
    assert "TTFT 2.0초" in text and "28.5 tok/s" in text and "21,374 토큰" in text


def test_turns_accumulate(tmp_path):
    """A session is a log; the second turn must not replace the first."""
    out = tmp_path / "t.md"
    storage.append_turn(out, user="첫 질문", answer="첫 답", tools=[],
                        metrics=METRICS, stamp="01:00:00")
    storage.append_turn(out, user="둘째 질문", answer="둘째 답", tools=[],
                        metrics=METRICS, stamp="02:00:00")
    text = out.read_text(encoding="utf-8")
    assert text.count("· 사용자") == 2 and "첫 질문" in text and "둘째 질문" in text
    assert text.count("# t") == 1          # the title header is written once


def test_a_turn_that_produced_nothing_writes_nothing(tmp_path):
    out = tmp_path / "t.md"
    storage.append_turn(out, user="", answer="", tools=[], metrics={})
    assert not out.exists()


def test_metrics_without_output_are_omitted():
    assert storage.format_metrics({"gen": 0, "gen_seconds": 0}) == ""
    assert "tok/s" in storage.format_metrics(METRICS)


@pytest.mark.asyncio
async def test_asking_a_question_writes_the_transcript():
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app.query_one("#prompt", PromptInput).text = "리듬게임 만들어줘"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.pause()
        out = storage.transcript_path(app.session_path)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "리듬게임 만들어줘" in text and "답변입니다" in text


@pytest.mark.asyncio
async def test_one_question_is_not_repeated_over_every_round():
    """A turn can take several tool rounds; the question is asked once."""
    app = AhaCodeApp()
    async with app.run_test() as pilot:
        app._turn_question = "한 번만"
        for _ in range(2):
            app.post_message(AhaCodeApp.ResponseComplete(
                [{"role": "assistant", "content": "라운드"}], "", 1, METRICS))
            await pilot.pause()
            await pilot.pause()
        text = storage.transcript_path(app.session_path).read_text(encoding="utf-8")
    assert text.count("한 번만") == 1 and text.count("라운드") == 2
