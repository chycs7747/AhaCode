import pytest

from ahacode.tools import plan


def test_todo_write_formats_a_checklist():
    out = plan.TODO_WRITE.execute({"items": [
        {"content": "reproduce the bug", "status": "done"},
        {"content": "write the fix", "status": "in_progress"},
        {"content": "add a test", "status": "pending"},
        {"content": "no-status defaults to pending"},
    ]})
    assert out.splitlines() == [
        "☑ reproduce the bug",
        "▶ write the fix",
        "☐ add a test",
        "☐ no-status defaults to pending",
    ]


def test_todo_write_is_read_only():
    assert plan.TODO_WRITE.requires_approval is False


def test_todo_write_empty():
    assert plan.TODO_WRITE.execute({"items": []}) == "(empty plan)"


# --- executable-step check -------------------------------------------------
# The regression this guards: a plan step that states an idea instead of an action
# is handed to a fresh sub-agent that can only finish by calling a tool, so it turns
# `write` into a scratchpad. See prompts.PLAN_SYSTEM for the full reasoning.

@pytest.mark.parametrize("step", [
    "Implement solution() using two-pointers over a pre-sorted subtree-sum array",
    "Verify all 4 examples pass (30/14/27/9)",
    "Write answer.py with solution() + __main__",
    "Run the suite and report failures",
    "solution() 작성 (루트 탐색, 반복 post-order로 재귀 깊이 회피)",  # verb-final, aside stripped
    "엣지 케이스 검증: k=1, k=n",                                    # verb ends the head clause
    "예시 4개 실행해서 결과 확인하기",
])
def test_actionable_steps_pass(step):
    assert not plan.non_actionable(step)


@pytest.mark.parametrize("step", [
    # The rejection that cost three planning turns: these all ended in print/출력.
    "main()에서 결과를 표로 출력한다",
    "알고리즘별 소요시간 테이블을 print",
    "결과를 results.json에 저장",
])
def test_korean_output_verbs_are_actionable(step):
    assert not plan.non_actionable(step)


@pytest.mark.parametrize("step", [
    # The exact step that produced 366 comment lines of derivation.
    "Algorithm: find root, compute subtree sums; answer(k) = min over X of max(X, ...)",
    "Performance: n=10,000 nodes, depth-10,000 one-sided chain",
    "알고리즘: 서브트리 합의 최소 최대값",
    "answer(k) = min over X",
    "",
])
def test_non_actionable_steps_are_flagged(step):
    assert plan.non_actionable(step)


def test_cancelled_is_a_fourth_state_and_counts_as_finished():
    assert plan.STATUSES == ("pending", "in_progress", "done", "cancelled")
    assert plan.mark("cancelled") == "✗"
    items = [{"content": "a", "status": "done"}, {"content": "b", "status": "cancelled"},
             {"content": "c", "status": "in_progress"}, {"content": "d"}]
    assert [it["content"] for it in plan.unfinished(items)] == ["c", "d"]


def test_todo_write_describes_the_status_discipline():
    desc = plan.TODO_WRITE.description
    assert "never on intent" in desc and "exactly ONE in_progress" in desc
    assert "cancelled" in plan.TODO_WRITE.parameters["properties"]["items"]["items"]["properties"]["status"]["enum"]


# --- the shape of `items` as it actually arrives -------------------------------

def test_items_sent_as_a_json_string_are_recovered():
    """Seen live: the model JSON-encoded the list twice, so `items` arrived as one
    string and the panel drew a checklist of single characters."""
    raw = '[{"content": "Write sort.py", "status": "done"}, {"content": "Run it"}]'
    items, note = plan.coerce_items(raw)
    assert [it["content"] for it in items] == ["Write sort.py", "Run it"]
    assert items[0]["status"] == "done" and items[1]["status"] == "pending"
    assert "string" in note


def test_bare_strings_become_pending_items():
    items, note = plan.coerce_items(["Write sort.py", " Run it "])
    assert items == [{"content": "Write sort.py", "status": "pending"},
                     {"content": "Run it", "status": "pending"}]
    assert "objects" in note


def test_a_string_that_is_not_json_yields_nothing_with_a_reason():
    items, note = plan.coerce_items("just some words")
    assert items == [] and "must be a JSON array" in note


def test_well_formed_items_pass_through_without_a_note():
    items, note = plan.coerce_items([{"content": "a", "status": "in_progress"}])
    assert items == [{"content": "a", "status": "in_progress"}] and note == ""


def test_unknown_status_is_read_as_pending():
    items, _ = plan.coerce_items([{"content": "a", "status": "finished?"}])
    assert items[0]["status"] == "pending"


def test_todo_write_tells_the_model_when_it_repaired_the_shape():
    out = plan.TODO_WRITE.execute({"items": '["Write sort.py"]'})
    assert out.splitlines()[0] == "☐ Write sort.py"
    assert "note:" in out
