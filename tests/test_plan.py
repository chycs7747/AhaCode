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
