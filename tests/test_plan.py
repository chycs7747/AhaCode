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
