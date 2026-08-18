from ahacode.session import ChatSession


def test_messages_accumulate_in_order():
    s = ChatSession()
    s.add_user("hello")
    s.add_assistant("hi there")
    s.add_user("how is the weather?")
    assert [m["role"] for m in s.messages] == ["user", "assistant", "user"]
    assert s.messages[0]["content"] == "hello"


def test_sessions_are_independent():
    """Histories must never leak between sessions (default_factory check)."""
    a, b = ChatSession(), ChatSession()
    a.add_user("only in A")
    assert b.messages == []
