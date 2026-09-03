from ahacode.session import ChatSession


def test_messages_accumulate_in_order():
    s = ChatSession()
    s.add_user("hello")
    s.add_user("how is the weather?")
    assert [m["content"] for m in s.messages] == ["hello", "how is the weather?"]
    assert s.messages[0]["role"] == "user"


def test_sessions_are_independent():
    """Histories must never leak between sessions (default_factory check)."""
    a, b = ChatSession(), ChatSession()
    a.add_user("only in A")
    assert b.messages == []
