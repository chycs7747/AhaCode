"""todo_write: the model's checklist, and the plan vocabulary every consumer
derives from (statuses, glyphs, the executability heuristic)."""

from __future__ import annotations

import json
import re

from ahacode.tools.base import Tool

# The plan vocabulary, defined once: the tool's schema, the pinned panel and the
# result file all derive from it.
STATUS_MARKS = {
    "pending": "☐",
    "in_progress": "▶",
    "done": "☑",
    "cancelled": "✗",  # no longer needed; distinct from done so a plan stays honest
}
STATUSES = tuple(STATUS_MARKS)  # the enum the tool advertises, in display order
PENDING, IN_PROGRESS, DONE, CANCELLED = STATUSES
FINISHED = frozenset({DONE, CANCELLED})  # states that need no more work


def coerce_items(raw) -> tuple[list[dict], str]:
    """Normalise whatever the model sent as `items` into [{content, status}, …].

    Repairs the shapes models do send: the list JSON-encoded a second time (one
    string), a single object, and bare strings instead of objects.

    Args:
        raw: The tool argument as received.

    Returns:
        (items, note); the note tells the model the shape it should have sent, and
        is "" when nothing needed fixing.
    """
    note = ""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
            note = "items was sent as a string; send a JSON array of objects"
        except json.JSONDecodeError:
            return [], "items must be a JSON array of {content, status} objects"
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return [], "items must be a JSON array of {content, status} objects"
    out: list[dict] = []
    for it in raw:
        if isinstance(it, str):
            if it.strip():
                out.append({"content": it.strip(), "status": PENDING})
                note = note or "items held bare strings; send {content, status} objects"
        elif isinstance(it, dict) and str(it.get("content", "")).strip():
            status = it.get("status") or PENDING
            out.append({**it, "content": str(it["content"]).strip(),
                        "status": status if status in STATUS_MARKS else PENDING})
    return out, note


def unfinished(items: list[dict]) -> list[dict]:
    """The items still owed: neither done nor cancelled."""
    return [it for it in items if it.get("status", PENDING) not in FINISHED]


def mark(status: str | None) -> str:
    """The glyph for a status; an unknown or missing one reads as pending."""
    return STATUS_MARKS.get(status or PENDING, STATUS_MARKS[PENDING])


# --- executability check ---------------------------------------------------
# A plan step is carried out by a session whose only way to finish it is a tool
# call, so a step must read as a DOING step: imperative verb first in English, last
# in Korean. A heuristic that only ever warns.
_EN_VERBS = frozenset("""
add benchmark build check clean commit compare compute confirm convert create delete
deploy design document draft drop ensure extend extract find fix generate handle
implement improve install introduce list load make measure migrate move parse patch
port print profile prove read refactor remove rename render replace report reproduce
run save scan set setup show simplify solve split test time trace update upgrade
validate verify wire write
""".split())

_KO_TAIL = re.compile(
    r"(작성|구현|실행|검증|확인|측정|수정|추가|삭제|제거|정리|테스트|분석|비교|배포|생성|변경|리팩터|정의"
    r"|출력|print|반환|저장|계산|호출|등록|설치|적용|표시|기록|로드)"
    r"(하기|하라|한다|해라|할 것)?\s*[.!]?$"
)
_HANGUL = re.compile(r"[가-힣]")


def non_actionable(step: str) -> bool:
    """Whether a step reads as a topic rather than a task.

    Args:
        step: The step text.

    Returns:
        True when the step neither starts with an English verb nor ends with a
        Korean one (a trailing parenthetical or a ": detail" tail is ignored).
    """
    text = step.strip()
    if not text:
        return True
    if _HANGUL.search(text):
        core = text
        while True:
            stripped = re.sub(r"\s*[(（][^()（）]*[)）]\s*$", "", core)
            if stripped == core:
                break
            core = stripped
        head = core.split(":")[0].strip()
        return not (_KO_TAIL.search(core) or _KO_TAIL.search(head))
    first = re.split(r"[^A-Za-z]+", text.lower(), maxsplit=1)[0]
    return first not in _EN_VERBS


def _todo_write(args: dict) -> str:
    """Render the checklist back as the tool's result."""
    items, note = coerce_items(args.get("items", []))
    if not items:
        return f"(empty plan){' — ' + note if note else ''}"
    lines = [f"{mark(it.get('status'))} {it['content']}" for it in items]
    return "\n".join(lines) + (f"\n(note: {note})" if note else "")


TODO_WRITE = Tool(
    name="todo_write",
    description=(
        # The "plan first" nudge lives on the tool rather than in the always-on
        # prompt, so it does not bias every atomic question.
        "Record or update the task list. Send the full list each time. If the work "
        "splits into three or more steps, lay it out here BEFORE making any change.\n"
        "Status rules: pending → in_progress → done, or cancelled if no longer needed. "
        "Mark a step in_progress before starting it and keep exactly ONE in_progress "
        "while work remains. Mark done only after the work is actually complete, "
        "including any verification it needs — never on intent. If a step is blocked "
        "or only partly done, leave it in_progress and add a follow-up step that "
        "names the blocker. Update in real time; do not batch completions."
    ),
    parameters={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "description": "The full task list, in order",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": (
                                "One EXECUTABLE step, imperative: a verb plus a concrete "
                                "artifact or checkable outcome (\"Write solver.py with "
                                "solve()\", \"Run the 4 examples and confirm 40/14/27/9\"). "
                                "Not a topic, a formula, or an idea — those are not steps."
                            ),
                        },
                        "status": {
                            "type": "string",
                            "enum": list(STATUSES),
                        },
                    },
                    "required": ["content"],
                },
            },
        },
        "required": ["items"],
    },
    execute=_todo_write,
)
