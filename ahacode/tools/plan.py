"""todo_write: record a structured plan (task list). Read-only in effect — it
just formats the steps back as a checklist, so it is safe in plan mode.

Stateless by design: like Claude Code's TodoWrite, the model sends the whole list
each time and we render it fresh. No shared-state mutation from the worker thread,
so no ctx is needed (that seam is deferred until subagents require it)."""

from __future__ import annotations

import re

from ahacode.tools.base import Tool

# The plan vocabulary, defined ONCE. Three places used to spell it out separately —
# this dict, the same dict again in widgets/todo_panel.py, and the schema's enum below
# — so a fourth status (or a changed glyph) had to be added in three files and would
# silently disagree if it was not. Everything downstream is derived from here: the
# marks, the enum the model is allowed to send, the default, and the terminal state.
STATUS_MARKS = {
    "pending": "☐",
    "in_progress": "▶",
    "done": "☑",
}
STATUSES = tuple(STATUS_MARKS)      # the enum the tool advertises, in display order
PENDING, IN_PROGRESS, DONE = STATUSES  # named keys, so no caller spells them literally


def mark(status: str | None) -> str:
    """The glyph for a status; an unknown or missing one reads as not-started."""
    return STATUS_MARKS.get(status or PENDING, STATUS_MARKS[PENDING])

# --- executability check ---------------------------------------------------
# Why a plan step must be a DOING step: /run hands each one to a fresh sub-agent
# whose only way to finish is a tool call. A step that states a fact or an idea has
# no legal completion, so the child reaches for `write` — the one tool that takes
# free text — and files its derivation as source comments (measured: 366 comment
# lines out of 467 for the step "Algorithm: find root, compute subtree sums; …").
#
# The rule is Claude Code's TodoWrite convention: content in imperative form. English
# puts the verb first, Korean puts it last, so each is checked at its own end. This is
# a HEURISTIC and only ever warns — an unlisted verb costs one line of text, never a
# blocked run.
_EN_VERBS = frozenset("""
add benchmark build check clean commit compare compute confirm convert create delete
deploy design document draft drop ensure extend extract find fix generate handle
implement improve install introduce list load make measure migrate move parse patch
port print profile prove read refactor remove rename render replace report reproduce
run save scan set setup show simplify solve split test time trace update upgrade
validate verify wire write
""".split())

# Korean is verb-final: the step ends in the action ("solution() 작성", "예시 4개 검증").
_KO_TAIL = re.compile(
    r"(작성|구현|실행|검증|확인|측정|수정|추가|삭제|제거|정리|테스트|분석|비교|배포|생성|변경|리팩터|정의"
    r"|출력|print|반환|저장|계산|호출|등록|설치|적용|표시|기록|로드)"
    r"(하기|하라|한다|해라|할 것)?\s*[.!]?$"
)
_HANGUL = re.compile(r"[가-힣]")


def non_actionable(step: str) -> bool:
    """True if `step` does not read as something a sub-agent could carry out.

    Imperative English starts with the verb; Korean ends with it. Anything else — a
    bare noun phrase ("Algorithm: …", "Performance: …"), a formula, a statement — is
    a topic, not a task.
    """
    text = step.strip()
    if not text:
        return True
    if _HANGUL.search(text):
        # A trailing parenthetical is detail, not the action: "solution() 작성 (루트
        # 탐색…)" is verb-final once the aside is dropped. Strip repeatedly so nested
        # or stacked asides all come off.
        core = text
        while True:
            stripped = re.sub(r"\s*[(（][^()（）]*[)）]\s*$", "", core)
            if stripped == core:
                break
            core = stripped
        # "엣지 케이스 검증: k=1, k=n" — the verb ends the head clause and the colon
        # introduces detail, so accept a verb-final head too.
        head = core.split(":")[0].strip()
        return not (_KO_TAIL.search(core) or _KO_TAIL.search(head))
    first = re.split(r"[^A-Za-z]+", text.lower(), maxsplit=1)[0]
    return first not in _EN_VERBS


def _todo_write(args: dict) -> str:
    items = args.get("items", [])
    lines = [f"{mark(it.get('status'))} {it['content']}" for it in items]
    return "\n".join(lines) or "(empty plan)"


TODO_WRITE = Tool(
    name="todo_write",
    description=(
        # Where the "plan first" nudge lives. Deliberately here and not in the
        # always-on system prompt: attached to the tool, the model reads it while
        # deciding to call this, instead of it biasing every atomic question.
        "Record or update the plan as a task list. Call this to lay out the steps "
        "before acting; send the full list each time (status: pending/in_progress/done). "
        "If the work splits into three or more steps, lay the plan out here BEFORE "
        "making any change."
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
                            # The shape is enforced here as well as in PLAN_SYSTEM: the
                            # model reads this while filling the argument in.
                            "description": (
                                "One EXECUTABLE step, imperative: a verb plus a concrete "
                                "artifact or checkable outcome (\"Write solver.py with "
                                "solve()\", \"Run the 4 examples and confirm 40/14/27/9\"). "
                                "Not a topic, a formula, or an idea — those are not steps."
                            ),
                        },
                        "status": {
                            "type": "string",
                            "enum": list(STATUSES),  # derived, never re-typed
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
