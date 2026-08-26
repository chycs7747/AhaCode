"""plan_submit: the model's explicit "planning is done" — validated, then written
to plans/{session}.md by the harness."""

from types import SimpleNamespace

import pytest

from ahacode import storage, tools
from ahacode.tools import plan_submit


@pytest.fixture(autouse=True)
def plans_in_tmp(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "PLANS_DIR", tmp_path / "plans")


def _ctx(tmp_path, name="2026-08-26_120000"):
    return SimpleNamespace(session_path=tmp_path / f"{name}.jsonl")


def _submit(tmp_path, **args):
    return plan_submit.PLAN_SUBMIT.execute(args, _ctx(tmp_path))


# --- validation: the reasons come back in words the model can act on -----------

def test_empty_steps_are_rejected(tmp_path):
    with pytest.raises(plan_submit.PlanRejected, match="steps is empty"):
        _submit(tmp_path, summary="s", steps=[])


def test_blank_steps_count_as_empty(tmp_path):
    with pytest.raises(plan_submit.PlanRejected, match="steps is empty"):
        _submit(tmp_path, summary="s", steps=["  ", ""])


def test_non_executable_step_is_accepted_with_a_note(tmp_path):
    """The executability check is a heuristic, so it warns instead of refusing —
    as a hard gate it rejected a valid Korean step three times (it ended in print)."""
    out = _submit(tmp_path, summary="s", steps=[
        "Write solver.py with solve()",
        "Algorithm: subtree sums via DFS",     # a topic, not an action
        "Run the 4 examples and confirm 40/14/27/9",
    ])
    assert out.startswith("Plan saved to")
    assert "Note: step 2 'Algorithm" in out
    assert "step 1" not in out and "step 3" not in out
    assert (storage.PLANS_DIR / "2026-08-26_120000.md").exists()


def test_a_rejection_writes_nothing(tmp_path):
    with pytest.raises(plan_submit.PlanRejected):
        _submit(tmp_path, summary="s", steps=[])
    assert not (storage.PLANS_DIR).exists()


def test_check_and_note_are_reusable_without_a_context():
    assert plan_submit.check([]) is not None
    assert plan_submit.check(["Algorithm: idea"]) is None       # accepted…
    assert "step 1" in plan_submit.note(["Algorithm: idea"])   # …with a note
    assert plan_submit.note(["Write x.py with main()"]) == ""


# --- success: the file is the plan -----------------------------------------------

def test_success_writes_the_plan_file_named_after_the_session(tmp_path):
    out = _submit(
        tmp_path, summary="Fix the parser",
        steps=["Write parser.py with parse()", "Run pytest and confirm 12 passed"],
        validation=["uv run pytest tests/test_parser.py"],
        body="The old parser choked on empty lines.",
    )
    path = storage.PLANS_DIR / "2026-08-26_120000.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Fix the parser\n")
    assert "1. [ ] Write parser.py with parse()" in text
    assert "2. [ ] Run pytest and confirm 12 passed" in text
    assert "## Validation\n\n- uv run pytest tests/test_parser.py" in text
    assert "## Notes\n\nThe old parser choked on empty lines." in text
    # the result names the file and tells the model the turn is over
    assert "2 steps" in out
    assert "stop here" in out


def test_optional_sections_are_omitted_when_empty(tmp_path):
    _submit(tmp_path, summary="s", steps=["Write a.py with f()"])
    text = (storage.PLANS_DIR / "2026-08-26_120000.md").read_text(encoding="utf-8")
    assert "## Validation" not in text and "## Notes" not in text


def test_resubmission_replaces_the_previous_plan(tmp_path):
    _submit(tmp_path, summary="v1", steps=["Write a.py with f()"])
    _submit(tmp_path, summary="v2", steps=["Write b.py with g()", "Run b.py and confirm 3"])
    text = (storage.PLANS_DIR / "2026-08-26_120000.md").read_text(encoding="utf-8")
    assert text.startswith("# v2\n") and "a.py" not in text


def test_without_a_session_there_is_nowhere_to_save(tmp_path):
    with pytest.raises(plan_submit.PlanRejected, match="no session"):
        plan_submit.PLAN_SUBMIT.execute(
            {"summary": "s", "steps": ["Write a.py with f()"]}, SimpleNamespace()
        )


# --- placement: plan mode only, never a sub-agent --------------------------------

def test_tool_is_not_in_the_global_registry():
    assert "plan_submit" not in tools.REGISTRY
    assert "plan_submit" not in tools.registry_for(0, 1)


def test_tool_contract():
    t = plan_submit.PLAN_SUBMIT
    assert t.wants_ctx is True            # needs the session to name the file
    assert t.requires_approval is False   # writing plans/ is the harness's own business
    assert set(t.parameters["required"]) == {"summary", "steps"}
    spec = tools.specs({"plan_submit": t})[0]["function"]
    assert spec["name"] == "plan_submit" and "steps" in spec["parameters"]["properties"]
