"""The structural plan runner: given an ordered plan, it delegates each step to a
fresh sub-agent in order and threads each concise result forward. Tested offline
with a fake delegate — no gateway, no UI — the same way subagent/agent are tested.
"""

from ahacode import orchestrator


def test_runs_each_step_in_order():
    seen = []

    def delegate(prompt, description):
        seen.append(description)
        return f"result of {description}"

    res = orchestrator.run_plan("solve X", ["design", "implement", "verify"], delegate)
    assert seen == ["design", "implement", "verify"]          # order preserved
    assert [p.description for p in res.phases] == ["design", "implement", "verify"]
    assert res.result == "result of verify"                   # final = last phase


def test_threads_prior_results_forward():
    """Each step's prompt carries the overall task + every earlier step's result,
    but NOT the earlier steps' reasoning — that is the anti-spiral property."""
    prompts = []

    def delegate(prompt, description):
        prompts.append(prompt)
        return f"[{description} done]"

    orchestrator.run_plan("BIG TASK", ["a", "b", "c"], delegate)

    # phase 1 sees the task and its own step, but no prior results section.
    assert "BIG TASK" in prompts[0] and "# Your phase\na" in prompts[0]
    assert "Results of earlier phases" not in prompts[0]
    # phase 2 sees phase 1's result; phase 3 sees both.
    assert "[a done]" in prompts[1]
    assert "[a done]" in prompts[2] and "[b done]" in prompts[2]


def test_empty_plan_yields_empty_result():
    res = orchestrator.run_plan("nothing to do", [], lambda p, d: "unused")
    assert res.phases == [] and res.result == ""


def test_cancellation_stops_between_phases():
    """is_cancelled is checked before each phase, so a mid-plan cancel runs no
    further sub-agents (the harness stays cooperative, like the agent loop)."""
    ran = []

    def delegate(prompt, description):
        ran.append(description)
        return "ok"

    # cancel after the first phase has run.
    res = orchestrator.run_plan(
        "task", ["one", "two", "three"], delegate,
        is_cancelled=lambda: len(ran) >= 1,
    )
    assert ran == ["one"]                       # stopped before phase 2
    assert res.result == "ok"                   # phase 1's result still returned


def test_phase_prompt_shape():
    prior = [orchestrator.PhaseResult("design", "use a heap")]
    p = orchestrator._phase_prompt("the task", "implement it", prior)
    assert "# Overall task\nthe task" in p
    assert "# Your phase\nimplement it" in p
    assert "## design\nuse a heap" in p
    assert "concise, self-contained result" in p
