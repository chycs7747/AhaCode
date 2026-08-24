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


def test_synthesize_reduces_phase_results_to_final():
    """With a synthesizer, the final result is a combined synthesis of ALL phases,
    not just the last phase's output (the reduce step)."""
    got = {}

    def synth(task, phases):
        got["task"] = task
        got["results"] = [p.result for p in phases]
        return "COMBINED ANSWER"

    res = orchestrator.run_plan(
        "big task", ["a", "b"], lambda p, d: f"result-{d}", synthesize=synth
    )
    assert res.result == "COMBINED ANSWER"          # reduced, not "result-b"
    assert got["task"] == "big task"
    assert got["results"] == ["result-a", "result-b"]  # synthesizer saw every phase
    assert [p.description for p in res.phases] == ["a", "b"]  # phases still recorded


def test_no_synthesize_keeps_last_phase_result():
    res = orchestrator.run_plan("t", ["a", "b"], lambda p, d: f"r-{d}")
    assert res.result == "r-b"  # map-only: last phase's result


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


def test_a_verbose_phase_result_is_capped_at_the_source():
    """A phase result is read twice — threaded into later phases AND by the
    synthesis — so an unbounded one is paid for many times over. Capping where it
    is produced bounds both readers with one change."""
    cap = orchestrator.MAX_RESULT_CHARS

    huge = "START" + "x" * 50_000 + "END"
    seen: list[str] = []

    def delegate(prompt, description):
        seen.append(prompt)
        return huge

    got = []
    out = orchestrator.run_plan(
        "task", ["one", "two"], delegate,
        synthesize=lambda t, phases: got.append(phases) or "final",
    )

    assert len(out.phases[0].result) < cap + 200
    assert "START" in out.phases[0].result and "END" in out.phases[0].result
    assert "elided" in out.phases[0].result
    # the second phase's prompt carries the capped result, not the raw 50k
    assert len(seen[1]) < cap + 1_000
    # and so does what the synthesis is handed
    assert all(len(p.result) < cap + 200 for p in got[0])


# --- incremental commit ----------------------------------------------------
# on_phase is the seam that lets the caller persist each phase as it lands. Before it,
# a long run held everything until the end: the parent session stayed empty (so it
# answered from a stale context) and a cancellation threw the finished work away.

def test_on_phase_fires_as_each_phase_lands():
    seen = []
    result = orchestrator.run_plan(
        "task", ["one", "two", "three"],
        delegate=lambda prompt, step: f"did {step}",
        on_phase=lambda phase: seen.append((phase.description, phase.result)),
    )
    assert seen == [("one", "did one"), ("two", "did two"), ("three", "did three")]
    assert [p.description for p in result.phases] == ["one", "two", "three"]


def test_on_phase_fires_before_the_next_phase_starts():
    """Ordering matters: the commit must happen between phases, not batched at the end,
    or a cancellation mid-run still loses the phase that just finished."""
    events = []
    def delegate(prompt, step):
        events.append(f"run:{step}")
        return step
    orchestrator.run_plan("t", ["a", "b"], delegate=delegate,
                          on_phase=lambda p: events.append(f"commit:{p.description}"))
    assert events == ["run:a", "commit:a", "run:b", "commit:b"]


def test_a_cancelled_run_keeps_the_phases_that_finished():
    """The regression: `if worker.is_cancelled: return` discarded a whole run, so
    asking the parent a question mid-run erased every completed phase."""
    committed = []
    stop = {"now": False}
    def delegate(prompt, step):
        if step == "two":
            stop["now"] = True  # user hits stop while phase two is running
        return f"did {step}"
    result = orchestrator.run_plan(
        "task", ["one", "two", "three"],
        delegate=delegate,
        is_cancelled=lambda: stop["now"],
        on_phase=lambda p: committed.append(p.description),
    )
    # phase two ran to completion, so it is kept; phase three never started
    assert committed == ["one", "two"]
    assert [p.description for p in result.phases] == ["one", "two"]


def test_cancelled_run_skips_the_synthesis():
    """A partial run must not be summarised as if it were the finished answer."""
    called = []
    result = orchestrator.run_plan(
        "task", ["one", "two"],
        delegate=lambda prompt, step: "x",
        is_cancelled=lambda: True,
        synthesize=lambda t, phases: called.append(t) or "SYNTH",
    )
    assert called == [] and result.result != "SYNTH"
