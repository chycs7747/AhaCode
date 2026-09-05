"""The assembled system prompts: identity first, shared rules, live environment."""

from dataclasses import replace

import pytest

from ahacode import config, prompts, subagent, workspace


@pytest.fixture(autouse=True)
def known_model():
    config.save(replace(config.DEFAULTS, name="qwen38"))


def test_act_system_has_the_key_sections():
    out = prompts.system.act()
    assert out.startswith("You are AhaCode")
    for marker in ("# Output", "# Editing code", "# Never", "Reply in the user's language",
                   "`path:line`", "IMPORTANT"):
        assert marker in out


def test_act_system_injects_live_environment():
    out = prompts.system.act()
    assert "# Environment" in out
    assert str(workspace.PROJECT_ROOT) in out
    assert "qwen38" in out


def test_act_system_has_no_few_shot_examples():
    assert "<example>" not in prompts.system.act()


def test_subagent_inherits_the_coding_rules():
    """A child holds the same write/edit/bash tools as the parent, so it carries the
    same rules — one shared constant, so the two cannot drift apart."""
    assert prompts.system.subagent() == (
        f"{prompts.system.IDENTITY}\n\n{prompts.system.SUBAGENT_ROLE}\n\n{prompts.system.CODING_RULES}"
    )
    assert prompts.system.CODING_RULES in prompts.system.act()
    assert "scratchpad" in prompts.system.subagent()


def test_subagent_run_uses_the_assembled_prompt():
    """subagent.run must resolve system.subagent() at call time, not freeze the bare
    framing constant as a default argument."""
    seen = {}

    def fake_stream(messages, specs):
        seen["system"] = messages[0]["content"]
        return iter(())

    subagent.run("do a thing", emit=lambda e: None, stream=fake_stream, registry={})
    assert seen["system"] == prompts.system.subagent()


def test_plan_system_demands_executable_steps():
    out = prompts.system.plan()
    assert out == f"{prompts.system.IDENTITY}\n\n{prompts.system.PLAN_MODE}"
    assert "PLAN MODE" in out and "EXECUTABLE" in out
    for marker in ("imperative verb", "artifact"):
        assert marker in out


def test_every_system_prompt_opens_with_the_same_identity():
    """A prompt that skips the identity leaves a local model to answer from its
    training data about who it is."""
    for out in (prompts.system.act(), prompts.system.plan(), prompts.system.subagent()):
        assert out.startswith(prompts.system.IDENTITY)
    assert "AhaCode" in prompts.system.IDENTITY and "cyh" in prompts.system.IDENTITY


def test_handoff_names_the_plan_file():
    """The impl session's first turn points at this session's plan, so the model
    reads the plan rather than reconstructing it."""
    assert "plans/x.md" in prompts.injected.handoff("plans/x.md")


def test_handoff_carries_one_gap_policy():
    """The impl session hears one rule for a plan that turns out wrong — hand it back
    (stop) or rewrite its own checklist (adapt) — never both."""
    stop = prompts.injected.handoff("plans/x.md")
    adapt = prompts.injected.handoff("plans/x.md", gap=prompts.injected.GAP_ADAPT)
    assert "plans/x.md" in stop and "plans/x.md" in adapt
    assert "do not start any other step" in stop and "cancel" not in stop
    assert "cancel" in adapt and "todo_write" in adapt
    assert "do not start any other step" not in adapt


def test_auto_continue_matches_the_policy_and_owns_no_status_rules():
    """The status choreography lives on todo_write alone; the continue turn only says
    which way to go when a step is stuck."""
    stop = prompts.injected.auto_continue()
    adapt = prompts.injected.auto_continue(gap=prompts.injected.GAP_ADAPT)
    assert "stop" in stop and "cancel" not in stop
    assert "cancel" in adapt and "stop" not in adapt
    for text in (stop, adapt):
        assert "in_progress" not in text


def test_turn_cap_and_interruption_use_the_harness_vocabulary():
    """A "step" is a plan step; the cap is on turns. The interruption notice is a plain
    user turn like the others, with no tag of its own."""
    assert "turn limit" in prompts.injected.MAX_TURNS
    assert "step limit" not in prompts.injected.MAX_TURNS
    assert not prompts.injected.INTERRUPTED.startswith("[")
