"""The assembled system prompts: identity first, shared rules, live environment."""

from dataclasses import replace

import pytest

from ahacode import config, prompts, subagent, workspace


@pytest.fixture(autouse=True)
def known_model():
    config.save(replace(config.DEFAULTS, name="qwen38"))


def test_act_system_has_the_key_sections():
    out = prompts.act_system()
    assert out.startswith("You are AhaCode")
    for marker in ("# Output", "# Editing code", "# Never", "Reply in the user's language",
                   "`path:line`", "IMPORTANT"):
        assert marker in out


def test_act_system_injects_live_environment():
    out = prompts.act_system()
    assert "# Environment" in out
    assert str(workspace.PROJECT_ROOT) in out
    assert "qwen38" in out


def test_act_system_has_no_few_shot_examples():
    assert "<example>" not in prompts.act_system()


def test_subagent_inherits_the_coding_rules():
    """A child holds the same write/edit/bash tools as the parent, so it carries the
    same rules — one shared constant, so the two cannot drift apart."""
    assert prompts.subagent_system() == (
        f"{prompts.IDENTITY}\n\n{prompts.SUBAGENT_SYSTEM}\n\n{prompts.CODING_RULES}"
    )
    assert prompts.CODING_RULES in prompts.act_system()
    assert "scratchpad" in prompts.subagent_system()


def test_subagent_run_uses_the_assembled_prompt():
    """subagent.run must resolve subagent_system() at call time, not freeze the bare
    framing constant as a default argument."""
    seen = {}

    def fake_stream(messages, specs):
        seen["system"] = messages[0]["content"]
        return iter(())

    subagent.run("do a thing", emit=lambda e: None, stream=fake_stream, registry={})
    assert seen["system"] == prompts.subagent_system()


def test_plan_system_demands_executable_steps():
    out = prompts.plan_system()
    assert out == f"{prompts.IDENTITY}\n\n{prompts.PLAN_SYSTEM}"
    assert "PLAN MODE" in out and "EXECUTABLE" in out
    for marker in ("imperative verb", "artifact"):
        assert marker in out


def test_every_system_prompt_opens_with_the_same_identity():
    """A prompt that skips the identity leaves a local model to answer from its
    training data about who it is."""
    for out in (prompts.act_system(), prompts.plan_system(), prompts.subagent_system()):
        assert out.startswith(prompts.IDENTITY)
    assert "AhaCode" in prompts.IDENTITY and "cyh" in prompts.IDENTITY
