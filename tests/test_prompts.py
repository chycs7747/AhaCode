"""Prompts module (roadmap: evidence-based prompt refit).

Covers the assembled prompt surface and the extension seams (per-model deltas,
sub-agent role addenda) that must grow without touching call sites.
"""

from dataclasses import replace

import pytest

from ahacode import config, prompts, storage, subagent


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Private config so prompts see a known model (env block reads config.load())."""
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    config.save(replace(config.DEFAULTS, name="qwen38"))
    yield


def test_act_system_has_the_key_sections():
    out = prompts.act_system()
    assert out.startswith("You are AhaCode")
    for marker in ("# Output", "# Editing code", "# Never", "Reply in the user's language",
                   "`path:line`", "IMPORTANT"):
        assert marker in out


def test_act_system_injects_live_environment():
    out = prompts.act_system()
    assert "# Environment" in out
    assert str(storage.PROJECT_ROOT) in out  # cwd
    assert "qwen38" in out                            # active model


def test_act_system_has_no_few_shot_examples():
    # Deliberate: examples anchor a reasoning model (measured). Length is a rule.
    assert "<example>" not in prompts.act_system()


def test_family_maps_model_ids():
    assert prompts._family("claude-sonnet-5") == "claude"
    assert prompts._family("gpt-5") == "gpt"
    assert prompts._family("codex-mini") == "gpt"
    assert prompts._family("qwen38") == "qwen"
    assert prompts._family(None) == "qwen"  # from config (qwen38)


def test_by_model_delta_layers_into_act(monkeypatch):
    # The BYOK seam: a family entry appears for that family only, no call-site change.
    monkeypatch.setitem(prompts.BY_MODEL, "claude",
                        {"examples": "EX_BLOCK_MARKER", "delegation": "DG_BLOCK_MARKER"})
    claude = prompts.act_system("claude-sonnet-5")
    assert "EX_BLOCK_MARKER" in claude and "DG_BLOCK_MARKER" in claude
    # qwen (the default) is untouched by a claude-only entry
    assert "EX_BLOCK_MARKER" not in prompts.act_system("qwen38")


def test_subagent_role_addendum_layers_in(monkeypatch):
    monkeypatch.setitem(prompts.ROLE_ADDENDA, "debug", "DEBUG_ADDENDUM")
    with_role = prompts.subagent_system("debug")
    assert prompts.SUBAGENT_SYSTEM in with_role and "DEBUG_ADDENDUM" in with_role
    # no role -> just the base, unchanged
    assert prompts.subagent_system() == prompts.SUBAGENT_SYSTEM


def test_plan_and_title_are_stable():
    assert prompts.plan_system() == prompts.PLAN_SYSTEM
    assert "PLAN MODE" in prompts.plan_system()
    assert prompts.title_system() == prompts.TITLE_SYSTEM


def test_subagent_module_reexports_the_prompt():
    # subagent.run(system=SUBAGENT_SYSTEM) must resolve to the prompts constant.
    assert subagent.SUBAGENT_SYSTEM is prompts.SUBAGENT_SYSTEM
