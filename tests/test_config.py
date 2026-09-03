"""config.toml: defaults on first run, values-only writes, two layers merged per key."""

from dataclasses import replace

from ahacode import config, workspace


def test_first_load_writes_the_global_file_with_defaults():
    assert not workspace.GLOBAL_CONFIG_PATH.exists()
    assert config.load() == config.DEFAULTS
    assert workspace.GLOBAL_CONFIG_PATH.exists()
    assert not workspace.CONFIG_PATH.exists()  # a project override is never created


def test_reads_user_values():
    workspace.GLOBAL_CONFIG_PATH.write_text(
        '[model]\nbase_url = "http://example:1234/v1"\nname = "my-model"\n'
        'api_key = "secret"\ntimeout = 5.0\n',
        encoding="utf-8",
    )
    cfg = config.load()
    assert (cfg.base_url, cfg.name, cfg.api_key, cfg.timeout) == (
        "http://example:1234/v1", "my-model", "secret", 5.0)


def test_partial_config_keeps_defaults_for_the_rest():
    workspace.GLOBAL_CONFIG_PATH.write_text('[model]\nname = "custom"\n', encoding="utf-8")
    cfg = config.load()
    assert cfg.name == "custom"
    assert cfg.base_url == config.DEFAULTS.base_url


def test_every_field_survives_a_save_and_load():
    cfg = replace(
        config.DEFAULTS, base_url="http://h:1/v1", name="m", api_key="k", timeout=7.5,
        thinking_token_budget=2048, reasoning_effort="high", no_think_after_tools=False,
        context_window=8192, subagent_depth=2, max_parallel_agents=3, impl_max_turns=0,
        auto_continue_stall=0, stall_rounds=0, plan_thinking_budget=8192,
        compact_threshold=0.5, keep_recent_messages=2, bash_timeout=45,
        allow_rules=("bash:uv run pytest*", "edit:ahacode/*"),
    )
    config.save(cfg)
    assert config.load() == cfg


def test_an_unset_per_mode_budget_is_omitted_and_reads_back_as_none():
    config.save(config.DEFAULTS)
    assert "plan_thinking_budget" not in workspace.GLOBAL_CONFIG_PATH.read_text(encoding="utf-8")
    assert config.load().plan_thinking_budget is None


def test_zero_survives_a_roundtrip():
    """0 means "off" for several knobs and must not come back as the default."""
    config.save(replace(config.DEFAULTS, context_window=0, thinking_token_budget=0))
    cfg = config.load()
    assert (cfg.context_window, cfg.thinking_token_budget) == (0, 0)


def test_a_project_override_is_merged_key_by_key():
    config.save(replace(config.DEFAULTS, base_url="http://global:1/v1", name="global-model"))
    workspace.CONFIG_PATH.write_text('[model]\nname = "project-model"\n', encoding="utf-8")
    cfg = config.load()
    assert cfg.name == "project-model"
    assert cfg.base_url == "http://global:1/v1"  # inherited from the global file


def test_first_load_leaves_an_existing_project_override_alone():
    """The default file is always the global one, never written over a project file."""
    workspace.CONFIG_PATH.write_text('[model]\nname = "project-model"\n', encoding="utf-8")
    assert config.load().name == "project-model"
    assert workspace.GLOBAL_CONFIG_PATH.exists()
    assert "project-model" in workspace.CONFIG_PATH.read_text(encoding="utf-8")


def test_save_goes_to_the_project_override_once_one_exists():
    workspace.CONFIG_PATH.write_text('[model]\nname = "project-model"\n', encoding="utf-8")
    config.save(replace(config.load(), name="changed"))
    assert "changed" in workspace.CONFIG_PATH.read_text(encoding="utf-8")
    assert "changed" not in workspace.GLOBAL_CONFIG_PATH.read_text(encoding="utf-8")


def test_thinking_budget_falls_back_to_global_then_per_mode():
    c = replace(config.DEFAULTS, thinking_token_budget=4096,
                plan_thinking_budget=8192, impl_thinking_budget=2048)
    assert c.thinking_budget_for("plan") == 8192
    assert c.thinking_budget_for("impl") == 2048
    assert c.thinking_budget_for("subagent") == 4096  # unset → global
    assert c.thinking_budget_for(None) == 4096  # plain act turn → global
