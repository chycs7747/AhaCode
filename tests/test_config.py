from ahacode import config


def test_creates_default_config_on_first_load(tmp_path):
    """First load must write a commented default file and return its values."""
    path = tmp_path / "config.toml"
    cfg = config.load(path)
    assert path.exists()
    assert cfg == config.ModelConfig(
        base_url=config.DEFAULT_BASE_URL,
        name=config.DEFAULT_MODEL,
        api_key=config.DEFAULT_API_KEY,
        timeout=config.DEFAULT_TIMEOUT,
    )


def test_reads_user_values(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[model]\nbase_url = "http://example:1234/v1"\nname = "my-model"\n'
        'api_key = "secret"\ntimeout = 5.0\n',
        encoding="utf-8",
    )
    cfg = config.load(path)
    assert cfg == config.ModelConfig("http://example:1234/v1", "my-model", "secret", 5.0)


def test_partial_config_falls_back_to_defaults(tmp_path):
    """A config with only some keys must still load, using defaults for the rest."""
    path = tmp_path / "config.toml"
    path.write_text('[model]\nname = "custom"\n', encoding="utf-8")
    cfg = config.load(path)
    assert cfg.name == "custom"
    assert cfg.base_url == config.DEFAULT_BASE_URL


def test_thinking_controls_default_and_roundtrip(tmp_path):
    """budget defaults to 4096, effort to medium, and both survive save/load."""
    from dataclasses import replace

    path = tmp_path / "config.toml"
    cfg = config.load(path)  # writes the commented default file
    assert cfg.thinking_token_budget == config.DEFAULT_THINKING_TOKEN_BUDGET == 4096
    assert cfg.reasoning_effort == config.DEFAULT_REASONING_EFFORT == "medium"

    config.save(replace(cfg, thinking_token_budget=2048, reasoning_effort="high"), path)
    cfg2 = config.load(path)
    assert cfg2.thinking_token_budget == 2048
    assert cfg2.reasoning_effort == "high"


def test_thinking_budget_zero_is_unbounded(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[model]\nthinking_token_budget = 0\n", encoding="utf-8")
    assert config.load(path).thinking_token_budget == 0
