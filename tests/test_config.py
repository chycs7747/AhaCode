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
