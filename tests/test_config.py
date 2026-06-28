"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest

from src.config import Config, load_config
from src.exceptions import ConfigError


def test_load_default_config() -> None:
    """The default config loads into a typed Config object."""
    config = load_config()
    assert isinstance(config, Config)
    assert config.seed == 42
    assert 0.0 < config.data.test_size < 1.0
    assert config.selection_metric == "pr_auc"


def test_models_section_has_all_five_models() -> None:
    """All five candidate models must be configured."""
    config = load_config()
    expected = {
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
    }
    assert expected.issubset(set(config.models.keys()))


def test_missing_config_raises(tmp_path) -> None:
    """A non-existent path raises ConfigError."""
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_invalid_config_raises(tmp_path) -> None:
    """A schema-violating config raises ConfigError."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: 1\ndata: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_malformed_yaml_raises(tmp_path) -> None:
    """Unparseable YAML raises ConfigError."""
    bad = tmp_path / "broken.yaml"
    bad.write_text("seed: : :\n  - [unbalanced", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)
