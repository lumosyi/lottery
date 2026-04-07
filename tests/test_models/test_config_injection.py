from __future__ import annotations

import pytest
import yaml

from lottery.cli import _get_model_kwargs
from lottery.config import AppConfig, load_config


def test_get_model_kwargs_injects_feature_builder_and_runtime_seed():
    config = AppConfig()
    config.features.window_sizes = [3, 7]
    config.features.sequence_length = 12
    config.runtime.seed = 99

    rf_kwargs = _get_model_kwargs(config, "rf")
    lstm_kwargs = _get_model_kwargs(config, "lstm")

    assert rf_kwargs["feature_builder"].window_sizes == [3, 7]
    assert rf_kwargs["random_state"] == 99
    assert lstm_kwargs["feature_builder"].window_sizes == [3, 7]
    assert lstm_kwargs["seq_len"] == 12


def test_load_config_normalizes_feature_windows(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "features": {
                    "window_sizes": [20, 5, 20, 10],
                    "sequence_length": 12,
                }
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.features.window_sizes == [5, 10, 20]


def test_load_config_rejects_invalid_filter_percentile(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "filters": {
                    "sum_range_percentile": 30,
                }
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="filters.sum_range_percentile"):
        load_config(config_path)
