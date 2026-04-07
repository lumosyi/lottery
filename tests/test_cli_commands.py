from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from lottery.cli import cli
from lottery.store.sqlite import SqliteStore


def _write_config(path: Path, db_path: Path, *, statistical_enabled: bool = True) -> None:
    config = {
        "data": {
            "db_path": str(db_path),
        },
        "models": {
            "statistical": {"enabled": statistical_enabled, "weight": 1.0},
            "random_forest": {"enabled": False},
            "xgboost": {"enabled": False},
            "lstm": {"enabled": False},
        },
        "runtime": {
            "seed": 7,
        },
    }
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")


def _write_multi_model_config(
    path: Path,
    db_path: Path,
    *,
    statistical_enabled: bool = True,
    rf_enabled: bool = True,
) -> None:
    config = {
        "data": {
            "db_path": str(db_path),
        },
        "models": {
            "statistical": {"enabled": statistical_enabled, "weight": 0.7},
            "random_forest": {"enabled": rf_enabled, "weight": 0.3},
            "xgboost": {"enabled": False},
            "lstm": {"enabled": False},
        },
        "runtime": {
            "seed": 7,
        },
    }
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")


def test_backtest_command_outputs_summary(tmp_path, sample_records):
    db_path = tmp_path / "lottery.db"
    with SqliteStore(db_path) as store:
        store.save(sample_records[:20])

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, db_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--config", str(config_path), "backtest", "--model", "statistical", "--holdout", "5"],
    )

    assert result.exit_code == 0, result.output
    assert "回测汇总" in result.output
    assert "[统计分析]" in result.output
    assert "[随机基线]" in result.output


def test_backtest_command_exports_json(tmp_path, sample_records):
    db_path = tmp_path / "lottery.db"
    with SqliteStore(db_path) as store:
        store.save(sample_records[:20])

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, db_path)
    output_path = tmp_path / "reports" / "backtest-report"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(config_path),
            "backtest",
            "--model",
            "statistical",
            "--holdout",
            "5",
            "--output",
            str(output_path),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    exported = output_path.with_suffix(".json")
    assert exported.exists()
    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert "metrics" in payload
    assert "baseline" in payload
    assert "cases" in payload
    assert "统计分析" in payload["cases"]
    assert payload["baseline"]["name"] == "随机基线"


def test_backtest_command_exports_csv(tmp_path, sample_records):
    db_path = tmp_path / "lottery.db"
    with SqliteStore(db_path) as store:
        store.save(sample_records[:20])

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, db_path)
    output_path = tmp_path / "reports" / "backtest-report.csv"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(config_path),
            "backtest",
            "--model",
            "statistical",
            "--holdout",
            "5",
            "--output",
            str(output_path),
            "--format",
            "csv",
        ],
    )

    assert result.exit_code == 0, result.output
    summary_path = output_path.with_suffix(".summary.csv")
    cases_path = output_path.with_suffix(".cases.csv")
    assert summary_path.exists()
    assert cases_path.exists()

    with summary_path.open(encoding="utf-8-sig", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    with cases_path.open(encoding="utf-8-sig", newline="") as handle:
        case_rows = list(csv.DictReader(handle))

    assert any(row["name"] == "统计分析" for row in summary_rows)
    assert any(row["name"] == "随机基线" for row in summary_rows)
    assert any(row["model"] == "统计分析" for row in case_rows)


def test_backtest_command_respects_step_and_min_history(tmp_path, sample_records):
    db_path = tmp_path / "lottery.db"
    with SqliteStore(db_path) as store:
        store.save(sample_records[:20])

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, db_path)
    output_path = tmp_path / "reports" / "sampled-backtest"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(config_path),
            "backtest",
            "--model",
            "statistical",
            "--holdout",
            "5",
            "--step",
            "2",
            "--min-history",
            "18",
            "--output",
            str(output_path),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["metadata"]["evaluated_periods"] == 2
    assert payload["metadata"]["step"] == 2
    assert payload["metadata"]["requested_min_history"] == 18
    assert payload["metadata"]["effective_min_history"] == 18
    metrics_by_name = {item["name"]: item for item in payload["metrics"]}
    assert metrics_by_name["统计分析"]["periods"] == 2
    assert metrics_by_name["随机基线"]["periods"] == 2


def test_predict_rejects_insufficient_history_for_explicit_model(tmp_path, sample_records, monkeypatch):
    db_path = tmp_path / "lottery.db"
    with SqliteStore(db_path) as store:
        store.save(sample_records[:20])

    config_path = tmp_path / "config.yaml"
    _write_multi_model_config(config_path, db_path)
    monkeypatch.setattr("lottery.cli._auto_update", lambda config: 0)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--config", str(config_path), "predict", "--model", "rf"],
    )

    assert result.exit_code != 0
    assert "历史数据仅 20 期" in result.output
    assert "随机森林(至少 60 期)" in result.output


def test_predict_all_skips_models_with_insufficient_history(tmp_path, sample_records, monkeypatch):
    db_path = tmp_path / "lottery.db"
    with SqliteStore(db_path) as store:
        store.save(sample_records[:20])

    config_path = tmp_path / "config.yaml"
    _write_multi_model_config(config_path, db_path)
    monkeypatch.setattr("lottery.cli._auto_update", lambda config: 0)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--config", str(config_path), "predict", "--model", "all", "--sets", "1"],
    )

    assert result.exit_code == 0, result.output
    assert "随机森林 需要至少 60 期历史数据，当前 20 期，已跳过" in result.output
    assert "模型: 统计分析" in result.output


def test_predict_rejects_explicitly_disabled_model(tmp_path, sample_records, monkeypatch):
    db_path = tmp_path / "lottery.db"
    with SqliteStore(db_path) as store:
        store.save(sample_records[:20])

    config_path = tmp_path / "config.yaml"
    _write_config(config_path, db_path, statistical_enabled=False)

    monkeypatch.setattr("lottery.cli._auto_update", lambda config: 0)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--config", str(config_path), "predict", "--model", "statistical"],
    )

    assert result.exit_code != 0
    assert "已禁用" in result.output
