"""回测辅助类型与统计函数。"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from lottery.constants import ALL_BLUE_BALLS, ALL_RED_BALLS, RED_BALL_COUNT
from lottery.types import LotteryRecord, Prediction

PRIZE_BUCKETS = ("4+0", "4+1", "5+0", "5+1", "6+0", "6+1")


@dataclass(slots=True)
class BacktestCase:
    """单期回测结果。"""

    issue: str
    red_hits: int
    blue_hit: bool
    bucket: str | None


@dataclass(slots=True)
class BacktestMetrics:
    """聚合后的回测指标。"""

    name: str
    periods: int
    skipped_periods: int
    avg_red_hits: float
    blue_hit_rate: float
    bucket_counts: dict[str, int] = field(default_factory=dict)


def case_to_dict(case: BacktestCase) -> dict[str, str | int | bool | None]:
    """将单期回测结果转换为可序列化字典。"""
    return {
        "issue": case.issue,
        "red_hits": case.red_hits,
        "blue_hit": case.blue_hit,
        "bucket": case.bucket,
    }


def metrics_to_dict(metrics: BacktestMetrics) -> dict[str, str | int | float | dict[str, int]]:
    """将聚合指标转换为可序列化字典。"""
    return {
        "name": metrics.name,
        "periods": metrics.periods,
        "skipped_periods": metrics.skipped_periods,
        "avg_red_hits": metrics.avg_red_hits,
        "blue_hit_rate": metrics.blue_hit_rate,
        "bucket_counts": {bucket: metrics.bucket_counts.get(bucket, 0) for bucket in PRIZE_BUCKETS},
    }


def evaluate_prediction(prediction: Prediction, actual: LotteryRecord) -> BacktestCase:
    """评估单组预测相对真实开奖的命中情况。"""
    red_hits = len(set(prediction.red_balls) & set(actual.red_balls))
    blue_hit = prediction.blue_ball == actual.blue_ball
    bucket = None
    if red_hits >= 4:
        bucket = f"{red_hits}+{1 if blue_hit else 0}"
    return BacktestCase(
        issue=actual.issue,
        red_hits=red_hits,
        blue_hit=blue_hit,
        bucket=bucket,
    )


def summarize_cases(name: str, cases: list[BacktestCase], skipped_periods: int = 0) -> BacktestMetrics:
    """将逐期结果汇总为核心指标。"""
    bucket_counts = {bucket: 0 for bucket in PRIZE_BUCKETS}
    for case in cases:
        if case.bucket:
            bucket_counts[case.bucket] += 1

    periods = len(cases)
    avg_red_hits = sum(case.red_hits for case in cases) / periods if periods else 0.0
    blue_hit_rate = sum(1 for case in cases if case.blue_hit) / periods if periods else 0.0
    return BacktestMetrics(
        name=name,
        periods=periods,
        skipped_periods=skipped_periods,
        avg_red_hits=round(avg_red_hits, 3),
        blue_hit_rate=round(blue_hit_rate, 4),
        bucket_counts=bucket_counts,
    )


def build_random_baseline_predictions(
    n_sets: int = 1,
    rng: random.Random | None = None,
) -> list[Prediction]:
    """生成用于回测比较的随机基线预测。"""
    picker = rng or random
    predictions: list[Prediction] = []
    for _ in range(n_sets):
        red_balls = tuple(sorted(picker.sample(ALL_RED_BALLS, RED_BALL_COUNT)))
        blue_ball = picker.choice(ALL_BLUE_BALLS)
        predictions.append(
            Prediction(
                red_balls=red_balls,
                blue_ball=blue_ball,
                score=0.0,
                source="随机基线",
            )
        )
    return predictions


def build_backtest_target_indices(
    total_records: int,
    holdout: int,
    *,
    step: int = 1,
    min_history: int = 10,
) -> list[int]:
    """根据参数计算需要评估的目标期索引。"""
    if total_records <= 0 or holdout <= 0:
        return []

    start_index = max(total_records - holdout, min_history)
    if start_index >= total_records:
        return []

    indices = list(range(start_index, total_records, step))
    if not indices:
        return []

    if indices[-1] != total_records - 1:
        indices.append(total_records - 1)
    return indices


def build_backtest_export_payload(
    metrics: list[BacktestMetrics],
    cases_by_model: dict[str, list[BacktestCase]],
    *,
    metadata: dict | None = None,
) -> dict:
    """构建回测导出负载。"""
    baseline = next((item for item in metrics if item.name == "随机基线"), None)
    payload = {
        "metrics": [metrics_to_dict(item) for item in metrics],
        "baseline": metrics_to_dict(baseline) if baseline is not None else None,
        "cases": {
            name: [case_to_dict(case) for case in cases]
            for name, cases in cases_by_model.items()
        },
    }
    if metadata:
        payload["metadata"] = metadata
    return payload


def export_backtest_results(
    output_path: str | Path,
    export_format: str,
    metrics: list[BacktestMetrics],
    cases_by_model: dict[str, list[BacktestCase]],
    *,
    metadata: dict | None = None,
) -> list[Path]:
    """导出回测结果到 JSON 或 CSV。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_backtest_export_payload(metrics, cases_by_model, metadata=metadata)

    if export_format == "json":
        json_path = path if path.suffix.lower() == ".json" else path.with_suffix(".json")
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return [json_path]

    if export_format == "csv":
        summary_path = path.with_suffix(".summary.csv")
        cases_path = path.with_suffix(".cases.csv")

        with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
            fieldnames = [
                "name",
                "periods",
                "skipped_periods",
                "avg_red_hits",
                "blue_hit_rate",
                *PRIZE_BUCKETS,
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for item in metrics:
                row = {
                    "name": item.name,
                    "periods": item.periods,
                    "skipped_periods": item.skipped_periods,
                    "avg_red_hits": item.avg_red_hits,
                    "blue_hit_rate": item.blue_hit_rate,
                }
                row.update({bucket: item.bucket_counts.get(bucket, 0) for bucket in PRIZE_BUCKETS})
                writer.writerow(row)

        with cases_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["model", "issue", "red_hits", "blue_hit", "bucket"],
            )
            writer.writeheader()
            for model_name, cases in cases_by_model.items():
                for case in cases:
                    writer.writerow(
                        {
                            "model": model_name,
                            "issue": case.issue,
                            "red_hits": case.red_hits,
                            "blue_hit": case.blue_hit,
                            "bucket": case.bucket or "",
                        }
                    )

        return [summary_path, cases_path]

    raise ValueError(f"不支持的导出格式: {export_format}")
