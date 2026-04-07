"""双色球智能预测系统 - CLI 命令定义"""

from __future__ import annotations

import importlib
import random
from pathlib import Path

import click
from loguru import logger

from lottery.analysis.report import AnalysisReport
from lottery.backtest import (
    PRIZE_BUCKETS,
    BacktestCase,
    BacktestMetrics,
    build_backtest_target_indices,
    build_random_baseline_predictions,
    export_backtest_results,
    evaluate_prediction,
    summarize_cases,
)
from lottery.config import AppConfig, load_config
from lottery.ensemble.engine import EnsembleEngine, EnsembleRunResult, PredictorRunResult
from lottery.ensemble.weighted_voting import WeightedVoting
from lottery.features.builder import FeatureBuilder
from lottery.fetcher.factory import FetcherFactory
from lottery.fetcher.web import WebFetcher
from lottery.models.base import BasePredictor
from lottery.runtime import set_random_seed
from lottery.store.sqlite import SqliteStore
from lottery.types import Prediction
from lottery.utils import dedupe_predictions
from lottery.visualization.charts import ChartRenderer
from lottery.visualization.cli_display import CliDisplay

_MODEL_DISPLAY_NAMES = {
    "statistical": "统计分析",
    "rf": "随机森林",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
}
_MODEL_MODULES = {
    "statistical": "lottery.models.statistical",
    "rf": "lottery.models.random_forest",
    "xgboost": "lottery.models.xgboost_model",
    "lstm": "lottery.models.lstm",
}
_OPTIONAL_MODEL_HINTS = {
    "xgboost": "pip install -e .[ml]",
    "lstm": "pip install -e .[dl]",
}


def _get_model_config(config: AppConfig, model_name: str):
    """获取模型配置对象。"""
    return {
        "statistical": config.models.statistical,
        "rf": config.models.random_forest,
        "xgboost": config.models.xgboost,
        "lstm": config.models.lstm,
    }.get(model_name)


def _build_feature_builder(config: AppConfig) -> FeatureBuilder:
    """按配置创建特征构建器。"""
    return FeatureBuilder(window_sizes=config.features.window_sizes)


def _get_model_kwargs(config: AppConfig, model_name: str) -> dict:
    """从配置中提取模型构造参数。"""
    mc = _get_model_config(config, model_name)
    kwargs = dict(mc.params if mc and mc.params else {})

    if model_name in {"rf", "xgboost", "lstm"}:
        kwargs["feature_builder"] = _build_feature_builder(config)

    if model_name in {"rf", "xgboost"} and config.runtime.seed is not None:
        kwargs.setdefault("random_state", config.runtime.seed)

    if model_name == "lstm":
        kwargs.setdefault("seq_len", config.features.sequence_length)

    return kwargs


def _auto_update(config: AppConfig) -> int:
    """自动增量更新（供其他命令内部调用）。"""
    with SqliteStore(config.data.db_path) as store:
        latest_issue = store.get_latest_issue()
        if latest_issue is None:
            return 0

        fetcher = WebFetcher(source_url=config.data.web_url)
        records = fetcher.fetch_since(latest_issue)
        if not records:
            return 0

        return store.save(records)


def _ensure_supported_strategy(config: AppConfig) -> None:
    """校验当前融合策略是否受支持。"""
    if config.ensemble.strategy != "weighted_voting":
        raise click.ClickException(
            f"当前仅支持 weighted_voting，收到: {config.ensemble.strategy}"
        )


def _register_models(model_names: list[str]) -> None:
    """按需导入模型模块以触发注册。"""
    for model_name in model_names:
        importlib.import_module(_MODEL_MODULES[model_name])


def _resolve_requested_models(
    config: AppConfig,
    model_names: tuple[str, ...],
) -> tuple[list[str], set[str]]:
    """解析最终要运行的模型列表。"""
    explicit_models: list[str] = []
    for model_name in model_names:
        if model_name != "all" and model_name not in explicit_models:
            explicit_models.append(model_name)

    disabled_explicit = [
        model_name
        for model_name in explicit_models
        if not _get_model_config(config, model_name).enabled
    ]
    if disabled_explicit:
        display_names = ", ".join(_MODEL_DISPLAY_NAMES[name] for name in disabled_explicit)
        raise click.ClickException(f"以下模型在配置中已禁用: {display_names}")

    if "all" in model_names or not explicit_models:
        resolved = [
            model_name
            for model_name in _MODEL_MODULES
            if _get_model_config(config, model_name).enabled
        ]
    else:
        resolved = explicit_models

    if not resolved:
        raise click.ClickException("当前没有启用的模型可运行")

    return resolved, set(explicit_models)


def _format_model_error(model_name: str, exc: Exception) -> str:
    """格式化模型不可用错误。"""
    display_name = _MODEL_DISPLAY_NAMES.get(model_name, model_name)
    hint = _OPTIONAL_MODEL_HINTS.get(model_name)
    if hint:
        return f"{display_name} 不可用: {exc}。安装命令: {hint}"
    return f"{display_name} 不可用: {exc}"


def _create_predictor(
    config: AppConfig,
    model_name: str,
    *,
    strict: bool,
) -> BasePredictor | None:
    """创建单个预测器，必要时对显式请求抛出清晰错误。"""
    from lottery.models.registry import PredictorRegistry

    try:
        return PredictorRegistry.create(model_name, **_get_model_kwargs(config, model_name))
    except Exception as exc:
        message = _format_model_error(model_name, exc)
        if strict:
            raise click.ClickException(message) from exc
        logger.warning(message)
        return None


def _build_predictors(
    config: AppConfig,
    model_names: list[str],
    explicit_requested: set[str],
) -> list[BasePredictor]:
    """批量创建预测器实例。"""
    predictors: list[BasePredictor] = []
    for model_name in model_names:
        predictor = _create_predictor(
            config,
            model_name,
            strict=model_name in explicit_requested,
        )
        if predictor is not None:
            predictors.append(predictor)
    return predictors


def _select_available_model_names(
    config: AppConfig,
    model_names: list[str],
    explicit_requested: set[str],
) -> list[str]:
    """筛选出当前环境下可实际运行的模型名称。"""
    available_names: list[str] = []
    for model_name in model_names:
        predictor = _create_predictor(
            config,
            model_name,
            strict=model_name in explicit_requested,
        )
        if predictor is not None:
            available_names.append(model_name)
    return available_names


def _estimate_model_min_history(config: AppConfig, model_name: str) -> int:
    """估算模型建议的最小历史期数。"""
    max_window = max(config.features.window_sizes or [5])

    if model_name == "statistical":
        return 10
    if model_name in {"rf", "xgboost"}:
        return max(max_window + 10, 20)
    if model_name == "lstm":
        return max(max_window + config.features.sequence_length + 10, 40)
    return 10


def _filter_history_compatible_model_names(
    config: AppConfig,
    model_names: list[str],
    explicit_requested: set[str],
    available_records: int,
) -> tuple[list[str], dict[str, int]]:
    """根据历史数据量筛选可运行模型。"""
    requirements = {
        model_name: _estimate_model_min_history(config, model_name) for model_name in model_names
    }
    insufficient = {
        model_name: required
        for model_name, required in requirements.items()
        if available_records < required
    }

    explicit_insufficient = {
        model_name: required
        for model_name, required in insufficient.items()
        if model_name in explicit_requested
    }
    if explicit_insufficient:
        message = ", ".join(
            f"{_MODEL_DISPLAY_NAMES[name]}(至少 {required} 期)"
            for name, required in explicit_insufficient.items()
        )
        raise click.ClickException(
            f"当前历史数据仅 {available_records} 期，无法运行以下显式指定模型: {message}"
        )

    for model_name, required in insufficient.items():
        logger.warning(
            f"{_MODEL_DISPLAY_NAMES[model_name]} 需要至少 {required} 期历史数据，当前 {available_records} 期，已跳过"
        )

    compatible = [model_name for model_name in model_names if model_name not in insufficient]
    if not compatible:
        required_text = ", ".join(
            f"{_MODEL_DISPLAY_NAMES[name]}(至少 {required} 期)"
            for name, required in requirements.items()
        )
        raise click.ClickException(
            f"当前历史数据仅 {available_records} 期，没有满足最小样本要求的模型: {required_text}"
        )

    return compatible, requirements


def _build_weight_map(config: AppConfig, predictors: list[BasePredictor]) -> dict[str, float]:
    """按已创建预测器生成归一化权重。"""
    weights: dict[str, float] = {}
    for predictor in predictors:
        model_name = next(
            key for key, display_name in _MODEL_DISPLAY_NAMES.items() if display_name == predictor.name
        )
        model_config = _get_model_config(config, model_name)
        weights[predictor.name] = model_config.weight

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return weights
    return {name: value / total_weight for name, value in weights.items()}


def _expanded_prediction_count(requested_sets: int) -> int:
    """为去重预留额外候选组数。"""
    if requested_sets <= 1:
        return 1
    return min(max(requested_sets * 2, requested_sets + 3), 50)


def _build_filter_pipeline(config: AppConfig, records):
    """根据配置和历史数据构建过滤管道。"""
    from lottery.filters.pipeline import FilterPipeline

    if not config.filters.enabled:
        return None

    return FilterPipeline.from_history(
        records,
        max_consecutive=config.filters.max_consecutive,
        repeat_recent=config.filters.repeat_check_recent,
        sum_percentile=config.filters.sum_range_percentile,
        exclude_extreme_odd_even=config.filters.exclude_extreme_odd_even,
        exclude_single_zone=config.filters.exclude_single_zone,
    )


def _print_filter_stats(stats: dict | None) -> None:
    """输出过滤统计信息。"""
    if not stats or stats["excluded"] == 0:
        return

    click.echo("\n概率排除统计:")
    click.echo(f"  过滤规则: {', '.join(stats['rules'])}")
    click.echo(f"  本次标记: {stats['excluded']}/{stats['total']} 组结果被标记")
    if stats["reasons"]:
        reasons_str = ", ".join(f"{k}({v}次)" for k, v in stats["reasons"].items())
        click.echo(f"  标记原因: {reasons_str}")


def _display_model_runs(model_runs: list[PredictorRunResult]) -> None:
    """统一打印各模型结果。"""
    for model_run in model_runs:
        click.echo(f"{'=' * 55}")
        click.echo(f"  模型: {model_run.name}")
        click.echo(f"{'=' * 55}")

        if model_run.error:
            click.echo(f"  运行失败: {model_run.error}", err=True)
            continue

        CliDisplay.print_prediction_table(model_run.predictions)
        click.echo()


def _load_records(config: AppConfig, *, auto_update: bool = False) -> list:
    """读取全部历史数据，必要时自动更新。"""
    if auto_update:
        new_count = _auto_update(config)
        if new_count > 0:
            click.echo(f"自动更新: 新增 {new_count} 期数据\n")

    with SqliteStore(config.data.db_path) as store:
        total = store.count()
        if total == 0:
            raise click.ClickException("数据库为空，请先运行 'lottery update' 采集数据")
        return store.load_all()


def _apply_filter_if_needed(config: AppConfig, predictions: list[Prediction], records, use_filter: bool):
    """按需对预测结果打标。"""
    if not use_filter:
        return predictions, None

    pipeline = _build_filter_pipeline(config, records)
    if pipeline is None:
        return predictions, None
    return pipeline.filter_predictions(predictions, records)


def _evaluate_primary_prediction(
    model_cases: dict[str, list[BacktestCase]],
    skipped_counts: dict[str, int],
    model_name: str,
    predictions: list[Prediction],
    actual_record,
) -> None:
    """评估并记录首组预测结果。"""
    if not predictions:
        skipped_counts[model_name] = skipped_counts.get(model_name, 0) + 1
        return
    model_cases.setdefault(model_name, []).append(evaluate_prediction(predictions[0], actual_record))


def _print_backtest_report(metrics: list[BacktestMetrics]) -> None:
    """输出回测摘要。"""
    if not metrics:
        return

    baseline = next((item for item in metrics if item.name == "随机基线"), None)
    click.echo(f"\n{'=' * 55}")
    click.echo("  回测汇总")
    click.echo(f"{'=' * 55}")

    for item in metrics:
        click.echo(f"\n[{item.name}]")
        click.echo(f"  评估期数: {item.periods} (跳过: {item.skipped_periods})")
        click.echo(f"  平均红球命中: {item.avg_red_hits:.3f}")
        click.echo(f"  蓝球命中率: {item.blue_hit_rate:.1%}")
        bucket_text = ", ".join(
            f"{bucket}:{item.bucket_counts.get(bucket, 0)}" for bucket in PRIZE_BUCKETS
        )
        click.echo(f"  奖级命中: {bucket_text}")
        if baseline and item.name != baseline.name:
            delta_red = item.avg_red_hits - baseline.avg_red_hits
            delta_blue = item.blue_hit_rate - baseline.blue_hit_rate
            click.echo(f"  对比随机基线: 红球 {delta_red:+.3f} | 蓝球 {delta_blue:+.1%}")


@click.group()
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    help="配置文件路径",
    type=click.Path(),
)
@click.option("--verbose", "-v", is_flag=True, help="显示详细日志")
@click.option("--seed", type=int, default=None, help="覆盖配置中的随机种子")
@click.pass_context
def cli(ctx: click.Context, config_path: str, verbose: bool, seed: int | None) -> None:
    """双色球智能预测系统。"""
    ctx.ensure_object(dict)

    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(lambda msg: click.echo(msg, err=True), level=level, format="{message}")

    try:
        config = load_config(config_path)
    except ValueError as exc:
        raise click.ClickException(f"配置文件无效: {exc}") from exc
    if seed is not None:
        config.runtime.seed = seed
    set_random_seed(config.runtime.seed)

    ctx.obj["config"] = config


@cli.command()
@click.option(
    "--source",
    type=click.Choice(["web", "csv"]),
    default=None,
    help="数据源类型（默认使用配置文件）",
)
@click.option("--count", default=100, help="采集期数", show_default=True)
@click.option("--csv-path", default=None, help="CSV 文件路径（source=csv 时使用）")
@click.pass_context
def fetch(ctx: click.Context, source: str | None, count: int, csv_path: str | None) -> None:
    """采集历史开奖数据。"""
    config: AppConfig = ctx.obj["config"]
    source = source or config.data.source

    click.echo(f"数据源: {source}，采集期数: {count}")

    kwargs = {}
    if source == "csv":
        kwargs["file_path"] = csv_path or config.data.csv_path
    elif source == "web":
        kwargs["source_url"] = config.data.web_url

    try:
        fetcher = FetcherFactory.create(source, **kwargs)
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(f"创建采集器失败: {exc}") from exc

    click.echo("正在采集数据...")
    records = fetcher.fetch_latest(count)

    if not records:
        click.echo("未获取到任何数据")
        return

    click.echo(f"采集到 {len(records)} 条记录")

    with SqliteStore(config.data.db_path) as store:
        inserted = store.save(records)

    click.echo(f"新增 {inserted} 条记录到数据库")
    click.echo("\n最近 5 期开奖:")
    for record in records[-5:]:
        click.echo(f"  {record}")


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """查看数据概览和系统状态。"""
    config: AppConfig = ctx.obj["config"]

    db_path = Path(config.data.db_path)
    click.echo("=" * 50)
    click.echo("  双色球智能预测系统 - 数据概览")
    click.echo("=" * 50)

    click.echo(f"\n数据库: {db_path}")
    if not db_path.exists():
        click.echo("  状态: 数据库尚未创建")
        click.echo("  提示: 运行 'lottery update' 采集数据")
        return

    with SqliteStore(config.data.db_path) as store:
        total = store.count()
        latest = store.get_latest_issue()

        click.echo(f"  总记录数: {total}")
        click.echo(f"  最新期号: {latest or '无'}")

        if total > 0:
            records = store.load_recent(5)
            click.echo(f"\n最近 {len(records)} 期开奖:")
            for record in records:
                click.echo(f"  {record}")

    click.echo("\n配置:")
    click.echo(f"  默认数据源: {config.data.source}")
    click.echo(f"  分析窗口: 最近 {config.analysis.default_recent} 期")
    click.echo(f"  特征窗口: {config.features.window_sizes}")
    click.echo(f"  LSTM 序列长度: {config.features.sequence_length}")
    click.echo(f"  随机种子: {config.runtime.seed}")

    click.echo("\n预测模型:")
    models = [
        ("统计分析", config.models.statistical),
        ("随机森林", config.models.random_forest),
        ("XGBoost", config.models.xgboost),
        ("LSTM", config.models.lstm),
    ]
    for name, model_config in models:
        status = "启用" if model_config.enabled else "禁用"
        click.echo(f"  {name}: {status} (权重: {model_config.weight})")

    click.echo(f"\n融合策略: {config.ensemble.strategy}")


@cli.command()
@click.pass_context
def update(ctx: click.Context) -> None:
    """增量更新: 自动获取最新开奖数据。"""
    config: AppConfig = ctx.obj["config"]
    fetcher = WebFetcher(source_url=config.data.web_url)

    with SqliteStore(config.data.db_path) as store:
        latest_issue = store.get_latest_issue()

        if latest_issue is None:
            click.echo("数据库为空，执行全量采集...")
            records = fetcher.fetch_latest(3000)
        else:
            latest_record = store.get_latest_record()
            click.echo(f"当前最新: {latest_record}")
            click.echo("检查新数据...")
            records = fetcher.fetch_since(latest_issue)

        if not records:
            click.echo("已是最新，无新数据")
            return

        inserted = store.save(records)
        total_after = store.count()

    click.echo("\n更新完成:")
    click.echo(f"  新增: {inserted} 期")
    click.echo(f"  总计: {total_after} 期")

    if inserted > 0:
        click.echo("\n最新开奖:")
        for record in records[-min(inserted, 5):]:
            click.echo(f"  {record}")


@cli.command()
@click.option("--recent", default=None, type=int, help="分析最近 N 期（默认使用配置文件）")
@click.option("--show-charts", is_flag=True, help="生成并保存统计图表")
@click.pass_context
def analyze(ctx: click.Context, recent: int | None, show_charts: bool) -> None:
    """运行统计分析。"""
    config: AppConfig = ctx.obj["config"]
    records = _load_records(config, auto_update=True)
    total = len(records)
    n = min(recent or config.analysis.default_recent, total)
    recent_records = records[-n:]

    click.echo(f"分析最近 {len(recent_records)} 期数据 (共 {total} 期)")

    report = AnalysisReport.default(
        hot_window=config.analysis.hot_window,
        cold_threshold=config.analysis.cold_threshold,
    )
    results = report.generate(recent_records)

    CliDisplay.print_analysis(results)

    if show_charts:
        click.echo("\n生成图表...")
        renderer = ChartRenderer(
            output_dir=config.output.charts_dir,
            style=config.output.chart_style,
        )
        saved = renderer.render_all(results)
        for path in saved:
            click.echo(f"  已保存: {path}")
        click.echo(f"共生成 {len(saved)} 张图表")


def _run_ensemble(
    config: AppConfig,
    requested_models: list[str],
    explicit_requested: set[str],
    records,
    sets: int,
    use_filter: bool,
) -> None:
    """融合模式: 多模型协同预测。"""
    _ensure_supported_strategy(config)
    predictors = _build_predictors(config, requested_models, explicit_requested)
    if not predictors:
        raise click.ClickException("无可用模型，请检查依赖或启用状态")

    candidate_sets = _expanded_prediction_count(sets)
    weight_map = _build_weight_map(config, predictors)
    engine = EnsembleEngine(
        predictors=predictors,
        strategy=WeightedVoting(weights=weight_map),
        filter_pipeline=_build_filter_pipeline(config, records) if use_filter else None,
    )
    result = engine.run(records, n_sets=candidate_sets, per_model_sets=max(candidate_sets, 3))

    for model_run in result.model_runs:
        if model_run.error:
            continue
        unique_predictions, duplicate_count = dedupe_predictions(model_run.predictions, limit=sets)
        model_run.predictions = unique_predictions
        if duplicate_count > 0:
            click.echo(f"  {model_run.name}: 已去重 {duplicate_count} 组重复号码")

    result.fused_predictions, fused_duplicate_count = dedupe_predictions(
        result.fused_predictions,
        limit=sets,
    )

    _display_model_runs(result.model_runs)

    if not result.fused_predictions:
        click.echo("所有模型均失败，无法生成融合结果")
        return

    if fused_duplicate_count > 0:
        click.echo(f"\n融合结果去重: 移除 {fused_duplicate_count} 组重复号码")
    click.echo(f"\n{'=' * 55}")
    click.echo("  多策略融合推荐")
    click.echo(f"{'=' * 55}")
    CliDisplay.print_prediction_table(result.fused_predictions)
    _print_filter_stats(result.filter_stats)
    click.echo(f"\n共生成 {len(result.fused_predictions)} 组融合推荐号码")


def _run_independent(
    config: AppConfig,
    requested_models: list[str],
    explicit_requested: set[str],
    records,
    sets: int,
    use_filter: bool,
) -> None:
    """非融合模式: 独立运行各模型。"""
    predictors = _build_predictors(config, requested_models, explicit_requested)
    if not predictors:
        raise click.ClickException("无可用模型，请检查依赖或启用状态")

    candidate_sets = _expanded_prediction_count(sets)
    all_predictions: list[Prediction] = []

    for predictor in predictors:
        click.echo(f"{'=' * 55}")
        click.echo(f"  模型: {predictor.name}")
        click.echo(f"{'=' * 55}")

        try:
            click.echo("  训练中...")
            predictor.train(records)
            predictions = predictor.predict(records, n_sets=candidate_sets)
            predictions, duplicate_count = dedupe_predictions(predictions, limit=sets)
            if duplicate_count > 0:
                click.echo(f"  去重: 移除 {duplicate_count} 组重复号码")
            predictions, filter_stats = _apply_filter_if_needed(config, predictions, records, use_filter)
            all_predictions.extend(predictions)

            CliDisplay.print_prediction_table(predictions)
            _print_filter_stats(filter_stats)
            click.echo()
        except Exception as exc:
            click.echo(f"  模型 {predictor.name} 运行失败: {exc}", err=True)
            logger.exception(f"模型 {predictor.name} 异常")

    if all_predictions:
        click.echo(f"\n共生成 {len(all_predictions)} 组推荐号码")


@cli.command()
@click.option(
    "--model",
    "model_names",
    multiple=True,
    type=click.Choice(["statistical", "rf", "xgboost", "lstm", "all"]),
    default=["all"],
    help="选择预测模型",
    show_default=True,
)
@click.option("--sets", default=5, type=click.IntRange(min=1, max=20), help="生成推荐组数", show_default=True)
@click.option("--ensemble/--no-ensemble", default=False, help="启用多策略融合模式")
@click.option("--filter/--no-filter", "use_filter", default=True, help="启用概率排除过滤")
@click.pass_context
def predict(
    ctx: click.Context,
    model_names: tuple[str, ...],
    sets: int,
    ensemble: bool,
    use_filter: bool,
) -> None:
    """生成预测号码。"""
    config: AppConfig = ctx.obj["config"]
    records = _load_records(config, auto_update=True)

    click.echo(f"使用 {len(records)} 期历史数据进行预测")
    click.echo("\n注意: 双色球是真随机过程，预测结果仅供参考，不构成投注建议。\n")

    requested_models, explicit_requested = _resolve_requested_models(config, model_names)
    _register_models(requested_models)
    requested_models = _select_available_model_names(config, requested_models, explicit_requested)
    requested_models, _ = _filter_history_compatible_model_names(
        config,
        requested_models,
        explicit_requested,
        len(records),
    )

    if ensemble:
        _run_ensemble(config, requested_models, explicit_requested, records, sets, use_filter)
    else:
        _run_independent(config, requested_models, explicit_requested, records, sets, use_filter)


@cli.command()
@click.option(
    "--model",
    "model_names",
    multiple=True,
    type=click.Choice(["statistical", "rf", "xgboost", "lstm", "all"]),
    default=["all"],
    help="选择参与回测的模型",
    show_default=True,
)
@click.option("--holdout", default=100, type=click.IntRange(min=1), help="回测最近 N 期", show_default=True)
@click.option("--sets", default=1, type=click.IntRange(min=1, max=20), help="每期生成组数", show_default=True)
@click.option("--step", default=1, type=click.IntRange(min=1), help="每隔 N 期采样一次回测", show_default=True)
@click.option(
    "--min-history",
    default=10,
    type=click.IntRange(min=1),
    help="开始评估前要求至少具备的历史期数",
    show_default=True,
)
@click.option("--ensemble/--no-ensemble", default=False, help="同时评估融合结果")
@click.option("--filter/--no-filter", "use_filter", default=False, help="回测时是否应用过滤打标")
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="导出回测结果到文件（未指定则仅输出终端摘要）",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv"]),
    default="json",
    show_default=True,
    help="回测结果导出格式（仅在指定 --output 时生效）",
)
@click.pass_context
def backtest(
    ctx: click.Context,
    model_names: tuple[str, ...],
    holdout: int,
    sets: int,
    step: int,
    min_history: int,
    ensemble: bool,
    use_filter: bool,
    output: Path | None,
    output_format: str,
) -> None:
    """使用 walk-forward 方式回测最近若干期表现。"""
    config: AppConfig = ctx.obj["config"]
    records = _load_records(config, auto_update=False)
    total = len(records)
    if total <= holdout:
        raise click.ClickException(f"数据不足，当前仅有 {total} 期，无法回测最近 {holdout} 期")

    requested_models, explicit_requested = _resolve_requested_models(config, model_names)
    _register_models(requested_models)
    requested_models = _select_available_model_names(config, requested_models, explicit_requested)
    requested_models, history_requirements = _filter_history_compatible_model_names(
        config,
        requested_models,
        explicit_requested,
        total,
    )
    if not requested_models:
        raise click.ClickException("无可用模型，请检查依赖或启用状态")

    effective_min_history = max(
        min_history,
        max(history_requirements[name] for name in requested_models),
    )
    target_indices = build_backtest_target_indices(
        total,
        holdout,
        step=step,
        min_history=effective_min_history,
    )
    if not target_indices:
        raise click.ClickException(
            "当前参数下没有可评估期数，请降低 --min-history 或缩小 --holdout"
        )

    skipped_for_history = holdout - len(target_indices)
    click.echo(
        f"开始回测最近 {holdout} 期数据（共 {total} 期，步长 {step}，最少历史 {effective_min_history} 期）"
    )
    if effective_min_history != min_history:
        click.echo(
            f"已按模型要求自动提升最小历史期数: {min_history} -> {effective_min_history}"
        )
    click.echo(
        f"实际评估 {len(target_indices)} 期"
        + (f"，因历史不足或采样跳过 {skipped_for_history} 期" if skipped_for_history > 0 else "")
    )
    if ensemble:
        _ensure_supported_strategy(config)

    model_cases: dict[str, list[BacktestCase]] = {"随机基线": []}
    skipped_counts: dict[str, int] = {"随机基线": 0}
    if ensemble:
        model_cases["融合推荐"] = []
        skipped_counts["融合推荐"] = 0

    baseline_rng = random.Random(config.runtime.seed)
    for target_index in target_indices:
        history = records[:target_index]
        actual_record = records[target_index]
        pipeline = _build_filter_pipeline(config, history) if use_filter else None

        baseline_predictions = build_random_baseline_predictions(n_sets=sets, rng=baseline_rng)
        _evaluate_primary_prediction(
            model_cases,
            skipped_counts,
            "随机基线",
            baseline_predictions,
            actual_record,
        )

        predictors = _build_predictors(config, requested_models, explicit_requested)
        if not predictors:
            raise click.ClickException("无可用模型，请检查依赖或启用状态")

        weight_map = _build_weight_map(config, predictors) if ensemble else None
        engine = EnsembleEngine(
            predictors=predictors,
            strategy=WeightedVoting(weights=weight_map) if ensemble else None,
            filter_pipeline=pipeline if ensemble and use_filter else None,
        )
        result: EnsembleRunResult = engine.run(
            history,
            n_sets=sets,
            per_model_sets=max(sets, 3) if ensemble else sets,
        )

        for model_run in result.model_runs:
            if model_run.error:
                skipped_counts[model_run.name] = skipped_counts.get(model_run.name, 0) + 1
                continue

            predictions = model_run.predictions
            if pipeline is not None and use_filter:
                predictions, _ = pipeline.filter_predictions(predictions, history)
            _evaluate_primary_prediction(
                model_cases,
                skipped_counts,
                model_run.name,
                predictions,
                actual_record,
            )

        if ensemble:
            _evaluate_primary_prediction(
                model_cases,
                skipped_counts,
                "融合推荐",
                result.fused_predictions,
                actual_record,
            )

    summary_order = [*(_MODEL_DISPLAY_NAMES[name] for name in requested_models)]
    if ensemble:
        summary_order.append("融合推荐")
    summary_order.append("随机基线")

    metrics: list[BacktestMetrics] = []
    for name in summary_order:
        metrics.append(
            summarize_cases(
                name,
                model_cases.get(name, []),
                skipped_periods=skipped_counts.get(name, 0),
            )
        )

    _print_backtest_report(metrics)

    if output is not None:
        ordered_cases = {name: model_cases.get(name, []) for name in summary_order}
        exported_paths = export_backtest_results(
            output,
            output_format,
            metrics,
            ordered_cases,
            metadata={
                "holdout": holdout,
                "evaluated_periods": len(target_indices),
                "step": step,
                "requested_min_history": min_history,
                "effective_min_history": effective_min_history,
                "sets": sets,
                "ensemble": ensemble,
                "filter_enabled": use_filter,
                "model_min_history": {
                    _MODEL_DISPLAY_NAMES[name]: history_requirements[name] for name in requested_models
                },
            },
        )
        click.echo("\n回测结果已导出:")
        for path in exported_paths:
            click.echo(f"  {path}")
