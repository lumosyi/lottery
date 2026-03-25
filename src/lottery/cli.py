"""双色球智能预测系统 - CLI 命令定义"""

from __future__ import annotations

from pathlib import Path

import click
from loguru import logger

from lottery.analysis.report import AnalysisReport
from lottery.config import AppConfig, load_config
from lottery.fetcher.factory import FetcherFactory
from lottery.fetcher.web import WebFetcher
from lottery.store.sqlite import SqliteStore
from lottery.types import Prediction
from lottery.visualization.charts import ChartRenderer
from lottery.visualization.cli_display import CliDisplay

# ---------- 模型注册名 <-> 显示名 / 配置映射 ----------

_MODEL_DISPLAY_NAMES = {
    "statistical": "统计分析",
    "rf": "随机森林",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
}


def _get_model_config(config: AppConfig, model_name: str):
    """获取模型配置对象"""
    return {
        "statistical": config.models.statistical,
        "rf": config.models.random_forest,
        "xgboost": config.models.xgboost,
        "lstm": config.models.lstm,
    }.get(model_name)


def _get_model_kwargs(config: AppConfig, model_name: str) -> dict:
    """从配置中提取模型构造参数"""
    mc = _get_model_config(config, model_name)
    return mc.params if mc and mc.params else {}


def _auto_update(config: AppConfig) -> int:
    """自动增量更新（供其他命令内部调用）

    Returns:
        新增记录数
    """
    with SqliteStore(config.data.db_path) as store:
        latest_issue = store.get_latest_issue()
        if latest_issue is None:
            return 0

        fetcher = WebFetcher(source_url=config.data.web_url)
        records = fetcher.fetch_since(latest_issue)
        if not records:
            return 0

        return store.save(records)


# ---------- CLI 入口 ----------

@click.group()
@click.option(
    "--config", "config_path", default="config.yaml",
    help="配置文件路径", type=click.Path(),
)
@click.option("--verbose", "-v", is_flag=True, help="显示详细日志")
@click.pass_context
def cli(ctx: click.Context, config_path: str, verbose: bool) -> None:
    """双色球智能预测系统

    综合统计分析、机器学习、深度学习多策略融合预测。

    \b
    注意: 双色球是真随机过程，任何预测方法都无法提高中奖概率。
    本系统仅供技术学习和数据分析参考。
    """
    ctx.ensure_object(dict)

    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(lambda msg: click.echo(msg, err=True), level=level, format="{message}")

    ctx.obj["config"] = load_config(config_path)


# ---------- fetch ----------

@cli.command()
@click.option("--source", type=click.Choice(["web", "csv"]), default=None,
              help="数据源类型（默认使用配置文件）")
@click.option("--count", default=100, help="采集期数", show_default=True)
@click.option("--csv-path", default=None, help="CSV 文件路径（source=csv 时使用）")
@click.pass_context
def fetch(ctx: click.Context, source: str | None, count: int, csv_path: str | None) -> None:
    """采集历史开奖数据"""
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
    except (ValueError, FileNotFoundError) as e:
        click.echo(f"创建采集器失败: {e}", err=True)
        raise SystemExit(1)

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


# ---------- info ----------

@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """查看数据概览和系统状态"""
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

    click.echo(f"\n配置:")
    click.echo(f"  默认数据源: {config.data.source}")
    click.echo(f"  分析窗口: 最近 {config.analysis.default_recent} 期")

    click.echo(f"\n预测模型:")
    models = [
        ("统计分析", config.models.statistical),
        ("随机森林", config.models.random_forest),
        ("XGBoost", config.models.xgboost),
        ("LSTM", config.models.lstm),
    ]
    for name, m in models:
        status = "启用" if m.enabled else "禁用"
        click.echo(f"  {name}: {status} (权重: {m.weight})")

    click.echo(f"\n融合策略: {config.ensemble.strategy}")


# ---------- update ----------

@cli.command()
@click.pass_context
def update(ctx: click.Context) -> None:
    """增量更新: 自动获取最新开奖数据

    检测数据库中的最新期号，仅从网络采集之后的新数据。
    首次运行（数据库为空）时自动采集全部历史数据。
    """
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

    click.echo(f"\n更新完成:")
    click.echo(f"  新增: {inserted} 期")
    click.echo(f"  总计: {total_after} 期")

    if inserted > 0:
        click.echo(f"\n最新开奖:")
        for r in records[-min(inserted, 5):]:
            click.echo(f"  {r}")


# ---------- analyze ----------

@cli.command()
@click.option("--recent", default=None, type=int, help="分析最近N期（默认使用配置文件）")
@click.option("--show-charts", is_flag=True, help="生成并保存统计图表")
@click.pass_context
def analyze(ctx: click.Context, recent: int | None, show_charts: bool) -> None:
    """运行统计分析"""
    config: AppConfig = ctx.obj["config"]

    # 自动增量更新
    new_count = _auto_update(config)
    if new_count > 0:
        click.echo(f"自动更新: 新增 {new_count} 期数据\n")

    with SqliteStore(config.data.db_path) as store:
        total = store.count()
        if total == 0:
            click.echo("数据库为空，请先运行 'lottery update' 采集数据")
            return

        n = min(recent or config.analysis.default_recent, total)
        records = store.load_recent(n)

    click.echo(f"分析最近 {len(records)} 期数据 (共 {total} 期)")

    report = AnalysisReport.default(
        hot_window=config.analysis.hot_window,
        cold_threshold=config.analysis.cold_threshold,
    )
    results = report.generate(records)

    CliDisplay.print_analysis(results)

    if show_charts:
        click.echo(f"\n生成图表...")
        renderer = ChartRenderer(
            output_dir=config.output.charts_dir,
            style=config.output.chart_style,
        )
        saved = renderer.render_all(results)
        for path in saved:
            click.echo(f"  已保存: {path}")
        click.echo(f"共生成 {len(saved)} 张图表")


# ---------- predict ----------

def _register_models(model_names: tuple[str, ...]) -> None:
    """延迟导入模型模块以触发注册表注册"""
    import lottery.models.statistical  # noqa: F401
    import lottery.models.random_forest  # noqa: F401
    import lottery.models.xgboost_model  # noqa: F401

    try:
        import lottery.models.lstm  # noqa: F401
    except ImportError:
        if "lstm" in model_names or "all" in model_names:
            click.echo("LSTM 模型需要 PyTorch，请安装: pip install torch", err=True)


def _build_filter_pipeline(config: AppConfig, records):
    """根据配置和历史数据构建过滤管道"""
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
    """输出过滤统计信息"""
    if not stats or stats["excluded"] == 0:
        return

    click.echo(f"\n概率排除统计:")
    click.echo(f"  过滤规则: {', '.join(stats['rules'])}")
    click.echo(f"  本次过滤: {stats['excluded']}/{stats['total']} 组被标记排除 (置信度已降权)")
    if stats["reasons"]:
        reasons_str = ", ".join(f"{k}({v}次)" for k, v in stats["reasons"].items())
        click.echo(f"  排除原因: {reasons_str}")


def _run_ensemble(
    config: AppConfig, available: list[str], records, sets: int, use_filter: bool,
) -> None:
    """融合模式: 多模型协同预测"""
    from lottery.ensemble.engine import EnsembleEngine
    from lottery.ensemble.weighted_voting import WeightedVoting
    from lottery.models.registry import PredictorRegistry

    # 构建权重映射（仅启用的模型）
    weight_map: dict[str, float] = {}
    predictors = []
    for model_name in available:
        try:
            kwargs = _get_model_kwargs(config, model_name)
            predictor = PredictorRegistry.create(model_name, **kwargs)
            predictors.append(predictor)

            mc = _get_model_config(config, model_name)
            display_name = _MODEL_DISPLAY_NAMES.get(model_name, model_name)
            if mc:
                weight_map[display_name] = mc.weight
        except Exception as e:
            click.echo(f"  创建 {model_name} 失败: {e}", err=True)

    if not predictors:
        click.echo("无可用模型")
        return

    # 归一化权重（确保总和为1）
    total_weight = sum(weight_map.values())
    if total_weight > 0:
        weight_map = {k: v / total_weight for k, v in weight_map.items()}

    # 构建过滤管道
    pipeline = _build_filter_pipeline(config, records) if use_filter else None

    engine = EnsembleEngine(
        predictors=predictors,
        strategy=WeightedVoting(weights=weight_map),
        filter_pipeline=pipeline,
    )
    fused, filter_stats = engine.run(records, n_sets=sets, per_model_sets=3)
    _print_filter_stats(filter_stats)
    click.echo(f"\n共生成 {len(fused)} 组融合推荐号码")


def _run_independent(
    config: AppConfig, available: list[str], records, sets: int, use_filter: bool,
) -> None:
    """非融合模式: 独立运行各模型"""
    from lottery.models.registry import PredictorRegistry

    pipeline = _build_filter_pipeline(config, records) if use_filter else None
    all_predictions: list[Prediction] = []

    for model_name in available:
        click.echo(f"{'=' * 55}")
        click.echo(f"  模型: {_MODEL_DISPLAY_NAMES.get(model_name, model_name)}")
        click.echo(f"{'=' * 55}")

        try:
            kwargs = _get_model_kwargs(config, model_name)
            predictor = PredictorRegistry.create(model_name, **kwargs)

            click.echo("  训练中...")
            predictor.train(records)

            predictions = predictor.predict(records, n_sets=sets)
            all_predictions.extend(predictions)

            CliDisplay.print_prediction_table(predictions)
            click.echo()

        except Exception as e:
            click.echo(f"  模型 {model_name} 运行失败: {e}", err=True)
            logger.exception(f"模型 {model_name} 异常")

    if all_predictions:
        # 对全部预测结果执行过滤
        if pipeline:
            all_predictions, filter_stats = pipeline.filter_predictions(all_predictions, records)
            _print_filter_stats(filter_stats)
        click.echo(f"\n共生成 {len(all_predictions)} 组推荐号码")


@cli.command()
@click.option(
    "--model", "model_names", multiple=True,
    type=click.Choice(["statistical", "rf", "xgboost", "lstm", "all"]),
    default=["all"], help="选择预测模型", show_default=True,
)
@click.option("--sets", default=5, type=click.IntRange(min=1, max=20),
              help="生成推荐组数", show_default=True)
@click.option("--ensemble/--no-ensemble", default=False, help="启用多策略融合模式")
@click.option("--filter/--no-filter", "use_filter", default=True, help="启用概率排除过滤")
@click.pass_context
def predict(ctx: click.Context, model_names: tuple[str, ...], sets: int, ensemble: bool, use_filter: bool) -> None:
    """生成预测号码"""
    config: AppConfig = ctx.obj["config"]

    # 自动增量更新
    new_count = _auto_update(config)
    if new_count > 0:
        click.echo(f"自动更新: 新增 {new_count} 期数据\n")

    # 加载数据
    with SqliteStore(config.data.db_path) as store:
        total = store.count()
        if total == 0:
            click.echo("数据库为空，请先运行 'lottery update' 采集数据")
            return
        records = store.load_all()

    click.echo(f"使用 {len(records)} 期历史数据进行预测")
    click.echo("\n注意: 双色球是真随机过程，预测结果仅供参考，不构成投注建议。\n")

    # 注册模型
    _register_models(model_names)

    from lottery.models.registry import PredictorRegistry
    available = PredictorRegistry.list_available() if "all" in model_names else list(model_names)

    if ensemble:
        _run_ensemble(config, available, records, sets, use_filter)
    else:
        _run_independent(config, available, records, sets, use_filter)
