"""融合引擎 - 协调多个预测器并融合结果"""

from __future__ import annotations

import click
from loguru import logger

from lottery.ensemble.base import EnsembleStrategy
from lottery.ensemble.weighted_voting import WeightedVoting
from lottery.filters.pipeline import FilterPipeline
from lottery.models.base import BasePredictor
from lottery.types import LotteryRecord, Prediction
from lottery.visualization.cli_display import CliDisplay


class EnsembleEngine:
    """融合引擎

    协调多个预测器的训练、预测和结果融合。
    可选注入 FilterPipeline 对最终结果进行概率排除过滤。
    """

    def __init__(
        self,
        predictors: list[BasePredictor],
        strategy: EnsembleStrategy | None = None,
        weights: dict[str, float] | None = None,
        filter_pipeline: FilterPipeline | None = None,
    ) -> None:
        self._predictors = predictors
        self._strategy = strategy or WeightedVoting(weights=weights)
        self._filter_pipeline = filter_pipeline

    def run(
        self,
        records: list[LotteryRecord],
        n_sets: int = 5,
        per_model_sets: int = 3,
    ) -> tuple[list[Prediction], dict | None]:
        """执行完整的预测流程

        Returns:
            (融合后的推荐列表, 过滤统计信息或None)
        """
        all_predictions: list[Prediction] = []

        for predictor in self._predictors:
            click.echo(f"\n  [{predictor.name}] 训练中...")
            try:
                predictor.train(records)
                preds = predictor.predict(records, n_sets=per_model_sets)
                all_predictions.extend(preds)

                click.echo(f"  [{predictor.name}] 生成 {len(preds)} 组预测")
                CliDisplay.print_prediction_table(preds)

            except Exception as e:
                logger.warning(f"[{predictor.name}] 运行失败: {e}")
                click.echo(f"  [{predictor.name}] 失败: {e}", err=True)

        if not all_predictions:
            click.echo("所有模型均失败，无法生成融合结果")
            return [], None

        # 融合
        click.echo(f"\n{'=' * 55}")
        click.echo(f"  多策略融合推荐")
        click.echo(f"{'=' * 55}")

        fused = self._strategy.fuse(all_predictions, n_sets=n_sets)

        # 过滤
        filter_stats = None
        if self._filter_pipeline:
            fused, filter_stats = self._filter_pipeline.filter_predictions(fused, records)

        click.echo(f"\n  综合 {len(self._predictors)} 个模型的 {len(all_predictions)} 组预测，"
                    f"融合生成 {len(fused)} 组最终推荐:")
        CliDisplay.print_prediction_table(fused)

        return fused, filter_stats
