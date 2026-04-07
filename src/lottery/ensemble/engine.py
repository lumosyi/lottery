"""融合引擎 - 协调多个预测器并融合结果"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from lottery.ensemble.base import EnsembleStrategy
from lottery.ensemble.weighted_voting import WeightedVoting
from lottery.filters.pipeline import FilterPipeline
from lottery.models.base import BasePredictor
from lottery.types import LotteryRecord, Prediction


@dataclass(slots=True)
class PredictorRunResult:
    """单个预测器的执行结果。"""

    name: str
    predictions: list[Prediction] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class EnsembleRunResult:
    """一次融合执行的完整结果。"""

    model_runs: list[PredictorRunResult] = field(default_factory=list)
    fused_predictions: list[Prediction] = field(default_factory=list)
    filter_stats: dict | None = None


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
        self._strategy = strategy if strategy is not None else (
            WeightedVoting(weights=weights) if weights is not None else None
        )
        self._filter_pipeline = filter_pipeline

    def run(
        self,
        records: list[LotteryRecord],
        n_sets: int = 5,
        per_model_sets: int = 3,
    ) -> EnsembleRunResult:
        """执行完整的预测流程。"""
        all_predictions: list[Prediction] = []
        model_runs: list[PredictorRunResult] = []

        for predictor in self._predictors:
            try:
                predictor.train(records)
                preds = predictor.predict(records, n_sets=per_model_sets)
                all_predictions.extend(preds)
                model_runs.append(PredictorRunResult(name=predictor.name, predictions=preds))
            except Exception as e:
                logger.warning(f"[{predictor.name}] 运行失败: {e}")
                model_runs.append(PredictorRunResult(name=predictor.name, error=str(e)))

        if not all_predictions:
            return EnsembleRunResult(model_runs=model_runs)

        if self._strategy is None:
            return EnsembleRunResult(model_runs=model_runs)

        fused = self._strategy.fuse(all_predictions, n_sets=n_sets)

        filter_stats = None
        if self._filter_pipeline:
            fused, filter_stats = self._filter_pipeline.filter_predictions(fused, records)

        return EnsembleRunResult(
            model_runs=model_runs,
            fused_predictions=fused,
            filter_stats=filter_stats,
        )
