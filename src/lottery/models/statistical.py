"""统计分析预测器

基于频率、遗漏值、冷热号等统计指标，为每个号码计算综合概率，
然后按概率加权随机采样生成推荐号码。
"""

from __future__ import annotations

import random

from loguru import logger

from lottery.constants import ALL_BLUE_BALLS, ALL_RED_BALLS, RED_BALL_COUNT
from lottery.features.transforms import calc_frequency, calc_missing
from lottery.models.base import BasePredictor
from lottery.models.registry import PredictorRegistry
from lottery.types import LotteryRecord, Prediction
from lottery.utils import weighted_sample_no_replace


@PredictorRegistry.register("statistical")
class StatisticalPredictor(BasePredictor):
    """基于统计分析的预测器

    综合考虑以下因素为每个号码打分:
    1. 历史频率（高频号倾向继续出现）
    2. 当前遗漏值（高遗漏号倾向回补）
    3. 近期趋势（近期频率与长期频率的偏差）

    最终将评分归一化为概率分布，进行加权采样。
    """

    def __init__(
        self,
        freq_weight: float = 0.3,
        missing_weight: float = 0.4,
        trend_weight: float = 0.3,
    ) -> None:
        self._freq_weight = freq_weight
        self._missing_weight = missing_weight
        self._trend_weight = trend_weight
        self._trained = False

    @property
    def name(self) -> str:
        return "统计分析"

    def train(self, records: list[LotteryRecord]) -> None:
        """统计预测器无需训练，仅标记就绪"""
        if len(records) < 10:
            raise ValueError("至少需要 10 期数据")
        self._trained = True
        logger.info(f"[{self.name}] 就绪，共 {len(records)} 期数据")

    def predict(self, records: list[LotteryRecord], n_sets: int = 1) -> list[Prediction]:
        """基于统计概率生成预测"""
        if not self._trained:
            self.train(records)

        red_probs = self._compute_red_probability(records)
        blue_probs = self._compute_blue_probability(records)

        predictions: list[Prediction] = []
        for _ in range(n_sets):
            # 按概率加权采样 6 个红球（不重复）
            red_balls = weighted_sample_no_replace(
                ALL_RED_BALLS, red_probs, RED_BALL_COUNT
            )
            # 按概率加权采样 1 个蓝球
            blue_ball = random.choices(ALL_BLUE_BALLS, weights=blue_probs, k=1)[0]

            # 评分基于所选号码的平均概率
            avg_prob = sum(red_probs[ALL_RED_BALLS.index(b)] for b in red_balls) / RED_BALL_COUNT
            score = min(avg_prob * 10, 0.95)  # 归一化到合理范围

            predictions.append(
                Prediction(
                    red_balls=tuple(sorted(red_balls)),
                    blue_ball=blue_ball,
                    score=round(score, 3),
                    source=self.name,
                )
            )

        return predictions

    def _compute_red_probability(self, records: list[LotteryRecord]) -> list[float]:
        """计算红球各号码的综合概率"""
        total = len(records)

        # 1. 长期频率得分
        freq = calc_frequency(records, "red")
        freq_scores = [freq.get(b, 0) / total for b in ALL_RED_BALLS]

        # 2. 遗漏值得分（遗漏越高，得分越高）
        missing = calc_missing(records, "red")
        max_missing = max(missing.values()) or 1
        missing_scores = [missing[b] / max_missing for b in ALL_RED_BALLS]

        # 3. 近期趋势得分（近10期频率 vs 长期频率的差异）
        recent_n = min(10, total)
        recent_freq = calc_frequency(records[-recent_n:], "red")
        trend_scores = []
        for b in ALL_RED_BALLS:
            long_rate = freq.get(b, 0) / total
            short_rate = recent_freq.get(b, 0) / recent_n
            # 近期频率高于长期 -> 正趋势
            trend_scores.append(max(short_rate - long_rate + 0.5, 0.1))

        # 加权综合
        combined = []
        for i in range(len(ALL_RED_BALLS)):
            score = (
                self._freq_weight * freq_scores[i]
                + self._missing_weight * missing_scores[i]
                + self._trend_weight * trend_scores[i]
            )
            combined.append(max(score, 0.01))

        # 归一化为概率分布
        total_score = sum(combined)
        return [s / total_score for s in combined]

    def _compute_blue_probability(self, records: list[LotteryRecord]) -> list[float]:
        """计算蓝球各号码的综合概率"""
        total = len(records)

        freq = calc_frequency(records, "blue")
        missing = calc_missing(records, "blue")
        max_missing = max(missing.values()) or 1

        combined = []
        for b in ALL_BLUE_BALLS:
            freq_score = freq.get(b, 0) / total
            missing_score = missing[b] / max_missing
            score = 0.4 * freq_score + 0.6 * missing_score
            combined.append(max(score, 0.01))

        total_score = sum(combined)
        return [s / total_score for s in combined]
