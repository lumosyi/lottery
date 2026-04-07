"""加权投票融合策略。

每个号码的得分 = 各预测器对该号码的推荐分数 * 预测器权重 * 评分。
取得分最高的 6 个红球和 1 个蓝球。
"""

from __future__ import annotations

from collections import defaultdict

from lottery.constants import ALL_BLUE_BALLS, ALL_RED_BALLS, RED_BALL_COUNT
from lottery.ensemble.base import EnsembleStrategy
from lottery.types import Prediction


class WeightedVoting(EnsembleStrategy):
    """加权投票融合

    为每个号码计算加权得分:
    - 号码在某预测组中出现 -> 获得该预测器的权重 * 评分
    - 按得分排序选取 Top 号码
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """
        Args:
            weights: 各预测器的权重，key 为预测器 source 名称
                     未指定的预测器默认权重 1.0
        """
        self._weights = weights or {}

    def fuse(self, predictions: list[Prediction], n_sets: int = 1) -> list[Prediction]:
        """加权投票融合"""
        if not predictions:
            return []

        # 计算红球和蓝球得分
        red_scores: dict[int, float] = defaultdict(float)
        blue_scores: dict[int, float] = defaultdict(float)

        for pred in predictions:
            w = self._weights.get(pred.source, 1.0)
            score = w * pred.score

            for ball in pred.red_balls:
                red_scores[ball] += score

            blue_scores[pred.blue_ball] += score

        # 确保所有号码都有得分（缺失的为0）
        for ball in ALL_RED_BALLS:
            red_scores.setdefault(ball, 0)
        for ball in ALL_BLUE_BALLS:
            blue_scores.setdefault(ball, 0)

        # 按得分排序
        sorted_red = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_blue = sorted(blue_scores.items(), key=lambda x: x[1], reverse=True)

        results: list[Prediction] = []
        for i in range(n_sets):
            if i == 0:
                # 第一组: 直接取 Top 6
                selected_red = [b for b, _ in sorted_red[:RED_BALL_COUNT]]
            else:
                # 后续组: 跳过已选的最高分号码，增加多样性
                offset = i * 2
                candidates = sorted_red[offset: offset + RED_BALL_COUNT + 5]
                selected_red = [b for b, _ in candidates[:RED_BALL_COUNT]]

                # 如果候选不足，从头部补充
                if len(selected_red) < RED_BALL_COUNT:
                    remaining = [
                        b for b, _ in sorted_red if b not in selected_red
                    ]
                    selected_red.extend(remaining[: RED_BALL_COUNT - len(selected_red)])

            blue_ball = sorted_blue[i % len(sorted_blue)][0]

            # 融合评分: 所选红球的平均加权得分归一化
            max_score = sorted_red[0][1] if sorted_red else 1.0
            avg_score = sum(red_scores[b] for b in selected_red) / RED_BALL_COUNT
            final_score = round(min(avg_score / max_score, 0.95) if max_score > 0 else 0.5, 3)

            results.append(
                Prediction(
                    red_balls=tuple(sorted(selected_red)),
                    blue_ball=blue_ball,
                    score=final_score,
                    source="融合推荐",
                    details={
                        "strategy": "weighted_voting",
                        "red_scores": {b: round(red_scores[b], 3) for b in selected_red},
                    },
                )
            )

        return results
