"""过滤器管道 — 串联多个规则，支持从历史数据自动推导阈值"""

from __future__ import annotations

from collections import Counter

import numpy as np
from loguru import logger

from lottery.filters.base import PredictionFilter
from lottery.filters.rules import (
    ConsecutiveFilter,
    OddEvenFilter,
    RepeatFilter,
    SumRangeFilter,
    ZoneFilter,
)
from lottery.types import LotteryRecord, Prediction


class FilterPipeline:
    """过滤器管道

    将多个 PredictionFilter 串联执行，对预测结果进行后处理。
    支持从历史数据自动推导合理的过滤阈值。
    """

    def __init__(self, filters: list[PredictionFilter]) -> None:
        self._filters = filters

    @property
    def rule_names(self) -> list[str]:
        """所有规则名称"""
        return [f.name for f in self._filters]

    def check(
        self,
        red_balls: tuple[int, ...],
        blue_ball: int,
        records: list[LotteryRecord],
    ) -> tuple[bool, list[str]]:
        """检查单组号码是否通过全部过滤规则

        Returns:
            (是否通过, 排除原因列表)
        """
        reasons: list[str] = []
        for f in self._filters:
            exclude, reason = f.should_exclude(red_balls, blue_ball, records)
            if exclude:
                reasons.append(reason)
        return len(reasons) == 0, reasons

    def filter_predictions(
        self,
        predictions: list[Prediction],
        records: list[LotteryRecord],
    ) -> tuple[list[Prediction], dict]:
        """过滤预测列表

        Args:
            predictions: 原始预测列表
            records: 历史记录

        Returns:
            (过滤后的预测列表, 过滤统计信息)
        """
        passed: list[Prediction] = []
        excluded_reasons: list[str] = []
        excluded_count = 0

        for pred in predictions:
            ok, reasons = self.check(pred.red_balls, pred.blue_ball, records)
            if ok:
                passed.append(pred)
            else:
                excluded_count += 1
                excluded_reasons.extend(reasons)
                # 在 details 中标注但仍保留（标记为被过滤）
                pred.details["filtered"] = True
                pred.details["filter_reasons"] = reasons
                pred.confidence *= 0.3  # 大幅降低置信度
                pred.source = f"{pred.source}[已排除]"
                passed.append(pred)

        # 统计
        reason_counter = Counter(
            r.split("(")[0] for r in excluded_reasons  # 取原因前缀分类
        )
        stats = {
            "total": len(predictions),
            "excluded": excluded_count,
            "passed": len(predictions) - excluded_count,
            "reasons": dict(reason_counter),
            "rules": self.rule_names,
        }

        if excluded_count > 0:
            logger.info(
                f"过滤统计: {excluded_count}/{len(predictions)} 组被标记排除"
            )

        return passed, stats

    @classmethod
    def from_history(
        cls,
        records: list[LotteryRecord],
        max_consecutive: int | None = None,
        repeat_recent: int = 10,
        sum_percentile: float = 95,
        exclude_extreme_odd_even: bool = True,
        exclude_single_zone: bool = True,
    ) -> "FilterPipeline":
        """从历史数据自动推导阈值并创建管道

        Args:
            records: 历史开奖记录
            max_consecutive: 最大连号阈值（None=自动推导）
            repeat_recent: 重复检查最近期数
            sum_percentile: 和值区间百分位
            exclude_extreme_odd_even: 是否排除全奇/全偶
            exclude_single_zone: 是否排除单区全出
        """
        filters: list[PredictionFilter] = []

        if len(records) < 10:
            logger.warning("历史数据不足 10 期，跳过过滤器构建")
            return cls(filters=[])

        # 1. 连号过滤 — 自动推导阈值
        if max_consecutive is None:
            max_len = 1
            for r in records:
                current = 1
                for i in range(1, len(r.red_balls)):
                    if r.red_balls[i] - r.red_balls[i - 1] == 1:
                        current += 1
                        max_len = max(max_len, current)
                    else:
                        current = 1
            # 阈值 = 历史最大连号 + 1，但至少为 4
            max_consecutive = max(max_len + 1, 4)
        filters.append(ConsecutiveFilter(max_length=max_consecutive))

        # 2. 重复过滤
        filters.append(RepeatFilter(recent_n=repeat_recent))

        # 3. 和值范围过滤
        sums = [sum(r.red_balls) for r in records]
        lower = 100 - sum_percentile
        min_sum = int(np.percentile(sums, lower))
        max_sum = int(np.percentile(sums, sum_percentile))
        filters.append(SumRangeFilter(min_sum=min_sum, max_sum=max_sum))

        # 4. 奇偶极端过滤
        if exclude_extreme_odd_even:
            filters.append(OddEvenFilter())

        # 5. 三区极端过滤
        if exclude_single_zone:
            filters.append(ZoneFilter())

        logger.info(
            f"过滤管道就绪: {len(filters)} 条规则 — "
            + ", ".join(f.name for f in filters)
        )
        return cls(filters=filters)
