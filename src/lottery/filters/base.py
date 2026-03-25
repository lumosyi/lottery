"""预测结果过滤器抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod

from lottery.types import LotteryRecord


class PredictionFilter(ABC):
    """预测结果过滤器基类

    每个过滤器实现一种排除规则，判断给定的号码组合是否应被排除。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """过滤器名称"""
        ...

    @abstractmethod
    def should_exclude(
        self,
        red_balls: tuple[int, ...],
        blue_ball: int,
        records: list[LotteryRecord],
    ) -> tuple[bool, str]:
        """判断组合是否应被排除

        Args:
            red_balls: 红球号码（升序）
            blue_ball: 蓝球号码
            records: 历史记录（用于重复检测等）

        Returns:
            (是否排除, 排除原因描述)
            如果不排除，原因为空字符串
        """
        ...
