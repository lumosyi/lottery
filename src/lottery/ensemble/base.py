"""融合策略抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod

from lottery.types import Prediction


class EnsembleStrategy(ABC):
    """融合策略抽象基类"""

    @abstractmethod
    def fuse(self, predictions: list[Prediction], n_sets: int = 1) -> list[Prediction]:
        """将多个预测器的结果融合为最终推荐

        Args:
            predictions: 各预测器生成的全部预测结果
            n_sets: 输出组数

        Returns:
            融合后的推荐号码列表
        """
        ...
