"""预测器抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from lottery.types import LotteryRecord, Prediction


class BasePredictor(ABC):
    """预测器抽象基类

    所有预测器需实现 name 属性、train 和 predict 方法。
    save/load 为可选覆盖。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """预测器名称"""
        ...

    @abstractmethod
    def train(self, records: list[LotteryRecord]) -> None:
        """使用历史数据训练模型

        Args:
            records: 按期号升序排列的历史记录
        """
        ...

    @abstractmethod
    def predict(self, records: list[LotteryRecord], n_sets: int = 1) -> list[Prediction]:
        """生成预测结果

        Args:
            records: 历史记录（用于构建预测特征）
            n_sets: 生成推荐组数

        Returns:
            预测结果列表
        """
        ...

    def save(self, path: Path) -> None:
        """持久化模型（可选覆盖）"""
        pass

    def load(self, path: Path) -> None:
        """加载已训练模型（可选覆盖）"""
        pass
