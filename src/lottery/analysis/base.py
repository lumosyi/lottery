"""分析器抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod

from lottery.types import AnalysisResult, LotteryRecord


class BaseAnalyzer(ABC):
    """统计分析器抽象基类

    所有分析器需实现 name 属性和 analyze 方法。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """分析器名称"""
        ...

    @abstractmethod
    def analyze(self, records: list[LotteryRecord]) -> AnalysisResult:
        """对历史记录执行分析

        Args:
            records: 按期号升序排列的历史记录

        Returns:
            分析结果
        """
        ...
