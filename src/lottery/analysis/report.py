"""分析报告聚合器"""

from __future__ import annotations

from lottery.analysis.base import BaseAnalyzer
from lottery.analysis.frequency import FrequencyAnalyzer
from lottery.analysis.hot_cold import HotColdAnalyzer
from lottery.analysis.missing_value import MissingValueAnalyzer
from lottery.analysis.odd_even import OddEvenAnalyzer
from lottery.analysis.pattern import PatternAnalyzer
from lottery.analysis.sum_value import SumAnalyzer
from lottery.analysis.zone import ZoneAnalyzer
from lottery.types import AnalysisResult, LotteryRecord


class AnalysisReport:
    """聚合多个分析器的结果报告

    可通过 default() 创建包含全部内置分析器的实例，
    也可自定义分析器组合。
    """

    def __init__(self, analyzers: list[BaseAnalyzer] | None = None) -> None:
        self._analyzers = analyzers or []

    def generate(self, records: list[LotteryRecord]) -> list[AnalysisResult]:
        """运行所有分析器并返回结果列表"""
        return [analyzer.analyze(records) for analyzer in self._analyzers]

    @classmethod
    def default(cls, hot_window: int = 10, cold_threshold: int = 1) -> "AnalysisReport":
        """创建包含所有内置分析器的默认报告"""
        return cls(
            analyzers=[
                FrequencyAnalyzer(),
                MissingValueAnalyzer(),
                HotColdAnalyzer(hot_window=hot_window, cold_threshold=cold_threshold),
                SumAnalyzer(),
                OddEvenAnalyzer(),
                ZoneAnalyzer(),
                PatternAnalyzer(),
            ]
        )
