"""频率统计分析器"""

from __future__ import annotations

from collections import Counter

from lottery.analysis.base import BaseAnalyzer
from lottery.constants import ALL_BLUE_BALLS, ALL_RED_BALLS
from lottery.types import AnalysisResult, LotteryRecord


class FrequencyAnalyzer(BaseAnalyzer):
    """频率统计分析器

    统计每个号码在历史数据中出现的次数和频率百分比。
    """

    @property
    def name(self) -> str:
        return "频率统计"

    def analyze(self, records: list[LotteryRecord]) -> AnalysisResult:
        total = len(records)
        if total == 0:
            return AnalysisResult(name=self.name, data={}, summary="无数据")

        # 统计红球频次
        red_counter: Counter[int] = Counter()
        for r in records:
            red_counter.update(r.red_balls)

        # 统计蓝球频次
        blue_counter: Counter[int] = Counter()
        for r in records:
            blue_counter[r.blue_ball] += 1

        # 构建完整频率数据（含零出现的号码）
        red_freq = {
            n: {"count": red_counter.get(n, 0), "rate": red_counter.get(n, 0) / total}
            for n in ALL_RED_BALLS
        }
        blue_freq = {
            n: {"count": blue_counter.get(n, 0), "rate": blue_counter.get(n, 0) / total}
            for n in ALL_BLUE_BALLS
        }

        # 找出最高频和最低频号码
        red_top3 = red_counter.most_common(3)
        blue_top3 = blue_counter.most_common(3)

        red_top_str = ", ".join(f"{n:02d}({c}次)" for n, c in red_top3)
        blue_top_str = ", ".join(f"{n:02d}({c}次)" for n, c in blue_top3)

        summary = (
            f"共分析 {total} 期 | "
            f"红球最热: {red_top_str} | "
            f"蓝球最热: {blue_top_str}"
        )

        return AnalysisResult(
            name=self.name,
            data={"red": red_freq, "blue": blue_freq, "total_periods": total},
            summary=summary,
        )
