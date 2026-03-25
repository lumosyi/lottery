"""奇偶比分析器"""

from __future__ import annotations

from collections import Counter

from lottery.analysis.base import BaseAnalyzer
from lottery.types import AnalysisResult, LotteryRecord


class OddEvenAnalyzer(BaseAnalyzer):
    """奇偶比分析器

    分析红球中奇数和偶数的个数比例分布。
    6个红球的奇偶比可能为: 6:0, 5:1, 4:2, 3:3, 2:4, 1:5, 0:6
    """

    @property
    def name(self) -> str:
        return "奇偶比分析"

    def analyze(self, records: list[LotteryRecord]) -> AnalysisResult:
        total = len(records)
        if total == 0:
            return AnalysisResult(name=self.name, data={}, summary="无数据")

        # 统计每期的奇偶比
        ratios: list[str] = []
        ratio_counter: Counter[str] = Counter()

        for r in records:
            odd_count = sum(1 for b in r.red_balls if b % 2 == 1)
            even_count = 6 - odd_count
            ratio = f"{odd_count}:{even_count}"
            ratios.append(ratio)
            ratio_counter[ratio] += 1

        # 按出现次数排序
        ratio_dist = {
            k: {"count": v, "rate": v / total}
            for k, v in ratio_counter.most_common()
        }

        # 最近一期
        latest_ratio = ratios[-1]

        # 最常见比例
        most_common = ratio_counter.most_common(1)[0]

        summary = (
            f"共分析 {total} 期 | "
            f"最新奇偶比: {latest_ratio} | "
            f"最常见: {most_common[0]}({most_common[1]}次, {most_common[1]/total:.1%})"
        )

        return AnalysisResult(
            name=self.name,
            data={
                "ratios": ratios,
                "distribution": ratio_dist,
                "latest": latest_ratio,
                "total_periods": total,
            },
            summary=summary,
        )
