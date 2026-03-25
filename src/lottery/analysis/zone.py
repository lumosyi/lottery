"""区间分布分析器"""

from __future__ import annotations

from collections import Counter

from lottery.analysis.base import BaseAnalyzer
from lottery.constants import ZONE_1, ZONE_2, ZONE_3
from lottery.types import AnalysisResult, LotteryRecord


class ZoneAnalyzer(BaseAnalyzer):
    """区间分布分析器

    将红球划分为三个区间:
    - 一区: 01-11
    - 二区: 12-22
    - 三区: 23-33

    分析每期红球在三区的分布比例。
    """

    _ZONE_NAMES = {1: "一区(01-11)", 2: "二区(12-22)", 3: "三区(23-33)"}

    @property
    def name(self) -> str:
        return "区间分布"

    def analyze(self, records: list[LotteryRecord]) -> AnalysisResult:
        total = len(records)
        if total == 0:
            return AnalysisResult(name=self.name, data={}, summary="无数据")

        # 统计每期的三区比
        zone_ratios: list[str] = []
        ratio_counter: Counter[str] = Counter()

        # 统计各区总出球数
        zone_totals = {1: 0, 2: 0, 3: 0}

        for r in records:
            z1 = sum(1 for b in r.red_balls if b in ZONE_1)
            z2 = sum(1 for b in r.red_balls if b in ZONE_2)
            z3 = sum(1 for b in r.red_balls if b in ZONE_3)

            zone_totals[1] += z1
            zone_totals[2] += z2
            zone_totals[3] += z3

            ratio = f"{z1}:{z2}:{z3}"
            zone_ratios.append(ratio)
            ratio_counter[ratio] += 1

        # 比例分布
        ratio_dist = {
            k: {"count": v, "rate": v / total}
            for k, v in ratio_counter.most_common()
        }

        # 各区占比
        total_balls = total * 6
        zone_rates = {
            self._ZONE_NAMES[z]: {
                "total": zone_totals[z],
                "rate": zone_totals[z] / total_balls,
            }
            for z in [1, 2, 3]
        }

        latest_ratio = zone_ratios[-1]
        most_common = ratio_counter.most_common(1)[0]

        summary = (
            f"共分析 {total} 期 | "
            f"最新三区比: {latest_ratio} | "
            f"最常见: {most_common[0]}({most_common[1]}次)"
        )

        return AnalysisResult(
            name=self.name,
            data={
                "zone_ratios": zone_ratios,
                "distribution": ratio_dist,
                "zone_rates": zone_rates,
                "latest": latest_ratio,
                "total_periods": total,
            },
            summary=summary,
        )
