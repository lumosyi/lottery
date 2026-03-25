"""和值分析器"""

from __future__ import annotations

from lottery.analysis.base import BaseAnalyzer
from lottery.types import AnalysisResult, LotteryRecord


class SumAnalyzer(BaseAnalyzer):
    """和值分析器

    分析红球和值（6个红球之和）的分布特征和走势。
    理论范围: 21 (1+2+3+4+5+6) ~ 183 (28+29+30+31+32+33)
    """

    @property
    def name(self) -> str:
        return "和值分析"

    def analyze(self, records: list[LotteryRecord]) -> AnalysisResult:
        total = len(records)
        if total == 0:
            return AnalysisResult(name=self.name, data={}, summary="无数据")

        sums = [sum(r.red_balls) for r in records]

        avg_sum = sum(sums) / total
        min_sum = min(sums)
        max_sum = max(sums)
        latest_sum = sums[-1]

        # 和值区间分布
        ranges = {
            "21-60": 0,
            "61-80": 0,
            "81-100": 0,
            "101-120": 0,
            "121-140": 0,
            "141-183": 0,
        }
        for s in sums:
            if s <= 60:
                ranges["21-60"] += 1
            elif s <= 80:
                ranges["61-80"] += 1
            elif s <= 100:
                ranges["81-100"] += 1
            elif s <= 120:
                ranges["101-120"] += 1
            elif s <= 140:
                ranges["121-140"] += 1
            else:
                ranges["141-183"] += 1

        # 最近 10 期走势
        recent_trend = sums[-10:] if total >= 10 else sums

        summary = (
            f"共分析 {total} 期 | "
            f"平均和值: {avg_sum:.1f} | "
            f"最新: {latest_sum} | "
            f"范围: {min_sum}-{max_sum}"
        )

        return AnalysisResult(
            name=self.name,
            data={
                "sums": sums,
                "avg": round(avg_sum, 1),
                "min": min_sum,
                "max": max_sum,
                "latest": latest_sum,
                "ranges": ranges,
                "recent_trend": recent_trend,
                "total_periods": total,
            },
            summary=summary,
        )
