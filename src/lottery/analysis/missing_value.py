"""遗漏值分析器"""

from __future__ import annotations

from lottery.analysis.base import BaseAnalyzer
from lottery.constants import ALL_BLUE_BALLS, ALL_RED_BALLS
from lottery.types import AnalysisResult, LotteryRecord


class MissingValueAnalyzer(BaseAnalyzer):
    """遗漏值分析器

    统计每个号码距上次出现的间隔期数（当前遗漏值），
    以及历史平均遗漏值和最大遗漏值。
    """

    @property
    def name(self) -> str:
        return "遗漏值分析"

    def analyze(self, records: list[LotteryRecord]) -> AnalysisResult:
        total = len(records)
        if total == 0:
            return AnalysisResult(name=self.name, data={}, summary="无数据")

        red_missing = self._calc_missing(records, ball_type="red")
        blue_missing = self._calc_missing(records, ball_type="blue")

        # 找出当前遗漏最大的号码
        red_max_current = max(red_missing.items(), key=lambda x: x[1]["current"])
        blue_max_current = max(blue_missing.items(), key=lambda x: x[1]["current"])

        summary = (
            f"共分析 {total} 期 | "
            f"红球最大遗漏: {red_max_current[0]:02d}({red_max_current[1]['current']}期) | "
            f"蓝球最大遗漏: {blue_max_current[0]:02d}({blue_max_current[1]['current']}期)"
        )

        return AnalysisResult(
            name=self.name,
            data={"red": red_missing, "blue": blue_missing, "total_periods": total},
            summary=summary,
        )

    @staticmethod
    def _calc_missing(
        records: list[LotteryRecord], ball_type: str
    ) -> dict[int, dict[str, int | float]]:
        """计算遗漏值统计

        Returns:
            {号码: {"current": 当前遗漏, "avg": 平均遗漏, "max": 最大遗漏}}
        """
        all_balls = ALL_RED_BALLS if ball_type == "red" else ALL_BLUE_BALLS
        total = len(records)

        result: dict[int, dict[str, int | float]] = {}

        for ball in all_balls:
            # 记录每次出现的位置
            appearances: list[int] = []
            for i, r in enumerate(records):
                if ball_type == "red":
                    if ball in r.red_balls:
                        appearances.append(i)
                else:
                    if ball == r.blue_ball:
                        appearances.append(i)

            if not appearances:
                result[ball] = {"current": total, "avg": total, "max": total}
                continue

            # 当前遗漏 = 最后一期到最新期的距离
            current = total - 1 - appearances[-1]

            # 计算每次间隔
            gaps: list[int] = [appearances[0]]  # 首次出现前的遗漏
            for j in range(1, len(appearances)):
                gaps.append(appearances[j] - appearances[j - 1] - 1)

            avg_gap = sum(gaps) / len(gaps) if gaps else 0
            max_gap = max(gaps) if gaps else 0

            result[ball] = {
                "current": current,
                "avg": round(avg_gap, 1),
                "max": max(max_gap, current),
            }

        return result
