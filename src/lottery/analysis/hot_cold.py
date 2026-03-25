"""冷热号分析器"""

from __future__ import annotations

from collections import Counter

from lottery.analysis.base import BaseAnalyzer
from lottery.constants import ALL_BLUE_BALLS, ALL_RED_BALLS
from lottery.types import AnalysisResult, LotteryRecord


class HotColdAnalyzer(BaseAnalyzer):
    """冷热号分析器

    基于最近 N 期（窗口期）的出现频率，将号码分为热号、温号、冷号三类。
    - 热号: 出现次数 >= 平均值 + 1
    - 冷号: 出现次数 <= cold_threshold 或 0 次
    - 温号: 介于两者之间
    """

    def __init__(self, hot_window: int = 10, cold_threshold: int = 1) -> None:
        self._hot_window = hot_window
        self._cold_threshold = cold_threshold

    @property
    def name(self) -> str:
        return "冷热号分析"

    def analyze(self, records: list[LotteryRecord]) -> AnalysisResult:
        total = len(records)
        if total == 0:
            return AnalysisResult(name=self.name, data={}, summary="无数据")

        # 取最近 N 期
        window = min(self._hot_window, total)
        recent = records[-window:]

        red_result = self._classify(recent, "red")
        blue_result = self._classify(recent, "blue")

        hot_red = sorted(red_result["hot"])
        cold_red = sorted(red_result["cold"])

        hot_str = ", ".join(f"{n:02d}" for n in hot_red[:5])
        cold_str = ", ".join(f"{n:02d}" for n in cold_red[:5])

        summary = (
            f"最近 {window} 期 | "
            f"红球热号: {hot_str} | "
            f"红球冷号: {cold_str}"
        )

        return AnalysisResult(
            name=self.name,
            data={
                "red": red_result,
                "blue": blue_result,
                "window": window,
            },
            summary=summary,
        )

    def _classify(
        self, records: list[LotteryRecord], ball_type: str
    ) -> dict[str, list[int]]:
        """将号码分类为热/温/冷

        基于相对分位数:
        - 热号: 出现次数 > 平均值（偏高区间）
        - 冷号: 出现次数 = 0 或低于平均值的一半
        - 温号: 介于两者之间
        """
        all_balls = ALL_RED_BALLS if ball_type == "red" else ALL_BLUE_BALLS

        counter: Counter[int] = Counter()
        for r in records:
            if ball_type == "red":
                counter.update(r.red_balls)
            else:
                counter[r.blue_ball] += 1

        # 计算平均出现次数
        counts = [counter.get(b, 0) for b in all_balls]
        avg = sum(counts) / len(counts) if counts else 0

        # 按出现次数排名分段: 前1/3热号，后1/3冷号，中间温号
        ball_counts = [(b, counter.get(b, 0)) for b in all_balls]
        ball_counts.sort(key=lambda x: x[1], reverse=True)

        n = len(ball_counts)
        hot_cutoff = n // 3
        cold_cutoff = 2 * n // 3

        hot: list[int] = [b for b, _ in ball_counts[:hot_cutoff]]
        warm: list[int] = [b for b, _ in ball_counts[hot_cutoff:cold_cutoff]]
        cold: list[int] = [b for b, _ in ball_counts[cold_cutoff:]]

        return {"hot": hot, "warm": warm, "cold": cold, "counts": dict(counter)}
