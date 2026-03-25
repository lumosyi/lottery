"""号码模式分析器 — 连号/重号/极端组合的历史统计

用于发现极低概率模式，为排除过滤器提供阈值依据。
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from lottery.analysis.base import BaseAnalyzer
from lottery.features.transforms import (
    calc_odd_even_ratio,
    calc_zone_ratio,
    count_consecutive,
    count_repeat,
)
from lottery.types import AnalysisResult, LotteryRecord


class PatternAnalyzer(BaseAnalyzer):
    """号码模式分析器

    统计历史数据中的:
    1. 连号模式（最大连号长度及分布）
    2. 重复开奖（完全相同红球/相邻期重号个数）
    3. 和值极端范围
    4. 奇偶/三区极端分布
    """

    @property
    def name(self) -> str:
        return "模式分析"

    def analyze(self, records: list[LotteryRecord]) -> AnalysisResult:
        total = len(records)
        if total == 0:
            return AnalysisResult(name=self.name, data={}, summary="无数据")

        # 1. 连号统计
        consecutive = self._analyze_consecutive(records)

        # 2. 重复开奖统计
        repeat = self._analyze_repeat(records)

        # 3. 和值统计
        sum_stats = self._analyze_sum_range(records)

        # 4. 奇偶极端
        odd_even = self._analyze_odd_even_extreme(records)

        # 5. 三区极端
        zone = self._analyze_zone_extreme(records)

        # 汇总摘要
        summary_parts = [
            f"{total} 期",
            f"最大连号: {consecutive['max_length']}连",
            f"和值95%区间: [{sum_stats['p2_5']}, {sum_stats['p97_5']}]",
        ]
        if repeat["full_repeat_count"] == 0:
            summary_parts.append("无完全重复开奖")

        return AnalysisResult(
            name=self.name,
            data={
                "consecutive": consecutive,
                "repeat": repeat,
                "sum_range": sum_stats,
                "odd_even_extreme": odd_even,
                "zone_extreme": zone,
                "total": total,
            },
            summary=" | ".join(summary_parts),
        )

    @staticmethod
    def _analyze_consecutive(records: list[LotteryRecord]) -> dict:
        """分析连号模式"""
        # 统计每期的最大连号长度（连续号码的最长段）
        max_lengths: list[int] = []
        consecutive_pairs: list[int] = []

        for r in records:
            pairs = count_consecutive(r.red_balls)
            consecutive_pairs.append(pairs)

            # 计算最大连号段长度
            max_len = 1
            current_len = 1
            for i in range(1, len(r.red_balls)):
                if r.red_balls[i] - r.red_balls[i - 1] == 1:
                    current_len += 1
                    max_len = max(max_len, current_len)
                else:
                    current_len = 1
            max_lengths.append(max_len)

        # 连号段长度分布
        length_dist = Counter(max_lengths)
        # 有连号的期数（最大段 >= 2）
        has_consecutive = sum(1 for l in max_lengths if l >= 2)

        return {
            "max_length": max(max_lengths),
            "length_distribution": dict(sorted(length_dist.items())),
            "has_consecutive_rate": round(has_consecutive / len(records), 4),
            "avg_pairs": round(np.mean(consecutive_pairs), 2),
            # 各长度出现次数
            "len_2": length_dist.get(2, 0),
            "len_3": length_dist.get(3, 0),
            "len_4": length_dist.get(4, 0),
            "len_5": length_dist.get(5, 0),
            "len_6": length_dist.get(6, 0),
        }

    @staticmethod
    def _analyze_repeat(records: list[LotteryRecord]) -> dict:
        """分析重复开奖"""
        # 完全相同的红球组合
        seen: dict[tuple[int, ...], list[str]] = {}
        for r in records:
            key = r.red_balls
            seen.setdefault(key, []).append(r.issue)

        full_repeats = {k: v for k, v in seen.items() if len(v) > 1}

        # 相邻期重号个数分布
        adjacent_repeat_counts: list[int] = []
        for i in range(1, len(records)):
            cnt = count_repeat(records[i].red_balls, records[i - 1].red_balls)
            adjacent_repeat_counts.append(cnt)

        repeat_dist = Counter(adjacent_repeat_counts)

        return {
            "full_repeat_count": len(full_repeats),
            "full_repeats": {
                " ".join(f"{b:02d}" for b in k): v
                for k, v in list(full_repeats.items())[:5]
            },
            "adjacent_repeat_distribution": dict(sorted(repeat_dist.items())),
            "avg_adjacent_repeat": round(np.mean(adjacent_repeat_counts), 2) if adjacent_repeat_counts else 0,
        }

    @staticmethod
    def _analyze_sum_range(records: list[LotteryRecord]) -> dict:
        """分析和值范围"""
        sums = [sum(r.red_balls) for r in records]
        return {
            "min": min(sums),
            "max": max(sums),
            "mean": round(np.mean(sums), 1),
            "std": round(np.std(sums), 1),
            "p2_5": int(np.percentile(sums, 2.5)),
            "p97_5": int(np.percentile(sums, 97.5)),
            "p5": int(np.percentile(sums, 5)),
            "p95": int(np.percentile(sums, 95)),
        }

    @staticmethod
    def _analyze_odd_even_extreme(records: list[LotteryRecord]) -> dict:
        """分析奇偶极端分布"""
        dist: Counter[str] = Counter()
        for r in records:
            odd, even = calc_odd_even_ratio(r.red_balls)
            dist[f"{odd}:{even}"] += 1

        total = len(records)
        return {
            "distribution": {k: {"count": v, "rate": round(v / total, 4)} for k, v in sorted(dist.items())},
            "all_odd_count": dist.get("6:0", 0),      # 全奇
            "all_even_count": dist.get("0:6", 0),      # 全偶
            "all_odd_rate": round(dist.get("6:0", 0) / total, 4),
            "all_even_rate": round(dist.get("0:6", 0) / total, 4),
        }

    @staticmethod
    def _analyze_zone_extreme(records: list[LotteryRecord]) -> dict:
        """分析三区极端分布"""
        dist: Counter[str] = Counter()
        for r in records:
            z1, z2, z3 = calc_zone_ratio(r.red_balls)
            dist[f"{z1}:{z2}:{z3}"] += 1

        total = len(records)
        # 单区全出: 6:0:0 / 0:6:0 / 0:0:6
        single_zone = dist.get("6:0:0", 0) + dist.get("0:6:0", 0) + dist.get("0:0:6", 0)

        return {
            "single_zone_count": single_zone,
            "single_zone_rate": round(single_zone / total, 4),
            # 也统计双区为0的情况（如 5:1:0, 0:5:1 等有一区为0）
            "has_empty_zone_count": sum(
                v for k, v in dist.items()
                if any(int(x) == 0 for x in k.split(":"))
            ),
        }
