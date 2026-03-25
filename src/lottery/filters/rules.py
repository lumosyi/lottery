"""具体排除规则实现"""

from __future__ import annotations

from lottery.constants import ZONE_1, ZONE_2, ZONE_3
from lottery.filters.base import PredictionFilter
from lottery.types import LotteryRecord


class ConsecutiveFilter(PredictionFilter):
    """连号过滤器 — 排除连号长度过大的组合

    例如 max_length=4 时，排除含 4 个及以上连续号码的组合。
    """

    def __init__(self, max_length: int = 4) -> None:
        self._max_length = max_length

    @property
    def name(self) -> str:
        return f"连号>={self._max_length}"

    def should_exclude(
        self, red_balls: tuple[int, ...], blue_ball: int,
        records: list[LotteryRecord],
    ) -> tuple[bool, str]:
        max_len = _max_consecutive_length(red_balls)
        if max_len >= self._max_length:
            return True, f"含{max_len}连号(阈值{self._max_length})"
        return False, ""


class RepeatFilter(PredictionFilter):
    """重复开奖过滤器 — 排除与最近 N 期红球完全相同的组合"""

    def __init__(self, recent_n: int = 10) -> None:
        self._recent_n = recent_n

    @property
    def name(self) -> str:
        return f"近{self._recent_n}期重复"

    def should_exclude(
        self, red_balls: tuple[int, ...], blue_ball: int,
        records: list[LotteryRecord],
    ) -> tuple[bool, str]:
        recent = records[-self._recent_n:] if len(records) >= self._recent_n else records
        for r in recent:
            if r.red_balls == red_balls:
                return True, f"与{r.issue}期红球完全相同"
        return False, ""


class SumRangeFilter(PredictionFilter):
    """和值范围过滤器 — 排除和值超出历史区间的组合"""

    def __init__(self, min_sum: int, max_sum: int) -> None:
        self._min_sum = min_sum
        self._max_sum = max_sum

    @property
    def name(self) -> str:
        return f"和值[{self._min_sum},{self._max_sum}]"

    def should_exclude(
        self, red_balls: tuple[int, ...], blue_ball: int,
        records: list[LotteryRecord],
    ) -> tuple[bool, str]:
        s = sum(red_balls)
        if s < self._min_sum:
            return True, f"和值{s}<{self._min_sum}"
        if s > self._max_sum:
            return True, f"和值{s}>{self._max_sum}"
        return False, ""


class OddEvenFilter(PredictionFilter):
    """奇偶极端过滤器 — 排除全奇或全偶的组合"""

    @property
    def name(self) -> str:
        return "全奇/全偶"

    def should_exclude(
        self, red_balls: tuple[int, ...], blue_ball: int,
        records: list[LotteryRecord],
    ) -> tuple[bool, str]:
        odd_count = sum(1 for b in red_balls if b % 2 == 1)
        if odd_count == 6:
            return True, "全奇数(6:0)"
        if odd_count == 0:
            return True, "全偶数(0:6)"
        return False, ""


class ZoneFilter(PredictionFilter):
    """三区极端过滤器 — 排除全部落在单区的组合"""

    @property
    def name(self) -> str:
        return "单区全出"

    def should_exclude(
        self, red_balls: tuple[int, ...], blue_ball: int,
        records: list[LotteryRecord],
    ) -> tuple[bool, str]:
        z1 = sum(1 for b in red_balls if b in ZONE_1)
        z2 = sum(1 for b in red_balls if b in ZONE_2)
        z3 = sum(1 for b in red_balls if b in ZONE_3)

        if z1 == 6:
            return True, "全落一区(6:0:0)"
        if z2 == 6:
            return True, "全落二区(0:6:0)"
        if z3 == 6:
            return True, "全落三区(0:0:6)"
        return False, ""


def _max_consecutive_length(red_balls: tuple[int, ...]) -> int:
    """计算红球中最长连续段的长度"""
    if not red_balls:
        return 0
    max_len = 1
    current = 1
    for i in range(1, len(red_balls)):
        if red_balls[i] - red_balls[i - 1] == 1:
            current += 1
            max_len = max(max_len, current)
        else:
            current = 1
    return max_len
