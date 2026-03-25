"""特征变换辅助函数"""

from __future__ import annotations

from collections import Counter

from lottery.constants import ALL_BLUE_BALLS, ALL_RED_BALLS, ZONE_1, ZONE_2, ZONE_3
from lottery.types import LotteryRecord


def calc_frequency(records: list[LotteryRecord], ball_type: str = "red") -> dict[int, int]:
    """计算号码出现频次"""
    counter: Counter[int] = Counter()
    for r in records:
        if ball_type == "red":
            counter.update(r.red_balls)
        else:
            counter[r.blue_ball] += 1
    return dict(counter)


def calc_missing(records: list[LotteryRecord], ball_type: str = "red") -> dict[int, int]:
    """计算每个号码的当前遗漏值（距最后一次出现的期数）"""
    all_balls = ALL_RED_BALLS if ball_type == "red" else ALL_BLUE_BALLS
    total = len(records)
    missing: dict[int, int] = {}

    for ball in all_balls:
        last_seen = -1
        for i in range(total - 1, -1, -1):
            if ball_type == "red":
                if ball in records[i].red_balls:
                    last_seen = i
                    break
            else:
                if ball == records[i].blue_ball:
                    last_seen = i
                    break
        missing[ball] = total - 1 - last_seen if last_seen >= 0 else total

    return missing


def calc_span(red_balls: tuple[int, ...]) -> int:
    """计算红球跨度（最大值 - 最小值）"""
    return red_balls[-1] - red_balls[0]


def calc_ac_value(red_balls: tuple[int, ...]) -> int:
    """计算 AC 值（号码复杂度指标）

    AC = 不同差值的个数 - (球数 - 1)
    差值: 所有两两组合的绝对差
    """
    diffs = set()
    balls = list(red_balls)
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            diffs.add(abs(balls[i] - balls[j]))
    return len(diffs) - (len(balls) - 1)


def count_consecutive(red_balls: tuple[int, ...]) -> int:
    """计算连号个数（相邻号码差为1的对数）"""
    count = 0
    for i in range(1, len(red_balls)):
        if red_balls[i] - red_balls[i - 1] == 1:
            count += 1
    return count


def calc_odd_even_ratio(red_balls: tuple[int, ...]) -> tuple[int, int]:
    """计算奇偶比 (奇数个数, 偶数个数)"""
    odd = sum(1 for b in red_balls if b % 2 == 1)
    return odd, len(red_balls) - odd


def calc_big_small_ratio(red_balls: tuple[int, ...]) -> tuple[int, int]:
    """计算大小比 (大号个数, 小号个数)，以17为分界"""
    big = sum(1 for b in red_balls if b >= 17)
    return big, len(red_balls) - big


def calc_zone_ratio(red_balls: tuple[int, ...]) -> tuple[int, int, int]:
    """计算三区比"""
    z1 = sum(1 for b in red_balls if b in ZONE_1)
    z2 = sum(1 for b in red_balls if b in ZONE_2)
    z3 = sum(1 for b in red_balls if b in ZONE_3)
    return z1, z2, z3


def count_repeat(
    current_balls: tuple[int, ...], prev_balls: tuple[int, ...]
) -> int:
    """计算与上期的重号个数"""
    return len(set(current_balls) & set(prev_balls))
