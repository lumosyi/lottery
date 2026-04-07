"""特征工程构建器

将 LotteryRecord 列表转换为模型可用的特征矩阵。
为每个号码构建是否出现的标签，以及丰富的上下文特征。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lottery.constants import ALL_BLUE_BALLS, ALL_RED_BALLS, RED_BALL_COUNT
from lottery.features.transforms import (
    calc_ac_value,
    calc_big_small_ratio,
    calc_frequency,
    calc_missing,
    calc_odd_even_ratio,
    calc_span,
    calc_zone_ratio,
    count_consecutive,
    count_repeat,
)
from lottery.types import LotteryRecord


class FeatureBuilder:
    """特征矩阵构建器

    将历史开奖数据转换为机器学习模型可用的特征矩阵。
    每一行对应一期开奖，特征包含该期之前的统计信息。

    特征类别:
    - 近N期频率特征（多个窗口）
    - 遗漏值特征
    - 和值/跨度/AC值
    - 奇偶比/大小比
    - 三区比
    - 连号/重号特征
    - 周期性特征（星期几、月份）
    """

    def __init__(self, window_sizes: list[int] | None = None) -> None:
        self._window_sizes = sorted(window_sizes or [5, 10, 20, 50])
        # 训练时记录的有效窗口，确保预测时特征一致
        self._effective_windows: list[int] | None = None

    @property
    def window_sizes(self) -> list[int]:
        """当前配置的窗口列表。"""
        return list(self._window_sizes)

    @property
    def effective_windows(self) -> list[int] | None:
        """训练后生效的窗口列表。"""
        if self._effective_windows is None:
            return None
        return list(self._effective_windows)

    def snapshot(self) -> dict[str, list[int] | None]:
        """导出特征构建器状态，便于持久化。"""
        return {
            "window_sizes": self.window_sizes,
            "effective_windows": self.effective_windows,
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, list[int] | None]) -> "FeatureBuilder":
        """从持久化状态恢复构建器。"""
        builder = cls(window_sizes=snapshot.get("window_sizes"))
        effective_windows = snapshot.get("effective_windows")
        builder._effective_windows = list(effective_windows) if effective_windows else None
        return builder

    def _get_effective_windows(self, data_size: int) -> list[int]:
        """根据数据量计算有效窗口"""
        if self._effective_windows is not None:
            return self._effective_windows
        effective = [w for w in self._window_sizes if w < data_size]
        if not effective:
            effective = [min(max(data_size - 1, 1), 5)]
        return effective

    def build(self, records: list[LotteryRecord]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """构建特征矩阵和标签矩阵

        Args:
            records: 按期号升序排列的历史记录

        Returns:
            (features_df, labels_df)
            - features_df: 每行是一期的特征，基于该期之前的数据计算
            - labels_df: 每行是一期的标签，红球列为 red_01~red_33 (0/1)，蓝球列为 blue_01~blue_16 (0/1)
        """
        # 计算并记录有效窗口
        self._effective_windows = self._get_effective_windows(len(records))

        min_history = max(self._effective_windows)
        if len(records) <= min_history:
            min_history = max(min(len(records) // 2, 5), 2)

        # 使用有效窗口
        original_windows = self._window_sizes
        self._window_sizes = self._effective_windows

        features_list: list[dict] = []
        labels_list: list[dict] = []

        for i in range(min_history, len(records)):
            # 当前期作为标签
            current = records[i]
            # 之前的历史作为特征来源
            history = records[:i]

            features = self._extract_features(history, current)
            labels = self._extract_labels(current)

            features_list.append(features)
            labels_list.append(labels)

        features_df = pd.DataFrame(features_list)
        labels_df = pd.DataFrame(labels_list)

        # 恢复原始窗口设置
        self._window_sizes = original_windows

        return features_df, labels_df

    def build_prediction_features(self, records: list[LotteryRecord]) -> pd.DataFrame:
        """为预测构建特征（基于全部历史数据，不需要标签）

        Args:
            records: 全部历史记录

        Returns:
            单行 DataFrame，用于模型预测下一期
        """
        from datetime import timedelta

        # 使用训练时记录的有效窗口，保证特征列一致
        original_windows = self._window_sizes
        self._window_sizes = self._get_effective_windows(len(records))

        last = records[-1]
        next_date = last.draw_date + timedelta(days=2)

        features = self._extract_features(records, _dummy_record(next_date))

        self._window_sizes = original_windows
        return pd.DataFrame([features])

    def _extract_features(
        self, history: list[LotteryRecord], current: LotteryRecord
    ) -> dict:
        """提取单期的全部特征"""
        features: dict = {}

        # 1. 多窗口频率特征
        for window in self._window_sizes:
            recent = history[-window:] if len(history) >= window else history
            red_freq = calc_frequency(recent, "red")
            blue_freq = calc_frequency(recent, "blue")

            for ball in ALL_RED_BALLS:
                features[f"red_freq_{window}_{ball:02d}"] = red_freq.get(ball, 0)
            for ball in ALL_BLUE_BALLS:
                features[f"blue_freq_{window}_{ball:02d}"] = blue_freq.get(ball, 0)

        # 2. 遗漏值特征
        red_missing = calc_missing(history, "red")
        blue_missing = calc_missing(history, "blue")
        for ball in ALL_RED_BALLS:
            features[f"red_missing_{ball:02d}"] = red_missing[ball]
        for ball in ALL_BLUE_BALLS:
            features[f"blue_missing_{ball:02d}"] = blue_missing[ball]

        # 3. 上一期的统计特征
        last = history[-1]
        features["last_sum"] = sum(last.red_balls)
        features["last_span"] = calc_span(last.red_balls)
        features["last_ac"] = calc_ac_value(last.red_balls)
        features["last_consecutive"] = count_consecutive(last.red_balls)

        odd, even = calc_odd_even_ratio(last.red_balls)
        features["last_odd"] = odd
        features["last_even"] = even

        big, small = calc_big_small_ratio(last.red_balls)
        features["last_big"] = big
        features["last_small"] = small

        z1, z2, z3 = calc_zone_ratio(last.red_balls)
        features["last_zone1"] = z1
        features["last_zone2"] = z2
        features["last_zone3"] = z3

        features["last_blue"] = last.blue_ball

        # 4. 重号特征（与上一期比较）
        if len(history) >= 2:
            features["last_repeat"] = count_repeat(
                history[-1].red_balls, history[-2].red_balls
            )
        else:
            features["last_repeat"] = 0

        # 5. 周期性特征
        features["weekday"] = current.draw_date.weekday()
        features["month"] = current.draw_date.month

        # 6. 近期和值趋势（最近5期的和值均值和标准差）
        recent_sums = [sum(r.red_balls) for r in history[-5:]]
        features["recent_sum_mean"] = np.mean(recent_sums)
        features["recent_sum_std"] = np.std(recent_sums) if len(recent_sums) > 1 else 0

        return features

    @staticmethod
    def _extract_labels(record: LotteryRecord) -> dict:
        """提取标签（每个号码是否出现）"""
        labels: dict = {}
        red_set = set(record.red_balls)
        for ball in ALL_RED_BALLS:
            labels[f"red_{ball:02d}"] = 1 if ball in red_set else 0
        for ball in ALL_BLUE_BALLS:
            labels[f"blue_{ball:02d}"] = 1 if ball == record.blue_ball else 0
        return labels

    def get_feature_names(self) -> list[str]:
        """获取特征名列表（用于模型解释）"""
        names: list[str] = []

        for window in self._window_sizes:
            for ball in ALL_RED_BALLS:
                names.append(f"red_freq_{window}_{ball:02d}")
            for ball in ALL_BLUE_BALLS:
                names.append(f"blue_freq_{window}_{ball:02d}")

        for ball in ALL_RED_BALLS:
            names.append(f"red_missing_{ball:02d}")
        for ball in ALL_BLUE_BALLS:
            names.append(f"blue_missing_{ball:02d}")

        names.extend([
            "last_sum", "last_span", "last_ac", "last_consecutive",
            "last_odd", "last_even", "last_big", "last_small",
            "last_zone1", "last_zone2", "last_zone3", "last_blue",
            "last_repeat", "weekday", "month",
            "recent_sum_mean", "recent_sum_std",
        ])

        return names


def _dummy_record(draw_date) -> LotteryRecord:
    """创建一个用于特征提取的虚拟记录（仅使用日期字段）"""
    from datetime import date as date_type

    return LotteryRecord(
        issue="predict",
        draw_date=draw_date if isinstance(draw_date, date_type) else date_type.today(),
        red_balls=(1, 2, 3, 4, 5, 6),
        blue_ball=1,
    )
