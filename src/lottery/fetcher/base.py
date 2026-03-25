"""数据采集器抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod

from lottery.types import LotteryRecord


class BaseFetcher(ABC):
    """数据采集器抽象基类

    子类需实现 fetch 和 fetch_latest 方法，
    分别用于采集指定范围和最近 N 期的数据。
    """

    @abstractmethod
    def fetch(
        self,
        start_issue: str | None = None,
        end_issue: str | None = None,
    ) -> list[LotteryRecord]:
        """采集指定范围的开奖数据

        Args:
            start_issue: 起始期号（含），None 表示不限
            end_issue: 结束期号（含），None 表示不限

        Returns:
            按期号升序排列的记录列表
        """
        ...

    @abstractmethod
    def fetch_latest(self, count: int = 100) -> list[LotteryRecord]:
        """采集最近 N 期数据

        Args:
            count: 期数

        Returns:
            按期号升序排列的记录列表
        """
        ...
