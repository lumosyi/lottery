"""数据存储接口定义"""

from __future__ import annotations

from typing import Protocol

from lottery.types import LotteryRecord


class DataStore(Protocol):
    """数据存储接口（结构化子类型）

    所有存储实现需满足此协议，无需显式继承。
    """

    def save(self, records: list[LotteryRecord]) -> int:
        """保存开奖记录，返回新增条数"""
        ...

    def load_all(self) -> list[LotteryRecord]:
        """加载全部记录（按期号升序）"""
        ...

    def load_recent(self, count: int) -> list[LotteryRecord]:
        """加载最近 N 期记录（按期号升序）"""
        ...

    def load_by_range(self, start_issue: str, end_issue: str) -> list[LotteryRecord]:
        """按期号范围加载记录"""
        ...

    def get_latest_issue(self) -> str | None:
        """获取最新期号，无数据时返回 None"""
        ...

    def count(self) -> int:
        """获取记录总数"""
        ...
