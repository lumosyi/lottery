"""SQLite 数据存储实现"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from loguru import logger

from lottery.types import LotteryRecord


class SqliteStore:
    """SQLite 存储实现

    使用内置 sqlite3 模块，零外部依赖。
    单表设计，满足双色球数据的全部查询需求。
    """

    def __init__(self, db_path: str | Path = "data/lottery.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_table()

    def _init_table(self) -> None:
        """创建数据表（如不存在）"""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS lottery_records (
                issue       TEXT PRIMARY KEY,
                draw_date   TEXT NOT NULL,
                red_1       INTEGER NOT NULL,
                red_2       INTEGER NOT NULL,
                red_3       INTEGER NOT NULL,
                red_4       INTEGER NOT NULL,
                red_5       INTEGER NOT NULL,
                red_6       INTEGER NOT NULL,
                blue_ball   INTEGER NOT NULL
            )
        """)
        self._conn.commit()

    def save(self, records: list[LotteryRecord]) -> int:
        """保存开奖记录，跳过已存在的期号，返回新增条数"""
        count_before = self.count()
        for record in records:
            try:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO lottery_records
                    (issue, draw_date, red_1, red_2, red_3, red_4, red_5, red_6, blue_ball)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.issue,
                        record.draw_date.isoformat(),
                        *record.red_balls,
                        record.blue_ball,
                    ),
                )
            except sqlite3.IntegrityError:
                logger.debug(f"期号 {record.issue} 已存在，跳过")
        self._conn.commit()
        inserted = self.count() - count_before
        logger.info(f"保存完成: 新增 {inserted} 条，共提交 {len(records)} 条")
        return inserted

    def get_latest_record(self) -> LotteryRecord | None:
        """获取最新一期完整记录"""
        cursor = self._conn.execute(
            "SELECT * FROM lottery_records ORDER BY issue DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return self._row_to_record(row) if row else None

    def get_oldest_issue(self) -> str | None:
        """获取最早期号"""
        cursor = self._conn.execute(
            "SELECT issue FROM lottery_records ORDER BY issue ASC LIMIT 1"
        )
        row = cursor.fetchone()
        return row["issue"] if row else None

    def load_all(self) -> list[LotteryRecord]:
        """加载全部记录（按期号升序）"""
        cursor = self._conn.execute(
            "SELECT * FROM lottery_records ORDER BY issue ASC"
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def load_recent(self, count: int) -> list[LotteryRecord]:
        """加载最近 N 期记录（返回按期号升序排列）"""
        cursor = self._conn.execute(
            "SELECT * FROM lottery_records ORDER BY issue DESC LIMIT ?",
            (count,),
        )
        rows = cursor.fetchall()
        return [self._row_to_record(row) for row in reversed(rows)]

    def load_by_range(self, start_issue: str, end_issue: str) -> list[LotteryRecord]:
        """按期号范围加载记录"""
        cursor = self._conn.execute(
            "SELECT * FROM lottery_records WHERE issue BETWEEN ? AND ? ORDER BY issue ASC",
            (start_issue, end_issue),
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_latest_issue(self) -> str | None:
        """获取最新期号"""
        cursor = self._conn.execute(
            "SELECT issue FROM lottery_records ORDER BY issue DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return row["issue"] if row else None

    def count(self) -> int:
        """获取记录总数"""
        cursor = self._conn.execute("SELECT COUNT(*) as cnt FROM lottery_records")
        return cursor.fetchone()["cnt"]

    def close(self) -> None:
        """关闭数据库连接"""
        self._conn.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> LotteryRecord:
        """将数据库行转换为 LotteryRecord"""
        return LotteryRecord(
            issue=row["issue"],
            draw_date=date.fromisoformat(row["draw_date"]),
            red_balls=(
                row["red_1"],
                row["red_2"],
                row["red_3"],
                row["red_4"],
                row["red_5"],
                row["red_6"],
            ),
            blue_ball=row["blue_ball"],
        )

    def __enter__(self) -> "SqliteStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()
