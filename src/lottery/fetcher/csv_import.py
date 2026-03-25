"""CSV 文件导入实现"""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

from loguru import logger

from lottery.fetcher.base import BaseFetcher
from lottery.types import LotteryRecord


class CsvFetcher(BaseFetcher):
    """CSV 文件导入器

    支持的 CSV 格式（需包含表头）：
    - 期号,开奖日期,红球1,红球2,红球3,红球4,红球5,红球6,蓝球
    - issue,draw_date,red_1,red_2,red_3,red_4,red_5,red_6,blue_ball
    """

    # 支持的列名映射
    _COLUMN_MAP = {
        # 期号
        "期号": "issue",
        "issue": "issue",
        "期次": "issue",
        # 日期
        "开奖日期": "draw_date",
        "draw_date": "draw_date",
        "日期": "draw_date",
        "date": "draw_date",
        # 红球
        "红球1": "red_1",
        "red_1": "red_1",
        "红球2": "red_2",
        "red_2": "red_2",
        "红球3": "red_3",
        "red_3": "red_3",
        "红球4": "red_4",
        "red_4": "red_4",
        "红球5": "red_5",
        "red_5": "red_5",
        "红球6": "red_6",
        "red_6": "red_6",
        # 蓝球
        "蓝球": "blue_ball",
        "blue_ball": "blue_ball",
        "蓝球1": "blue_ball",
    }

    def __init__(self, file_path: str | Path) -> None:
        self._file_path = Path(file_path)
        if not self._file_path.exists():
            raise FileNotFoundError(f"CSV 文件不存在: {self._file_path}")

    def fetch(
        self,
        start_issue: str | None = None,
        end_issue: str | None = None,
    ) -> list[LotteryRecord]:
        """从 CSV 加载数据，按期号范围过滤"""
        records = self._load_all()

        if start_issue:
            records = [r for r in records if r.issue >= start_issue]
        if end_issue:
            records = [r for r in records if r.issue <= end_issue]

        records.sort(key=lambda r: r.issue)
        logger.info(f"从 CSV 加载 {len(records)} 条记录")
        return records

    def fetch_latest(self, count: int = 100) -> list[LotteryRecord]:
        """从 CSV 加载最近 N 期数据"""
        records = self._load_all()
        records.sort(key=lambda r: r.issue)
        return records[-count:]

    def _load_all(self) -> list[LotteryRecord]:
        """加载 CSV 中的全部记录"""
        records: list[LotteryRecord] = []

        with open(self._file_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV 文件无表头")

            # 映射列名
            col_map = self._resolve_columns(reader.fieldnames)

            for line_num, row in enumerate(reader, start=2):
                try:
                    record = self._parse_row(row, col_map)
                    records.append(record)
                except (ValueError, KeyError) as e:
                    logger.warning(f"第 {line_num} 行解析失败: {e}")

        return records

    def _resolve_columns(self, fieldnames: list[str]) -> dict[str, str]:
        """将 CSV 列名映射为标准字段名"""
        col_map: dict[str, str] = {}
        for name in fieldnames:
            stripped = name.strip()
            if stripped in self._COLUMN_MAP:
                col_map[name] = self._COLUMN_MAP[stripped]
            else:
                logger.debug(f"忽略未识别的列: {stripped}")
        return col_map

    @staticmethod
    def _parse_row(row: dict[str, str], col_map: dict[str, str]) -> LotteryRecord:
        """解析单行 CSV 数据为 LotteryRecord"""
        # 反转映射: 标准字段名 -> CSV 列名
        reverse_map: dict[str, str] = {v: k for k, v in col_map.items()}

        issue = row[reverse_map["issue"]].strip()
        date_str = row[reverse_map["draw_date"]].strip()

        # 解析日期（支持多种格式）
        draw_date = _parse_date(date_str)

        red_balls = tuple(
            int(row[reverse_map[f"red_{i}"]].strip()) for i in range(1, 7)
        )
        blue_ball = int(row[reverse_map["blue_ball"]].strip())

        return LotteryRecord(
            issue=issue,
            draw_date=draw_date,
            red_balls=tuple(sorted(red_balls)),
            blue_ball=blue_ball,
        )


def _parse_date(date_str: str) -> date:
    """解析多种日期格式"""
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y年%m月%d日"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"无法解析日期: {date_str}")
