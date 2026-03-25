"""数据采集器工厂"""

from __future__ import annotations

from pathlib import Path

from lottery.fetcher.base import BaseFetcher
from lottery.fetcher.csv_import import CsvFetcher
from lottery.fetcher.web import WebFetcher


class FetcherFactory:
    """根据数据源类型创建对应的采集器"""

    @staticmethod
    def create(source_type: str, **kwargs) -> BaseFetcher:
        """创建数据采集器

        Args:
            source_type: 数据源类型，"web" 或 "csv"
            **kwargs: 传递给具体采集器的参数

        Returns:
            BaseFetcher 实例

        Raises:
            ValueError: 未知的数据源类型
        """
        match source_type:
            case "web":
                return WebFetcher(
                    source_url=kwargs.get("source_url"),
                    retry=kwargs.get("retry", 3),
                    timeout=kwargs.get("timeout", 15),
                )
            case "csv":
                file_path = kwargs.get("file_path") or kwargs.get("csv_path")
                if not file_path:
                    raise ValueError("CSV 模式需要指定 file_path 参数")
                return CsvFetcher(file_path=Path(file_path))
            case _:
                raise ValueError(
                    f"未知的数据源类型: {source_type}，支持: web, csv"
                )
