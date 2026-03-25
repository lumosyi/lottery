"""网络爬取实现 - 从福彩官网 API 获取历史开奖数据"""

from __future__ import annotations

import time
from datetime import date, datetime

import requests
from loguru import logger

from lottery.fetcher.base import BaseFetcher
from lottery.types import LotteryRecord


class WebFetcher(BaseFetcher):
    """网络爬取器

    数据源: 中国福利彩票官网 API
    接口: /cwl_admin/front/cwlkj/search/kjxx/findDrawNotice
    """

    _DEFAULT_URL = (
        "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
    )

    def __init__(
        self,
        source_url: str | None = None,
        retry: int = 3,
        timeout: int = 15,
    ) -> None:
        self._url = source_url or self._DEFAULT_URL
        self._retry = retry
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Referer": "https://www.cwl.gov.cn/",
                "Accept": "application/json",
            }
        )

    def fetch(
        self,
        start_issue: str | None = None,
        end_issue: str | None = None,
    ) -> list[LotteryRecord]:
        """从官网 API 采集指定范围的数据"""
        all_records: list[LotteryRecord] = []
        page_no = 1
        page_size = 50

        while True:
            params = {
                "name": "ssq",
                "issueCount": "",
                "issueStart": start_issue or "",
                "issueEnd": end_issue or "",
                "dayStart": "",
                "dayEnd": "",
                "pageNo": page_no,
                "pageSize": page_size,
                "week": "",
                "systemType": "PC",
            }

            data = self._request_with_retry(params)
            if not data:
                break

            result = data.get("result")
            if not result:
                break

            for item in result:
                try:
                    record = self._parse_item(item)
                    all_records.append(record)
                except (ValueError, KeyError) as e:
                    logger.warning(f"解析记录失败: {e}, 原始数据: {item}")

            # 检查是否还有下一页
            total = data.get("countNum", 0)
            if page_no * page_size >= total:
                break

            page_no += 1
            time.sleep(0.5)  # 请求间隔，避免过于频繁

        all_records.sort(key=lambda r: r.issue)
        logger.info(f"从网络采集 {len(all_records)} 条记录")
        return all_records

    def fetch_latest(self, count: int = 100) -> list[LotteryRecord]:
        """采集最近 N 期数据"""
        all_records: list[LotteryRecord] = []
        page_no = 1
        page_size = min(count, 50)

        while len(all_records) < count:
            params = {
                "name": "ssq",
                "issueCount": "",
                "issueStart": "",
                "issueEnd": "",
                "dayStart": "",
                "dayEnd": "",
                "pageNo": page_no,
                "pageSize": page_size,
                "week": "",
                "systemType": "PC",
            }

            data = self._request_with_retry(params)
            if not data or not data.get("result"):
                break

            for item in data["result"]:
                try:
                    record = self._parse_item(item)
                    all_records.append(record)
                except (ValueError, KeyError) as e:
                    logger.warning(f"解析记录失败: {e}")

            if len(data["result"]) < page_size:
                break

            page_no += 1
            time.sleep(0.5)

        all_records.sort(key=lambda r: r.issue)
        return all_records[:count] if len(all_records) > count else all_records

    def fetch_since(self, last_issue: str) -> list[LotteryRecord]:
        """增量采集: 获取指定期号之后的全部新数据

        Args:
            last_issue: 数据库中最新的期号，将采集此期号之后的所有数据

        Returns:
            新记录列表（按期号升序）
        """
        # 先获取最近一批数据，过滤出比 last_issue 更新的
        all_new: list[LotteryRecord] = []
        page_no = 1
        page_size = 50
        found_old = False

        while not found_old:
            params = {
                "name": "ssq",
                "issueCount": "",
                "issueStart": "",
                "issueEnd": "",
                "dayStart": "",
                "dayEnd": "",
                "pageNo": page_no,
                "pageSize": page_size,
                "week": "",
                "systemType": "PC",
            }

            data = self._request_with_retry(params)
            if not data or not data.get("result"):
                break

            for item in data["result"]:
                try:
                    record = self._parse_item(item)
                    if record.issue <= last_issue:
                        found_old = True
                        break
                    all_new.append(record)
                except (ValueError, KeyError) as e:
                    logger.warning(f"解析记录失败: {e}")

            if not found_old and len(data["result"]) == page_size:
                page_no += 1
                time.sleep(0.5)
            else:
                break

        all_new.sort(key=lambda r: r.issue)
        logger.info(f"增量采集: 在 {last_issue} 之后发现 {len(all_new)} 条新记录")
        return all_new

    def _request_with_retry(self, params: dict) -> dict | None:
        """带重试的 HTTP 请求"""
        for attempt in range(1, self._retry + 1):
            try:
                resp = self._session.get(
                    self._url, params=params, timeout=self._timeout
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                logger.warning(f"请求失败 (第 {attempt}/{self._retry} 次): {e}")
                if attempt < self._retry:
                    time.sleep(2 * attempt)

        logger.error("请求全部失败，放弃")
        return None

    @staticmethod
    def _parse_item(item: dict) -> LotteryRecord:
        """解析 API 返回的单条记录

        API 返回格式示例:
        {
            "code": "2024001",
            "date": "2024-01-02(二)",
            "red": "01,05,16,18,25,30",
            "blue": "14",
            ...
        }
        """
        issue = item["code"]

        # 解析日期（移除括号中的星期信息）
        date_str = item["date"].split("(")[0].strip()
        draw_date = datetime.strptime(date_str, "%Y-%m-%d").date()

        # 解析红球
        red_str = item["red"]
        red_balls = tuple(sorted(int(x) for x in red_str.split(",")))

        # 解析蓝球
        blue_ball = int(item["blue"])

        return LotteryRecord(
            issue=issue,
            draw_date=draw_date,
            red_balls=red_balls,
            blue_ball=blue_ball,
        )
