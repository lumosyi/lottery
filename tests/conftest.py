from __future__ import annotations

from datetime import date, timedelta

import pytest

from lottery.types import LotteryRecord


def make_records(count: int = 40) -> list[LotteryRecord]:
    offsets = [0, 3, 7, 11, 17, 23]
    start = date(2024, 1, 1)
    records: list[LotteryRecord] = []

    for index in range(count):
        reds = sorted((((index + offset) % 33) + 1) for offset in offsets)
        records.append(
            LotteryRecord(
                issue=f"2024{index + 1:03d}",
                draw_date=start + timedelta(days=index * 2),
                red_balls=tuple(reds),
                blue_ball=(index % 16) + 1,
            )
        )

    return records


@pytest.fixture
def sample_records() -> list[LotteryRecord]:
    return make_records()
