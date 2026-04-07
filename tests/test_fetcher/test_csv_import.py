from __future__ import annotations

import pytest

from lottery.fetcher.csv_import import CsvFetcher


def test_csv_fetcher_fails_fast_when_required_columns_missing(tmp_path):
    csv_path = tmp_path / "broken.csv"
    csv_path.write_text("issue,draw_date,red_1\n2024001,2024-01-01,1\n", encoding="utf-8")

    fetcher = CsvFetcher(csv_path)

    with pytest.raises(ValueError, match="CSV 缺少必要列"):
        fetcher.fetch_latest()
