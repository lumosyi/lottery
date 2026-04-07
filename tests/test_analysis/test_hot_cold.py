from __future__ import annotations

from lottery.analysis.hot_cold import HotColdAnalyzer


def test_hot_cold_uses_cold_threshold(sample_records):
    records = sample_records[:6]

    threshold_zero = HotColdAnalyzer(hot_window=6, cold_threshold=0).analyze(records)
    threshold_one = HotColdAnalyzer(hot_window=6, cold_threshold=1).analyze(records)

    cold_zero = set(threshold_zero.data["red"]["cold"])
    cold_one = set(threshold_one.data["red"]["cold"])

    assert cold_zero < cold_one
