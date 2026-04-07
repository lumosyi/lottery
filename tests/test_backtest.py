from __future__ import annotations

from lottery.backtest import build_backtest_target_indices


def test_build_backtest_target_indices_respects_step_and_min_history():
    indices = build_backtest_target_indices(
        total_records=20,
        holdout=10,
        step=3,
        min_history=12,
    )

    assert indices == [12, 15, 18, 19]


def test_build_backtest_target_indices_returns_empty_when_no_eligible_periods():
    indices = build_backtest_target_indices(
        total_records=20,
        holdout=5,
        step=1,
        min_history=20,
    )

    assert indices == []
