from __future__ import annotations

from lottery.filters.pipeline import FilterPipeline
from lottery.filters.rules import OddEvenFilter
from lottery.types import Prediction


def test_filter_pipeline_marks_predictions_without_mutating_input(sample_records):
    original = Prediction(
        red_balls=(1, 3, 5, 7, 9, 11),
        blue_ball=1,
        score=0.82,
        source="测试模型",
    )
    pipeline = FilterPipeline([OddEvenFilter()])

    filtered_predictions, stats = pipeline.filter_predictions([original], sample_records)

    assert stats["excluded"] == 1
    assert original.score == 0.82
    assert original.source == "测试模型"
    assert original.details == {}

    marked = filtered_predictions[0]
    assert marked.score == 0.82
    assert marked.source == "测试模型"
    assert marked.details["filtered"] is True
    assert marked.details["filter_reasons"] == ["全奇数(6:0)"]
