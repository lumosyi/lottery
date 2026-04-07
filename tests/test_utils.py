from __future__ import annotations

from lottery.types import Prediction
from lottery.utils import dedupe_predictions


def test_dedupe_predictions_removes_duplicate_number_sets():
    predictions = [
        Prediction(red_balls=(1, 2, 3, 4, 5, 6), blue_ball=7, score=0.8, source="A"),
        Prediction(red_balls=(1, 2, 3, 4, 5, 6), blue_ball=7, score=0.7, source="B"),
        Prediction(red_balls=(1, 2, 3, 4, 5, 8), blue_ball=7, score=0.6, source="A"),
    ]

    unique_predictions, duplicate_count = dedupe_predictions(predictions)

    assert duplicate_count == 1
    assert len(unique_predictions) == 2
    assert unique_predictions[0].source == "A"
    assert unique_predictions[1].red_balls == (1, 2, 3, 4, 5, 8)
