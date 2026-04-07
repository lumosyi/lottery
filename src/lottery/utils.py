"""公共工具函数"""

from __future__ import annotations

import random

import numpy as np
from loguru import logger

from lottery.types import Prediction


def weighted_sample_no_replace(
    population: list[int], weights: list[float], k: int, rng: random.Random | None = None
) -> list[int]:
    """带权重的不重复采样

    从 population 中按 weights 概率采样 k 个不重复元素。

    Args:
        population: 候选元素列表
        weights: 各元素的权重（无需归一化）
        k: 采样个数

    Returns:
        采样结果列表（无序）
    """
    if len(population) != len(weights):
        raise ValueError("population 和 weights 长度必须一致")
    if k < 0 or k > len(population):
        raise ValueError("k 超出可采样范围")

    picker = rng or random
    selected: list[int] = []
    remaining_items = list(population)
    remaining_weights = list(weights)

    for _ in range(k):
        if sum(remaining_weights) <= 0:
            raise ValueError("weights 总和必须大于 0")
        pos = picker.choices(range(len(remaining_items)), weights=remaining_weights, k=1)[0]
        selected.append(remaining_items.pop(pos))
        remaining_weights.pop(pos)

    return selected


def dedupe_predictions(
    predictions: list[Prediction],
    *,
    limit: int | None = None,
) -> tuple[list[Prediction], int]:
    """按号码组合去重预测结果，保留首次出现的条目。"""
    seen: set[tuple[tuple[int, ...], int]] = set()
    unique_predictions: list[Prediction] = []
    duplicate_count = 0

    for prediction in predictions:
        key = (prediction.red_balls, prediction.blue_ball)
        if key in seen:
            duplicate_count += 1
            continue

        seen.add(key)
        unique_predictions.append(prediction)
        if limit is not None and len(unique_predictions) >= limit:
            break

    return unique_predictions, duplicate_count


def set_random_seed(seed: int | None) -> None:
    """统一设置随机种子，便于复现实验。"""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    logger.debug(f"随机种子已设置为 {seed}")
