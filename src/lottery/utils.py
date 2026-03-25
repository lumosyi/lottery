"""公共工具函数"""

from __future__ import annotations

import random


def weighted_sample_no_replace(
    population: list[int], weights: list[float], k: int
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
    selected: list[int] = []
    # 使用索引操作避免重复 list.index() 查找
    indices = list(range(len(population)))
    remaining_weights = list(weights)

    for _ in range(k):
        chosen_idx = random.choices(indices, weights=remaining_weights, k=1)[0]
        pos = indices.index(chosen_idx)
        selected.append(population[chosen_idx])
        indices.pop(pos)
        remaining_weights.pop(pos)

    return selected
