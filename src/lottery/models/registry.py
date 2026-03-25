"""预测器注册表"""

from __future__ import annotations

from typing import Any

from lottery.models.base import BasePredictor


class PredictorRegistry:
    """预测器注册表 - 管理所有可用预测器

    支持通过装饰器注册和按名称创建预测器。
    """

    _registry: dict[str, type[BasePredictor]] = {}

    @classmethod
    def register(cls, name: str):
        """装饰器: 注册预测器类

        Usage:
            @PredictorRegistry.register("statistical")
            class StatisticalPredictor(BasePredictor):
                ...
        """
        def wrapper(predictor_cls: type[BasePredictor]):
            cls._registry[name] = predictor_cls
            return predictor_cls
        return wrapper

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BasePredictor:
        """按名称创建预测器实例

        Args:
            name: 注册名称
            **kwargs: 传递给构造函数的参数

        Returns:
            预测器实例

        Raises:
            KeyError: 未注册的预测器名称
        """
        if name not in cls._registry:
            raise KeyError(
                f"未注册的预测器: {name}，可用: {cls.list_available()}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        """列出所有已注册的预测器名称"""
        return list(cls._registry.keys())
