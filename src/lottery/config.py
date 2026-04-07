"""配置加载与管理"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """数据源配置"""

    source: str = "web"
    web_url: str = (
        "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
    )
    csv_path: str = "data/raw/history.csv"
    db_path: str = "data/lottery.db"


@dataclass
class AnalysisConfig:
    """统计分析配置"""

    default_recent: int = 100
    hot_window: int = 10
    cold_threshold: int = 5


@dataclass
class FeatureConfig:
    """特征工程配置"""

    window_sizes: list[int] = field(default_factory=lambda: [5, 10, 20, 50])
    sequence_length: int = 30


@dataclass
class ModelItemConfig:
    """单个模型配置"""

    enabled: bool = True
    weight: float = 0.25
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelsConfig:
    """模型配置集合"""

    statistical: ModelItemConfig = field(default_factory=lambda: ModelItemConfig(weight=0.2))
    random_forest: ModelItemConfig = field(
        default_factory=lambda: ModelItemConfig(weight=0.25, params={"n_estimators": 200})
    )
    xgboost: ModelItemConfig = field(
        default_factory=lambda: ModelItemConfig(
            weight=0.3, params={"max_depth": 6, "n_estimators": 300}
        )
    )
    lstm: ModelItemConfig = field(
        default_factory=lambda: ModelItemConfig(
            weight=0.25,
            params={"hidden_size": 128, "num_layers": 2, "epochs": 100, "learning_rate": 0.001},
        )
    )


@dataclass
class EnsembleConfig:
    """融合策略配置"""

    strategy: str = "weighted_voting"


@dataclass
class FiltersConfig:
    """概率排除过滤器配置"""

    enabled: bool = True
    max_consecutive: int = 4
    repeat_check_recent: int = 10
    sum_range_percentile: float = 95
    exclude_extreme_odd_even: bool = True
    exclude_single_zone: bool = True


@dataclass
class OutputConfig:
    """输出配置"""

    charts_dir: str = "output/charts"
    chart_style: str = "seaborn-v0_8"


@dataclass
class RuntimeConfig:
    """运行时配置"""

    seed: int | None = 42


@dataclass
class AppConfig:
    """应用全局配置"""

    data: DataConfig = field(default_factory=DataConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _ensure_int(name: str, value: Any, *, minimum: int = 0) -> int:
    """校验整数配置。"""
    if not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数，当前: {value!r}")
    if value < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}，当前: {value}")
    return value


def _normalize_window_sizes(value: Any) -> list[int]:
    """规范化窗口配置，自动去重并升序。"""
    if not isinstance(value, list) or not value:
        raise ValueError(f"features.window_sizes 必须是非空整数列表，当前: {value!r}")

    normalized = sorted({_ensure_int("features.window_sizes[]", item, minimum=1) for item in value})
    return normalized


def _ensure_weight(name: str, value: Any) -> float:
    """校验模型权重。"""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数字，当前: {value!r}")
    if value < 0:
        raise ValueError(f"{name} 不能为负数，当前: {value}")
    return float(value)


def _ensure_percentile(name: str, value: Any) -> float:
    """校验百分位配置。"""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数字，当前: {value!r}")
    numeric = float(value)
    if not 50 <= numeric <= 100:
        raise ValueError(f"{name} 必须在 50~100 之间，当前: {value}")
    return numeric


def _parse_model_item(raw: dict[str, Any]) -> ModelItemConfig:
    """从原始字典解析单个模型配置，将非标准字段放入 params"""
    known_keys = {"enabled", "weight"}
    return ModelItemConfig(
        enabled=raw.get("enabled", True),
        weight=_ensure_weight("models.*.weight", raw.get("weight", 0.25)),
        params={k: v for k, v in raw.items() if k not in known_keys},
    )


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """加载 YAML 配置文件

    Args:
        config_path: 配置文件路径，默认为项目根目录下的 config.yaml

    Returns:
        解析后的 AppConfig 实例

    Raises:
        FileNotFoundError: 配置文件不存在时
    """
    path = Path(config_path)
    if not path.exists():
        # 配置文件不存在时使用默认配置
        return AppConfig()

    with open(path, encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f) or {}

    config = AppConfig()

    # 数据配置
    if "data" in raw:
        d = raw["data"]
        config.data = DataConfig(
            source=d.get("source", config.data.source),
            web_url=d.get("web_url", config.data.web_url),
            csv_path=d.get("csv_path", config.data.csv_path),
            db_path=d.get("db_path", config.data.db_path),
        )

    # 分析配置
    if "analysis" in raw:
        a = raw["analysis"]
        config.analysis = AnalysisConfig(
            default_recent=_ensure_int(
                "analysis.default_recent",
                a.get("default_recent", config.analysis.default_recent),
                minimum=1,
            ),
            hot_window=_ensure_int(
                "analysis.hot_window",
                a.get("hot_window", config.analysis.hot_window),
                minimum=1,
            ),
            cold_threshold=_ensure_int(
                "analysis.cold_threshold",
                a.get("cold_threshold", config.analysis.cold_threshold),
                minimum=0,
            ),
        )

    # 特征配置
    if "features" in raw:
        fe = raw["features"]
        config.features = FeatureConfig(
            window_sizes=_normalize_window_sizes(fe.get("window_sizes", config.features.window_sizes)),
            sequence_length=_ensure_int(
                "features.sequence_length",
                fe.get("sequence_length", config.features.sequence_length),
                minimum=2,
            ),
        )

    # 模型配置
    if "models" in raw:
        m = raw["models"]
        if "statistical" in m:
            config.models.statistical = _parse_model_item(m["statistical"])
        if "random_forest" in m:
            config.models.random_forest = _parse_model_item(m["random_forest"])
        if "xgboost" in m:
            config.models.xgboost = _parse_model_item(m["xgboost"])
        if "lstm" in m:
            config.models.lstm = _parse_model_item(m["lstm"])

    # 融合配置
    if "ensemble" in raw:
        config.ensemble = EnsembleConfig(
            strategy=raw["ensemble"].get("strategy", config.ensemble.strategy),
        )

    # 输出配置
    if "output" in raw:
        o = raw["output"]
        config.output = OutputConfig(
            charts_dir=o.get("charts_dir", config.output.charts_dir),
            chart_style=o.get("chart_style", config.output.chart_style),
        )

    # 过滤器配置
    if "filters" in raw:
        fi = raw["filters"]
        config.filters = FiltersConfig(
            enabled=fi.get("enabled", config.filters.enabled),
            max_consecutive=_ensure_int(
                "filters.max_consecutive",
                fi.get("max_consecutive", config.filters.max_consecutive),
                minimum=2,
            ),
            repeat_check_recent=_ensure_int(
                "filters.repeat_check_recent",
                fi.get("repeat_check_recent", config.filters.repeat_check_recent),
                minimum=1,
            ),
            sum_range_percentile=_ensure_percentile(
                "filters.sum_range_percentile",
                fi.get("sum_range_percentile", config.filters.sum_range_percentile),
            ),
            exclude_extreme_odd_even=fi.get("exclude_extreme_odd_even", config.filters.exclude_extreme_odd_even),
            exclude_single_zone=fi.get("exclude_single_zone", config.filters.exclude_single_zone),
        )

    # 运行时配置
    if "runtime" in raw:
        rt = raw["runtime"]
        seed = rt.get("seed", config.runtime.seed)
        if seed is not None:
            seed = _ensure_int("runtime.seed", seed, minimum=0)
        config.runtime = RuntimeConfig(
            seed=seed,
        )

    return config
