"""双色球核心数据类型定义"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from lottery.constants import BLUE_BALL_MAX, BLUE_BALL_MIN, RED_BALL_COUNT, RED_BALL_MAX, RED_BALL_MIN


@dataclass(frozen=True, slots=True)
class LotteryRecord:
    """单条双色球开奖记录（不可变值对象）

    Attributes:
        issue: 期号，如 "2024001"
        draw_date: 开奖日期
        red_balls: 红球号码（升序元组，6个）
        blue_ball: 蓝球号码
    """

    issue: str
    draw_date: date
    red_balls: tuple[int, ...]
    blue_ball: int

    def __post_init__(self) -> None:
        if len(self.red_balls) != RED_BALL_COUNT:
            raise ValueError(f"红球数量必须为 {RED_BALL_COUNT}，当前: {len(self.red_balls)}")
        if not all(RED_BALL_MIN <= r <= RED_BALL_MAX for r in self.red_balls):
            raise ValueError(f"红球号码必须在 {RED_BALL_MIN}-{RED_BALL_MAX} 之间")
        if sorted(self.red_balls) != list(self.red_balls):
            raise ValueError("红球必须按升序排列")
        if len(set(self.red_balls)) != RED_BALL_COUNT:
            raise ValueError("红球号码不能重复")
        if not (BLUE_BALL_MIN <= self.blue_ball <= BLUE_BALL_MAX):
            raise ValueError(f"蓝球号码必须在 {BLUE_BALL_MIN}-{BLUE_BALL_MAX} 之间")

    def __str__(self) -> str:
        red_str = " ".join(f"{r:02d}" for r in self.red_balls)
        return f"[{self.issue}] {red_str} | {self.blue_ball:02d}"


@dataclass(slots=True)
class Prediction:
    """单次预测结果

    Attributes:
        red_balls: 推荐红球号码（升序元组）
        blue_ball: 推荐蓝球号码
        score: 评分 0.0~1.0
        source: 来源模型名称
        details: 附加信息
    """

    red_balls: tuple[int, ...]
    blue_ball: int
    score: float
    source: str
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        red_str = " ".join(f"{r:02d}" for r in self.red_balls)
        return f"[{self.source}] {red_str} | {self.blue_ball:02d} (评分: {self.score:.2%})"


@dataclass(slots=True)
class AnalysisResult:
    """单项分析结果

    Attributes:
        name: 分析名称
        data: 分析数据（灵活结构）
        summary: 人类可读摘要
    """

    name: str
    data: dict
    summary: str
