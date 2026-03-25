"""matplotlib 图表渲染器"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lottery.constants import ALL_BLUE_BALLS, ALL_RED_BALLS
from lottery.types import AnalysisResult

# 非交互式后端（支持无显示器环境），需在 plt 使用前设置
matplotlib.use("Agg")

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "STSong", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


class ChartRenderer:
    """matplotlib 图表渲染器

    将分析结果渲染为各类统计图表并保存到指定目录。
    """

    def __init__(
        self,
        output_dir: str | Path = "output/charts",
        style: str = "seaborn-v0_8",
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        try:
            plt.style.use(style)
        except OSError:
            pass  # 风格不可用时使用默认
        # 样式加载后重新设置中文字体（避免被样式覆盖）
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "STSong", "DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["axes.unicode_minus"] = False

    def render_all(self, results: list[AnalysisResult]) -> list[Path]:
        """渲染所有分析结果的图表

        Returns:
            生成的图表文件路径列表
        """
        saved: list[Path] = []
        for result in results:
            path = self._render_one(result)
            if path:
                saved.append(path)
        return saved

    def _render_one(self, result: AnalysisResult) -> Path | None:
        """根据分析类型渲染对应图表"""
        renderers = {
            "频率统计": self._plot_frequency,
            "遗漏值分析": self._plot_missing_value,
            "冷热号分析": self._plot_hot_cold,
            "和值分析": self._plot_sum_trend,
            "奇偶比分析": self._plot_odd_even,
            "区间分布": self._plot_zone,
        }
        renderer = renderers.get(result.name)
        if renderer and result.data:
            return renderer(result)
        return None

    def _plot_frequency(self, result: AnalysisResult) -> Path:
        """频率柱状图（红球 + 蓝球）"""
        data = result.data
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        # 红球频率
        red = data["red"]
        balls = ALL_RED_BALLS
        counts = [red[b]["count"] for b in balls]
        colors = ["#e74c3c" if c >= np.mean(counts) else "#fad7d7" for c in counts]
        ax1.bar([str(b) for b in balls], counts, color=colors)
        ax1.set_title("红球出现频率", fontsize=14)
        ax1.set_xlabel("号码")
        ax1.set_ylabel("次数")
        ax1.axhline(y=np.mean(counts), color="#666", linestyle="--", label=f"平均: {np.mean(counts):.1f}")
        ax1.legend()

        # 蓝球频率
        blue = data["blue"]
        balls_b = ALL_BLUE_BALLS
        counts_b = [blue[b]["count"] for b in balls_b]
        colors_b = ["#3498db" if c >= np.mean(counts_b) else "#d4e6f1" for c in counts_b]
        ax2.bar([str(b) for b in balls_b], counts_b, color=colors_b)
        ax2.set_title("蓝球出现频率", fontsize=14)
        ax2.set_xlabel("号码")
        ax2.set_ylabel("次数")
        ax2.axhline(y=np.mean(counts_b), color="#666", linestyle="--", label=f"平均: {np.mean(counts_b):.1f}")
        ax2.legend()

        plt.tight_layout()
        path = self._output_dir / "frequency.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_missing_value(self, result: AnalysisResult) -> Path:
        """遗漏值柱状图"""
        data = result.data
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        # 红球遗漏
        red = data["red"]
        balls = ALL_RED_BALLS
        current = [red[b]["current"] for b in balls]
        avg = [red[b]["avg"] for b in balls]

        x = np.arange(len(balls))
        width = 0.35
        ax1.bar(x - width / 2, current, width, label="当前遗漏", color="#e74c3c")
        ax1.bar(x + width / 2, avg, width, label="平均遗漏", color="#95a5a6")
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(b) for b in balls])
        ax1.set_title("红球遗漏值", fontsize=14)
        ax1.legend()

        # 蓝球遗漏
        blue = data["blue"]
        balls_b = ALL_BLUE_BALLS
        current_b = [blue[b]["current"] for b in balls_b]
        avg_b = [blue[b]["avg"] for b in balls_b]

        x_b = np.arange(len(balls_b))
        ax2.bar(x_b - width / 2, current_b, width, label="当前遗漏", color="#3498db")
        ax2.bar(x_b + width / 2, avg_b, width, label="平均遗漏", color="#95a5a6")
        ax2.set_xticks(x_b)
        ax2.set_xticklabels([str(b) for b in balls_b])
        ax2.set_title("蓝球遗漏值", fontsize=14)
        ax2.legend()

        plt.tight_layout()
        path = self._output_dir / "missing_value.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_hot_cold(self, result: AnalysisResult) -> Path:
        """冷热号热力图"""
        data = result.data
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for ax, ball_type, title, all_balls, color_map in [
            (ax1, "red", "红球冷热分布", ALL_RED_BALLS, {
                "hot": "#e74c3c", "warm": "#f39c12", "cold": "#3498db"
            }),
            (ax2, "blue", "蓝球冷热分布", ALL_BLUE_BALLS, {
                "hot": "#2980b9", "warm": "#f39c12", "cold": "#bdc3c7"
            }),
        ]:
            info = data[ball_type]
            hot_set = set(info["hot"])
            cold_set = set(info["cold"])

            colors = []
            for b in all_balls:
                if b in hot_set:
                    colors.append(color_map["hot"])
                elif b in cold_set:
                    colors.append(color_map["cold"])
                else:
                    colors.append(color_map["warm"])

            counts_data = info.get("counts", {})
            values = [counts_data.get(b, 0) for b in all_balls]

            ax.bar([str(b) for b in all_balls], values, color=colors)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("号码")
            ax.set_ylabel(f"最近 {data['window']} 期出现次数")

        # 图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#e74c3c", label="热号"),
            Patch(facecolor="#f39c12", label="温号"),
            Patch(facecolor="#3498db", label="冷号"),
        ]
        fig.legend(handles=legend_elements, loc="upper center", ncol=3, fontsize=11)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        path = self._output_dir / "hot_cold.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_sum_trend(self, result: AnalysisResult) -> Path:
        """和值走势图 + 区间分布"""
        data = result.data
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 走势图
        sums = data["sums"]
        ax1.plot(range(len(sums)), sums, color="#e74c3c", linewidth=1, alpha=0.7)
        ax1.axhline(y=data["avg"], color="#666", linestyle="--", label=f"平均: {data['avg']}")
        ax1.fill_between(range(len(sums)), sums, data["avg"], alpha=0.1, color="#e74c3c")
        ax1.set_title("和值走势", fontsize=14)
        ax1.set_xlabel("期数")
        ax1.set_ylabel("和值")
        ax1.legend()

        # 区间分布饼图
        ranges = data["ranges"]
        labels = list(ranges.keys())
        values = list(ranges.values())
        colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]
        ax2.pie(values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax2.set_title("和值区间分布", fontsize=14)

        plt.tight_layout()
        path = self._output_dir / "sum_value.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_odd_even(self, result: AnalysisResult) -> Path:
        """奇偶比分布图"""
        data = result.data
        fig, ax = plt.subplots(figsize=(10, 5))

        dist = data["distribution"]
        labels = list(dist.keys())
        counts = [d["count"] for d in dist.values()]
        colors = plt.cm.RdYlBu(np.linspace(0.2, 0.8, len(labels)))

        bars = ax.bar(labels, counts, color=colors)
        ax.set_title("奇偶比分布 (奇:偶)", fontsize=14)
        ax.set_xlabel("奇偶比")
        ax.set_ylabel("出现次数")

        # 在柱上标注百分比
        for bar, info in zip(bars, dist.values()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{info['rate']:.1%}",
                ha="center",
                fontsize=9,
            )

        plt.tight_layout()
        path = self._output_dir / "odd_even.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_zone(self, result: AnalysisResult) -> Path:
        """区间分布饼图 + 三区比柱状图"""
        data = result.data
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 各区占比饼图
        zone_rates = data["zone_rates"]
        labels = list(zone_rates.keys())
        values = [z["total"] for z in zone_rates.values()]
        colors = ["#e74c3c", "#f39c12", "#3498db"]
        ax1.pie(values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax1.set_title("三区出球占比", fontsize=14)

        # 三区比分布 Top 8
        dist = data["distribution"]
        top_items = list(dist.items())[:8]
        labels2 = [k for k, _ in top_items]
        counts2 = [v["count"] for _, v in top_items]
        ax2.barh(labels2[::-1], counts2[::-1], color="#2ecc71")
        ax2.set_title("三区比分布 (Top 8)", fontsize=14)
        ax2.set_xlabel("出现次数")

        plt.tight_layout()
        path = self._output_dir / "zone.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
