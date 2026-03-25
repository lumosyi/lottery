"""命令行格式化输出"""

from __future__ import annotations

import click

from lottery.types import AnalysisResult, LotteryRecord, Prediction


class CliDisplay:
    """命令行表格和格式化输出工具"""

    # 红球/蓝球显示颜色
    RED = "red"
    BLUE = "blue"

    @staticmethod
    def print_records(records: list[LotteryRecord], limit: int = 10) -> None:
        """表格形式展示开奖记录"""
        shown = records[-limit:]
        click.echo(f"\n{'期号':<12} {'红球':^25} {'蓝球':>4}")
        click.echo("-" * 45)
        for r in shown:
            red_str = " ".join(f"{b:02d}" for b in r.red_balls)
            click.echo(f"{r.issue:<12} {red_str:^25} {r.blue_ball:02d}")

    @staticmethod
    def print_analysis(results: list[AnalysisResult]) -> None:
        """展示多项分析结果"""
        for result in results:
            click.echo(f"\n{'=' * 55}")
            click.echo(f"  {result.name}")
            click.echo(f"{'=' * 55}")
            click.echo(f"  {result.summary}")

            # 根据分析类型展示详细数据
            if result.name == "频率统计":
                _print_frequency(result)
            elif result.name == "遗漏值分析":
                _print_missing_value(result)
            elif result.name == "冷热号分析":
                _print_hot_cold(result)
            elif result.name == "和值分析":
                _print_sum_value(result)
            elif result.name == "奇偶比分析":
                _print_odd_even(result)
            elif result.name == "区间分布":
                _print_zone(result)
            elif result.name == "模式分析":
                _print_pattern(result)

    @staticmethod
    def print_prediction(prediction: Prediction) -> None:
        """展示单组预测结果"""
        click.echo(f"  {prediction}")

    @staticmethod
    def print_prediction_table(predictions: list[Prediction]) -> None:
        """表格形式展示多组预测"""
        click.echo(f"\n{'序号':<4} {'来源':<12} {'红球':^25} {'蓝球':>4} {'置信度':>8}")
        click.echo("-" * 60)
        for i, p in enumerate(predictions, 1):
            red_str = " ".join(f"{b:02d}" for b in p.red_balls)
            click.echo(
                f"{i:<4} {p.source:<12} {red_str:^25} {p.blue_ball:02d} {p.confidence:>7.1%}"
            )


def _print_frequency(result: AnalysisResult) -> None:
    """展示频率统计详情"""
    data = result.data
    if not data:
        return

    # 红球频率 Top 10
    red = data["red"]
    sorted_red = sorted(red.items(), key=lambda x: x[1]["count"], reverse=True)
    click.echo(f"\n  红球出现频率 (Top 10):")
    click.echo(f"  {'号码':>4}  {'次数':>4}  {'频率':>6}  {'柱状图'}")
    for ball, info in sorted_red[:10]:
        bar = "█" * int(info["rate"] * 50)
        click.echo(f"  {ball:>4}  {info['count']:>4}  {info['rate']:>5.1%}  {bar}")

    # 蓝球频率
    blue = data["blue"]
    sorted_blue = sorted(blue.items(), key=lambda x: x[1]["count"], reverse=True)
    click.echo(f"\n  蓝球出现频率 (Top 5):")
    for ball, info in sorted_blue[:5]:
        bar = "█" * int(info["rate"] * 30)
        click.echo(f"  {ball:>4}  {info['count']:>4}  {info['rate']:>5.1%}  {bar}")


def _print_missing_value(result: AnalysisResult) -> None:
    """展示遗漏值详情"""
    data = result.data
    if not data:
        return

    # 红球当前遗漏值 Top 10
    red = data["red"]
    sorted_red = sorted(red.items(), key=lambda x: x[1]["current"], reverse=True)
    click.echo(f"\n  红球当前遗漏值 (Top 10):")
    click.echo(f"  {'号码':>4}  {'当前遗漏':>6}  {'平均遗漏':>6}  {'最大遗漏':>6}")
    for ball, info in sorted_red[:10]:
        click.echo(
            f"  {ball:>4}  {info['current']:>6}  {info['avg']:>6}  {info['max']:>6}"
        )


def _print_hot_cold(result: AnalysisResult) -> None:
    """展示冷热号详情"""
    data = result.data
    if not data:
        return

    for ball_type, label in [("red", "红球"), ("blue", "蓝球")]:
        info = data[ball_type]
        hot = sorted(info["hot"])
        warm = sorted(info["warm"])
        cold = sorted(info["cold"])

        click.echo(f"\n  {label}:")
        click.echo(f"    热号({len(hot)}): {_format_balls(hot)}")
        click.echo(f"    温号({len(warm)}): {_format_balls(warm)}")
        click.echo(f"    冷号({len(cold)}): {_format_balls(cold)}")


def _print_sum_value(result: AnalysisResult) -> None:
    """展示和值分析详情"""
    data = result.data
    if not data:
        return

    click.echo(f"\n  和值区间分布:")
    ranges = data["ranges"]
    total = data["total_periods"]
    for range_name, count in ranges.items():
        bar = "█" * int(count / total * 40)
        click.echo(f"  {range_name:>10}  {count:>4}  {count/total:>5.1%}  {bar}")

    click.echo(f"\n  最近走势: {' -> '.join(str(s) for s in data['recent_trend'])}")


def _print_odd_even(result: AnalysisResult) -> None:
    """展示奇偶比详情"""
    data = result.data
    if not data:
        return

    click.echo(f"\n  奇偶比分布 (奇:偶):")
    dist = data["distribution"]
    total = data["total_periods"]
    for ratio, info in dist.items():
        bar = "█" * int(info["rate"] * 40)
        click.echo(f"  {ratio:>6}  {info['count']:>4}  {info['rate']:>5.1%}  {bar}")


def _print_zone(result: AnalysisResult) -> None:
    """展示区间分布详情"""
    data = result.data
    if not data:
        return

    # 各区占比
    click.echo(f"\n  各区出球占比:")
    for zone_name, info in data["zone_rates"].items():
        bar = "█" * int(info["rate"] * 40)
        click.echo(f"  {zone_name}  {info['total']:>4}  {info['rate']:>5.1%}  {bar}")

    # 三区比分布 Top 5
    click.echo(f"\n  三区比分布 (Top 5):")
    dist = data["distribution"]
    for i, (ratio, info) in enumerate(dist.items()):
        if i >= 5:
            break
        click.echo(f"  {ratio:>8}  {info['count']:>4}  {info['rate']:>5.1%}")


def _format_balls(balls: list[int]) -> str:
    """格式化号码列表"""
    if not balls:
        return "(无)"
    return " ".join(f"{b:02d}" for b in balls)


def _print_pattern(result: AnalysisResult) -> None:
    """展示模式分析详情"""
    data = result.data
    if not data:
        return

    total = data["total"]

    # 连号统计
    cons = data["consecutive"]
    click.echo(f"\n  连号统计 ({total} 期):")
    click.echo(f"    历史最大连号: {cons['max_length']}连")
    click.echo(f"    含连号期数占比: {cons['has_consecutive_rate']:.1%}")
    items = []
    for length in [2, 3, 4, 5, 6]:
        cnt = cons.get(f"len_{length}", 0)
        if cnt > 0 or length <= 4:
            items.append(f"{length}连: {cnt}次({cnt/total:.1%})")
    click.echo(f"    分布: {' | '.join(items)}")

    # 排除建议
    threshold = min(cons["max_length"] + 1, 5)
    click.echo(f"    -> 建议: 排除 {threshold} 连号及以上组合")

    # 重复开奖统计
    rep = data["repeat"]
    click.echo(f"\n  重复开奖统计:")
    if rep["full_repeat_count"] == 0:
        click.echo(f"    {total} 期中无完全相同红球组合")
    else:
        click.echo(f"    完全重复: {rep['full_repeat_count']} 组")

    click.echo(f"    相邻期平均重号: {rep['avg_adjacent_repeat']:.1f} 个")
    adj = rep["adjacent_repeat_distribution"]
    adj_str = " | ".join(f"{k}个:{v}次" for k, v in sorted(adj.items()))
    click.echo(f"    相邻期重号分布: {adj_str}")

    # 和值范围
    sr = data["sum_range"]
    click.echo(f"\n  和值范围:")
    click.echo(f"    最小: {sr['min']}  最大: {sr['max']}  平均: {sr['mean']}")
    click.echo(f"    95%区间: [{sr['p2_5']}, {sr['p97_5']}]")
    click.echo(f"    -> 建议: 排除和值 < {sr['p2_5']} 或 > {sr['p97_5']} 的组合")

    # 奇偶极端
    oe = data["odd_even_extreme"]
    click.echo(f"\n  奇偶极端:")
    click.echo(f"    全奇(6:0): {oe['all_odd_count']}次({oe['all_odd_rate']:.2%})")
    click.echo(f"    全偶(0:6): {oe['all_even_count']}次({oe['all_even_rate']:.2%})")

    # 三区极端
    ze = data["zone_extreme"]
    click.echo(f"\n  三区极端:")
    click.echo(f"    单区全出(6:0:0等): {ze['single_zone_count']}次({ze['single_zone_rate']:.2%})")
