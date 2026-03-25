# 双色球智能预测系统

综合统计分析、机器学习、深度学习多策略融合的双色球号码预测工具。

> **声明**：双色球是真随机过程，任何预测方法都无法提高中奖概率。本系统仅供技术学习和数据分析参考，不构成投注建议。

## 环境要求

- Python 3.10+
- Windows / macOS / Linux

## 安装

```bash
# 进入项目目录
cd D:\python_project\lottery

# 创建虚拟环境（如已有可跳过）
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 安装项目（基础依赖，含统计分析 + 随机森林 + XGBoost）
pip install -e .

# 安装 LSTM 深度学习支持（可选，约 200MB）
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 快速开始

```bash
# 1. 首次使用 - 采集全部历史数据（约 2000 期，需联网）
lottery update

# 2. 查看数据概览
lottery info

# 3. 统计分析（自动拉取最新数据）
lottery analyze

# 4. 生成预测号码（全模型融合）
lottery predict --ensemble
```

## 命令详解

### `lottery update` — 增量更新数据

自动检测数据库中的最新期号，仅从福彩官网采集之后的新数据。首次运行时自动全量采集。

```bash
lottery update
```

输出示例：

```
当前最新: [2026029] 06 19 22 23 28 31 | 05
检查新数据...

更新完成:
  新增: 1 期
  总计: 1988 期

最新开奖:
  [2026030] 10 11 14 19 22 24 | 04
```

### `lottery fetch` — 手动采集数据

从指定数据源采集历史数据，支持网络爬取和 CSV 导入。

```bash
# 从网络采集最近 200 期
lottery fetch --source web --count 200

# 从 CSV 文件导入
lottery fetch --source csv --csv-path data/raw/history.csv
```

**CSV 文件格式**（支持中英文表头）：

```csv
期号,开奖日期,红球1,红球2,红球3,红球4,红球5,红球6,蓝球
2024001,2024-01-02,1,5,16,18,25,30,14
```

### `lottery info` — 查看系统状态

显示数据库信息、模型配置和融合策略。

```bash
lottery info
```

### `lottery analyze` — 统计分析

对历史数据进行 6 种维度的统计分析，执行前自动拉取最新数据。

```bash
# 分析最近 200 期
lottery analyze --recent 200

# 分析并生成图表（保存到 output/charts/）
lottery analyze --recent 500 --show-charts
```

**分析维度**：

| 分析项 | 说明 |
|--------|------|
| 频率统计 | 每个号码的出现次数和频率 |
| 遗漏值分析 | 每个号码距上次出现的间隔期数 |
| 冷热号分析 | 按近期出现频率分为热号/温号/冷号 |
| 和值分析 | 红球和值的分布特征和走势 |
| 奇偶比分析 | 红球奇数/偶数个数的分布 |
| 区间分布 | 红球在一区(01-11)/二区(12-22)/三区(23-33)的分布 |

**生成的图表**（`--show-charts`）：

```
output/charts/
├── frequency.png       # 频率柱状图
├── missing_value.png   # 遗漏值对比图
├── hot_cold.png        # 冷热号分布图
├── sum_value.png       # 和值走势 + 区间饼图
├── odd_even.png        # 奇偶比分布图
└── zone.png            # 三区分布图
```

### `lottery predict` — 生成预测号码

使用多种模型生成推荐号码，执行前自动拉取最新数据。

```bash
# 全部模型独立预测，每个模型 5 组
lottery predict --model all --sets 5

# 单模型预测
lottery predict --model statistical
lottery predict --model rf
lottery predict --model xgboost
lottery predict --model lstm

# 全模型融合推荐（推荐使用）
lottery predict --ensemble --sets 5

# 指定模型组合 + 融合
lottery predict --model rf --model xgboost --ensemble --sets 3

# 关闭概率排除过滤（输出原始结果）
lottery predict --ensemble --no-filter --sets 5
```

**可用模型**：

| 模型 | 名称 | 说明 |
|------|------|------|
| `statistical` | 统计分析 | 基于频率、遗漏值、近期趋势的概率采样 |
| `rf` | 随机森林 | 49 个独立分类器预测每个号码出现概率 |
| `xgboost` | XGBoost | 梯度提升树，自动处理样本不平衡 |
| `lstm` | LSTM | 序列预测，捕获时间序列模式（需安装 PyTorch） |

**融合模式**（`--ensemble`）：

多个模型各自生成预测后，通过加权投票融合为最终推荐。各模型权重可在 `config.yaml` 中调整。

**概率排除过滤**（默认启用）：

预测结果会经过概率排除过滤器后处理，排除极低概率的号码组合：

| 规则 | 排除条件 | 依据 |
|------|----------|------|
| 连号过滤 | 含 4+ 连续号码 | 历史 4 连号仅 18 次(0.9%)，6 连号从未出现 |
| 重复过滤 | 与近 10 期完全相同 | 1988 期中仅 1 组完全重复 |
| 和值过滤 | 和值超出 95% 区间 | 历史 95% 区间约 [59, 141] |
| 奇偶过滤 | 全奇(6:0)或全偶(0:6) | 各约 1%，概率极低 |
| 三区过滤 | 全落单区(6:0:0) | 仅 3 次(0.15%) |

被排除的组合不会被丢弃，而是**降低置信度并标注原因**，用户可自行判断。使用 `--no-filter` 关闭过滤。

## 配置文件

项目根目录的 `config.yaml` 控制全部运行时参数：

```yaml
data:
  source: "web"           # 默认数据源: web | csv
  db_path: "data/lottery.db"

analysis:
  default_recent: 100     # 默认分析期数
  hot_window: 10          # 冷热号窗口

models:
  statistical:
    enabled: true
    weight: 0.2           # 融合权重
  random_forest:
    enabled: true
    weight: 0.25
    n_estimators: 200
  xgboost:
    enabled: true
    weight: 0.3
    max_depth: 6
    n_estimators: 300
  lstm:
    enabled: true
    weight: 0.25
    hidden_size: 128
    epochs: 100

ensemble:
  strategy: "weighted_voting"
```

## 项目结构

```
lottery/
├── config.yaml                 # 运行时配置
├── pyproject.toml              # 项目依赖
├── data/
│   └── lottery.db              # SQLite 数据库（自动创建）
├── output/charts/              # 分析图表输出
└── src/lottery/
    ├── cli.py                  # CLI 命令入口
    ├── types.py                # 核心数据类型
    ├── config.py               # 配置加载
    ├── fetcher/                # 数据采集（网络 / CSV）
    ├── store/                  # 数据存储（SQLite）
    ├── analysis/               # 6 种统计分析器
    ├── features/               # 特征工程
    ├── models/                 # 预测模型（统计 / RF / XGBoost / LSTM）
    ├── ensemble/               # 多策略融合引擎
    └── visualization/          # 图表渲染 + 命令行输出
```

## 常用工作流

### 日常预测

```bash
# 一条命令搞定：自动更新数据 → 全模型训练 → 融合推荐
lottery predict --ensemble --sets 5
```

### 深度分析后预测

```bash
# 先看统计分析
lottery analyze --recent 300 --show-charts

# 再生成预测
lottery predict --ensemble
```

### 仅更新数据

```bash
lottery update
```

## 常见问题

**Q: LSTM 模型报错 `ImportError`？**

安装 PyTorch CPU 版本：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Q: 网络采集失败？**

检查网络连接，程序内置 3 次重试机制。也可用 CSV 手动导入：
```bash
lottery fetch --source csv --csv-path your_data.csv
```

**Q: 如何调整模型权重？**

编辑 `config.yaml` 中对应模型的 `weight` 字段，融合时会自动归一化。

**Q: 图表中文显示为方块？**

系统需安装中文字体（Microsoft YaHei 或 SimHei）。Linux 用户可执行：
```bash
sudo apt install fonts-wqy-microhei
```
