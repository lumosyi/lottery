"""双色球常量定义"""

# 红球范围
RED_BALL_MIN = 1
RED_BALL_MAX = 33
RED_BALL_COUNT = 6

# 蓝球范围
BLUE_BALL_MIN = 1
BLUE_BALL_MAX = 16

# 红球三区划分
ZONE_1 = range(1, 12)   # 01-11
ZONE_2 = range(12, 23)  # 12-22
ZONE_3 = range(23, 34)  # 23-33

# 全部红球号码
ALL_RED_BALLS = list(range(RED_BALL_MIN, RED_BALL_MAX + 1))
# 全部蓝球号码
ALL_BLUE_BALLS = list(range(BLUE_BALL_MIN, BLUE_BALL_MAX + 1))
