"""随机森林预测器

对每个号码独立建模：预测下一期该号码是否出现。
红球 33 个二分类器 + 蓝球 16 个二分类器 = 49 个模型。
取概率最高的 6 个红球和 1 个蓝球作为推荐。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from lottery.constants import ALL_BLUE_BALLS, ALL_RED_BALLS, RED_BALL_COUNT
from lottery.features.builder import FeatureBuilder
from lottery.models.base import BasePredictor
from lottery.models.registry import PredictorRegistry
from lottery.types import LotteryRecord, Prediction
from lottery.utils import weighted_sample_no_replace


@PredictorRegistry.register("rf")
class RandomForestPredictor(BasePredictor):
    """随机森林预测器

    为每个号码训练独立的 RandomForestClassifier，
    预测该号码在下一期出现的概率。
    """

    def __init__(
        self,
        n_estimators: int = 200,
        feature_builder: FeatureBuilder | None = None,
        random_state: int = 42,
    ) -> None:
        self._n_estimators = n_estimators
        self._feature_builder = feature_builder or FeatureBuilder()
        self._random_state = random_state
        self._red_models: dict[int, RandomForestClassifier] = {}
        self._blue_models: dict[int, RandomForestClassifier] = {}
        self._trained = False

    @property
    def name(self) -> str:
        return "随机森林"

    def train(self, records: list[LotteryRecord]) -> None:
        """训练 49 个独立的随机森林分类器"""
        logger.info(f"[{self.name}] 开始训练，{len(records)} 期数据")

        features_df, labels_df = self._feature_builder.build(records)
        X = features_df.values

        # 训练红球模型 + 蓝球模型（共 49 个）
        all_targets = [(b, f"red_{b:02d}", self._red_models) for b in ALL_RED_BALLS] + \
                      [(b, f"blue_{b:02d}", self._blue_models) for b in ALL_BLUE_BALLS]

        for ball, col, model_dict in tqdm(all_targets, desc=f"  [{self.name}] 训练", unit="模型"):
            y = labels_df[col].values
            model = RandomForestClassifier(
                n_estimators=self._n_estimators,
                random_state=self._random_state,
                n_jobs=-1,
                class_weight="balanced",
            )
            model.fit(X, y)
            model_dict[ball] = model

        self._trained = True
        logger.info(f"[{self.name}] 训练完成，共 {len(self._red_models) + len(self._blue_models)} 个模型")

    def predict(self, records: list[LotteryRecord], n_sets: int = 1) -> list[Prediction]:
        """基于概率排名生成预测"""
        if not self._trained:
            raise RuntimeError("模型未训练，请先调用 train()")

        X_pred = self._feature_builder.build_prediction_features(records).values

        # 获取每个号码的出现概率
        red_probs = {}
        for ball, model in self._red_models.items():
            prob = model.predict_proba(X_pred)[0]
            # predict_proba 返回 [P(不出现), P(出现)]
            red_probs[ball] = prob[1] if len(prob) > 1 else prob[0]

        blue_probs = {}
        for ball, model in self._blue_models.items():
            prob = model.predict_proba(X_pred)[0]
            blue_probs[ball] = prob[1] if len(prob) > 1 else prob[0]

        predictions: list[Prediction] = []
        for i in range(n_sets):
            # 取概率最高的 6 个红球
            sorted_red = sorted(red_probs.items(), key=lambda x: x[1], reverse=True)

            if i == 0:
                # 第一组：直接取 Top 6
                selected_red = [b for b, _ in sorted_red[:RED_BALL_COUNT]]
            else:
                # 后续组：基于概率加权随机采样，增加多样性
                balls = [b for b, _ in sorted_red]
                weights = [p for _, p in sorted_red]
                selected_red = weighted_sample_no_replace(balls, weights, RED_BALL_COUNT)

            # 蓝球取概率最高的（或随机）
            sorted_blue = sorted(blue_probs.items(), key=lambda x: x[1], reverse=True)
            blue_ball = sorted_blue[i % len(sorted_blue)][0]

            # 置信度：所选红球的平均概率
            avg_prob = np.mean([red_probs[b] for b in selected_red])
            confidence = round(float(avg_prob), 3)

            predictions.append(
                Prediction(
                    red_balls=tuple(sorted(selected_red)),
                    blue_ball=blue_ball,
                    confidence=confidence,
                    source=self.name,
                    details={"red_probs": {b: round(red_probs[b], 4) for b in selected_red}},
                )
            )

        return predictions

    def save(self, path: Path) -> None:
        """保存模型到文件"""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "red_models": self._red_models,
            "blue_models": self._blue_models,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"[{self.name}] 模型已保存到 {path}")

    def load(self, path: Path) -> None:
        """加载已训练的模型"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._red_models = data["red_models"]
        self._blue_models = data["blue_models"]
        self._trained = True
        logger.info(f"[{self.name}] 模型已加载")


