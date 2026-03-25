"""LSTM 深度学习预测器

将历史开奖序列视为时间序列，使用 LSTM 网络捕获序列模式。
输入: 近 N 期的特征序列
输出: 每个号码在下一期出现的概率
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from lottery.constants import ALL_BLUE_BALLS, ALL_RED_BALLS, RED_BALL_COUNT
from lottery.features.builder import FeatureBuilder
from lottery.models.base import BasePredictor
from lottery.models.registry import PredictorRegistry
from lottery.types import LotteryRecord, Prediction
from lottery.utils import weighted_sample_no_replace

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LSTMNetwork(nn.Module):
    """LSTM 网络结构"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 49,  # 33 red + 16 blue
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid(),  # 输出概率 0~1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


@PredictorRegistry.register("lstm")
class LSTMPredictor(BasePredictor):
    """LSTM 序列预测器

    将历史开奖序列视为时间序列，用 LSTM 捕获序列模式。
    """

    def __init__(
        self,
        seq_len: int = 30,
        hidden_size: int = 128,
        num_layers: int = 2,
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        feature_builder: FeatureBuilder | None = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch 未安装。请运行: pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )
        self._seq_len = seq_len
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._epochs = epochs
        self._lr = learning_rate
        self._batch_size = batch_size
        self._feature_builder = feature_builder or FeatureBuilder()
        self._model: LSTMNetwork | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._trained = False

    @property
    def name(self) -> str:
        return "LSTM"

    def train(self, records: list[LotteryRecord]) -> None:
        """训练 LSTM 模型"""
        logger.info(f"[{self.name}] 开始训练，{len(records)} 期数据，设备: {self._device}")

        # 构建特征和标签
        features_df, labels_df = self._feature_builder.build(records)
        X_all = features_df.values.astype(np.float32)
        y_all = labels_df.values.astype(np.float32)

        # 构建序列样本: 每个样本是连续 seq_len 期的特征 -> 预测下一期标签
        seq_len = min(self._seq_len, len(X_all) - 1)
        X_seq, y_seq = self._build_sequences(X_all, y_all, seq_len)

        if len(X_seq) == 0:
            raise ValueError("数据不足以构建序列样本")

        logger.info(
            f"[{self.name}] 序列样本: {len(X_seq)} 条, "
            f"特征维度: {X_seq.shape[2]}, 序列长度: {seq_len}"
        )

        # 转为 PyTorch 张量
        X_tensor = torch.FloatTensor(X_seq).to(self._device)
        y_tensor = torch.FloatTensor(y_seq).to(self._device)

        # 划分训练集/验证集 (90/10)
        split = int(len(X_tensor) * 0.9)
        train_dataset = TensorDataset(X_tensor[:split], y_tensor[:split])
        val_X, val_y = X_tensor[split:], y_tensor[split:]

        train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=True
        )

        # 初始化模型
        input_size = X_seq.shape[2]
        output_size = y_seq.shape[1]
        self._model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            output_size=output_size,
        ).to(self._device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        criterion = nn.BCELoss()

        # 训练循环
        best_val_loss = float("inf")
        patience = 15
        patience_counter = 0

        for epoch in range(1, self._epochs + 1):
            self._model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证
            self._model.eval()
            with torch.no_grad():
                val_outputs = self._model(val_X)
                val_loss = criterion(val_outputs, val_y).item() if len(val_X) > 0 else 0

            if epoch % 20 == 0 or epoch == 1:
                logger.info(
                    f"[{self.name}] Epoch {epoch}/{self._epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"[{self.name}] 早停于 Epoch {epoch}")
                    break

        self._trained = True
        logger.info(f"[{self.name}] 训练完成")

    def predict(self, records: list[LotteryRecord], n_sets: int = 1) -> list[Prediction]:
        """基于 LSTM 概率输出生成预测"""
        if not self._trained or self._model is None:
            raise RuntimeError("模型未训练，请先调用 train()")

        # 构建预测输入序列
        features_df, _ = self._feature_builder.build(records)
        X_all = features_df.values.astype(np.float32)

        seq_len = min(self._seq_len, len(X_all))
        # 取最后 seq_len 期特征作为输入
        X_input = X_all[-seq_len:].reshape(1, seq_len, -1)
        X_tensor = torch.FloatTensor(X_input).to(self._device)

        # 预测
        self._model.eval()
        with torch.no_grad():
            probs = self._model(X_tensor).cpu().numpy()[0]

        # 分离红球和蓝球概率
        red_probs = {ball: probs[i] for i, ball in enumerate(ALL_RED_BALLS)}
        blue_probs = {ball: probs[len(ALL_RED_BALLS) + i] for i, ball in enumerate(ALL_BLUE_BALLS)}

        predictions: list[Prediction] = []
        for i in range(n_sets):
            sorted_red = sorted(red_probs.items(), key=lambda x: x[1], reverse=True)

            if i == 0:
                selected_red = [b for b, _ in sorted_red[:RED_BALL_COUNT]]
            else:
                balls = [b for b, _ in sorted_red]
                weights = [max(p, 0.01) for _, p in sorted_red]
                selected_red = weighted_sample_no_replace(balls, weights, RED_BALL_COUNT)

            sorted_blue = sorted(blue_probs.items(), key=lambda x: x[1], reverse=True)
            blue_ball = sorted_blue[i % len(sorted_blue)][0]

            avg_prob = float(np.mean([red_probs[b] for b in selected_red]))
            confidence = round(avg_prob, 3)

            predictions.append(
                Prediction(
                    red_balls=tuple(sorted(selected_red)),
                    blue_ball=blue_ball,
                    confidence=confidence,
                    source=self.name,
                )
            )

        return predictions

    def save(self, path: Path) -> None:
        """保存模型权重"""
        if self._model is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)
        logger.info(f"[{self.name}] 模型已保存到 {path}")

    def load(self, path: Path) -> None:
        """加载模型权重（需要先知道输入维度）"""
        if self._model is None:
            raise RuntimeError("需要先构建模型结构再加载权重")
        self._model.load_state_dict(torch.load(path, map_location=self._device))
        self._trained = True
        logger.info(f"[{self.name}] 模型已加载")

    @staticmethod
    def _build_sequences(
        X: np.ndarray, y: np.ndarray, seq_len: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """将特征矩阵转换为序列样本

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签矩阵 (n_samples, n_labels)
            seq_len: 序列长度

        Returns:
            (X_seq, y_seq)
            - X_seq: (n_sequences, seq_len, n_features)
            - y_seq: (n_sequences, n_labels)
        """
        X_seq: list[np.ndarray] = []
        y_seq: list[np.ndarray] = []

        for i in range(seq_len, len(X)):
            X_seq.append(X[i - seq_len: i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)


