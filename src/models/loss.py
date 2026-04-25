import torch
import torch.nn as nn


class EarlyStopping:
    """早停回调"""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max',
    ):
        """
        Args:
            patience: 容忍epoch数
            min_delta: 最小改善阈值
            mode: 'max'或'min'，监控指标是增大还是减小
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """检查是否应该早停"""
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class MetricTracker:
    """指标跟踪器"""

    def __init__(self):
        self.history = {}
        self.current_epoch = {}

    def update(self, metrics: dict):
        """更新当前epoch的指标"""
        for key, value in metrics.items():
            if key not in self.current_epoch:
                self.current_epoch[key] = []
            if isinstance(value, (int, float)):
                self.current_epoch[key].append(value)

    def commit_epoch(self):
        """提交当前epoch，计算平均值"""
        for key, values in self.current_epoch.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(sum(values) / len(values) if values else 0)
        self.current_epoch = {}

    def get_best(self, metric: str, mode: str = 'max') -> float:
        """获取最佳指标"""
        if metric not in self.history or not self.history[metric]:
            return 0.0
        if mode == 'max':
            return max(self.history[metric])
        else:
            return min(self.history[metric])

    def get_last(self, metric: str) -> float:
        """获取最后一个指标值"""
        if metric not in self.history or not self.history[metric]:
            return 0.0
        return self.history[metric][-1]

    def summary(self) -> dict:
        """获取训练摘要"""
        summary = {}
        for key in self.history:
            if self.history[key]:
                summary[key] = {
                    'last': self.history[key][-1],
                    'mean': sum(self.history[key]) / len(self.history[key]),
                    'best': max(self.history[key]) if key.endswith('acc') or key.endswith('f1') else min(self.history[key]),
                }
        return summary