import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class ThreeWayDecision:
    """
    三支决策逻辑实现

    基于概率阈值将样本分配到三个区域:
    - POS: 正域 (正常)
    - BND: 边界域 (不确定)
    - NEG: 负域 (异常)
    """

    def __init__(self, alpha: float = 0.85, beta: float = 0.35):
        """
        Args:
            alpha: 正域阈值 (默认0.85)
            beta: 负域阈值 (默认0.35)
        """
        self.alpha = alpha
        self.beta = beta

    def __call__(
        self,
        normal_prob: torch.Tensor,
        p_abnormal: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行三支决策

        Args:
            normal_prob: 正常概率 [batch_size]
            p_abnormal: 异常概率 (可选，默认用1-normal_prob)

        Returns:
            pos_mask: 正域掩码 (正常)
            bnd_mask: 边界域掩码 (不确定)
            neg_mask: 负域掩码 (异常)
        """
        if p_abnormal is None:
            p_abnormal = 1 - normal_prob

        pos_mask = (normal_prob >= self.alpha).long()
        neg_mask = (p_abnormal >= (1 - self.beta)).long()
        bnd_mask = torch.ones_like(pos_mask) - pos_mask - neg_mask
        bnd_mask = bnd_mask.clamp(min=0).long()

        return pos_mask, bnd_mask, neg_mask

    def get_decisions(
        self,
        normal_prob: torch.Tensor,
        p_abnormal: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        获取决策结果

        Args:
            normal_prob: 正常概率 [batch_size]
            p_abnormal: 异常概率 (可选)

        Returns:
            decisions: 0=正常, 1=异常, 2=不确定
        """
        pos_mask, bnd_mask, neg_mask = self(normal_prob, p_abnormal)

        decisions = torch.zeros_like(normal_prob, dtype=torch.long) + 2  # 默认不确定
        decisions[pos_mask.bool()] = 0
        decisions[neg_mask.bool()] = 1

        return decisions

    def get_decision_labels(
        self,
        decisions: torch.Tensor,
    ) -> list:
        """将决策转换为标签"""
        label_map = {0: '正常', 1: '异常', 2: '不确定'}
        return [label_map[d.item()] for d in decisions]


class MCDropoutUncertainty:
    """
    MC Dropout不确定性估计

    通过多次前向传播估计预测的不确定性
    """

    def __init__(self, model: nn.Module, num_samples: int = 20):
        """
        Args:
            model: 带Dropout的模型
            num_samples: MC采样次数
        """
        self.model = model
        self.num_samples = num_samples

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        执行MC Dropout预测

        Args:
            x: 输入张量 [batch_size, ...]

        Returns:
            dict包含:
                - mean: 平均预测概率
                - std: 预测标准差 (不确定性)
                - predictions: 所有采样的预测
        """
        self.model.enable_mc_dropout()

        predictions = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(x)
                predictions.append(output['normal_prob'])

        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size]

        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        self.model.disable_mc_dropout()

        return {
            'mean': mean,
            'std': std,
            'predictions': predictions,
        }

    def get_uncertainty_mask(
        self,
        x: torch.Tensor,
        threshold: float = 0.1,
    ) -> torch.Tensor:
        """
        获取高不确定性样本的掩码

        Args:
            x: 输入张量
            threshold: 不确定性阈值

        Returns:
            uncertain_mask: 高不确定性样本掩码
        """
        result = self.predict(x)
        uncertain_mask = result['std'] > threshold
        return uncertain_mask


class TemperatureScaler:
    """
    温度缩放校准

    用于校准模型输出的概率，使其更接近真实概率
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        拟合温度参数

        Args:
            logits: 模型logits
            labels: 真实标签
        """
        import numpy as np
        from scipy.optimize import minimize

        def loss_fn(T):
            scaled_logits = logits / T.item()
            probs = torch.softmax(scaled_logits, dim=1)
            # NLL Loss
            nll = F.cross_entropy(scaled_logits, labels)
            return nll.item()

        # 简单二分搜索找到最佳温度
        low, high = 0.5, 5.0
        for _ in range(20):
            mid = (low + high) / 2
            if loss_fn(torch.tensor(mid)) < loss_fn(torch.tensor(mid - 0.1)):
                low = mid
            else:
                high = mid

        self.temperature = (low + high) / 2

    def scale(self, logits: torch.Tensor) -> torch.Tensor:
        """缩放logits"""
        return logits / self.temperature

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """预测概率（已校准）"""
        scaled = self.scale(logits)
        return F.softmax(scaled, dim=1)


class ThresholdTuner:
    """
    三支决策阈值调优器

    用于找到最佳的alpha和beta阈值
    """

    def __init__(self, decision_maker: ThreeWayDecision):
        self.decision_maker = decision_maker

    def tune_on_validation(
        self,
        val_probs: torch.Tensor,
        val_labels: torch.Tensor,
        alpha_range: Tuple[float, float] = (0.7, 0.95),
        beta_range: Tuple[float, float] = (0.15, 0.5),
        metric: str = 'f1_macro',
    ) -> Dict:
        """
        在验证集上调优阈值

        Args:
            val_probs: 验证集预测概率
            val_labels: 验证集真实标签 (0=正常, 1=异常)
            alpha_range: alpha搜索范围
            beta_range: beta搜索范围
            metric: 优化指标

        Returns:
            最佳阈值配置
        """
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

        best_metric = 0
        best_alpha, best_beta = 0.85, 0.35

        # 网格搜索
        for alpha in torch.arange(alpha_range[0], alpha_range[1], 0.05):
            for beta in torch.arange(beta_range[0], beta_range[1], 0.05):
                if alpha <= beta:
                    continue

                self.decision_maker.alpha = alpha.item()
                self.decision_maker.beta = beta.item()

                decisions = self.decision_maker.get_decisions(val_probs)

                # 将三支决策映射到二分类 (正常=0, 异常=1, 不确定=1 or 忽略)
                # 方案：将不确定视为预测为异常，需要人工审核
                pred_binary = (decisions >= 1).long()  # 异常(1)和不确定(2)都视为需要审核

                # 计算指标
                if metric == 'accuracy':
                    acc = accuracy_score(val_labels.cpu(), pred_binary.cpu())
                    score = acc
                elif metric == 'f1_macro':
                    # 将不确定归类为异常后计算F1
                    f1 = f1_score(val_labels.cpu(), pred_binary.cpu(), average='macro')
                    score = f1
                else:
                    f1 = f1_score(val_labels.cpu(), pred_binary.cpu(), average='macro')
                    score = f1

                if score > best_metric:
                    best_metric = score
                    best_alpha, best_beta = alpha.item(), beta.item()

        # 恢复最佳阈值
        self.decision_maker.alpha = best_alpha
        self.decision_maker.beta = best_beta

        return {
            'alpha': best_alpha,
            'beta': best_beta,
            f'best_{metric}': best_metric,
        }

    def analyze_boundary_size(
        self,
        probs: torch.Tensor,
    ) -> Dict:
        """分析边界域大小"""
        decisions = self.decision_maker.get_decisions(probs)

        n_total = len(decisions)
        n_boundary = (decisions == 2).sum().item()
        n_positive = (decisions == 0).sum().item()
        n_negative = (decisions == 1).sum().item()

        return {
            'total': n_total,
            'positive_region': n_positive,
            'negative_region': n_negative,
            'boundary_region': n_boundary,
            'boundary_ratio': n_boundary / n_total if n_total > 0 else 0,
        }