import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeWayDecisionClassifier(nn.Module):
    """
    三支决策分类器

    将样本分为三个区域:
    - Positive: P(normal) >= alpha -> 正常
    - Negative: P(normal) <= beta -> 异常
    - Boundary: beta < P(normal) < alpha -> 不确定(人工审核)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_scenes: int = 2,
        dropout: float = 0.3,
        alpha: float = 0.85,
        beta: float = 0.35,
    ):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_scenes: 场景类别数
            dropout: Dropout比率
            alpha: 正域阈值
            beta: 负域阈值
        """
        super().__init__()

        self.alpha = alpha
        self.beta = beta

        # 场景分类器
        self.scene_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_scenes),
        )

        # 特征检测器（用于晨读场景）
        self.reading_feature_dim = 3  # person, classroom, projector
        self.reading_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.reading_feature_dim),
        )

        # 特征检测器（用于晨跑场景）
        self.running_feature_dim = 4  # person, playground, sky, trees
        self.running_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.running_feature_dim),
        )

        # 正常/异常分类头
        self.anomaly_classifier = nn.Sequential(
            nn.Linear(input_dim + num_scenes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # 输出正常概率
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, input_dim]

        Returns:
            dict包含:
                - scene_probs: 场景概率
                - reading_features: 晨读特征检测logits
                - running_features: 晨跑特征检测logits
                - normal_prob: 正常概率
                - decision: 三支决策结果
        """
        # 场景分类
        scene_logits = self.scene_classifier(x)
        scene_probs = F.softmax(scene_logits, dim=1)

        # 晨读特征检测
        reading_logits = self.reading_detector(x)
        reading_probs = torch.sigmoid(reading_logits)

        # 晨跑特征检测
        running_logits = self.running_detector(x)
        running_probs = torch.sigmoid(running_logits)

        # 异常分类（结合场景特征）
        combined = torch.cat([x, scene_probs], dim=1)
        normal_logits = self.anomaly_classifier(combined)
        normal_prob = torch.sigmoid(normal_logits).squeeze(-1)  # [batch_size]

        # 三支决策
        decisions = self._three_way_decision(normal_prob)

        return {
            'scene_probs': scene_probs,
            'scene_logits': scene_logits,
            'reading_features': reading_probs,
            'running_features': running_probs,
            'normal_prob': normal_prob,
            'decisions': decisions,
        }

    def _three_way_decision(self, normal_prob: torch.Tensor) -> torch.Tensor:
        """
        三支决策

        Args:
            normal_prob: 正常概率 [batch_size]

        Returns:
            decisions: 0=正常, 1=异常, 2=不确定
        """
        decisions = torch.full_like(normal_prob, 2, dtype=torch.long)  # 默认不确定

        # 正常区域: P(normal) >= alpha
        decisions[normal_prob >= self.alpha] = 0

        # 异常区域: P(normal) <= beta
        decisions[normal_prob <= self.beta] = 1

        return decisions

    def set_thresholds(self, alpha: float, beta: float):
        """设置三支决策阈值"""
        self.alpha = alpha
        self.beta = beta

    def get_decision_label(self, decision: int) -> str:
        """获取决策标签"""
        labels = {0: '正常', 1: '异常', 2: '不确定'}
        return labels.get(decision, '未知')


class BayesianThreeWayClassifier(nn.Module):
    """
    贝叶斯三支决策分类器（带MC Dropout不确定性估计）
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_scenes: int = 2,
        dropout: float = 0.5,  # MC Dropout需要较大的dropout
        alpha: float = 0.85,
        beta: float = 0.35,
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta

        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 场景分类器
        self.scene_classifier = nn.Linear(hidden_dim, num_scenes)

        # 异常分类器
        self.anomaly_classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict:
        """前向传播（MC Dropout会在训练和启用时生效）"""
        features = self.feature_extractor(x)

        scene_logits = self.scene_classifier(features)
        anomaly_logit = self.anomaly_classifier(features)

        normal_prob = torch.sigmoid(anomaly_logit).squeeze(-1)
        scene_probs = F.softmax(scene_logits, dim=1)

        decisions = self._three_way_decision(normal_prob)

        return {
            'features': features,
            'scene_probs': scene_probs,
            'normal_prob': normal_prob,
            'decisions': decisions,
        }

    def _three_way_decision(self, normal_prob: torch.Tensor) -> torch.Tensor:
        decisions = torch.full_like(normal_prob, 2, dtype=torch.long)
        decisions[normal_prob >= self.alpha] = 0
        decisions[normal_prob <= self.beta] = 1
        return decisions

    def enable_mc_dropout(self):
        """启用MC Dropout"""
        for m in self.modules():
            if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
                m.train()

    def disable_mc_dropout(self):
        """禁用MC Dropout"""
        for m in self.modules():
            if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
                m.eval()