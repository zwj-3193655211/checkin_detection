import pytest
import torch
from src.models.three_way_decision import ThreeWayDecision


class TestThreeWayDecision:
    """三支决策逻辑测试"""

    def test_initialization(self):
        """测试初始化"""
        twd = ThreeWayDecision(alpha=0.85, beta=0.35)
        assert twd.alpha == 0.85
        assert twd.beta == 0.35

    def test_positive_region(self):
        """测试正域划分"""
        twd = ThreeWayDecision(alpha=0.85, beta=0.35)
        probs = torch.tensor([0.9, 0.95, 0.8, 0.5])
        pos_mask, bnd_mask, neg_mask = twd(probs)

        assert pos_mask.sum() == 2  # 0.9, 0.95 >= 0.85
        assert bnd_mask.sum() == 1  # 0.8 在 (0.35, 0.85) 之间
        assert neg_mask.sum() == 1  # 0.5 <= 0.35

    def test_negative_region(self):
        """测试负域划分"""
        twd = ThreeWayDecision(alpha=0.85, beta=0.35)
        probs = torch.tensor([0.2, 0.3, 0.35, 0.4])
        pos_mask, bnd_mask, neg_mask = twd(probs)

        # 0.2, 0.3 <= (1 - 0.35) = 0.65 -> negative
        # 但我们的实现是 P(normal) <= beta
        assert neg_mask.sum() == 3  # 0.2, 0.3, 0.35 <= 0.35
        assert bnd_mask.sum() == 1  # 0.4 在边界

    def test_boundary_region(self):
        """测试边界域"""
        twd = ThreeWayDecision(alpha=0.85, beta=0.35)
        probs = torch.tensor([0.4, 0.5, 0.6, 0.7, 0.8])
        pos_mask, bnd_mask, neg_mask = twd(probs)

        # 都在边界域
        assert bnd_mask.sum() == 5
        assert pos_mask.sum() == 0
        assert neg_mask.sum() == 0

    def test_get_decisions(self):
        """测试决策方法"""
        twd = ThreeWayDecision(alpha=0.85, beta=0.35)
        probs = torch.tensor([0.9, 0.5, 0.2])
        decisions = twd.get_decisions(probs)

        assert decisions[0].item() == 0  # 正常
        assert decisions[1].item() == 2  # 不确定
        assert decisions[2].item() == 1  # 异常

    def test_get_decision_labels(self):
        """测试标签映射"""
        twd = ThreeWayDecision(alpha=0.85, beta=0.35)
        decisions = torch.tensor([0, 1, 2])
        labels = twd.get_decision_labels(decisions)

        assert labels == ['正常', '异常', '不确定']

    def test_single_threshold(self):
        """测试单一阈值情况"""
        twd = ThreeWayDecision(alpha=0.7, beta=0.3)
        probs = torch.tensor([0.8, 0.5, 0.2])
        decisions = twd.get_decisions(probs)

        assert decisions[0].item() == 0  # >= 0.7
        assert decisions[1].item() == 2  # (0.3, 0.7)
        assert decisions[2].item() == 1  # <= 0.3


class TestClassifier:
    """分类器测试"""

    def test_model_initialization(self):
        """测试模型初始化"""
        from src.models.classifier import ThreeWayDecisionClassifier

        model = ThreeWayDecisionClassifier(
            input_dim=512,
            hidden_dim=256,
            num_scenes=2,
        )

        assert model is not None
        assert model.alpha == 0.85
        assert model.beta == 0.35

    def test_model_forward(self):
        """测试模型前向传播"""
        from src.models.classifier import ThreeWayDecisionClassifier

        model = ThreeWayDecisionClassifier(input_dim=512)
        x = torch.randn(4, 512)

        outputs = model(x)

        assert 'scene_probs' in outputs
        assert 'normal_prob' in outputs
        assert 'decisions' in outputs
        assert outputs['scene_probs'].shape == (4, 2)
        assert outputs['normal_prob'].shape == (4,)
        assert outputs['decisions'].shape == (4,)

    def test_threshold_setting(self):
        """测试阈值设置"""
        from src.models.classifier import ThreeWayDecisionClassifier

        model = ThreeWayDecisionClassifier()
        model.set_thresholds(alpha=0.9, beta=0.2)

        assert model.alpha == 0.9
        assert model.beta == 0.2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])