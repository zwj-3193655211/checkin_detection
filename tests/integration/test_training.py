import pytest
import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestIntegration:
    """集成测试"""

    def test_end_to_end_decision(self):
        """端到端测试：从概率到决策"""
        from src.models.three_way_decision import ThreeWayDecision

        twd = ThreeWayDecision(alpha=0.85, beta=0.35)

        # 测试各种概率
        test_probs = [
            0.95,  # 正常
            0.85,  # 正常 (边界)
            0.6,   # 不确定
            0.35,  # 不确定/异常边界
            0.2,   # 异常
            0.05,  # 异常
        ]

        for prob in test_probs:
            decisions = twd.get_decisions(torch.tensor([prob]))
            assert decisions.shape == (1,)

    def test_model_with_decision(self):
        """测试模型与决策器的集成"""
        from src.models.classifier import ThreeWayDecisionClassifier
        from src.models.three_way_decision import ThreeWayDecision

        model = ThreeWayDecisionClassifier(input_dim=512)
        decision_maker = ThreeWayDecision(alpha=0.85, beta=0.35)

        # 模拟输入
        x = torch.randn(8, 512)

        # 模型预测
        outputs = model(x)

        # 三支决策
        decisions = decision_maker.get_decisions(outputs['normal_prob'])

        assert decisions.shape == (8,)
        assert all(d.item() in [0, 1, 2] for d in decisions)


class TestTrainerLogic:
    """训练逻辑测试"""

    def test_early_stopping(self):
        """测试早停逻辑"""
        from src.models.loss import EarlyStopping

        es = EarlyStopping(patience=5, mode='max')

        # 模拟改进
        assert not es(0.5)
        assert not es(0.6)
        assert not es(0.7)
        assert es(0.65)  # 没有改进，counter=1
        assert not es(0.75)  # 改进，重置counter

        # 连续不改进
        es2 = EarlyStopping(patience=3, mode='max')
        assert not es2(0.5)
        assert not es2(0.45)  # 下降，不算改进
        assert not es2(0.4)
        assert es2(0.4)  # 第三次，触发早停

    def test_metric_tracker(self):
        """测试指标跟踪"""
        from src.models.loss import MetricTracker

        tracker = MetricTracker()

        tracker.update({'loss': [0.5, 0.4], 'acc': [0.8]})
        tracker.commit_epoch()

        tracker.update({'loss': [0.3], 'acc': [0.9]})
        tracker.commit_epoch()

        assert 'loss' in tracker.history
        assert 'acc' in tracker.history
        assert len(tracker.history['loss']) == 2
        assert tracker.get_last('loss') == 0.3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])