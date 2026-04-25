import pytest
import torch
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import CheckinDataset, LabelsManager
from src.data.transforms import get_train_transforms, get_val_transforms


class TestCheckinDataset:
    """数据集测试"""

    def test_initialization(self, tmp_path):
        """测试数据集初始化"""
        # 创建临时图片
        from PIL import Image
        for i in range(5):
            img = Image.new('RGB', (224, 224), color=(255, 0, 0))
            img.save(tmp_path / f"test_{i}.jpg")

        dataset = CheckinDataset(data_dir=str(tmp_path))

        assert len(dataset) == 5
        assert dataset.img_size == 224

    def test_getitem(self, tmp_path):
        """测试获取单个样本"""
        from PIL import Image

        # 创建测试图片
        img = Image.new('RGB', (300, 200), color=(0, 255, 0))
        img.save(tmp_path / "test.jpg")

        dataset = CheckinDataset(data_dir=str(tmp_path))
        item = dataset[0]

        assert 'image' in item
        assert 'filename' in item
        assert item['filename'] == "test.jpg"

    def test_feature_mask(self):
        """测试特征掩码"""
        dataset = CheckinDataset(data_dir=".")

        # 晨读特征
        reading_mask = dataset.get_feature_mask(
            CheckinDataset.SCENE_MORNING_READING,
            ['person', 'classroom']
        )
        assert reading_mask.shape == (3,)
        assert reading_mask[0] == 1  # person
        assert reading_mask[1] == 1  # classroom
        assert reading_mask[2] == 0  # projector

        # 晨跑特征
        running_mask = dataset.get_feature_mask(
            CheckinDataset.SCENE_MORNING_RUNNING,
            ['person', 'sky']
        )
        assert running_mask.shape == (4,)
        assert running_mask[0] == 1  # person
        assert running_mask[1] == 0  # playground
        assert running_mask[2] == 1  # sky
        assert running_mask[3] == 0  # trees

    def test_collate_fn(self):
        """测试collate函数"""
        batch = [
            {
                'image': torch.randn(3, 224, 224),
                'filename': 'test1.jpg',
                'confidence': 0.9,
            },
            {
                'image': torch.randn(3, 224, 224),
                'filename': 'test2.jpg',
                'confidence': 0.8,
            },
        ]

        result = CheckinDataset.collate_fn(batch)

        assert 'images' in result
        assert 'filenames' in result
        assert 'confidences' in result
        assert result['images'].shape == (2, 3, 224, 224)
        assert len(result['filenames']) == 2


class TestLabelsManager:
    """标签管理器测试"""

    def test_initialization(self, tmp_path):
        """测试初始化"""
        labels_file = tmp_path / "labels.json"
        manager = LabelsManager(str(labels_file))

        assert manager.labels == {}

    def test_add_label(self, tmp_path):
        """测试添加标签"""
        labels_file = tmp_path / "labels.json"
        manager = LabelsManager(str(labels_file))

        manager.add_label(
            filename="test.jpg",
            scene="morning_reading",
            features=["person", "classroom"],
            label="normal",
            confidence=0.95,
        )

        assert "test.jpg" in manager.labels
        assert manager.labels["test.jpg"]["scene"] == "morning_reading"
        assert manager.labels["test.jpg"]["label"] == "normal"

    def test_get_stats(self, tmp_path):
        """测试统计信息"""
        labels_file = tmp_path / "labels.json"
        manager = LabelsManager(str(labels_file))

        # 添加测试数据
        manager.add_label("img1.jpg", "morning_reading", [], "normal", 1.0)
        manager.add_label("img2.jpg", "morning_reading", [], "abnormal", 1.0)
        manager.add_label("img3.jpg", "morning_running", [], "normal", 1.0)

        stats = manager.get_stats()

        assert stats['total'] == 3
        assert stats['normal'] == 2
        assert stats['abnormal'] == 1
        assert stats['morning_reading'] == 2
        assert stats['morning_running'] == 1


class TestTransforms:
    """数据增强测试"""

    def test_train_transforms(self):
        """测试训练数据增强"""
        transform = get_train_transforms()
        assert transform is not None

    def test_val_transforms(self):
        """测试验证数据增强"""
        transform = get_val_transforms()
        assert transform is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])