import os
import json
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd


class CheckinDataset(Dataset):
    """晨读晨练签到打卡数据集"""

    # 场景类型
    SCENE_MORNING_READING = "morning_reading"
    SCENE_MORNING_RUNNING = "morning_running"

    # 标签类型
    LABEL_NORMAL = "normal"
    LABEL_ABNORMAL = "abnormal"
    LABEL_UNDECIDED = "undecided"

    # 晨读特征
    READING_FEATURES = ["person", "classroom", "projector"]
    # 晨跑特征
    RUNNING_FEATURES = ["person", "playground", "sky", "trees"]

    def __init__(
        self,
        data_dir: str,
        labels_file: Optional[str] = None,
        transform=None,
        img_size: int = 224,
    ):
        """
        Args:
            data_dir: 图片目录路径
            labels_file: 标签JSON文件路径
            transform: 数据增强transform
            img_size: 统一图像尺寸
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform

        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # 加载标签
        if labels_file and os.path.exists(labels_file):
            with open(labels_file, 'r', encoding='utf-8') as f:
                self.labels = json.load(f)
        else:
            self.labels = {}

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本"""
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)

        # 加载图像
        image = Image.open(img_path).convert('RGB')

        # 调整尺寸
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)

        # 应用transform
        if self.transform:
            image = self.transform(image)
        else:
            # 默认转换为tensor
            import torchvision.transforms as T
            image = T.ToTensor()(image)

        # 获取标签
        label_info = self.labels.get(img_name, {})

        return {
            'image': image,
            'filename': img_name,
            'scene': label_info.get('scene', self.SCENE_MORNING_READING),
            'features': label_info.get('features', []),
            'label': label_info.get('label', self.LABEL_UNDECIDED),
            'confidence': label_info.get('confidence', 0.0),
        }

    def get_feature_mask(self, scene: str, features: List[str]) -> torch.Tensor:
        """获取特征掩码"""
        if scene == self.SCENE_MORNING_READING:
            feature_list = self.READING_FEATURES
        else:
            feature_list = self.RUNNING_FEATURES

        mask = torch.tensor([1 if f in features else 0 for f in feature_list], dtype=torch.float32)
        return mask

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """自定义collate函数"""
        images = torch.stack([item['image'] for item in batch])
        filenames = [item['filename'] for item in batch]
        scenes = [item['scene'] for item in batch]
        labels = [item['label'] for item in batch]
        confidences = torch.tensor([item['confidence'] for item in batch])

        return {
            'images': images,
            'filenames': filenames,
            'scenes': scenes,
            'labels': labels,
            'confidences': confidences,
        }


class LabelsManager:
    """标签管理器"""

    def __init__(self, labels_file: str):
        self.labels_file = labels_file
        self.labels = self._load_labels()

    def _load_labels(self) -> Dict:
        """加载标签文件"""
        if os.path.exists(self.labels_file):
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_labels(self):
        """保存标签到文件"""
        with open(self.labels_file, 'w', encoding='utf-8') as f:
            json.dump(self.labels, f, ensure_ascii=False, indent=2)

    def add_label(
        self,
        filename: str,
        scene: str,
        features: List[str],
        label: str,
        confidence: float = 1.0
    ):
        """添加或更新标签"""
        self.labels[filename] = {
            'scene': scene,
            'features': features,
            'label': label,
            'confidence': confidence,
        }

    def get_label(self, filename: str) -> Optional[Dict]:
        """获取单个文件的标签"""
        return self.labels.get(filename)

    def get_stats(self) -> Dict:
        """获取标签统计信息"""
        stats = {
            'total': len(self.labels),
            'normal': 0,
            'abnormal': 0,
            'undecided': 0,
            'morning_reading': 0,
            'morning_running': 0,
        }

        for label_info in self.labels.values():
            label = label_info.get('label', 'undecided')
            scene = label_info.get('scene', 'morning_reading')

            if label == 'normal':
                stats['normal'] += 1
            elif label == 'abnormal':
                stats['abnormal'] += 1
            else:
                stats['undecided'] += 1

            if scene == 'morning_reading':
                stats['morning_reading'] += 1
            else:
                stats['morning_running'] += 1

        return stats