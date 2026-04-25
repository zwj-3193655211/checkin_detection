"""
数据增强模块

提供多种数据增强策略:
1. 基础增强 (torchvision)
2. 测试时增强 (TTA)
"""

import torchvision.transforms as T
import numpy as np
from PIL import Image
from typing import Callable, List, Optional


DEFAULT_SIZE = 224


def get_train_transforms(img_size: int = DEFAULT_SIZE) -> T.Compose:
    """获取训练时的数据增强 (torchvision)"""
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size: int = DEFAULT_SIZE) -> T.Compose:
    """获取验证/测试时的transform (torchvision)"""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_inference_transforms(img_size: int = DEFAULT_SIZE) -> T.Compose:
    """获取推理时的transform（无数据增强）"""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_tta_transforms(img_size: int = DEFAULT_SIZE):
    """获取测试时增强 (TTA) 的多个transform"""
    base = get_inference_transforms(img_size)
    transforms = [
        base,
        T.Compose([T.RandomHorizontalFlip(p=1.0), T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    ]
    return transforms