"""
双MLP模型定义

包含:
- MLPClassifier: 主分类器 (512 -> 256 -> 128 -> 2)
"""
import torch.nn as nn


class MLPClassifier(nn.Module):
    """MLP主分类器，输出2类（晨读/晨跑）"""

    def __init__(self, input_dim=512, hidden_dim=256, output_dim=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)
