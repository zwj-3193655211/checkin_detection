"""
优化版特征预测器 - 解决置信度过高问题

核心改进:
1. 温度缩放推理 - 降低预测置信度
2. 添加最大熵正则化 - 防止过度自信
3. 使用Focal Loss - 处理类别不平衡
4. 改进模型结构 - 添加残差连接和层归一化
5. 输出约束 - 限制最大置信度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPFeaturesOptimized(nn.Module):
    """优化版MLP特征预测器 - 匹配已保存的权重结构"""

    def __init__(self, input_dim=512, hidden_dim=512, output_dim=11, dropout=0.3, temperature=1.8):
        super().__init__()
        self.temperature = temperature  # 推理时的温度缩放因子

        # 匹配保存的权重结构: 512 -> 512 -> 256 -> 128 -> 11
        self.fc1 = nn.Linear(input_dim, hidden_dim)           # 512 -> 512
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)    # 512 -> 256
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)  # 256 -> 128
        self.ln3 = nn.LayerNorm(hidden_dim // 4)
        self.fc_out = nn.Linear(hidden_dim // 4, output_dim)  # 128 -> 11
        self.residual = nn.Linear(hidden_dim, hidden_dim // 4)  # 512 -> 128

        self.dropout = nn.Dropout(dropout)
        self.confidence_clamp = 0.95  # 最大置信度限制

    def forward(self, x, inference=False):
        """
        Args:
            x: 输入特征
            inference: 是否为推理模式（应用温度缩放）
        """
        out = self.fc1(x)
        out = self.ln1(out)
        out = F.silu(out)
        out = self.dropout(out)
        residual_input = out

        out = self.fc2(out)
        out = self.ln2(out)
        out = F.silu(out)
        out = self.dropout(out)

        residual = self.residual(residual_input)
        out = self.fc3(out)
        out = self.ln3(out)
        out = out + residual
        out = F.silu(out)
        out = self.dropout(out)

        logits = self.fc_out(out)

        if inference:
            logits = logits / self.temperature
            probs = torch.sigmoid(logits)
            probs = torch.clamp(probs, max=self.confidence_clamp)
            return probs

        return logits


class ConfidenceRegularizedLoss(nn.Module):
    """带置信度正则化的损失函数（仅训练时使用）"""

    def __init__(self, alpha=0.1, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        focal_weight = torch.pow(1 - probs, self.gamma) * targets + torch.pow(probs, self.gamma) * (1 - targets)
        focal_loss = (focal_weight * bce_loss).mean()

        entropy_reg = -probs * torch.log(probs + 1e-10) - (1 - probs) * torch.log(1 - probs + 1e-10)
        entropy_reg = entropy_reg.mean()

        loss = focal_loss - self.alpha * entropy_reg
        return loss
