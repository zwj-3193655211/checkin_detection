# 晨读晨练签到检测系统 - 训练数据集

## 📋 数据集简介

这是一个用于训练晨读晨练打卡检测模型的数据集，包含CLIP视觉特征和人工标注的标签。

## 📁 数据集结构

```
checkin_detection/
├── data/
│   ├── clip_features_cpu.csv  # CLIP特征向量（2040样本 × 513列）
│   └── labels.json                    # 标签数据
├── outputs/
│   └── mlp_model.pt                  # 预训练模型（可选）
├── train_mlp.py                      # 训练脚本
└── README.md                         # 本文件
```

## 📊 数据统计

| 类别 | 数量 | 比例 |
|------|------|------|
| 晨读 | 1491 | 73.1% |
| 晨跑 | 525 | 25.7% |
| 异常 | 24 | 1.2% |
| **总计** | **2040** | **100%** |

## 🔧 快速开始

### 1. 环境准备

```bash
# 创建conda环境（可选）
conda create -n checkin_detection python=3.10
conda activate checkin_detection

# 安装依赖
pip install torch torchvision
pip install scikit-learn pandas numpy
pip install git+https://github.com/openai/CLIP.git
```

### 2. 下载数据集

下载本仓库，目录结构如下：

```
your_project/
├── data/
│   ├── clip_features_cpu.csv  # 特征向量
│   └── labels.json                    # 标签
├── outputs/                           # 创建空目录
├── train_mlp.py                      # 训练脚本
└── README.md
```

### 3. 训练模型

```bash
python train_mlp.py
```

训练过程：
```
============================================================
使用CLIP特征训练MLP分类器
============================================================
加载CLIP特征: data/clip_features_cpu.csv
  格式: CSV
  特征维度: torch.Size([2040, 512])
  样本数量: 2040
加载标注: data/labels.json
标注数量: 2040

准备数据集...
有效样本数: 2040
类别分布:
  晨读: 1491
  晨跑: 525
  异常: 24

数据集划分:
  训练集: 1468
  验证集: 164
  测试集: 408

开始训练MLP...
...
模型已保存: outputs/mlp_model.pt
最佳验证准确率: 99.39%
```

### 4. 测试模型

```bash
python test_system.py
```

## 📊 数据格式

### CLIP特征 (clip_features_cpu.csv)

```csv
filename,f_0,f_1,f_2,...,f_511
1-2026-04-13.jpeg,-0.552221,-0.144429,-0.095771,0.129928,...,...
1-2026-04-14.jpeg,-0.341436,-0.338564,-0.081769,0.400537,...,...
```

- **filename**: 图片文件名
- **f_0 ~ f_511**: 512维CLIP特征向量
- **特征维度**: 2040样本 × 513列
- **文件大小**: ~11 MB

### 标签数据 (labels.json)

```json
{
  "_schema": "checkin_labels_v2",
  "labels": {
    "1-2026-04-13.jpeg": {
      "label": "晨读",
      "scene": "morning_reading",
      "is_normal": "normal",
      "features": {
        "人脸": true,
        "蓝色桌子": true,
        "教室": true,
        "投影幕布": true
      }
    },
    "1-2026-04-14.jpeg": {
      "label": "晨跑",
      "features": {
        "人脸": true,
        "跑道": true,
        "天空": true,
        "绿地": true
      }
    }
  }
}
```

## 🔬 技术细节

### CLIP特征提取

- **模型**: OpenAI CLIP ViT-B/32
- **特征维度**: 512维
- **归一化**: 未归一化（与MLP训练一致）

### MLP模型架构

```
MLP(
  (net): Sequential(
    (0): Linear(512 → 256) + ReLU + Dropout(0.3)
    (1): Linear(256 → 128) + ReLU + Dropout(0.3)
    (2): Linear(128 → 3)
  )
)
```

### 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam (lr=0.001, weight_decay=1e-4) |
| 学习率调度 | StepLR (每20轮 × 0.5) |
| Batch Size | 32 |
| Epochs | 100 |
| 早停策略 | 验证准确率最优 |

## 📈 性能指标

使用三支决策后的检测效果：

| 指标 | 结果 | 目标 |
|------|------|------|
| 漏检率 | 0.00% | 0% ✅ |
| 人工审核率 | 4.22% | <30% ✅ |
| 自动通过率 | 95.78% | >70% ✅ |
| 分类准确率 | 99.51% | >95% ✅ |

### 三支决策阈值

```python
ALPHA_ACCEPT = 0.9996  # 自动接受阈值
ALPHA_REJECT = 0.20     # 自动拒绝阈值
```

## ⚠️ 注意事项

1. **CSV格式 vs PT格式**
   - CSV格式可读性更好，方便调试
   - PT格式加载更快，适合生产环境
   - `train_mlp.py` 优先使用CSV

2. **特征归一化**
   - CLIP特征**不需要归一化**
   - 与训练时保持一致才能获得最佳效果

3. **异常类样本较少**
   - 仅24张异常样本（1.2%）
   - 使用三支决策确保漏检率为零

## 📝 许可

本数据集仅供学术研究和教育目的使用。

## 👤 作者

晨读晨练签到检测系统开发团队

## 📅 更新日期

2026-05-07
