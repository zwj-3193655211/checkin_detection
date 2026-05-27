# 晨读晨练签到检测系统

基于CLIP+MLP的晨读晨练打卡检测系统，采用三支决策规则实现零漏检。

## 📊 性能指标

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| 漏检率 | 0% | 0.00% | ✅ |
| 人工审核率 | <30% | ~24% | ✅ |
| 分类准确率 | >95% | ~99% | ✅ |

## 🎯 核心特性

- **二分类MLP**：仅区分晨读/晨跑，不直接识别异常
- **11维特征预测**：可解释性强，提供分类依据
- **四规则决策系统**：规则1/2拦截 + 规则3/4放行
- **零漏检**：所有异常样本均进入待审核

## 📁 项目结构

```
checkin_detection/
├── data/
│   ├── clip_features_cpu.csv  # CLIP特征向量
│   ├── labels.json           # 标签数据
│   ├── split_config.json     # 数据集划分
│   ├── mlp_classifier.pt     # MLP主分类器（二分类）
│   └── mlp_features_optimized.pt  # MLP特征预测器（11维，优化版）
├── src/
│   ├── checkin_system.py     # 检测系统主程序
│   └── models/
│       ├── mlp.py            # MLP模型定义
│       └── mlp_features_optimized.py  # 优化版特征预测器
├── outputs/                  # 检测报告
├── train_mlp_binary.py       # 二分类训练脚本
├── test_current.py           # 当前配置测试脚本
└── README.md                 # 本文件
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建conda环境
conda create -n checkin_detection python=3.10
conda activate checkin_detection

# 安装依赖
pip install torch torchvision
pip install scikit-learn pandas numpy pillow
pip install git+https://github.com/openai/CLIP.git
```

### 2. 运行检测系统

```bash
# 启动GUI系统
python src/checkin_system.py
```

### 3. 测试当前配置

```bash
# 运行规则验证测试
python test_current.py
```

## 📋 三支决策规则

| 规则 | 触发条件 | 决策 |
|------|---------|------|
| 规则1 | 置信度 < 0.88 | 待审核 |
| 规则2 | 特征数 < 3 | 待审核 |
| 规则3 | 晨跑 且 特征数 ≥ 6 | 自动通过 |

**决策流程**：
1. 先检查规则3/4（高置信度放行）
2. 再检查规则1/2（拦截低质量样本）
3. 否则自动通过

## 📊 数据集统计

| 类别 | 数量 | 比例 |
|------|------|------|
| 晨读 | 1491张 | 73.1% |
| 晨跑 | 525张 | 25.7% |
| 异常 | 24张 | 1.2% |
| **总计** | **2040张** | **100%** |

## 🔧 核心参数

```python
TEMPERATURE = 5.0        # 温度缩放
ALPHA_ACCEPT = 0.88      # 置信度阈值
FEATURE_THRESHOLD = 0.50 # 特征阈值
MIN_FEATURES = 3         # 最少特征数
RULE3_THRESH = 6         # 晨跑自动通过阈值
```

## 📝 特征设计（11维）

| 索引 | 特征名 | 类别 |
|------|--------|------|
| 0 | 人脸 | 公共 |
| 1 | 蓝色桌子 | 晨读 |
| 2 | 教室 | 晨读 |
| 3 | 投影幕布 | 晨读 |
| 4 | 跑道 | 晨跑 |
| 5 | 天空 | 晨跑 |
| 6 | 绿地 | 晨跑 |
| 7 | 树木 | 晨跑 |
| 8 | 旗杆 | 晨跑 |
| 9 | 号码布 | 晨跑 |
| 10 | 主席台 | 晨跑 |

## 📖 文档

完整技术文档请参阅 [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)

## ⚠️ 注意事项

1. **CLIP特征不需要归一化**
2. **训练集不包含异常样本**，异常通过规则识别

## 📅 更新日期

2026-05-21