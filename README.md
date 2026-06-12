# 晨读晨练签到打卡检测系统

基于 CLIP + 双 MLP 的晨读晨练打卡检测系统，采用温度缩放与四规则三支决策实现 **零漏检**。

## 性能指标

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| 漏检率 | 0% | 0.00% | ✅ |
| 人工审核率 | <30% | ~24% | ✅ |
| 分类准确率 | >95% | ~99% | ✅ |

## 核心特性

- **CLIP ViT-B/32 特征提取**：利用预训练视觉-语言模型提取 512 维高质量特征，无需从头训练
- **双 MLP 架构**：主分类器（二分类：晨读/晨跑）+ 特征预测器（11 维可解释特征），将大预测任务拆分为独立要素识别
- **温度缩放（Temperature Scaling）**：T=5.0，平滑置信度分布，防止模型过度自信，使异常样本更容易被规则拦截
- **四规则三支决策系统**：结合置信度与特征匹配数，科学地降低漏检率和人工审核率
- **零漏检**：所有异常样本均进入待审核队列，绝无遗漏

## 项目结构

```
checkin_detection/
├── data/
│   ├── raw/                          # 原始打卡图片（~2057张）
│   ├── clip_features_cpu.csv         # CLIP ViT-B/32 512维特征向量
│   ├── labels.json                   # 人工标注（主标签 + 11维特征）
│   ├── split_config.json             # 训练/验证/测试集划分
│   ├── mlp_classifier.pt             # MLP主分类器权重（二分类）
│   ├── mlp_features_optimized.pt     # MLP特征预测器权重（11维，优化版）
│   └── analyze_split.py              # 数据集划分统计分析
├── src/
│   ├── checkin_system.py             # 主程序：tkinter GUI + 预测 + 人工审核
│   └── models/
│       ├── mlp.py                    # MLP主分类器模型定义
│       └── mlp_features_optimized.py # 优化版11维特征预测器（残差+FocalLoss）
├── scripts/
│   ├── preprocessing.py              # 数据预处理（质量检查、CLIP特征提取、数据集划分）
│   └── feature_label_tool.py         # 图形化人工标注工具（tkinter GUI）
├── outputs/                          # 早期实验模型（ResNet18/50、CLIP分类器等，已废弃）
├── train_mlp.py                      # 统一训练脚本（双MLP）
├── tune_thresholds.py                # 阈值参数网格搜索优化
├── test_current.py                   # 当前配置规则验证脚本
├── environment.yml                   # Conda环境配置
├── requirements.txt                  # Pip依赖列表
├── setup.bat                         # Windows安装脚本
├── setup.sh                          # Linux/Mac安装脚本
├── README.md                         # 本文件
├── TECHNICAL_DOCUMENTATION.md        # 详细技术文档
└── ARCHITECTURE.md                   # 架构图文档（Mermaid）
```

## 快速开始

### 1. 环境准备

```bash
# 方式一：使用conda
conda env create -f environment.yml
conda activate checkin_detection

# 方式二：手动安装
conda create -n checkin_detection python=3.10
conda activate checkin_detection
pip install torch torchvision
pip install scikit-learn pandas numpy pillow
pip install git+https://github.com/openai/CLIP.git

# 方式三：Windows一键安装
setup.bat
```

### 2. 运行检测系统

```bash
python src/checkin_system.py
```

### 3. 训练模型（可选）

```bash
# 训练双MLP模型
python train_mlp.py

# 参数调优（网格搜索最优阈值）
python tune_thresholds.py

# 验证当前规则配置
python test_current.py
```

### 4. 人工标注工具

```bash
python scripts/feature_label_tool.py
```

## 四规则三支决策

系统使用四条规则按优先级顺序进行决策：

| 规则 | 触发条件 | 决策 | 说明 |
|------|---------|------|------|
| 规则1 | 晨跑 且 置信度 >= 0.85 且 匹配特征 >= 5 | 自动通过 | 晨跑快速通过 |
| 规则2 | 晨读 且 置信度 >= 0.85 且 匹配特征 >= 3 | 自动通过 | 晨读快速通过 |
| 规则3 | 匹配特征 < 3 | 待审核 | 少特征拦截 |
| 规则4 | 置信度 < 0.85 | 待审核 | 低置信拦截 |
| 默认 | 以上均不满足 | 自动通过 | — |

## 数据集统计

| 类别 | 数量 | 比例 |
|------|------|------|
| 晨读 | 1491张 | 73.1% |
| 晨跑 | 525张 | 25.7% |
| 异常 | 24张 | 1.2% |
| **总计** | **2040张** | **100%** |

## 核心参数

```python
TEMPERATURE = 5.0           # 温度缩放（平滑置信度分布，T↑ → 置信度更分散 → 异常更易被拦截）
ALPHA_ACCEPT = 0.85         # 置信度阈值
ALPHA_AUTO_PASS = 0.85      # 自动通过最低置信度
FEATURE_THRESHOLD = 0.60    # 特征存在判定阈值
MIN_FEATURES = 3            # 最少匹配特征数
RUN_FEATURE_THRESH = 5      # 晨跑快速通过特征数
READ_FEATURE_THRESH = 3     # 晨读快速通过特征数
```

## 11维特征设计

| 索引 | 特征名 | 类别 | 说明 |
|------|--------|------|------|
| 0 | 人脸 | 公共 | 判断是否有人在画面中 |
| 1 | 蓝色桌子 | 晨读 | 教室课桌特征 |
| 2 | 教室 | 晨读 | 教室场景特征 |
| 3 | 投影幕布 | 晨读 | 教室设备特征 |
| 4 | 跑道 | 晨跑 | 塑胶跑道特征 |
| 5 | 天空 | 晨跑 | 户外天空特征 |
| 6 | 绿地 | 晨跑 | 草地/绿化带特征 |
| 7 | 树木 | 晨跑 | 户外树木特征 |
| 8 | 旗杆 | 晨跑 | 操场标志特征 |
| 9 | 号码布 | 晨跑 | 晨跑号码布特征（权重×2） |
| 10 | 主席台 | 晨跑 | 操场主席台特征 |

## 技术文档

- 完整技术文档（含实验历程与方案演进）：[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)
- 系统架构图：[ARCHITECTURE.md](ARCHITECTURE.md)

## 注意事项

1. **CLIP特征不需要归一化**——直接使用 `encode_image` 的原始输出
2. **训练集不包含异常样本**——异常通过低置信度/少特征规则自然拦截
3. **号码布特征权重翻倍**——`feature_probs[9] = min(feature_probs[9] * 2.0, 1.0)`，这是晨跑的强判别特征
4. **特征预测器使用 Focal Loss**——应对正负样本不平衡，alpha=0.15, gamma=2.0

## 更新日期

2026-06-09
