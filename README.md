# 晨读晨练签到打卡检测系统

基于三支决策(Three-Way Decision)的签到检测系统，用于自动识别正常/异常的晨读和晨跑打卡图片。

## 系统概述

系统将图片分为三类：
- **正常 (Accept)**: 置信度 ≥ α，直接通过
- **异常 (Reject)**: 置信度 ≤ β，直接拒绝
- **不确定 (Uncertain)**: β < 置信度 < α，需要人工审核

## 场景特征

### 晨读场景
- 人物
- 教室
- 投影仪

### 晨跑场景
- 操场
- 人物
- 天空
- 树木

## 环境配置

```bash
# 创建conda环境
conda env create -f environment.yml

# 激活环境
conda activate checkin_detection
```

## 项目结构

```
checkin_detection/
├── data/               # 数据目录
│   ├── raw/            # 原始图片目录（统一存储位置）
│   ├── labels.json     # 标签文件
│   ├── dataset_stats.json  # 数据集统计
│   └── cache/          # 特征缓存
├── src/               # 源代码
│   ├── checkin_system.py   # 主程序界面
│   ├── train_resnet.py     # 训练模型
│   ├── data/               # 数据处理
│   ├── models/             # 模型定义
│   └── features/           # 特征提取
├── scripts/           # 工具脚本
│   ├── preprocessing.py    # 预处理
│   ├── integrate_new_data.py  # 数据整合
│   └── complete_labels.py     # 标签处理
├── configs/           # 配置文件
├── outputs/           # 输出目录（模型、报告）
└── tests/             # 测试
```

## 使用方法

### 1. 启动主程序（可视化界面）
```bash
python src/checkin_system.py
```

主程序功能：
- 📁 选择数据包文件夹
- 🔍 开始预测（自动识别晨读/晨跑）
- ✏️ 人工审核（审核不确定的图片）
- 📊 生成报告

### 2. 训练模型
```bash
python src/train_resnet.py
```

训练会从 `data/raw/` 目录读取图片，使用 `data/labels.json` 中的标签。

### 3. 数据预处理
```bash
python scripts/preprocessing.py
```

预处理会分析 `data/raw/` 目录中的图片。

### 4. 整合新数据
```bash
python scripts/integrate_new_data.py
```

## 数据目录说明

### data/raw/ 目录
- **位置**：`checkin_detection/data/raw/`
- **用途**：统一存储所有图片
  - 用于训练模型
  - 用于预处理和分析
  - 用于数据管理
- **说明**：所有图片都存储在这个目录，统一管理

**注意**：所有脚本统一使用 `data/raw/` 目录作为图片存储位置。

## 数据集统计

- 总图片数：2040 张
- 晨读：1503 张 (73.7%)
- 晨跑：505 张 (24.8%)
- 异常：29 张 (1.4%)

数据划分：
- 训练集：1383 张
- 验证集：189 张
- 测试集：468 张

## 技术栈

- PyTorch 2.0+
- ResNet18 (预训练模型)
- CLIP (ViT-B/32)
- Tkinter (GUI界面)
- Albumentations

## 作者

晨读晨练签到打卡项目组
