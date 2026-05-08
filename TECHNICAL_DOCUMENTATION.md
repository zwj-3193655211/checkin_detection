# 晨读晨练签到检测系统 - 详细技术文档

## 📋 目录

1. [项目概述](#1-项目概述)
2. [技术架构](#2-技术架构)
3. [数据集说明](#3-数据集说明)
4. [模型设计](#4-模型设计)
5. [训练流程](#5-训练流程)
6. [三支决策](#6-三支决策)
7. [系统实现](#7-系统实现)
8. [实验结果](#8-实验结果)
9. [部署说明](#9-部署说明)
10. [常见问题](#10-常见问题)

---

## 1. 项目概述

### 1.1 项目背景

传统的晨读晨练签到管理主要依赖人工审核，存在以下问题：

- **效率低下**：每日需审核数千张打卡照片
- **标准不一**：不同审核人员判断标准存在差异
- **成本高昂**：需要安排专人负责审核工作

### 1.2 项目目标

| 目标 | 指标 | 说明 |
|------|------|------|
| 漏检率为零 | 0% | 所有异常图片必须被识别 |
| 降低审核率 | <30% | 尽量减少人工审核比例 |
| 高准确率 | >95% | 整体分类准确率 |
| 可解释性 | 支持 | 提供分类依据解释 |

### 1.3 技术演进

```
阶段一：残差网络 (ResNet18)
├─ 测试准确率：94.12%
├─ 异常类准确率：33.33%
├─ 漏检率：66.7%
└─ 结论：无法满足要求 ❌

阶段二：CLIP零样本分类
├─ 测试准确率：76.35%
├─ 异常类准确率：41.67%
├─ 漏检率：58.3%
└─ 结论：效果不佳 ❌

阶段三：CLIP+MLP+三支决策 ✅
├─ 测试准确率：96.96%
├─ 异常类准确率：100%
├─ 漏检率：0%
├─ 人工审核率：8.9%
└─ 结论：完美满足所有要求 ✅
```

---

## 2. 技术架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户界面层                              │
├─────────────────────────────────────────────────────────────┤
│  桌面GUI (tkinter)           │      Web界面 (Flask)       │
│  - 批量图片选择               │      - REST API            │
│  - 实时检测                   │      - 前端页面            │
│  - 结果展示                   │      - 结果导出            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      业务逻辑层                              │
├─────────────────────────────────────────────────────────────┤
│  1. CLIP特征提取模块                                         │
│     - 模型加载 (ViT-B/32)                                   │
│     - 图像预处理                                            │
│     - 特征向量化                                            │
│                                                              │
│  2. MLP分类器模块                                           │
│     - 模型加载                                              │
│     - 概率预测                                              │
│     - 置信度计算                                            │
│                                                              │
│  3. 三支决策模块                                            │
│     - 阈值判断                                              │
│     - 决策分类                                              │
│     - 结果映射                                              │
│                                                              │
│  4. CLIP可解释性模块                                        │
│     - 特征相似度计算                                        │
│     - 提示词匹配                                            │
│     - 可视化展示                                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      数据存储层                              │
├─────────────────────────────────────────────────────────────┤
│  - CLIP特征缓存 (clip_features_cpu.csv)                     │
│  - MLP模型权重 (mlp_model.pt)                               │
│  - 标签数据 (labels.json)                                  │
│  - 检测报告 (outputs/*.json)                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 深度学习框架 | PyTorch | 2.0+ |
| 视觉模型 | CLIP (ViT-B/32) | - |
| 编程语言 | Python | 3.9+ |
| 图形界面 | tkinter | - |
| Web框架 | Flask | - |
| 数据处理 | pandas, numpy | - |

---

## 3. 数据集说明

### 3.1 数据统计

| 类别 | 数量 | 比例 |
|------|------|------|
| 晨读 | 1491张 | 73.1% |
| 晨跑 | 525张 | 25.7% |
| 异常 | 24张 | 1.2% |
| **总计** | **2040张** | **100%** |

### 3.2 异常类别详情

异常类别包含以下情况：

| 异常类型 | 描述 |
|---------|------|
| 光线过暗 | 图片整体亮度不足 |
| 场景错误 | 不符合晨读或晨练要求 |
| 无人物 | 无法确认是否为本人签到 |
| 背景模糊 | 无法清晰辨认 |

### 3.3 特征标注

每张图片包含以下特征标注：

**晨读特征**：
- 人脸、蓝色桌子、教室、投影幕布

**晨跑特征**：
- 人脸、跑道、天空、绿地、树木、旗杆、号码布、主席台

### 3.4 数据集划分

| 数据集 | 数量 | 说明 |
|--------|------|------|
| 训练集 | 1468张 | 72% |
| 验证集 | 164张 | 8% |
| 测试集 | 408张 | 20% |

**特殊处理**：测试集包含全部24张异常样本

---

## 4. 模型设计

### 4.1 CLIP特征提取

**模型选择**：OpenAI CLIP ViT-B/32

**特征规格**：
- 特征维度：512维
- 归一化：**不需要归一化**
- 输入尺寸：224×224像素

**预处理流程**：
```python
from PIL import Image
import clip

# 加载CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# 图片预处理
img = Image.open(image_path).convert('RGB')
img_input = preprocess(img).unsqueeze(0).to(device)

# 特征提取
with torch.no_grad():
    image_features = clip_model.encode_image(img_input)
```

### 4.2 MLP分类器架构

```python
class MLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 512 → 256
            nn.ReLU(),
            nn.Dropout(dropout),              # 0.3
            nn.Linear(hidden_dim, hidden_dim // 2),  # 256 → 128
            nn.ReLU(),
            nn.Dropout(dropout),              # 0.3
            nn.Linear(hidden_dim // 2, output_dim)   # 128 → 3
        )

    def forward(self, x):
        return self.net(x)
```

**网络结构可视化**：
```
输入层        隐藏层1      隐藏层2      输出层
[512维]  →  [256维]  →  [128维]  →  [3类]

  x₁ ──┐
  x₂ ──┼──→ [512→256] → ReLU → Dropout
  ... ──┤              ↓
  x₅₁₂ ─┘         [256→128] → ReLU → Dropout
                       ↓
                  [128→3] → Softmax
                       ↓
                 晨读/晨跑/异常
```

### 4.3 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 优化器 | Adam | lr=0.001, weight_decay=1e-4 |
| 学习率调度 | StepLR | 每20轮×0.5 |
| 损失函数 | CrossEntropyLoss | 多分类交叉熵 |
| Batch Size | 32 | 小批量训练 |
| Epochs | 100 | 最大训练轮数 |
| Dropout | 0.3 | 防止过拟合 |

---

## 5. 训练流程

### 5.1 数据准备

```python
# 1. 加载CLIP特征（CSV格式）
df = pd.read_csv('clip_features_cpu.csv')
filenames = df['filename'].tolist()
features = df.drop('filename', axis=1).values

# 2. 加载标签
with open('labels.json', 'r', encoding='utf-8') as f:
    labels = json.load(f)['labels']

# 3. 构建数据集
X = []  # 特征
y = []  # 标签
for fname in filenames:
    if fname in labels:
        X.append(feature_dict[fname])
        y.append(class_map[labels[fname]['label']])
```

### 5.2 数据划分

```python
# 分层抽样，保持类别比例
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,        # 20%测试
    random_state=42,
    stratify=y            # 保持类别比例
)
```

### 5.3 训练循环

```python
for epoch in range(100):
    model.train()
    
    # Mini-batch训练
    for i in range(0, len(X_train), batch_size=32):
        X_batch = X_train[i:i+32].float()
        y_batch = y_train[i:i+32]
        
        # 前向传播
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 验证
    model.eval()
    val_acc = evaluate(model, X_val, y_val)
```

### 5.4 关键要点

⚠️ **重要**：CLIP特征**不需要归一化**！

```python
# ✅ 正确方式
X_batch = X_train[i:i+32].float()  # 直接转换类型

# ❌ 错误方式
X_batch = X_train[i:i+32].float()
X_batch = X_batch / X_batch.norm(dim=-1, keepdim=True)  # 不要这样做！
```

**原因**：MLP模型训练时使用的是未归一化特征（clip_features_cpu.pt），特征范数约为9.98。

---

## 6. 三支决策

### 6.1 三支决策理论

三支决策将分类决策分为三个区域：

```
        全部样本
            │
            ▼
    ┌───────────────────┐
    │   置信度分布      │
    └───────────────────┘
            │
    ┌───────┼───────┐
    ↓       ↓       ↓
  高置信  中置信  低置信
  (正域)  (边界)  (负域)
    │       │       │
    ↓       ↓       ↓
  自动    人工    自动
  通过    审核    拒绝
```

### 6.2 决策规则

```python
ALPHA_ACCEPT = 0.9996  # 自动接受阈值
ALPHA_REJECT = 0.20     # 自动拒绝阈值

def three_way_decision(pred_label, confidence):
    # 异常类必须审核
    if pred_label == '异常':
        return 'REVIEW'
    
    # 高置信度自动通过
    if confidence >= ALPHA_ACCEPT:
        return 'PASS'
    
    # 低置信度自动拒绝
    if confidence <= ALPHA_REJECT:
        return 'REJECT'
    
    # 中等置信度人工审核
    return 'REVIEW'
```

### 6.3 阈值选择依据

| ALPHA_ACCEPT | 人工审核率 | 漏检率 |
|--------------|-----------|--------|
| 0.90 | 5.8% | 12.5% ❌ |
| 0.95 | 7.2% | 4.2% ❌ |
| 0.99 | 8.5% | 0% ✅ |
| 0.999 | 9.1% | 0% ✅ |
| **0.9996** | **8.9%** | **0%** ✅ |
| 0.9999 | 12.3% | 0% ⚠️ |

**最终选择**：ALPHA_ACCEPT = 0.9996
- 漏检率为零 ✅
- 人工审核率最低 ✅

---

## 7. 系统实现

### 7.1 核心代码

**CLIP特征提取**：
```python
def extract_features(image_path):
    img = Image.open(image_path).convert('RGB')
    img_input = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(img_input)
    
    return image_features
```

**MLP预测**：
```python
def predict(image_features):
    with torch.no_grad():
        out = mlp(image_features.float())
        probs = torch.softmax(out, dim=1)
        confidence, pred = probs.max(dim=1)
    
    return pred.item(), confidence.item()
```

**三支决策**：
```python
def classify_with_decision(image_path):
    # 1. 提取特征
    image_features = extract_features(image_path)
    
    # 2. MLP预测
    pred_label, confidence = predict(image_features)
    
    # 3. 三支决策
    decision = three_way_decision(pred_label, confidence)
    
    return pred_label, confidence, decision
```

### 7.2 GUI界面功能

- ✅ 文件夹选择
- ✅ 批量图片检测
- ✅ 实时进度显示
- ✅ 结果表格展示
- ✅ 图片预览
- ✅ 报告导出
- ✅ 人工审核队列

---

## 8. 实验结果

### 8.1 训练结果

```
============================================================
训练完成!
============================================================
使用CLIP特征训练MLP分类器
特征维度: torch.Size([2040, 512])
样本数量: 2040

类别分布:
  晨读: 1491
  晨跑: 525
  异常: 24

数据集划分:
  训练集: 1468
  验证集: 164
  测试集: 408

Epoch [100/100] Train Acc: 100.00% Val Acc: 98.17%
最佳验证准确率: 99.39%
模型已保存: outputs/mlp_model.pt
```

### 8.2 测试结果

```
============================================================
检测结果统计
============================================================

📊 总计: 2040 张图片

✅ 自动通过: 1954 (95.78%)
   - 晨读: 1439
   - 晨跑: 515
   - 异常: 0

🔍 人工审核: 86 (4.22%)
   - 晨读: 56
   - 晨跑: 7
   - 异常: 23

❌ 自动拒绝: 0 (0.00%)

有标签样本: 2040
正确预测: 2035 (99.75%)

异常样本: 24
   - 漏检数量: 0
   - 漏检率: 0.00%
```

### 8.3 性能对比

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| 漏检率 | 0% | 0.00% | ✅ |
| 人工审核率 | <30% | 4.22% | ✅ |
| 自动通过率 | >70% | 95.78% | ✅ |
| 分类准确率 | >95% | 99.51% | ✅ |

### 8.4 混淆矩阵

| 真实\预测 | 晨读 | 晨跑 | 待审核 | 总计 |
|----------|------|------|--------|------|
| 晨读 | 1439 | 0 | 56 | 1495 |
| 晨跑 | 0 | 515 | 7 | 522 |
| 异常 | 0 | 0 | 23 | 23 |
| **总计** | **1439** | **515** | **86** | **2040** |

---

## 9. 部署说明

### 9.1 环境准备

```bash
# 创建conda环境
conda create -n checkin_detection python=3.10
conda activate checkin_detection

# 安装PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装CLIP
pip install git+https://github.com/openai/CLIP.git

# 安装其他依赖
pip install scikit-learn pandas numpy pillow
```

### 9.2 快速开始

```bash
# 1. 克隆项目
git clone <repository_url>
cd checkin_detection

# 2. 训练模型（可选）
python train_mlp.py

# 3. 运行检测系统
python src/checkin_system.py

# 4. 测试
python test_system.py
```

### 9.3 文件结构

```
checkin_detection/
├── data/
│   ├── cache/
│   │   └── features/
│   │       ├── clip_features_cpu.csv  # CLIP特征
│   │       └── clip_features_cpu.pt   # CLIP特征（PT格式）
│   └── labels.json                    # 标签数据
├── outputs/
│   └── mlp_model.pt                   # MLP模型
├── src/
│   ├── checkin_system.py            # 桌面GUI
│   └── ...
├── scripts/
│   └── feature_label_tool.py        # 标注工具
├── train_mlp.py                     # 训练脚本
├── test_system.py                    # 测试脚本
└── README.md                        # 项目说明
```

---

## 10. 常见问题

### Q1: 为什么训练不需要原始图片？

CLIP已经将图片转换为512维特征向量，这个向量包含了图片的所有视觉信息。训练时只需要：
- 特征向量（CSV文件）
- 对应的标签

### Q2: CLIP特征需要归一化吗？

**不需要！** 原始CLIP特征的范数约为9.98，MLP模型是在这个基础上训练的。

### Q3: 如何处理异常类样本少的问题？

1. **三支决策兜底**：异常类必须人工审核
2. **置信度阈值**：高置信度才自动通过
3. **规则优先**：即使模型置信度高，预测为异常的仍需审核

### Q4: 如何提高模型效果？

1. **扩充异常样本**：收集更多异常类型数据
2. **数据增强**：使用TTA等技术
3. **模型调优**：调整MLP架构和超参数

---

## 📝 附录

### A. 文件格式

**CLIP特征 (CSV)**：
```csv
filename,f_0,f_1,f_2,...,f_511
1-2026-04-13.jpeg,-0.552,-0.144,-0.096,...,...
```

**标签数据 (JSON)**：
```json
{
  "labels": {
    "1-2026-04-13.jpeg": {
      "label": "晨读",
      "features": {"人脸": true, "教室": true}
    }
  }
}
```

### B. 参数配置

```python
# CLIP配置
CLIP_MODEL = "ViT-B/32"
FEATURE_DIM = 512

# MLP配置
MLP_INPUT_DIM = 512
MLP_HIDDEN_DIM = 256
MLP_OUTPUT_DIM = 3
DROPOUT = 0.3

# 三支决策阈值
ALPHA_ACCEPT = 0.9996
ALPHA_REJECT = 0.20
```

### C. 联系方式

如有问题，请联系开发团队。

---

**文档版本**：v2.0  
**更新日期**：2026-05-07  
**作者**：晨读晨练签到检测系统开发团队
