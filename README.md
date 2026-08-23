# 晨读晨练签到打卡检测系统（checkin_detection）

## 项目简介

基于 **SigLIP2-NaFlex-B/16** 提取 768 维视觉特征（原生宽高比零裁切 NaFlex，免从头训练），设计双 MLP 架构——主分类器（晨读/晨跑二分类）+ 特征预测器（11 维可解释特征），将复杂判定拆解为要素识别；引入温度缩放（T=5.0）平滑置信度、缓解过度自信；构建四规则三支决策系统，结合置信度与特征匹配数科学降低漏检与人工审核负担。

在约 2057 张标注数据上实现漏检率 0.00%、分类准确率约 99%、人工审核率约 15%；已在 GitHub 开源（github.com/zwj-3193655211/checkin_detection）。

---

## 近期工程优化（feature-naflex 分支）

在算法内核之上完成的工程化与体验改进，主线围绕 **SigLIP2-NaFlex-B/16** 展开：

1. **采用 SigLIP2-NaFlex-B/16 为主编码器（原生宽高比零裁切）**
   - 相比 CLIP ViT-B/32 的固定正方形 CenterCrop，SigLIP2-NaFlex 支持原生宽高比输入（NaFlex 可变 patch 网格），对晨跑全景、晨读教室等宽幅场景避免裁切带来的信息损失；768 维语义表征优于 CLIP 的 512 维。
   - 仍保留与 CLIP 在 GUI 下拉框一键热切换的能力，便于对比与部署（切换后自动重载编码器与双 MLP 头并动态刷新阈值策略）。

2. **逐维度阈值调优 + 安全下限（面向 SigLIP 优化）**
   - 放弃「单一全局 floor」，对 11 个特征维度各自以 Youden's J（TPR − FPR）独立寻优，让「打了标签的图更易通过、没打标签的图更不易通过」。
   - 叠加用户硬性要求的 **0.60 安全下限**：平均维度准确率仍保持 95.79%（远高于默认 88.44%），漏检恢复为 0.00%。

3. **修复切换大模型时的窗口卡死**
   - 根因：在主线程同步加载 SigLIP 大模型，阻塞 Tkinter 事件循环被判「未响应」。
   - 改为后台线程异步加载 + 主线程回刷 UI，加载期间禁用下拉框并显示状态栏提示，失败则回退原模型。

4. **一键启动脚本 `launch.bat`**
   - 用专属 conda 环境（`checkin_detection`）绝对路径调用，纯 ASCII 规避 GBK 解码中文 `.bat` 导致双击闪退的问题。

---

## 核心特性

- **SigLIP2-NaFlex 零裁切特征提取（主编码器）**：采用 SigLIP2-NaFlex-B/16，原生宽高比输入（NaFlex 可变 patch 网格）、768 维语义表征，免从头训练；可选 CLIP ViT-B/32 热切换。
- **可解释判定**：11 维特征预测器输出「人脸 / 蓝色桌子 / 教室 / 投影幕布 / 跑道 / 天空 / 绿地 / 树木 / 旗杆 / 号码布 / 主席台」，决策可追溯到具体要素。
- **温度缩放**：分类与特征 logit 分别除以温度（T=5.0 / T=1.8）平滑概率分布，缓解过度自信。
- **四规则三支决策**：自动通过 / 待审核 / 自动拒绝，结合置信度与特征匹配数，平衡漏检红线与人工审核负担。
- **编码器热切换**：SigLIP 与 CLIP 运行时无缝切换，便于对比与部署。

## 算法架构

```
输入图片（原生宽高比，零裁切）
   │
   ▼
[SigLIP2-NaFlex-B/16 视觉编码器(主)]  → 768 维图像特征
   │   （可选 CLIP ViT-B/32 → 512 维）
   ├─► [MLPClassifier]        → 晨读 / 晨跑 二分类 softmax
   └─► [MLPFeatures(11维)]    → 11 维特征 sigmoid（可解释）
   │
   ▼
[四规则三支决策]  ← 置信度 + 特征匹配数 + 逐维度阈值(含0.60下限)
   ├─ 自动通过
   ├─ 待审核（人工校正）
   └─ 自动拒绝 / 标记异常
```

- **主分类器**：输入 768（SigLIP）/ 512（CLIP）维特征，隐藏 256，输出 2（晨读/晨跑），softmax 前除以 `CLASSIFIER_TEMPERATURE=5.0`。
- **特征预测器**：输入 768 / 512 维特征，隐藏 512，输出 11，推理时已含 sigmoid + 温度缩放（`FEATURE_TEMPERATURE=1.8`）。
- **三支决策**：规则 1/2（自动通过：高置信度 + 足够特征）、规则 3（特征过少 → 待审核）、规则 4（置信度不足 → 待审核）。

## 性能结果

测试集（含 12 张异常）上的三支决策对比（详见 `EXPERIMENT_REPORT_NAFLEX.md`）：

| 编码器 / 配置 | 审核率 | 自动通过 | 漏检率 |
|---|---|---|---|
| CLIP-ViT-B/32（default 0.66/0.60） | 43.30% | 56.70% | 0.00% |
| CLIP-ViT-B/32（tuned） | 6.85% | 93.15% | 0.00% |
| SigLIP2-NaFlex-B/16（default） | 43.30% | 56.70% | 0.00% |
| **推荐：SigLIP2-NaFlex + 逐维度阈值 + 0.60 下限** | **14.95%** | **85.05%** | **0.00%** |

> 漏检（异常被放行）为一票否决项，必须 = 0%。
> 简介中的「人工审核率约 15%」即上表推荐配置（SigLIP2-NaFlex + 逐维度阈值 + 0.60 下限）口径：实测审核率 14.95%、自动通过 85.05%、漏检 0.00%。

## 快速开始

### 方式一：一键启动（推荐）

直接双击根目录的 `launch.bat`，自动调用专属 conda 环境 `checkin_detection` 运行 GUI（默认加载 SigLIP2-NaFlex 编码器）。

### 方式二：手动启动

```bash
# 1. 创建并激活专属环境（首次）
conda create -n checkin_detection python=3.10.20
conda activate checkin_detection
pip install -r requirements.txt

# 2. 启动 GUI
python src/checkin_system.py
```

启动后在 GUI 中：

1. 选择「识别模型」（默认 SigLIP2-NaFlex，可切换 CLIP）下拉框；
2. 点击「1 选择数据目录」选择待检测图片文件夹；
3. 点击「2 开始判别」执行预测与三支决策；
4. 点击「3 人工审核」校正待审核图片；
5. 点击「4 生成报告」导出 JSON 检测报告。

## 环境要求

- Python 3.10（专属环境 `checkin_detection`）
- PyTorch 2.11（CPU）、torchvision、transformers、timm、safetensors、Pillow、python-docx、openai/clip
- 详见 `requirements.txt` 与 `environment.yml`（已钉版本，并注明 Pillow/torch DLL 冲突、CLIP git 依赖等坑）

## 目录结构

```
checkin_detection/
├── launch.bat              # 一键启动脚本（专属环境绝对路径）
├── requirements.txt        # 依赖清单（钉版本）
├── environment.yml         # conda 环境定义
├── src/
│   ├── checkin_system.py   # GUI 主程序 + 三支决策 + 模型热切换
│   ├── encoders.py         # SigLIP / CLIP / ViT-Tiny 编码器工厂
│   ├── config.py           # 全局阈值 / 温度 / 决策参数
│   └── models/             # 双 MLP 头定义
├── scripts/                # 训练、调参、对比实验脚本
├── data/                   # 标注数据、模型权重、调优结果
├── outputs/                # 检测报告输出
└── EXPERIMENT_REPORT_NAFLEX.md  # 编码器公平对比与阈值调优实验报告
```

## 开源

- 仓库：https://github.com/zwj-3193655211/checkin_detection
- 分支：`feature-naflex`（SigLIP2-NaFlex 主编码器 + 工程优化）
