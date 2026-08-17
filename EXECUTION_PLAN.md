# 摆脱 CLIP 依赖：NaFlex 编码器 + 自训 Transformer 执行计划

> 版本：v1.0（2026-08-18）
> 前置调研：硬件实测、依赖可用性实测、网络连通性实测、数据集尺寸统计均已验证，证据见 §2。

---

## 0. 结论摘要

| 问题 | 结论 | 关键证据 |
|---|---|---|
| Q1 裁切丢信息 | **确认存在且严重**，NaFlex 可彻底解决 | 实测平均丢 35.6% 像素，最高 56.2% |
| Q2 训练专用模型 | **可行**，但"从零训练"不可行，走"换编码器 + 端到端微调"两步 | 数据仅 2067 张；ViT 从零需百万级 |
| Q3 笔记本算力 | **够用**，8GB 显存可跑全部 planned 实验 | SigLIP2-B/16 微调估算 4-6GB；ViT-Tiny <3GB |
| Q4 评价与回退 | 三 seed 均值±标准差 + 配对检验；git tag 一键回退 | 见 §3.4、§5 |

---

## 1. 背景与目标

**现状**：CLIP ViT-B/32（冻结）提 512 维特征 → MLP 双头（2 分类 + 11 维特征）→ 三支决策。

**问题**：
1. CLIP 预处理 Resize+CenterCrop 到 224×224，非方形图被裁切丢信息；
2. CLIP 为通用模型，对本项目（晨读/晨练打卡场景）不够专精，且冻结不可适应；
3. 需确认本机（RTX 5060 Laptop 8GB）能否承担训练；
4. 需要严谨的对比实验设计与失败回退方案。

**目标**：在不劣化现有三支决策指标（漏检率、审核率）的前提下，用原生宽高比编码 + 专属微调模型替代 CLIP，并形成完整的对比实验报告。

---

## 2. 调研证据（全部实测）

### 2.1 硬件实测

| 项目 | 实测值 | 判定 |
|---|---|---|
| GPU | RTX 5060 Laptop，8151 MiB（空闲 7005 MiB），驱动 610.88 | ✅ |
| **GPU 架构** | **Blackwell，compute capability 12.0（sm_120）** | ⚠️ 关键约束：torch 必须 ≥2.7 且为 cu128+ 构建，当前环境装的 2.11.0+cpu 不行 |
| 内存 | 31.2 GB（可用 20.4 GB） | ✅ 充裕 |
| 磁盘 | C 盘剩 47 GB，D 盘剩 177 GB（conda 环境在 D 盘） | ✅ cu 版 torch 约 3 GB，无压力 |
| 散热 | 笔记本形态 | ⚠️ 长时间满载会降频，训练按"短 epoch + 分段"设计 |

### 2.2 软件依赖实测（checkin_detection 环境）

| 依赖 | 现状 | 可用版本 | 结论 |
|---|---|---|---|
| torch | 2.11.0+**cpu**（这就是之前特征只能 CPU 提取的原因） | **`D:\VidPic Studio\.venv` 已有 2.11.0+cu128，GPU 实测通过** | **直接复用 VidPic venv，免 3GB 下载** |
| transformers | 未安装 | VidPic venv 已有 **4.57.6**（SigLIP2 支持验证 ✅） | 同上 |
| timm | 未安装 | 1.0.28 | venv 中补装（ViT-Tiny 权重托管在 HF，见网络预案） |
| pandas/PIL | 已有 | venv 已有 pandas 3.0.5 / PIL 12.3.0 | ✅ 训练脚本仅依赖 torch/numpy/pandas |
| pypi | — | 实测可达（HTTP 200） | ✅ |

### 2.3 模型权重下载实测（2026-08-18 二次复测，含 VPN 场景）

| 源 | 实测 | 结论 |
|---|---|---|
| HF 小文件（config.json）经 VPN 代理（127.0.0.1:7897） | 200，1.1s | ✅ 可用 |
| HF 大文件（model.safetensors，LFS CDN）经 VPN 代理 | 206 但 **0 字节下行** | ❌ 当前代理节点对 HF 文件 CDN 不通 |
| **ModelScope `google/siglip2-base-patch16-naflex`（官方镜像，直连无需代理）** | **实测 5.5 MB/s，文件列表验证：model.safetensors 1431.5 MB + tokenizer 共 1.47 GB，约 4-5 分钟下完** | ✅ **主用方案** |
| hf-mirror.com 直连 | 超时 | ❌（沙箱视角；你本机或可用作备选） |
| download.pytorch.org / pypi.org | 可达（pypi 走代理 200） | ✅ |

> 权重获取优先级：**ModelScope 直连（已验证）> HF+VPN 代理（小文件可，大文件看代理节点）> hf-mirror > 浏览器手动下载**。注意：你 PowerShell 里设的 `$env:HTTP_PROXY` 只对当前终端生效，跑下载脚本时需在同一终端或显式设置。

### 2.4 数据集尺寸统计实测（n=800 抽样）

- 主流尺寸：1080×1440、1260×1680、1170×1560（3:4 竖拍），另有 1260×2803 超长图；
- CLIP CenterCrop **平均丢弃 35.6% 像素，最高 56.2%**；
- 被裁区域集中在画面上下边缘——恰好是天空/旗杆/绿地/跑道/主席台等 11 维判定特征的所在位置。

### 2.5 NaFlex 技术要点（SigLIP 2 论文 arXiv:2502.14786）

- 出处澄清：NaFlex 由 Google 在 SigLIP 2 中提出；智谱 GLM-5V-Turbo 的 CogViT 在对比预训练阶段采用了该方案。开源权重为 Google 官方 `google/siglip2-*-naflex`（Apache-2.0，以 HF 模型卡为准）。
- 原理：图片不裁方，缩放至高宽均为 patch(16) 的整数倍、token 数 ≤ 目标序列长度（B/16 上限 512）；位置编码双线性插值到非方形网格；padding token 在注意力中被 mask。
- **论文明示的边界**：NaFlex 在训练过的分辨率之间插值良好，但**外推差**（序列长度别超 512）；B/16 变体未用自蒸馏/掩码预测损失（对下游微调无影响）。
- 对本项目数据的映射（3:4 → 240×320 = 15×20 = 300 token；1260×2803 → 约 144×320 = 180 token），**全部图零裁切、长宽比失真 ≤ (16-1)/宽 ≈ 6%**。

---

## 3. 四个核心问题的技术论证

### 3.1 Q1：图片尺寸不一，如何避免裁切损失

**结论：主用 NaFlex 预处理（零裁切），备选 letterbox。**

| 方案 | 裁切损失 | 长宽比失真 | 需要训练 | 备注 |
|---|---|---|---|---|
| 现状 CenterCrop | 平均 35.6% | 0 | 否 | 基线 |
| 三裁块平均（CLIP） | 0 | 0 | 否（仅重提特征） | 推理 3 倍开销，特征平均会模糊语义；仅作对照 |
| letterbox 补边 | 0 | 0 | 建议 | 冻结 CLIP 吃补边图属分布外输入，特征劣化；只配自训模型 |
| **NaFlex（SigLIP2）** | **0** | ≤6% | 否（可先冻结后微调） | 论文验证的原生比例方案，首选 |

### 3.2 Q2：CLIP 不够专精，如何训练最适合的模型

**结论：不搞从零训练（2000 张图 vs ViT 从零需百万级，无归纳偏置必败）。采用"淘汰制"两阶段：**

- **路线 A（第一阶段，低成本）**：冻结 SigLIP2-NaFlex-B/16 提 768 维特征 → 复用现有 MLP 双头训练管线（`input_dim` 512→768）。一次性同时解决 Q1+Q2 的一半，1 小时内出结果。
- **路线 B（第二阶段，按需）**：ViT-Tiny（ImageNet 预训练，5.5M 参数）+ 双输出头，端到端微调，输入 224×320 竖版（保留原生比例）。这是"自己训练的模型"，作为报告的核心创新点。
- **路线 C（可选加分）**：端到端微调 SigLIP2-NaFlex-B/16（86M，变长序列工程复杂度较高）。仅当 A/B 都不达标时启动。
- 淘汰逻辑：A 达标且 B 不超 A → 报告以 A 为主线、B 为对比实验；B 超 A → B 为主线。两条路线代码互不干扰，不存在互相拖累。

### 3.3 Q3：这台笔记本是否有足够能力（显存/时间预算）

显存估算（fp16/AMP，seq≤512，**Phase 1 冒烟测试实测校准**）：

| 任务 | 估算显存 | 8GB 判定 | 预估耗时 |
|---|---|---|---|
| SigLIP2-B/16 冻结推理提特征（batch 32） | 1.5–2.5 GB | ✅ | GPU 约 3–5 分钟 / 2067 张（CPU 兜底约 1 小时） |
| MLP 双头训练（复用现有脚本） | <1 GB | ✅ | 每模型约 5–15 分钟 |
| ViT-Tiny 端到端微调（batch 64，AMP） | 2–3 GB | ✅ | 100 epoch 约 20–40 分钟 |
| SigLIP2-B/16 端到端微调（batch 8–16，AMP） | 4–6 GB（开梯度检查点可再降约 1.5 GB） | ⚠️ 可行但紧 | 1–2 小时 |

理论依据：B/16 为 86M 参数，AMP 训练显存 ≈ 权重(0.17GB) + fp32 主副本(0.34GB) + Adam 动量方差(0.69GB) ≈ 1.2GB，其余为激活值（seq 300 量级、batch ≤16 时约 2–4GB）。**注意显存估算必须在 Phase 1 用实测数字替换，若实测爆显存按 §6 R3 降级。**

### 3.4 Q4：评价指标、对比实验、改进与回退

**指标体系（沿用并扩展现有三支决策评估）：**

| 层级 | 指标 | 说明 |
|---|---|---|
| 主指标 | 漏检率（异常/错类被自动通过） | 越低越好，**一票否决项**：新模型漏检率 > 基线即不采用 |
| 主指标 | 审核率（1−自动通过率） | 在漏检率不劣化前提下越低越好 |
| 主指标 | 自动通过正确率 | 三支决策端到端正确性 |
| 次指标 | 分类 accuracy / macro-F1 / 混淆矩阵 | 主分类头 |
| 次指标 | 11 维特征逐项 F1（阳性类）+ 宏平均 | 特征头，验证裁切修复是否兑现 |
| 诊断指标 | α（自动通过门槛）扫描曲线 | 漏检-审核 tradeoff，比单点阈值更公平 |

**对比实验矩阵：**

- 固定 `split_config.json`（seed 42 的 70/15/15 划分，异常样本不进训练集），**所有模型吃完全相同的样本划分**；
- 每个配置跑 **3 个 seed（42/3407/2026）**，报均值±标准差——test 集仅约 310 张，单 seed 波动可达 ±2%，不控方差会得出错误结论；
- 对照组：① CLIP+CenterCrop 基线（现有）② CLIP+三裁块平均（隔离"裁切"单一变量）③ SigLIP2-NaFlex 冻结+MLP（隔离"编码器"变量）④ ViT-Tiny 端到端 ⑤（可选）SigLIP2 微调；
- 统计检验：分类差异用逐样本配对 McNemar 检验；阈值类指标因调参自由度大，以效应量+多 seed 稳定性为主，不迷信 p 值；
- **阈值必须重调**：换了特征空间，`FEATURE_THRESHOLD_*` 不能沿用旧值，每组实验用同一套阈值搜索流程（复用 `tune_feature_thresholds.py`）。

**效果不如基线时的改进路径（按顺序尝试，每步只改一个变量）：**
1. NaFlex 序列长度 256→512（更多信息）；2. 换 so400m/14（400M，推理仍可行）；3. 解锁端到端微调（路线 C）；4. 数据诊断——重点人工检查 1260×2803 超长图是否为截图/拼图，必要时剔除或单独建模；5. 三裁块 CLIP 对照组如果反而最好，说明瓶颈不在裁切，回到数据侧找原因。

**回退方案：**
- 代码层：git tag 基线 + 分支开发，随时 `git checkout` 回退（见 §4）；
- 模型层：现有 `data/mlp_classifier.pt`、`mlp_features_optimized.pt`、`clip_features_cpu.csv`、`tuned_thresholds.json` 原地不动；新产物一律新命名（`siglip_features.csv`、`mlp_classifier_siglip.pt` 等）；
- 系统层：`src/config.py` 的编码器选择抽成可配置项，GUI 一行切回 CLIP 基线。

---

## 4. Git 策略

**现状**：仓库已在 `git@github.com:zwj-3193655211/checkin_detection.git`；main 领先 origin 6 个提交；工作区有 2 个已删除的旧报告 docx 未提交。

```
Phase 0（立刻做，半小时）:
  1. 确认 2 个 docx 删除是有意的（若是误删: git restore 恢复）
  2. git add -A && git commit -m "chore: 清理旧版报告文档"
  3. git push                          # 把领先的 6+1 个提交推上去
  4. git tag v2.1-clip-baseline        # 基线锚点：CLIP 方案的最终状态
  5. git push --tags
  6. git checkout -b feat/naflex-transformer   # 所有新工作在分支上做

合并规则：每个 Phase 验收通过 → merge 回 main 并打 tag（p1-env-ready、p2-naflex-mlp、p3-vit-tiny、p4-final）
回退规则：任何 Phase 失败 → git checkout v2.1-clip-baseline 即恢复全部代码；
         模型权重不受 git 管理（.gitignore 已排除），靠"新产物不覆盖旧文件"保证可回退
```

---

## 5. 分阶段执行计划

### Phase 0：Git 基线固化（0.5 小时）
见 §4。**验收**：`git log origin/main..main` 为空；tag 已推送。

### Phase 1：环境复用 + GPU 冒烟测试（0.5 小时）

> **更新（2026-08-18）**：实测发现 `D:\VidPic Studio\.venv` 已具备全部核心依赖，直接复用，免去 3GB torch 下载。

**复用环境实测结果（`D:\VidPic Studio\.venv`，python 3.13.5，体积 4.9GB）**：
- torch 2.11.0+**cu128**，`cuda.is_available()=True`，RTX 5060（sm_120）实测 matmul 通过 ✅
- transformers **4.57.6**（正好是计划锁定版本），SigLIP2 支持已验证 ✅
- pandas 3.0.5 / numpy 2.5.2 / PIL 12.3.0 ✅（本项目训练脚本仅依赖这些）
- 缺：timm、modelscope（小包，pip 补装即可）；sklearn 不需要

**执行方式**：
```bash
# 1. 备份 checkin_detection 环境清单（回退依据，venv 复用不改动它）
conda activate checkin_detection && pip freeze > requirements_backup_20260818.txt
# 2. 在 VidPic venv 中补装两个小包（增量安装，不动现有包）
"D:\VidPic Studio\.venv\Scripts\python.exe" -m pip install timm==1.0.28 modelscope
# 3. 之后所有新脚本统一用该解释器运行：
#    "D:\VidPic Studio\.venv\Scripts\python.exe" scripts/xxx.py
#    （旧 GUI/旧脚本继续用 checkin_detection conda 环境，互不干扰）
```

**复用的风险与对策**：这是与 VidPic Studio 共享的环境，若对方日后升级 transformers 可能破坏锁定版本。对策：所有新脚本入口处加版本断言（`assert transformers.__version__ == '4.57.6'`，冒烟测试负责把关）。若想完全隔离：`robocopy "D:\VidPic Studio\.venv" .venv_naflex /E` 克隆一份（4.9GB，D 盘空间充足），`python -m pip` 在克隆环境中可正常工作。

**冒烟测试脚本（`scripts/smoke_test_gpu.py`，新建，用 VidPic venv 运行）**：
1. 版本断言 + `torch.cuda.get_device_capability()==(12,0)`；
2. 从 ModelScope 拉 SigLIP2-NaFlex-B/16（正确仓库 ID：`google/siglip2-base-patch16-naflex`，**无需代理**；`modelscope download --model google/siglip2-base-patch16-naflex model.safetensors preprocessor_config.json config.json`，约 1.4GB / 5 分钟），本地路径加载；
3. 用 3 张真实图片（一张 3:4、一张超长图、一张方形）跑 `get_image_features`，打印实际输入尺寸/token 数；
4. batch 32 推理实测显存峰值，替换 §3.3 的估算值。
**验收**：全部通过。**失败处理**：权重拉不下→按 R1 四重预案（注意 VidPic 项目经验：你本机 hf-mirror 直连可用， clash 代理是坏的，下载前别挂代理）。**回退点**：venv 复用方案对两个项目都是增量改动，无需回退；checkin_detection conda 环境全程未动。

### Phase 2：NaFlex 特征重提 + MLP 重训（2 小时）——最低成本验证 Q1
1. 新建 `scripts/extract_siglip_features.py`（仿 `preprocessing.py` 结构）：NaFlex 预处理、seq_len=512、batch 32、GPU；产出 `data/siglip_features.csv`（filename + feat_0..feat_767），**不触碰旧 CSV**；
2. 抽查 10 张图的 NaFlex 实际输入尺寸，人工确认零裁切；
3. `train_mlp.py` 加 `--features-csv` 参数（默认旧值，零破坏）：`input_dim` 512→768，其余超参不动；
4. 阈值重调（复用 `tune_feature_thresholds.py`），产出 `tuned_thresholds_siglip.json`；
5. 跑 3 seed，记录指标。
**验收**：漏检率 ≤ 基线 且 审核率或特征宏 F1 至少一项显著改善（多 seed 方向一致）。**达标**→ merge，进入 Phase 3；**不达标**→ 按 §3.4 改进路径 1→2 尝试，仍不达标则 Phase 3 照常进行（两路线独立）。

### Phase 3：自训 ViT-Tiny 端到端（3 小时）——报告核心创新点
1. 新建 `src/models/vit_tiny_dual.py`：timm `vit_tiny_patch16_224.augreg_in21k_ft_in1k` 骨干（位置编码插值到 14×20 网格）+ 双输出头（2 分类 + 11 sigmoid），推理接口与现有 MLP 对齐；
2. 新建 `train_vit_tiny.py`：图像级数据增强（RandAugment 轻度 + ColorJitter + RandomResizedCrop 保守版，**不破坏上下语义**，禁止水平翻转类增强以外的大幅几何变换）、分层学习率（骨干 1e-4 / 头 1e-3）、AMP、cosine 调度、早停；输入统一 Resize 到 224×320（等比+轻微补边，比例失真 <1%）；
3. 跑 3 seed；接入三支决策评估。
**验收**：训练曲线正常收敛（无震荡/过拟合早停失效）；指标进入 §6 对比矩阵统一评估。
**回退点**：新文件独立，删文件即回退。

### Phase 4：对比实验矩阵 + 显著性分析（2 小时）
1. `scripts/run_experiment_matrix.py`：统一入口跑 §3.4 的全部配置 × 3 seed，输出汇总 CSV；
2. `scripts/analyze_results.py`：均值±std、McNemar 检验、α 扫描曲线、逐特征 F1 对比图；
3. 生成 `EXPERIMENT_REPORT_NAFLEX.md`。
**验收**：结论有数据支撑，方向性结论在 3 seed 上一致。

### Phase 5：系统接入与报告（2 小时）
1. `src/encoders.py`：编码器抽象（CLIP / SigLIP-NaFlex / ViT-Tiny 三实现，统一 `encode(image)->feature` 接口）；
2. `src/config.py` 加 `ENCODER` 配置项，GUI `checkin_system.py` 改为调用抽象层——**一行配置切回基线**；
3. 更新实训报告：架构图、实验对比表、结论与讨论（预训练表征 vs 专用微调的权衡分析）。

**总时间预算：约 10.5 小时有效工时，建议分 2–3 天完成（中间让笔记本散热）。**

---

## 6. 风险清单与预案

| # | 风险 | 概率 | 预案 |
|---|---|---|---|
| R1 | HF 大文件下载不通（实测：VPN 代理下 LFS CDN 0 字节下行） | 低 | **主用 ModelScope 直连（google/siglip2-base-patch16-naflex，实测 5.5MB/s，无需代理）**；备选：HF+VPN（换代理节点后重试大文件）、hf-mirror、浏览器手动下载后本地加载。timm 的 ViT-Tiny 权重（~22MB）同理，ModelScope 有 timm 官方镜像 |
| R2 | Blackwell sm_120 与部分旧版库不兼容 | 低 | 锁 torch≥2.11+cu128、transformers 4.57.6；Phase 1 冒烟测试先行 |
| R3 | SigLIP2 微调爆显存（若启动路线 C） | 中 | 开梯度检查点；batch 减半+梯度累积；序列长度降到 256 |
| R4 | 笔记本长时间满载降频/过热 | 中 | 训练分段（每 20 epoch 落盘）；晚间执行；必要时限制功耗 |
| R5 | 新方案指标不及基线 | 中 | 本计划的核心就是允许失败：§3.4 改进路径 + 全程新文件不覆盖 + git tag 一键回退；对照组 ②（三裁块 CLIP）帮助定位瓶颈归属 |
| R6 | 超长图（1260×2803）本身是截图/异常数据 | 低 | Phase 2 第 2 步人工抽查时一并确认；必要时剔除后重跑 split（打新 tag，不动基线划分） |
| R7 | transformers 4.57 与教程代码（多为 4.4x 时代）API 有差异 | 低 | 以官方 model card 代码为准；冒烟测试覆盖加载→编码全链路 |

---

## 7. 立即可执行的下一步

1. 确认工作区 2 个 docx 的删除意图 → 执行 Phase 0（git 基线）；
2. 执行 Phase 1（环境 + 冒烟测试，实测显存回填本计划 §3.3）；
3. Phase 2 结果出来后再决定 Phase 3 的投入力度。
