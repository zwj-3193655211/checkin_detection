# Findings & Decisions

## Requirements

### 用户需求
1. **Web界面** - 替代tkinter，支持远程访问
2. **GradCAM可视化** - 模型可解释性，增强结果可信度
3. **数据增强** - 提升模型泛化能力
4. **三支决策逻辑** - 已完成（自动调优alpha=0.70, beta=0.15）

### 当前系统状态
- ResNet18预训练模型微调
- 三分类：晨读、晨跑、异常
- 测试准确率：96.73%
- 三支决策已集成
- 异常类准确率低（33.33%，仅3个样本）

## Research Findings

### Web框架对比
| 框架 | 优点 | 缺点 |
|------|------|------|
| Flask | 轻量、灵活 | 需手动处理异步、无自动文档 |
| FastAPI | 自动文档、类型安全、异步 | 学习曲线 |
| Django | 功能完整 | 过于庞大 |
| Gradio | 快速原型 | 定制化有限 |

**结论**: FastAPI最适合

### GradCAM实现方案
- ResNet模型有forward_hook和backward_hook
- 可以提取最后一层卷积层的梯度
- 计算CAM热力图
- torchvision有现成实现可参考

### 数据增强库对比
| 库 | 优点 | 缺点 |
|----|------|------|
| albumentations | 性能好、丰富 | 需额外安装 |
| torchvision.transforms | 内置、简单 | 性能一般 |

**结论**: albumentations优先

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| FastAPI作为Web框架 | 自动Swagger文档、类型安全、异步性能好 |
| 简单HTML/JS前端 | 无需构建复杂前端，快速实现 |
| Docker部署 | 一键部署、环境隔离 |
| GradCAM使用hook机制 | torchvision原生支持，实现简单 |
| albumentations数据增强 | 性能优于torchvision transforms |
| 三支决策解耦设计 | 便于单独测试和维护 |

## 项目结构更新

```
checkin_detection/
├── src/
│   ├── train_resnet.py      # 训练脚本（待增加数据增强）
│   ├── checkin_system.py    # Tkinter GUI（已有）
│   ├── web_app.py           # 【新增】FastAPI Web应用
│   ├── gradcam.py           # 【新增】GradCAM可视化
│   └── models/
│       └── three_way_decision.py  # 三支决策（已有）
├── templates/               # 【新增】HTML模板
│   └── index.html
├── Dockerfile               # 【新增】Docker配置
├── docker-compose.yml       # 【新增】Docker Compose
└── requirements.txt         # 【更新】添加fastapi, uvicorn
```

## Issues Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| 无 | - | - |
