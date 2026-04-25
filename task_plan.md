# Task Plan: 晨读晨练签到检测系统功能增强

## Goal
在现有ResNet18模型基础上，实现Web界面、GradCAM可视化和数据增强三大功能，同时保持三支决策逻辑集成。

## Current Phase
Phase 1: 需求分析与技术选型

## Phases

### Phase 1: 需求分析与技术选型
- [x] 三支决策逻辑已实现（alpha=0.70, beta=0.15自动调优）
- [ ] 分析Web界面技术选型（Flask vs FastAPI vs Gradio）
- [ ] 分析GradCAM可视化实现方案
- [ ] 分析数据增强策略
- [ ] 确定技术栈优先级
- **Status:** in_progress

### Phase 2: Web界面开发
- [ ] 选择Web框架（推荐FastAPI）
- [ ] 设计API接口：
  - POST /predict - 单张图片预测
  - POST /predict/batch - 批量预测
  - GET /health - 健康检查
- [ ] 实现前端页面：
  - 图片上传组件
  - 预测结果显示
  - 置信度展示
  - 人工审核入口
- [ ] 集成三支决策逻辑
- [ ] 添加CORS支持（跨域访问）
- [ ] Docker化部署配置
- **Status:** pending

### Phase 3: GradCAM可视化开发
- [ ] 实现GradCAM核心算法
- [ ] 集成到checkin_system.py
- [ ] 在Web界面中显示热力图叠加
- [ ] 支持点击查看详细解释
- [ ] 导出可视化结果
- **Status:** pending

### Phase 4: 数据增强实现
- [ ] 在train_resnet.py中集成数据增强
- [ ] 实现以下增强策略：
  - 随机水平翻转
  - 随机旋转（±15°）
  - 颜色抖动（亮度、对比度、饱和度）
  - 随机裁剪
- [ ] 添加数据增强开关配置
- [ ] 重新训练模型验证效果
- **Status:** pending

### Phase 5: 测试与部署
- [ ] Web界面功能测试
- [ ] GradCAM可视化测试
- [ ] 数据增强效果验证
- [ ] 整体集成测试
- [ ] GitHub更新与发布
- **Status:** pending

## Key Questions

1. Web界面选择Flask还是FastAPI？
   - Flask: 轻量简单，但需要手动处理异步
   - FastAPI: 现代、自动化文档、性能更好
   - 结论: FastAPI更适合

2. GradCAM需要哪些依赖？
   - torchvision已有hook机制
   - 只需要numpy和matplotlib用于可视化
   - 无需额外安装

3. 数据增强会影响三支决策阈值吗？
   - 需要重新训练后重新调优阈值
   - 计划在Phase 4完成后重新训练

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Web框架选FastAPI | 自动Swagger文档、类型安全、异步支持好 |
| 使用Flask+Vite/HTML | 简单易用，无需构建复杂前端 |
| GradCAM使用ResNetHook | torchvision内置支持，实现简单 |
| 数据增强使用albumentations | 性能优于torchvision transforms |
| 三支决策作为独立模块 | 与现有系统解耦，易于维护 |

## 技术栈

```
Web界面:
- Backend: FastAPI + Uvicorn
- Frontend: HTML + JavaScript (无框架)
- 部署: Docker + docker-compose

GradCAM:
- 使用torchvision ResNet hooks
- matplotlib生成热力图
- PIL合成叠加图

数据增强:
- albumentations库
- torchvision.transforms作为备选
```

## 预计工作流程

```
Phase 1 (1-2次对话)
    ↓
Phase 2: Web界面 (3-5次对话)
    ↓
Phase 3: GradCAM (2-3次对话)
    ↓
Phase 4: 数据增强 (2-3次对话)
    ↓
Phase 5: 测试与部署 (1-2次对话)
```
