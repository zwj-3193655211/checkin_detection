# outputs 目录

此目录用于存储训练输出的模型文件和评估报告。

## 文件说明

- `resnet18_best.pt` - 最佳验证准确率的模型
- `resnet18.pt` - 训练结束时的模型
- `evaluation_report.json` - 评估报告

## 注意事项

1. 模型文件(.pt)较大，Git LFS建议超过50MB的文件
2. 首次使用需要运行 `python src/train_resnet.py` 训练模型
3. 或从发布页下载预训练模型
