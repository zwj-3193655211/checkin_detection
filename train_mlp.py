"""
使用CLIP特征训练MLP分类器

训练流程：
1. 加载预提取的CLIP特征 (clip_features_cpu.pt)
2. 加载标注数据 (labels.json)
3. 训练MLP分类器 (512 -> 256 -> 128 -> 3)
4. 保存模型 (mlp_model.pt)
"""

import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def load_clip_features(features_path):
    """加载预提取的CLIP特征（支持.pt和.csv格式）"""
    print(f"加载CLIP特征: {features_path}")
    
    if features_path.suffix == '.csv':
        # CSV格式
        print("  格式: CSV")
        df = pd.read_csv(features_path)
        filenames = df['filename'].tolist()
        features = df.drop('filename', axis=1).values
        features = torch.tensor(features, dtype=torch.float32)
        print(f"  特征维度: {features.shape}")
        print(f"  样本数量: {len(features)}")
        return {'features': features, 'filenames': filenames}
    else:
        # PT格式
        print("  格式: PT")
        data = torch.load(features_path)
        print(f"  特征维度: {data['features'].shape}")
        print(f"  样本数量: {len(data['features'])}")
        return data


def load_labels(labels_path):
    """加载标注数据"""
    print(f"加载标注: {labels_path}")
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)

    # 处理不同的JSON结构
    if 'labels' in labels_data:
        labels = labels_data['labels']
    else:
        labels = labels_data

    print(f"标注数量: {len(labels)}")
    return labels


def prepare_dataset(features_data, labels, class_map={'晨读': 0, '晨跑': 1, '异常': 2}):
    """准备训练数据"""
    print("\n准备数据集...")

    X = []  # 特征
    y = []  # 标签
    filenames = []  # 文件名

    # 构建文件名到特征的映射
    feature_dict = {}
    for i, fname in enumerate(features_data['filenames']):
        feature_dict[fname] = features_data['features'][i]

    # 遍历标注数据
    for fname, info in labels.items():
        if fname in feature_dict:
            label_name = info.get('label')
            if label_name in class_map:
                X.append(feature_dict[fname])
                y.append(class_map[label_name])
                filenames.append(fname)

    X = torch.stack(X)
    y = torch.tensor(y, dtype=torch.long)

    print(f"有效样本数: {len(X)}")
    print(f"类别分布:")
    for label_name, label_idx in class_map.items():
        count = (y == label_idx).sum().item()
        print(f"  {label_name}: {count}")

    return X, y, filenames


def train_mlp(X_train, y_train, X_val, y_val, output_path='outputs/mlp_model.pt'):
    """训练MLP模型"""
    print("\n开始训练MLP...")

    # 创建模型
    model = MLP(input_dim=512, hidden_dim=256, output_dim=3, dropout=0.3)
    print(model)

    # 定义损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 转换为张量
    X_train = X_train.float()
    y_train = y_train.long()
    X_val = X_val.float()
    y_val = y_val.long()

    # 训练参数
    epochs = 100
    batch_size = 32
    best_val_acc = 0

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Mini-batch训练
        indices = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        # 学习率调整
        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            _, val_predicted = torch.max(val_outputs, 1)
            val_acc = (val_predicted == y_val).sum().item() / len(y_val)

        train_acc = correct / total

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {total_loss/len(X_train)*batch_size:.4f} "
                  f"Train Acc: {train_acc*100:.2f}% "
                  f"Val Acc: {val_acc*100:.2f}%")

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 保存模型
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\n✅ 模型已保存: {output_path}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")

    return model


def evaluate_model(model, X_test, y_test, class_names=['晨读', '晨跑', '异常']):
    """评估模型"""
    print("\n模型评估:")
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.float())
        _, predicted = torch.max(outputs, 1)

    # 总体准确率
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f"测试集准确率: {accuracy*100:.2f}%")

    # 各类别准确率
    print("\n各类别准确率:")
    for i, class_name in enumerate(class_names):
        mask = (y_test == i)
        if mask.sum() > 0:
            class_acc = (predicted[mask] == y_test[mask]).sum().item() / mask.sum().item()
            print(f"  {class_name}: {class_acc*100:.2f}% ({mask.sum().item()}个样本)")


def main():
    print("=" * 60)
    print("使用CLIP特征训练MLP分类器")
    print("=" * 60)

    # 路径配置
    project_root = Path(__file__).parent
    # CLIP特征文件路径（CSV格式优先，与labels.json同目录）
    csv_path = project_root / 'data' / 'clip_features_cpu.csv'
    pt_path = project_root / 'data' / 'clip_features_cpu.pt'
    
    if csv_path.exists():
        features_path = csv_path
    elif pt_path.exists():
        features_path = pt_path
    else:
        print(f"\n❌ 错误: CLIP特征文件不存在")
        print(f"   CSV路径: {csv_path}")
        print(f"   PT路径: {pt_path}")
        print(f"\n请先运行特征提取脚本生成特征文件")
        return
    
    labels_path = project_root / 'data' / 'labels.json'
    output_path = project_root / 'outputs' / 'mlp_model.pt'

    # 检查文件是否存在
    if not features_path.exists():
        print(f"\n❌ 错误: CLIP特征文件不存在")
        print(f"   路径: {features_path}")
        print(f"\n请先运行特征提取脚本生成特征文件")
        return

    if not labels_path.exists():
        print(f"\n❌ 错误: 标注文件不存在")
        print(f"   路径: {labels_path}")
        return

    # 1. 加载数据
    features_data = load_clip_features(features_path)
    labels = load_labels(labels_path)

    # 2. 准备数据集
    X, y, filenames = prepare_dataset(features_data, labels)

    # 3. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    print(f"\n数据集划分:")
    print(f"  训练集: {len(X_train)}")
    print(f"  验证集: {len(X_val)}")
    print(f"  测试集: {len(X_test)}")

    # 4. 训练模型
    model = train_mlp(X_train, y_train, X_val, y_val, output_path)

    # 5. 评估模型
    evaluate_model(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
