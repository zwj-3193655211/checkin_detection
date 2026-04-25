"""
三支决策阈值调优脚本
加载已保存的最佳模型，多次调整阈值找到最优配置
"""
import json
import os
import random
import numpy as np
import torch
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.append(os.path.dirname(__file__))
from models.three_way_decision import ThreeWayDecision, ThresholdTuner


class CustomDS(Dataset):
    def __init__(self, data_dir, labels):
        self.data = []
        for item in labels:
            img_path = os.path.join(data_dir, item['image'])
            if os.path.exists(img_path):
                self.data.append((img_path, item['label']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = Image.open(path).convert('RGB')
        transform = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        img = transform(image=None)['image']
        return img, label


def get_validation_probs(model, val_loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs)
            all_labels.append(labels)

    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)


def main():
    print("=" * 60)
    print("三支决策阈值调优")
    print("=" * 60)

    labels_file = r'c:\Users\31936\Desktop\晨读晨练签到打卡检测\checkin_detection\data\labels.json'
    data_dir = r'c:\Users\31936\Desktop\晨读晨练签到打卡检测\checkin_detection\data\raw'
    model_path = r'c:\Users\31936\Desktop\晨读晨练签到打卡检测\checkin_detection\outputs\resnet18_best.pt'

    labels_data = json.load(open(labels_file, encoding='utf-8'))['labels']
    labels = [{'image': k, **v} for k, v in labels_data.items()]

    dataset = CustomDS(data_dir, labels)

    n = len(dataset.data)
    random.seed(42)
    random.shuffle(dataset.data)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    val_data = dataset.data[train_size:train_size+val_size]
    test_data = dataset.data[train_size+val_size:]

    class ValDS(Dataset):
        def __init__(self, data_list):
            self.data = data_list
            self.classes = {'晨读': 0, '晨跑': 1, '异常': 2}
            self.transform = A.Compose([
                A.Resize(height=224, width=224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            path, label_name = self.data[idx]
            label = self.classes.get(label_name, 0)
            img = Image.open(path).convert('RGB')
            img = np.array(img)
            img = self.transform(image=img)['image']
            return img, label

    val_ds = ValDS(val_data)
    test_ds = ValDS(test_data)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"✅ 模型加载完成: {model_path}")

    val_probs, val_labels = get_validation_probs(model, val_loader, device)
    test_probs, test_labels = get_validation_probs(model, test_loader, device)

    print(f"\n📊 验证集: {len(val_data)} 样本")
    print(f"📊 测试集: {len(test_data)} 样本")

    binary_labels = (val_labels >= 2).long()
    test_binary_labels = (test_labels >= 2).long()

    decision_maker = ThreeWayDecision(alpha=0.85, beta=0.35)
    tuner = ThresholdTuner(decision_maker)

    print("\n🔧 正在调优阈值...")
    result = tuner.tune_on_validation(
        val_probs[:, 0],
        binary_labels,
        alpha_range=(0.6, 0.95),
        beta_range=(0.01, 0.5),
        metric='recall_safe'
    )

    print(f"\n✅ 阈值调优完成:")
    print(f"   最优 alpha: {result['alpha']:.2f}")
    print(f"   最优 beta: {result['beta']:.2f}")
    print(f"   安全召回率: {result['best_recall_safe']:.4f}")

    normal_probs = test_probs[:, 0]
    decisions = decision_maker.get_decisions(normal_probs)

    n_correct = 0
    class_correct = {'晨读': 0, '晨跑': 0, '异常': 0}
    class_total = {'晨读': 0, '晨跑': 0, '异常': 0}

    label_names = ['晨读', '晨跑', '异常']
    for i in range(len(decisions)):
        true_label = test_labels[i].item()
        pred_decision = decisions[i].item()

        name = label_names[true_label]
        class_total[name] += 1

        if pred_decision == 0:
            if true_label == 0:
                n_correct += 1
                class_correct[name] += 1
        else:
            if true_label >= 1:
                n_correct += 1
                class_correct[name] += 1

    test_acc = 100.0 * n_correct / len(decisions)

    print(f"\n🎯 测试集评估 (使用最优阈值):")
    print(f"   总体准确率: {test_acc:.2f}%")
    print(f"   各类别准确率:")
    for name in ['晨读', '晨跑', '异常']:
        if class_total[name] > 0:
            print(f"     {name}: {100.0*class_correct[name]/class_total[name]:.2f}% ({class_correct[name]}/{class_total[name]})")

    boundary_result = tuner.analyze_boundary_size(test_probs[:, 0])
    print(f"   边界域样本比例: {boundary_result['boundary_ratio']*100:.1f}%")


if __name__ == '__main__':
    main()
