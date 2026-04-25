"""
使用预训练ResNet18微调 - 带验证和测试
"""
import json
import os
import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class CustomDS(Dataset):
    def __init__(self, data_dir, labels):
        self.data = []
        self.classes = {'晨读': 0, '晨跑': 1, '异常': 2}
        for fn, info in labels.items():
            label_name = info.get('label')
            if label_name in self.classes:
                path = os.path.join(data_dir, fn)
                if os.path.exists(path):
                    self.data.append((path, self.classes[label_name]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = Image.open(path).convert('RGB')
        img = transforms.Resize((224, 224))(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img, label


def evaluate(model, loader, device='cpu'):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    class_correct = {'晨读': 0, '晨跑': 0, '异常': 0}
    class_total = {'晨读': 0, '晨跑': 0, '异常': 0}
    
    with torch.no_grad():
        for img, label in loader:
            img, label = img.to(device), label.to(device)
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            
            # 各类别准确率
            for i, l in enumerate(label):
                class_name = ['晨读', '晨跑', '异常'][l.item()]
                class_total[class_name] += 1
                if predicted[i] == l:
                    class_correct[class_name] += 1
    
    accuracy = 100 * correct / total if total > 0 else 0
    
    return accuracy, class_correct, class_total


def main():
    print("=" * 60)
    print("ResNet18 训练与评估")
    print("=" * 60)
    
    # 加载数据
    labels_file = r'c:\Users\31936\Desktop\晨读晨练签到打卡检测\checkin_detection\data\labels.json'
    data_dir = r'c:\Users\31936\Desktop\晨读晨练签到打卡检测\checkin_detection\data\raw'
    output_dir = r'c:\Users\31936\Desktop\晨读晨练签到打卡检测\checkin_detection\outputs'
    
    print(f"\n📂 数据目录: {data_dir}")
    print(f"📄 标签文件: {labels_file}")
    
    labels = json.load(open(labels_file, encoding='utf-8'))['labels']
    print(f"📊 总标注数: {len(labels)}")
    
    # 创建完整数据集
    full_dataset = CustomDS(data_dir, labels)
    
    # 划分数据集 (70% 训练, 15% 验证, 15% 测试)
    all_data = full_dataset.data.copy()
    random.shuffle(all_data)
    
    n = len(all_data)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size+val_size]
    test_data = all_data[train_size+val_size:]
    
    print(f"\n📊 数据划分:")
    print(f"  训练集: {len(train_data)} 张 ({len(train_data)/n*100:.1f}%)")
    print(f"  验证集: {len(val_data)} 张 ({len(val_data)/n*100:.1f}%)")
    print(f"  测试集: {len(test_data)} 张 ({len(test_data)/n*100:.1f}%)")
    
    # 创建数据集类
    class TrainDS(Dataset):
        def __init__(self, data_list):
            self.data = data_list
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            path, label = self.data[idx]
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            return img, label
    
    train_ds = TrainDS(train_data)
    val_ds = TrainDS(val_data)
    test_ds = TrainDS(test_data)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # 加载预训练模型
    print("\n🔄 加载预训练 ResNet18...")
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 3)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    print("\n🚀 开始训练...")
    print("-" * 60)
    
    best_val_acc = 0
    num_epochs = 5
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_acc = 100 * correct / total
        
        # 验证
        val_acc, _, _ = evaluate(model, val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - 训练准确率: {train_acc:.2f}% - 验证准确率: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(output_dir, 'resnet18_best.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✅ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
    
    print("-" * 60)
    print("训练完成！")
    
    # 加载最佳模型进行测试
    print("\n📊 在测试集上评估...")
    print("=" * 60)
    model.load_state_dict(torch.load(best_model_path))
    test_acc, class_correct, class_total = evaluate(model, test_loader)
    
    print(f"\n🎯 最终测试结果:")
    print(f"  总体准确率: {test_acc:.2f}%")
    print(f"\n  各类别准确率:")
    for class_name in ['晨读', '晨跑', '异常']:
        if class_total[class_name] > 0:
            acc = 100 * class_correct[class_name] / class_total[class_name]
            print(f"    {class_name}: {acc:.2f}% ({class_correct[class_name]}/{class_total[class_name]})")
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, 'resnet18.pt')
    torch.save(model.state_dict(), final_model_path)
    
    print(f"\n✅ 模型已保存:")
    print(f"  最佳模型: {best_model_path}")
    print(f"  最终模型: {final_model_path}")
    
    # 保存评估报告
    report = {
        'best_val_accuracy': best_val_acc,
        'test_accuracy': test_acc,
        'class_accuracy': {
            name: 100 * class_correct[name] / class_total[name] if class_total[name] > 0 else 0
            for name in ['晨读', '晨跑', '异常']
        },
        'dataset_size': {
            'train': len(train_data),
            'val': len(val_data),
            'test': len(test_data)
        },
        'class_counts': {
            'test': class_total
        }
    }
    
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 评估报告: {report_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
