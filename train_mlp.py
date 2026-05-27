"""
统一训练脚本 - 训练双MLP模型

功能:
1. 训练二分类器MLPClassifier（晨读/晨跑）
2. 训练优化版特征预测器MLPFeaturesOptimized（11维特征）

核心改进:
- 特征预测器使用ConfidenceRegularizedLoss防止过度自信
- 推理时使用温度缩放降低置信度
- 使用Focal Loss处理类别不平衡
- 特征级数据增强
"""
import torch
import torch.nn as nn
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.models.mlp import MLPClassifier
from src.models.mlp_features_optimized import MLPFeaturesOptimized, ConfidenceRegularizedLoss

torch.manual_seed(42)
np.random.seed(42)

FEATURE_INDEX = {
    "人脸": 0, "蓝色桌子": 1, "教室": 2, "投影幕布": 3,
    "跑道": 4, "天空": 5, "绿地": 6, "树木": 7,
    "旗杆": 8, "号码布": 9, "主席台": 10,
}

CLASS_FEATURES = {
    '晨读': ['人脸', '蓝色桌子', '教室', '投影幕布'],
    '晨跑': ['人脸', '跑道', '天空', '绿地', '树木', '旗杆', '号码布', '主席台']
}

TEMPERATURE = 5.0
ALPHA_ACCEPT = 0.85    # 优化后
ALPHA_AUTO_PASS = 0.85 # 优化后
FEATURE_THRESHOLD = 0.60 # 优化后
MIN_FEATURES = 3
RUN_FEATURE_THRESH = 5
READ_FEATURE_THRESH = 3


def validate_with_three_way_decision(classifier, features_model, X_val, y_main_val, y_features_val, epoch=0, is_final=False):
    """使用三支决策进行端到端验证"""
    classifier.eval()
    features_model.eval()

    with torch.no_grad():
        logits = classifier(X_val.float())
        scaled_probs = torch.softmax(logits / TEMPERATURE, dim=1)
        confidences, preds = scaled_probs.max(dim=1)

        feat_probs = features_model(X_val.float(), inference=True)

    normal_mask = (y_main_val >= 0)
    anomaly_mask = (y_main_val == -1)
    normal_total = normal_mask.sum().item()
    anomaly_total = anomaly_mask.sum().item()

    rule1_count = 0
    rule2_count = 0
    rule3_count = 0
    rule4_count = 0
    review_count = 0
    miss_count = 0
    detail_results = []

    for i in range(len(X_val)):
        pred_label = '晨读' if preds[i] == 0 else '晨跑'
        confidence = confidences[i].item()
        true_label = y_main_val[i].item()

        class_feats = CLASS_FEATURES[pred_label]
        matched = [f for f, fidx in FEATURE_INDEX.items()
                   if feat_probs[i, fidx].item() > FEATURE_THRESHOLD and f in class_feats]
        matched_count = len(matched)

        r1 = pred_label == '晨跑' and confidence >= ALPHA_AUTO_PASS and matched_count >= RUN_FEATURE_THRESH
        r2 = pred_label == '晨读' and confidence >= ALPHA_AUTO_PASS and matched_count >= READ_FEATURE_THRESH
        r3 = matched_count < MIN_FEATURES
        r4 = confidence < ALPHA_ACCEPT

        if r1:
            decision = 0
            rule1_count += 1
        elif r2:
            decision = 0
            rule2_count += 1
        elif r3:
            decision = 1
            rule3_count += 1
        elif r4:
            decision = 1
            rule4_count += 1
        else:
            decision = 0

        is_anomaly = (true_label == -1)
        is_review = (decision == 1)

        if is_anomaly and not is_review:
            miss_count += 1
        if is_review:
            review_count += 1

        detail_results.append({
            'pred': pred_label,
            'true': true_label,
            'confidence': confidence,
            'matched_count': matched_count,
            'decision': '待审核' if decision == 1 else '自动通过',
            'is_anomaly': is_anomaly,
            'is_review': is_review
        })

    miss_rate = miss_rate_val = miss_count / anomaly_total * 100 if anomaly_total > 0 else 0
    review_rate = review_count / len(X_val) * 100
    normal_pass = normal_total - rule1_count - rule2_count
    normal_pass_rate = normal_pass / normal_total * 100 if normal_total > 0 else 0

    if is_final or (epoch % 10 == 0 and epoch > 0):
        print(f"\n  [三支决策验证]")
        print(f"    漏检率: {miss_rate:.2f}% (漏检{miss_count}/{anomaly_total})")
        print(f"    审核率: {review_rate:.2f}% (待审核{review_count}/{len(X_val)})")
        print(f"    正常通过率: {normal_pass_rate:.2f}% (通过{normal_pass}/{normal_total})")
        print(f"    规则触发: 规则1={rule1_count}, 规则2={rule2_count}, 规则3={rule3_count}, 规则4={rule4_count}")

    return {
        'miss_rate': miss_rate,
        'review_rate': review_rate,
        'normal_pass_rate': normal_pass_rate,
        'rule1': rule1_count,
        'rule2': rule2_count,
        'rule3': rule3_count,
        'rule4': rule4_count,
        'miss_count': miss_count
    }


def load_data():
    """加载特征和标签数据"""
    project_root = Path(__file__).parent

    df = pd.read_csv(project_root / 'data' / 'clip_features_cpu.csv')
    filenames = df['filename'].tolist()
    features = torch.tensor(df.drop('filename', axis=1).values, dtype=torch.float32)

    with open(project_root / 'data' / 'labels.json', 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
        labels = labels_data.get('labels', labels_data)

    with open(project_root / 'data' / 'split_config.json', 'r', encoding='utf-8') as f:
        split_config = json.load(f)

    return features, filenames, labels, split_config


def prepare_data(features, filenames, labels, split_config):
    """准备训练数据"""
    feature_dict = {fname: features[i] for i, fname in enumerate(filenames)}

    def prepare_split(file_list, is_train=False):
        X = []
        y_main = []
        y_features = []
        filenames_list = []

        for fname in file_list:
            if fname in feature_dict:
                label_info = labels.get(fname, {})
                label_name = label_info.get('label', '未知')

                if is_train and label_name not in ['晨读', '晨跑']:
                    continue

                if label_name == '晨读':
                    main_label = 0
                elif label_name == '晨跑':
                    main_label = 1
                else:
                    main_label = -1

                feature_label = [0] * 11
                saved_features = label_info.get('features', {})

                for feature_key, feature_idx in FEATURE_INDEX.items():
                    if saved_features.get(feature_key, False):
                        feature_label[feature_idx] = 1

                if sum(feature_label) == 0:
                    if saved_features.get('人脸', False):
                        feature_label[0] = 1
                    if label_name == '晨读':
                        for key, idx in [('蓝色桌子', 1), ('教室', 2), ('投影幕布', 3)]:
                            if saved_features.get(key, False):
                                feature_label[idx] = 1
                    elif label_name == '晨跑':
                        for key, idx in [('跑道', 4), ('天空', 5), ('绿地', 6), ('树木', 7),
                                        ('旗杆', 8), ('号码布', 9), ('主席台', 10)]:
                            if saved_features.get(key, False):
                                feature_label[idx] = 1

                X.append(feature_dict[fname])
                y_main.append(main_label)
                y_features.append(feature_label)
                filenames_list.append(fname)

        X = torch.stack(X)
        y_main = torch.tensor(y_main, dtype=torch.long)
        y_features = torch.tensor(y_features, dtype=torch.float32)

        return X, y_main, y_features, filenames_list

    X_train, y_main_train, y_features_train, _ = prepare_split(split_config['train_files'], is_train=True)
    X_val, y_main_val, y_features_val, _ = prepare_split(split_config['val_files'], is_train=False)
    X_test, y_main_test, y_features_test, _ = prepare_split(split_config['test_files'], is_train=False)

    print(f"\n训练集: {len(X_train)} 样本")
    print(f"  晨读: {(y_main_train == 0).sum().item()}, 晨跑: {(y_main_train == 1).sum().item()}")

    print(f"验证集: {len(X_val)} 样本")
    val_normal = (y_main_val >= 0).sum().item()
    val_anomaly = (y_main_val == -1).sum().item()
    print(f"  正常: {val_normal}, 异常: {val_anomaly}")

    print(f"测试集: {len(X_test)} 样本")
    test_normal = (y_main_test >= 0).sum().item()
    test_anomaly = (y_main_test == -1).sum().item()
    print(f"  正常: {test_normal}, 异常: {test_anomaly}")

    return {
        'train': (X_train, y_main_train, y_features_train),
        'val': (X_val, y_main_val, y_features_val),
        'test': (X_test, y_main_test, y_features_test)
    }


def augment_features(X, noise_std=0.02):
    """对CLIP特征添加高斯噪声作为数据增强"""
    if noise_std > 0:
        noise = torch.randn_like(X) * noise_std
        return X + noise
    return X


def train_classifier(X_train, y_train, X_val, y_val, output_path='data/mlp_classifier.pt'):
    """训练二分类器MLP（晨读/晨跑）"""
    print("\n" + "=" * 60)
    print("训练二分类器（晨读/晨跑）")
    print("=" * 60)

    train_mask = (y_train >= 0)
    X_train_filtered = X_train[train_mask]
    y_train_filtered = y_train[train_mask]

    val_mask = (y_val >= 0)
    X_val_filtered = X_val[val_mask]
    y_val_filtered = y_val[val_mask]

    model = MLPClassifier(input_dim=512, hidden_dim=256, output_dim=2, dropout=0.3)
    print(f"\n模型结构:\n{model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    epochs = 100
    batch_size = 32
    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        indices = torch.randperm(len(X_train_filtered))

        for i in range(0, len(X_train_filtered), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_filtered[batch_idx]
            y_batch = y_train_filtered[batch_idx]

            outputs = model(X_batch.float())
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_filtered.float())
            val_pred = val_outputs.argmax(dim=1)
            val_acc = (val_pred == y_val_filtered).sum().item() / len(y_val_filtered)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(X_train_filtered)*batch_size:.4f} Val Acc: {val_acc*100:.2f}%")

    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), output_path)
    print(f"\n[OK] 二分类器已保存: {output_path}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")

    return model


def train_features(X_train, y_train, X_val, y_val, y_main_val, classifier, output_path='data/mlp_features_optimized.pt'):
    """训练优化版特征预测器（使用三支决策验证）"""
    print("\n" + "=" * 60)
    print("训练优化版特征预测器")
    print("目标: 提高准确性，降低过高置信度")
    print("=" * 60)

    model = MLPFeaturesOptimized(
        input_dim=512,
        hidden_dim=512,
        output_dim=11,
        dropout=0.25,
        temperature=1.8
    )
    print(f"\n模型结构:\n{model}")

    criterion = ConfidenceRegularizedLoss(alpha=0.15, gamma=2.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    patience = 25
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    epochs = 250
    batch_size = 32
    noise_std = 0.02

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        indices = torch.randperm(len(X_train))

        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = augment_features(X_train[batch_idx], noise_std)
            y_batch = y_train[batch_idx]

            outputs = model(X_batch.float(), inference=False)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.float(), inference=False)
            val_loss = criterion(val_outputs, y_val).item()

            val_probs = model(X_val.float(), inference=True)
            val_preds = (val_probs > 0.5).float()
            val_acc = (val_preds == y_val).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_miss_rate = float('inf')
            patience_counter = 0
        else:
            patience_counter += 1

        val_metrics = validate_with_three_way_decision(classifier, model, X_val, y_main_val, y_val, epoch=epoch+1)

        if val_metrics['miss_rate'] < best_miss_rate:
            best_miss_rate = val_metrics['miss_rate']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            print(f"  训练损失: {total_loss/len(X_train)*batch_size:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  验证准确率: {val_acc*100:.2f}%")
            print(f"  当前学习率: {scheduler.get_last_lr()[0]:.6f}")

        if patience_counter >= patience:
            print(f"\n早停触发，当前轮次: {epoch+1}")
            break

    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), output_path)
    print(f"\n[OK] 优化版特征预测器已保存: {output_path}")

    return model


def evaluate_model(classifier, features_model, X_test, y_main_test, y_features_test):
    """评估模型"""
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)

    classifier.eval()
    features_model.eval()

    with torch.no_grad():
        logits = classifier(X_test.float())
        probs = torch.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)

        feat_probs = features_model(X_test.float(), inference=True)
        feat_preds = (feat_probs > 0.5).float()

    normal_mask = (y_main_test >= 0)
    anomaly_mask = (y_main_test == -1)

    if normal_mask.sum() > 0:
        normal_acc = (preds[normal_mask] == y_main_test[normal_mask]).float().mean().item()
        print(f"\n正常样本（晨读+晨跑）准确率: {normal_acc*100:.2f}%")

    if anomaly_mask.sum() > 0:
        anomaly_confs = confidences[anomaly_mask]
        print(f"\n异常样本检测:")
        print(f"  异常样本数: {len(anomaly_confs)}")
        print(f"  平均置信度: {anomaly_confs.mean().item():.4f}")
        print(f"  置信度范围: [{anomaly_confs.min().item():.4f}, {anomaly_confs.max().item():.4f}]")

    print(f"\n特征预测准确率: {(feat_preds == y_features_test).float().mean().item()*100:.2f}%")

    print(f"\n各特征准确率:")
    for feature_name, feature_idx in FEATURE_INDEX.items():
        feat_acc = (feat_preds[:, feature_idx] == y_features_test[:, feature_idx]).float().mean().item()
        avg_prob = feat_probs[:, feature_idx].mean().item()
        print(f"  {feature_name}: 准确率={feat_acc*100:.2f}%, 平均置信度={avg_prob:.4f}")

    print("\n" + "=" * 60)
    print("【三支决策端到端验证】")
    print("=" * 60)
    final_metrics = validate_with_three_way_decision(classifier, features_model, X_test, y_main_test, y_features_test, is_final=True)
    print(f"\n  规则说明:")
    print(f"    规则1: 晨跑 且 置信度 >= {ALPHA_AUTO_PASS} 且 特征数 >= {RUN_FEATURE_THRESH} → 自动通过")
    print(f"    规则2: 晨读 且 置信度 >= {ALPHA_AUTO_PASS} 且 特征数 >= {READ_FEATURE_THRESH} → 自动通过")
    print(f"    规则3: 特征数 < {MIN_FEATURES} → 待审核")
    print(f"    规则4: 置信度 < {ALPHA_ACCEPT} → 待审核")


def main():
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("=" * 60)
    print("统一训练脚本 - 训练双MLP模型")
    print("=" * 60)

    features, filenames, labels, split_config = load_data()
    dataset = prepare_data(features, filenames, labels, split_config)

    X_train, y_main_train, y_features_train = dataset['train']
    X_val, y_main_val, y_features_val = dataset['val']
    X_test, y_main_test, y_features_test = dataset['test']

    classifier = train_classifier(X_train, y_main_train, X_val, y_main_val)
    features_model = train_features(X_train, y_features_train, X_val, y_features_val, y_main_val, classifier)

    evaluate_model(classifier, features_model, X_test, y_main_test, y_features_test)

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
