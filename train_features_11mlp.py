"""
训练11个独立MLP特征预测器（每个特征专精一个二分类任务）

方案对比:
- baseline: MLPFeaturesOptimized 共享网络输出11维（512→512→256→128→11）
- 本方案:   11个独立小MLP，每个 512→128→1，各自独立训练

每个特征独立配置:
- Focal Loss alpha: 按该特征正负样本比设置（少数类加权）
- 温度: 独立温度缩放（默认统一1.8，可独立微调）
- 阈值: 验证集上网格搜索每特征最优阈值（对比时可选）

用法（checkin_detection 环境）:
    python train_features_11mlp.py
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    CLASSIFIER_TEMPERATURE, FEATURE_TEMPERATURE,
    FEATURE_THRESHOLD_READ, FEATURE_THRESHOLD_RUN,
    ALPHA_AUTO_PASS, ALPHA_REVIEW, MIN_FEATURES,
    RUN_FEATURE_THRESH, READ_FEATURE_THRESH,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "mlp_features_11x_v2.pt"

# ==================== 特征定义 ====================
FEATURE_INDEX = {
    "人脸": 0, "蓝色桌子": 1, "教室": 2, "投影幕布": 3,
    "跑道": 4, "天空": 5, "绿地": 6, "树木": 7,
    "旗杆": 8, "号码布": 9, "主席台": 10,
}
FEATURE_NAMES = list(FEATURE_INDEX.keys())  # 11个特征名

CLASS_FEATURES = {
    '晨读': ['人脸', '蓝色桌子', '教室', '投影幕布'],
    '晨跑': ['人脸', '跑道', '天空', '绿地', '树木', '旗杆', '号码布', '主席台']
}


# ==================== 模型定义 ====================
class SingleFeatureMLP(nn.Module):
    """单个特征的专用MLP: 512 -> 128 -> 1"""

    def __init__(self, input_dim=512, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.silu(self.ln1(self.fc1(x)))
        out = self.dropout(out)
        out = F.silu(self.ln2(self.fc2(out)))
        out = self.dropout(out)
        return self.fc_out(out)  # (N, 1) logits


class MLPEnsembleFeatures(nn.Module):
    """11个独立MLP集成，接口与 MLPFeaturesOptimized 兼容"""

    def __init__(self, input_dim=512, hidden_dim=128, num_features=11,
                 dropout=0.3, temperatures=None):
        super().__init__()
        self.heads = nn.ModuleList([
            SingleFeatureMLP(input_dim, hidden_dim, dropout)
            for _ in range(num_features)
        ])
        # 每个特征独立的推理温度
        if temperatures is None:
            temperatures = [FEATURE_TEMPERATURE] * num_features
        self.temperatures = temperatures

    def forward(self, x, inference=False):
        """返回 (N, 11) 特征概率，接口与原版一致"""
        outs = []
        for i, head in enumerate(self.heads):
            logits = head(x)
            if inference:
                logits = logits / self.temperatures[i]
            outs.append(torch.sigmoid(logits))
        return torch.cat(outs, dim=1)


# ==================== 数据加载 ====================
def load_data():
    """加载特征、标签、划分（与 train_mlp.py 一致）"""
    df = pd.read_csv(DATA_DIR / 'clip_features_cpu.csv')
    features = torch.tensor(df.drop('filename', axis=1).values, dtype=torch.float32)
    filenames = df['filename'].tolist()

    with open(DATA_DIR / 'labels.json', 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    labels = labels_data.get('labels', labels_data)

    with open(DATA_DIR / 'split_config.json', 'r', encoding='utf-8') as f:
        split_config = json.load(f)

    feature_dict = {fname: features[i] for i, fname in enumerate(filenames)}

    def prepare_split(file_list):
        X, y_main, y_features = [], [], []
        for fname in file_list:
            if fname not in feature_dict:
                continue
            info = labels.get(fname, {})
            label = info.get('label', '未知')
            if label == '异常':
                y_main.append(-1)
            elif label == '晨跑':
                y_main.append(1)
            else:
                y_main.append(0)

            feat = [0] * 11
            saved = info.get('features', {})
            for key, idx in FEATURE_INDEX.items():
                if saved.get(key, False):
                    feat[idx] = 1
            X.append(feature_dict[fname])
            y_features.append(feat)
        return (torch.stack(X), torch.tensor(y_main, dtype=torch.long),
                torch.tensor(y_features, dtype=torch.float32))

    X_train, y_main_train, yf_train = prepare_split(split_config['train_files'])
    X_val, y_main_val, yf_val = prepare_split(split_config['val_files'])
    X_test, y_main_test, yf_test = prepare_split(split_config['test_files'])
    return (X_train, y_main_train, yf_train), (X_val, y_main_val, yf_val), (X_test, y_main_test, yf_test)


def feature_alpha(y_feature):
    """按正类占比设置 Focal Loss alpha（少数类加权）

    v2: 限制在 [0.3, 0.7]，避免对极端稀疏特征（号码布 4% 正类）
    加权过重导致误报激增。
    """
    pos = y_feature.float().mean().item()
    alpha = 1.0 - pos  # 正类越多 alpha 越小
    return min(max(alpha, 0.30), 0.70)


class FocalLossWithAlpha(nn.Module):
    """带样本权重的Focal Loss + 熵正则化（单特征二分类）"""

    def __init__(self, alpha=0.5, gamma=2.0, entropy_beta=0.15):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.entropy_beta = entropy_beta

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        focal_w = torch.pow(1 - probs, self.gamma) * targets + torch.pow(probs, self.gamma) * (1 - targets)
        # alpha 加权: 正类 weight=alpha, 负类 weight=1-alpha
        w = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = (w * focal_w * bce).mean()
        # 熵正则化：防止过度自信（与原版 ConfidenceRegularizedLoss 一致）
        eps = 1e-10
        entropy = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps)).mean()
        return focal_loss - self.entropy_beta * entropy


# ==================== 训练单个特征 ====================
def train_one_feature(idx, feature_name, X_train, yf_train, X_val, yf_val,
                      alpha, hidden_dim=256, dropout=0.3, epochs=200,
                      patience=25, seed=42):
    """训练一个特征专用MLP，返回 (模型state, 最佳验证AUROC/准确率)

    v2: 模型容量 256 隐层，epochs 200
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    y_train = yf_train[:, idx].unsqueeze(1)
    y_val = yf_val[:, idx].unsqueeze(1)

    model = SingleFeatureMLP(512, hidden_dim, dropout)
    criterion = FocalLossWithAlpha(alpha=alpha, gamma=2.0, entropy_beta=0.15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    best_val_loss = float('inf')
    best_state = None
    best_acc = 0.0
    patience_counter = 0
    batch_size = 64

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_train))
        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            bidx = indices[i:i + batch_size]
            X_batch = X_train[bidx] + torch.randn_like(X_train[bidx]) * 0.02
            logits = model(X_batch.float())
            loss = criterion(logits, y_train[bidx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(bidx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(X_val.float())
            v_loss = criterion(v_logits, y_val).item()
            v_probs = torch.sigmoid(v_logits)
            v_preds = (v_probs > 0.5).float()
            v_acc = (v_preds == y_val).float().mean().item()

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_acc = v_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_state, best_acc, best_val_loss


# ==================== 三支决策评估（兼容原版接口） ====================
def evaluate_three_way(classifier, features_model, X, y_main, verbose=False):
    """三支决策端到端评估，features_model 须返回 (N,11) 概率"""
    classifier.eval()
    features_model.eval()
    with torch.no_grad():
        logits = classifier(X.float())
        scaled = torch.softmax(logits / CLASSIFIER_TEMPERATURE, dim=1)
        confidences, preds = scaled.max(dim=1)
        feat_probs = features_model(X.float(), inference=True)

    normal_total = (y_main >= 0).sum().item()
    anomaly_total = (y_main == -1).sum().item()
    rule1 = rule2 = rule3 = rule4 = 0
    review_count = miss_count = 0
    review_normal = 0

    for i in range(len(X)):
        pred_label = '晨读' if preds[i] == 0 else '晨跑'
        confidence = confidences[i].item()
        true_label = y_main[i].item()

        class_feats = CLASS_FEATURES[pred_label]
        matched = 0
        for f, fidx in FEATURE_INDEX.items():
            thresh = FEATURE_THRESHOLD_READ if fidx < 4 else FEATURE_THRESHOLD_RUN
            if feat_probs[i, fidx].item() > thresh and f in class_feats:
                matched += 1

        r1 = pred_label == '晨跑' and confidence >= ALPHA_AUTO_PASS and matched >= RUN_FEATURE_THRESH
        r2 = pred_label == '晨读' and confidence >= ALPHA_AUTO_PASS and matched >= READ_FEATURE_THRESH
        r3 = matched < MIN_FEATURES
        r4 = confidence < ALPHA_REVIEW

        if r1:
            rule1 += 1; decision = 0
        elif r2:
            rule2 += 1; decision = 0
        elif r3:
            rule3 += 1; decision = 1
        elif r4:
            rule4 += 1; decision = 1
        else:
            decision = 0

        if true_label == -1 and decision == 0:
            miss_count += 1
        if decision == 1:
            review_count += 1
            if true_label != -1:
                review_normal += 1

    miss_rate = miss_count / anomaly_total * 100 if anomaly_total > 0 else 0
    review_rate = review_count / len(X) * 100
    pass_rate = (normal_total - review_normal) / len(X) * 100 if len(X) > 0 else 0

    if verbose:
        print(f"    漏检率: {miss_rate:.2f}% ({miss_count}/{anomaly_total})")
        print(f"    审核率: {review_rate:.2f}% ({review_count}/{len(X)})")
        print(f"    自动通过率: {pass_rate:.2f}% ({normal_total - review_normal}/{len(X)})")
        print(f"    规则: R1={rule1} R2={rule2} R3={rule3} R4={rule4}")

    return {'miss_rate': miss_rate, 'review_rate': review_rate,
            'pass_rate': pass_rate, 'miss_count': miss_count,
            'rule1': rule1, 'rule2': rule2, 'rule3': rule3, 'rule4': rule4}


def feature_accuracy(features_model, X, yf, verbose=False):
    """各特征准确率"""
    with torch.no_grad():
        probs = features_model(X.float(), inference=True)
    preds = (probs > 0.5).float()
    results = {}
    for name, idx in FEATURE_INDEX.items():
        acc = (preds[:, idx] == yf[:, idx]).float().mean().item()
        results[name] = acc
    if verbose:
        for name, acc in results.items():
            print(f"    {name}: {acc * 100:.2f}%")
    return results


# ==================== 主流程 ====================
def main():
    print("=" * 60)
    print("训练 11 个独立 MLP 特征预测器")
    print("=" * 60)

    (X_train, y_main_train, yf_train), (X_val, y_main_val, yf_val), \
        (X_test, y_main_test, yf_test) = load_data()
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 训练主分类器（复用 train_mlp.py 逻辑）
    from train_mlp import train_classifier
    classifier = train_classifier(X_train, y_main_train, X_val, y_main_val)
    torch.save(classifier.state_dict(), DATA_DIR / "mlp_classifier.pt")
    print("[OK] 主分类器已保存: data/mlp_classifier.pt")

    # 训练 11 个特征模型
    model = MLPEnsembleFeatures(num_features=11, hidden_dim=256)
    state_dict = {}
    per_feature_info = []

    for idx, name in enumerate(FEATURE_NAMES):
        alpha = feature_alpha(yf_train[:, idx])
        print(f"\n--- 特征[{idx}] {name} (alpha={alpha:.3f}, "
              f"正类={yf_train[:, idx].sum().item()}/{len(yf_train)}) ---")
        state, acc, vloss = train_one_feature(idx, name, X_train, yf_train,
                                              X_val, yf_val, alpha)
        state_dict[f'head_{idx}'] = state
        per_feature_info.append({
            'feature': name, 'idx': idx, 'alpha': alpha,
            'val_acc': acc, 'val_loss': vloss
        })
        print(f"  完成: 验证准确率={acc * 100:.2f}%, 验证损失={vloss:.4f}")

    # 保存模型（state_dict + 元信息）
    # 统一保存为 heads.{idx}.{param} 格式，与 ModuleList 加载兼容
    save_data = {
        'state_dict': state_dict,
        'temperatures': model.temperatures,
        'info': per_feature_info,
        'model': '11-independent-MLP',
    }
    torch.save(save_data, OUTPUT_PATH)
    print(f"\n[OK] 11-MLP 集成已保存: {OUTPUT_PATH}")

    # 加载并评估
    def load_ensemble():
        m = MLPEnsembleFeatures(num_features=11, hidden_dim=256)
        sd = torch.load(OUTPUT_PATH, map_location='cpu')
        full = {}
        for k, v in sd['state_dict'].items():
            if k.startswith('heads.'):
                full[k] = v
            elif k.startswith('head_'):
                idx = k.split('_')[1]
                for pk, pv in v.items():
                    full[f'heads.{idx}.{pk}'] = pv
        m.load_state_dict(full)
        m.temperatures = sd['temperatures']
        return m

    ensemble = load_ensemble()

    print("\n" + "=" * 60)
    print("【验证集评估 - 11 独立 MLP】")
    print("=" * 60)
    evaluate_three_way(classifier, ensemble, X_val, y_main_val, verbose=True)
    print("  各特征准确率:")
    feature_accuracy(ensemble, X_val, yf_val, verbose=True)

    print("\n" + "=" * 60)
    print("【测试集评估 - 11 独立 MLP】")
    print("=" * 60)
    evaluate_three_way(classifier, ensemble, X_test, y_main_test, verbose=True)
    print("  各特征准确率:")
    feature_accuracy(ensemble, X_test, yf_test, verbose=True)


if __name__ == '__main__':
    main()
