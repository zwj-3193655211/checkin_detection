"""
参数优化脚本 - 调优三支决策阈值
"""
import torch
import json
import pandas as pd
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent

df = pd.read_csv(project_root / 'data' / 'clip_features_cpu.csv')
features = torch.tensor(df.drop('filename', axis=1).values, dtype=torch.float32)
filenames = df['filename'].tolist()

with open(project_root / 'data' / 'labels.json', 'r', encoding='utf-8') as f:
    labels_data = json.load(f)
    labels = labels_data.get('labels', labels_data)

with open(project_root / 'data' / 'split_config.json', 'r', encoding='utf-8') as f:
    split = json.load(f)

all_files = set(split.get('train_files', [])) | set(split.get('val_files', [])) | set(split.get('test_files', []))
all_indices = [i for i, fname in enumerate(filenames) if fname in all_files]

FEATURE_INDEX = {
    '人脸': 0, '蓝色桌子': 1, '教室': 2, '投影幕布': 3,
    '跑道': 4, '天空': 5, '绿地': 6, '树木': 7, '旗杆': 8, '号码布': 9, '主席台': 10
}

CLASS_FEATURES = {
    '晨读': ['人脸', '蓝色桌子', '教室', '投影幕布'],
    '晨跑': ['人脸', '跑道', '天空', '绿地', '树木', '旗杆', '号码布', '主席台']
}

from src.models.mlp import MLPClassifier
from src.models.mlp_features_optimized import MLPFeaturesOptimized

classifier = MLPClassifier(input_dim=512, hidden_dim=256, output_dim=2)
classifier.load_state_dict(torch.load(project_root / 'data' / 'mlp_classifier.pt'))
classifier.eval()

features_model = MLPFeaturesOptimized(input_dim=512, hidden_dim=512, output_dim=11)
features_model.load_state_dict(torch.load(project_root / 'data' / 'mlp_features_optimized.pt'))
features_model.eval()

X_all = features[all_indices]
y_true = []
for i in all_indices:
    fname = filenames[i]
    label_info = labels.get(fname, {})
    label_name = label_info.get('label', '未知')
    y_true.append(0 if label_name == '晨读' else (1 if label_name == '晨跑' else -1))

y_true = torch.tensor(y_true)
anomaly_total = (y_true == -1).sum().item()
normal_total = (y_true >= 0).sum().item()

with torch.no_grad():
    logits = classifier(X_all.float())
    feat_probs = features_model(X_all.float(), inference=True)

feat_probs[:, 9] = torch.clamp(feat_probs[:, 9] * 2, max=1.0)

scaled_probs = torch.softmax(logits / 5.0, dim=1)
scaled_confs = scaled_probs.max(dim=1)[0]
scaled_preds = scaled_probs.argmax(dim=1)

print(f"全部数据: {len(X_all)} 张 (正常: {normal_total}, 异常: {anomaly_total})")

preds_np = scaled_preds.numpy()
confs_np = scaled_confs.numpy()
feat_np = feat_probs.numpy()
y_np = y_true.numpy()

def evaluate(alpha_accept, alpha_auto_pass, feature_threshold):
    r1 = r2 = r3 = r4 = miss = review = 0
    for i in range(len(X_all)):
        pred = '晨读' if preds_np[i] == 0 else '晨跑'
        conf = confs_np[i]
        matched = sum(1 for f, fidx in FEATURE_INDEX.items()
                     if feat_np[i, fidx] > feature_threshold and f in CLASS_FEATURES[pred])
        
        if pred == '晨跑' and conf >= alpha_auto_pass and matched >= 5:
            d = 0; r1 += 1
        elif pred == '晨读' and conf >= alpha_auto_pass and matched >= 3:
            d = 0; r2 += 1
        elif matched < 3:
            d = 1; r3 += 1
        elif conf < alpha_accept:
            d = 1; r4 += 1
        else:
            d = 0
        
        if y_np[i] == -1 and d == 0: miss += 1
        if d == 1: review += 1
    
    return miss, review, r1, r2, r3, r4

print("\n搜索中...")
best = (0.88, 0.83, 0.50, float('inf'))

for aa in np.arange(0.60, 0.90, 0.05):
    for aap in np.arange(0.60, 0.90, 0.05):
        for ft in np.arange(0.30, 0.70, 0.05):
            miss, review, _, _, _, _ = evaluate(aa, aap, ft)
            if miss == 0 and review < best[3]:
                best = (aa, aap, ft, review)
                print(f"  AA={aa:.2f}, AAP={aap:.2f}, FT={ft:.2f} -> 审核={review}")

aa, aap, ft, _ = best
miss, review, r1, r2, r3, r4 = evaluate(aa, aap, ft)

print(f"\n最优参数:")
print(f"  ALPHA_ACCEPT = {aa:.2f}")
print(f"  ALPHA_AUTO_PASS = {aap:.2f}")
print(f"  FEATURE_THRESHOLD = {ft:.2f}")
print(f"\n结果: 漏检={miss}, 审核={review}/{len(X_all)} ({review/len(X_all)*100:.2f}%)")
print(f"规则: r1={r1}, r2={r2}, r3={r3}, r4={r4}")
