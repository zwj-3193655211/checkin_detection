"""
测试检测系统 - 验证data/raw目录的检测效果
"""
import os
import json
from pathlib import Path
import torch
import clip
from PIL import Image
import torch.nn as nn

# MLP模型定义
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.net(x)

# 参数
ALPHA_ACCEPT = 0.9996
ALPHA_REJECT = 0.20
id2label = {0: '晨读', 1: '晨跑', 2: '异常'}

# 加载模型
print("加载CLIP模型...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

print("加载MLP模型...")
mlp = MLP()
mlp.load_state_dict(torch.load('outputs/mlp_model.pt', map_location=device))
mlp.eval()

# 加载标签
print("加载标签...")
with open('data/labels.json', 'r', encoding='utf-8') as f:
    labels_data = json.load(f)
if 'labels' in labels_data:
    labels = labels_data['labels']
else:
    labels = labels_data

class_map = {'晨读': 0, '晨跑': 1, '异常': 2}

# 获取data/raw中的所有图片
data_dir = Path('data/raw')
image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

print(f"\n总图片数: {len(image_files)}")

# 测试所有图片
stats = {
    'total': len(image_files),
    'auto_pass': {'晨读': 0, '晨跑': 0, '异常': 0},
    'review': {'晨读': 0, '晨跑': 0, '异常': 0},
    'reject': {'晨读': 0, '晨跑': 0, '异常': 0},
    'missed_abnormal': 0,
    'wrong_pred': 0,
    'details': []
}

print("\n开始检测...")
for i, fname in enumerate(image_files):
    if (i + 1) % 100 == 0:
        print(f"  进度: {i+1}/{len(image_files)}")
    
    img_path = data_dir / fname
    
    # CLIP特征提取
    img = Image.open(img_path).convert('RGB')
    img_input = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(img_input)
    
    # MLP预测
    with torch.no_grad():
        out = mlp(image_features.float())
        probs = torch.softmax(out, dim=1)
        conf, pred = probs.max(dim=1)
    
    pred_label = id2label[pred.item()]
    confidence = conf.item()
    
    # 三支决策
    if pred_label == '异常':
        decision = 'review'
    elif confidence >= ALPHA_ACCEPT:
        decision = 'auto_pass'
    elif confidence <= ALPHA_REJECT:
        decision = 'reject'
    else:
        decision = 'review'
    
    # 统计
    stats[decision][pred_label] += 1
    
    # 检查是否为异常漏检
    true_label = labels.get(fname, {}).get('label', '未知')
    if true_label == '异常' and decision != 'review':
        stats['missed_abnormal'] += 1
    
    # 检查预测是否正确
    if true_label != pred_label:
        stats['wrong_pred'] += 1
    
    stats['details'].append({
        'file': fname,
        'true_label': true_label,
        'pred_label': pred_label,
        'confidence': confidence,
        'decision': decision
    })

# 输出结果
print("\n" + "=" * 60)
print("检测结果统计")
print("=" * 60)

total_auto = sum(stats['auto_pass'].values())
total_review = sum(stats['review'].values())
total_reject = sum(stats['reject'].values())

print(f"\n📊 总计: {stats['total']} 张图片")
print(f"\n✅ 自动通过: {total_auto} ({total_auto/stats['total']*100:.2f}%)")
print(f"   - 晨读: {stats['auto_pass']['晨读']}")
print(f"   - 晨跑: {stats['auto_pass']['晨跑']}")
print(f"   - 异常: {stats['auto_pass']['异常']}")

print(f"\n🔍 人工审核: {total_review} ({total_review/stats['total']*100:.2f}%)")
print(f"   - 晨读: {stats['review']['晨读']}")
print(f"   - 晨跑: {stats['review']['晨跑']}")
print(f"   - 异常: {stats['review']['异常']}")

print(f"\n❌ 自动拒绝: {total_reject} ({total_reject/stats['total']*100:.2f}%)")
print(f"   - 晨读: {stats['reject']['晨读']}")
print(f"   - 晨跑: {stats['reject']['晨跑']}")
print(f"   - 异常: {stats['reject']['异常']}")

# 统计有标签的样本
labeled_stats = {
    'total': 0,
    'correct': 0,
    'abnormal_total': 0,
    'abnormal_missed': 0
}

for detail in stats['details']:
    if detail['true_label'] in class_map:
        labeled_stats['total'] += 1
        if detail['true_label'] == detail['pred_label']:
            labeled_stats['correct'] += 1
        if detail['true_label'] == '异常':
            labeled_stats['abnormal_total'] += 1
            if detail['decision'] != 'review':
                labeled_stats['abnormal_missed'] += 1

print(f"\n{'=' * 60}")
print("有标签样本统计")
print("=" * 60)
print(f"有标签样本: {labeled_stats['total']}")
print(f"正确预测: {labeled_stats['correct']} ({labeled_stats['correct']/labeled_stats['total']*100:.2f}%)")
print(f"\n⚠️ 异常样本: {labeled_stats['abnormal_total']}")
print(f"   - 漏检数量: {labeled_stats['abnormal_missed']}")
print(f"   - 漏检率: {labeled_stats['abnormal_missed']/labeled_stats['abnormal_total']*100:.2f}%")

print(f"\n{'=' * 60}")
print("最终结果")
print("=" * 60)
print(f"✅ 漏检率: {labeled_stats['abnormal_missed']/labeled_stats['abnormal_total']*100:.2f}%")
print(f"✅ 人工审核率: {total_review/stats['total']*100:.2f}%")
print(f"✅ 自动通过率: {total_auto/stats['total']*100:.2f}%")
