# -*- coding: utf-8 -*-
"""Phase 2 阈值重调：SigLIP-NaFlex 特征空间的最优阈值搜索 + 与 CLIP 基线公平对比

两边都在验证集上做坐标下降搜索（零漏检优先、最小审核率），
然后在测试集上对比——避免"用默认阈值冤枉新特征空间"。

用法（checkin_detection 环境）:
    python tune_siglip_thresholds.py
输出: data/tuned_thresholds_siglip.json（不覆盖旧 tuned_thresholds.json）
"""
import json
from pathlib import Path

import torch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

from train_features_11mlp import FEATURE_INDEX, FEATURE_NAMES, CLASS_FEATURES
from src.models.mlp import MLPClassifier
from src.models.mlp_features_optimized import MLPFeaturesOptimized
from src.config import (
    CLASSIFIER_TEMPERATURE, ALPHA_AUTO_PASS, ALPHA_REVIEW,
    MIN_FEATURES, RUN_FEATURE_THRESH, READ_FEATURE_THRESH,
    FEATURE_THRESHOLD_READ, FEATURE_THRESHOLD_RUN,
)
from train_mlp import prepare_data


def load_split(features_csv):
    df = pd.read_csv(DATA_DIR / features_csv)
    filenames = df['filename'].tolist()
    features = torch.tensor(df.drop('filename', axis=1).values, dtype=torch.float32)
    labels = json.load(open(DATA_DIR / 'labels.json', encoding='utf-8')).get('labels')
    split_config = json.load(open(DATA_DIR / 'split_config.json', encoding='utf-8'))
    return prepare_data(features, filenames, labels, split_config)


def load_models(suffix):
    import os
    cls = MLPClassifier(input_dim=768 if suffix else 512, hidden_dim=256, output_dim=2, dropout=0.3)
    cls.load_state_dict(torch.load(DATA_DIR / f'mlp_classifier{suffix}.pt', map_location='cpu'))
    cls.eval()
    feat = MLPFeaturesOptimized(input_dim=768 if suffix else 512, hidden_dim=512,
                                output_dim=11, dropout=0.25)
    feat.load_state_dict(torch.load(DATA_DIR / f'mlp_features_optimized{suffix}.pt', map_location='cpu'))
    feat.eval()
    return cls, feat


def decide(logits, feat_probs, y_main, thresholds):
    scaled = torch.softmax(logits / CLASSIFIER_TEMPERATURE, dim=1)
    confidences, preds = scaled.max(dim=1)
    normal_total = (y_main >= 0).sum().item()
    anomaly_total = (y_main == -1).sum().item()
    review_count = miss_count = review_normal = 0
    for i in range(len(feat_probs)):
        pred_label = '晨读' if preds[i] == 0 else '晨跑'
        confidence = confidences[i].item()
        true_label = y_main[i].item()
        matched = 0
        for f, fidx in FEATURE_INDEX.items():
            if feat_probs[i, fidx].item() > thresholds[fidx] and f in CLASS_FEATURES[pred_label]:
                matched += 1
        r1 = pred_label == '晨跑' and confidence >= ALPHA_AUTO_PASS and matched >= RUN_FEATURE_THRESH
        r2 = pred_label == '晨读' and confidence >= ALPHA_AUTO_PASS and matched >= READ_FEATURE_THRESH
        r3 = matched < MIN_FEATURES
        r4 = confidence < ALPHA_REVIEW
        decision = 0 if (r1 or r2) else (1 if (r3 or r4) else 0)
        if true_label == -1 and decision == 0:
            miss_count += 1
        if decision == 1:
            review_count += 1
            if true_label != -1:
                review_normal += 1
    return {'miss_rate': (miss_count / anomaly_total * 100) if anomaly_total > 0 else 0.0,
            'review_rate': review_count / len(feat_probs) * 100,
            'pass_rate': (normal_total - review_normal) / len(feat_probs) * 100 if len(feat_probs) else 0.0,
            'miss_count': miss_count}


def coordinate_search(logits, feat_probs, y_main, thresholds0, rounds=3):
    grid = [round(0.30 + 0.02 * i, 2) for i in range(31)]
    thresh = list(thresholds0)
    for r in range(rounds):
        for idx in range(11):
            best_t, best_s = thresh[idx], None
            for t in grid:
                thresh[idx] = t
                m = decide(logits, feat_probs, y_main, thresh)
                score = m['review_rate'] + 100 * m['miss_rate']
                if best_s is None or score < best_s:
                    best_s, best_t = score, t
            thresh[idx] = best_t
        m = decide(logits, feat_probs, y_main, thresh)
        print(f"    第{r+1}轮: score={m['review_rate'] + 100*m['miss_rate']:.2f} "
              f"review={m['review_rate']:.2f}% miss={m['miss_rate']:.2f}%")
    return thresh


def evaluate(name, cls, feat, split, thresholds, tag):
    X, y_main, _ = split
    with torch.no_grad():
        logits = cls(X.float())
        probs = feat(X.float(), inference=True)
    m = decide(logits, probs, y_main, thresholds)
    print(f"  [{tag}] {name}: review={m['review_rate']:.2f}% pass={m['pass_rate']:.2f}% "
          f"miss={m['miss_rate']:.2f}% ({m['miss_count']}例)")
    return m


def main():
    print("=" * 70)
    print("SigLIP-NaFlex vs CLIP 基线：各自最优阈值的公平对比")
    print("=" * 70)

    init_thresh = [FEATURE_THRESHOLD_READ if i < 4 else FEATURE_THRESHOLD_RUN for i in range(11)]

    for tag, csv, suffix in [("CLIP基线", "clip_features_cpu.csv", ""),
                             ("SigLIP-NaFlex", "siglip_features.csv", "_siglip")]:
        print(f"\n========== {tag} ==========")
        ds = load_split(csv)
        cls, feat = load_models(suffix)
        (X_tr, ym_tr, _), (X_v, ym_v, _), (X_t, ym_t, _) = ds['train'], ds['val'], ds['test']
        with torch.no_grad():
            logits_v, prob_v = cls(X_v.float()), feat(X_v.float(), inference=True)
        print("  默认阈值:")
        evaluate(tag, cls, feat, ds['val'], init_thresh, "val默认")
        print("  阈值搜索(val):")
        thresh = coordinate_search(logits_v, prob_v, ym_v, init_thresh)
        print("  最优阈值:", {FEATURE_NAMES[i]: thresh[i] for i in range(11)})
        evaluate(tag, cls, feat, ds['test'], thresh, "test最优")
        evaluate(tag, cls, feat, ds['test'], init_thresh, "test默认")
        out_name = 'tuned_thresholds_siglip.json' if suffix else 'tuned_thresholds.json'
        with open(DATA_DIR / out_name, 'w', encoding='utf-8') as f:
            json.dump({FEATURE_NAMES[i]: thresh[i] for i in range(11)}, f, ensure_ascii=False, indent=2)
        print(f"  已保存: data/{out_name}")

if __name__ == '__main__':
    main()
