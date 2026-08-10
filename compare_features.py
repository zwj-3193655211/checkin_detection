"""
对比评估：原版共享 MLP 特征预测器 vs 11 独立 MLP 特征预测器

用法（checkin_detection 环境）:
    python compare_features.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

from src.models.mlp import MLPClassifier
from src.models.mlp_features_optimized import MLPFeaturesOptimized
from train_features_11mlp import (
    MLPEnsembleFeatures, load_data, evaluate_three_way,
    feature_accuracy, FEATURE_INDEX, FEATURE_NAMES,
    CLASS_FEATURES,
)
from src.config import CLASSIFIER_TEMPERATURE


def load_classifier(path=DATA_DIR / "mlp_classifier.pt"):
    model = MLPClassifier()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def load_baseline_features(path=DATA_DIR / "mlp_features_optimized.pt"):
    model = MLPFeaturesOptimized()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def load_ensemble(path=DATA_DIR / "mlp_features_11x_v2.pt"):
    sd = torch.load(path, map_location='cpu')
    model = MLPEnsembleFeatures(num_features=11, hidden_dim=256)
    full = {}
    for k, v in sd['state_dict'].items():
        if k.startswith('heads.'):
            full[k] = v
        elif k.startswith('head_'):
            idx = k.split('_')[1]
            for pk, pv in v.items():
                full[f'heads.{idx}.{pk}'] = pv
    model.load_state_dict(full)
    model.temperatures = sd['temperatures']
    model.eval()
    return model


def main():
    print("=" * 70)
    print("对比评估: 共享MLP特征预测器 vs 11独立MLP特征预测器")
    print("=" * 70)

    (X_train, _, yf_train), (X_val, y_main_val, yf_val), \
        (X_test, y_main_test, yf_test) = load_data()

    classifier = load_classifier()
    baseline = load_baseline_features()
    ensemble = load_ensemble()

    print(f"\n模型文件: baseline={DATA_DIR / 'mlp_features_optimized.pt'}")
    print(f"          11-MLP = {DATA_DIR / 'mlp_features_11x_v2.pt'}")

    for name, X, y_main, yf in [("验证集", X_val, y_main_val, yf_val),
                                ("测试集", X_test, y_main_test, yf_test)]:
        print("\n" + "=" * 70)
        print(f"【{name}】三支决策指标")
        print("=" * 70)
        print("\n  --- 原版共享MLP ---")
        m1 = evaluate_three_way(classifier, baseline, X, y_main, verbose=True)
        print("\n  --- 11独立MLP ---")
        m2 = evaluate_three_way(classifier, ensemble, X, y_main, verbose=True)

        print(f"\n  {'指标':<12} {'共享MLP':<12} {'11独立MLP':<12} 对比")
        for k in ['miss_rate', 'review_rate', 'pass_rate']:
            d = m1[k] - m2[k]
            mark = "▲更优" if d > 0.01 else ("▼更差" if d < -0.01 else "持平")
            print(f"  {k:<12} {m1[k]:<12.2f} {m2[k]:<12.2f} {mark}")

        print(f"\n  --- {name}各特征准确率 ---")
        acc1 = feature_accuracy(baseline, X, yf)
        acc2 = feature_accuracy(ensemble, X, yf)
        print(f"  {'特征':<8} {'共享MLP':<10} {'11独立MLP':<10} 对比")
        better = 0
        for feat in FEATURE_NAMES:
            a1, a2 = acc1[feat] * 100, acc2[feat] * 100
            mark = "▲" if a2 > a1 + 0.5 else ("▼" if a2 < a1 - 0.5 else "=")
            if a2 > a1 + 0.5:
                better += 1
            print(f"  {feat:<8} {a1:<10.2f} {a2:<10.2f} {mark}")
        print(f"\n  11独立MLP 优势特征数: {better}/11")


if __name__ == '__main__':
    main()
