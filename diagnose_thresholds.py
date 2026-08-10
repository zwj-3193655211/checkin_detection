"""
诊断：阈值优化后三支决策实际靠哪些规则在起作用？
检查匹配特征数分布 + 各规则触发 + 阈值0.3是否导致规则退化
"""
import json
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

from train_features_11mlp import load_data, FEATURE_INDEX, FEATURE_NAMES, CLASS_FEATURES
from src.models.mlp import MLPClassifier
from src.models.mlp_features_optimized import MLPFeaturesOptimized
from src.config import (
    CLASSIFIER_TEMPERATURE, ALPHA_AUTO_PASS, ALPHA_REVIEW,
    MIN_FEATURES, RUN_FEATURE_THRESH, READ_FEATURE_THRESH,
)


def load_classifier(path=DATA_DIR / "mlp_classifier.pt"):
    m = MLPClassifier()
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m


def load_baseline(path=DATA_DIR / "mlp_features_optimized.pt"):
    m = MLPFeaturesOptimized()
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m


def diagnose(classifier, features_model, X, y_main, thresholds, tag):
    """详细诊断三支决策，输出规则触发 + 匹配特征数分布 + 异常样本详情"""
    with torch.no_grad():
        logits = classifier(X.float())
        scaled = torch.softmax(logits / CLASSIFIER_TEMPERATURE, dim=1)
        confidences, preds = scaled.max(dim=1)
        feat_probs = features_model(X.float(), inference=True)

    rule1 = rule2 = rule3 = rule4 = 0
    review = 0
    matched_hist = {}  # 匹配特征数 -> 计数
    anomaly_details = []

    for i in range(len(X)):
        pred_label = '晨读' if preds[i] == 0 else '晨跑'
        conf = confidences[i].item()
        true_label = y_main[i].item()

        matched = 0
        for f, fidx in FEATURE_INDEX.items():
            if feat_probs[i, fidx].item() > thresholds[fidx] and f in CLASS_FEATURES[pred_label]:
                matched += 1
        matched_hist[matched] = matched_hist.get(matched, 0) + 1

        r1 = pred_label == '晨跑' and conf >= ALPHA_AUTO_PASS and matched >= RUN_FEATURE_THRESH
        r2 = pred_label == '晨读' and conf >= ALPHA_AUTO_PASS and matched >= READ_FEATURE_THRESH
        r3 = matched < MIN_FEATURES
        r4 = conf < ALPHA_REVIEW

        decision = 0
        if r1: rule1 += 1
        elif r2: rule2 += 1
        elif r3: rule3 += 1; decision = 1
        elif r4: rule4 += 1; decision = 1

        if true_label == -1:
            anomaly_details.append({
                'conf': round(conf, 3), 'pred': pred_label,
                'matched': matched, 'decision': '审核' if decision else '通过',
            })

    print(f"\n===== {tag} =====")
    print(f"规则触发: R1={rule1} R2={rule2} R3={rule3} R4={rule4}")
    print(f"待审核: {review}, 匹配特征数分布: {dict(sorted(matched_hist.items()))}")
    print(f"异常样本详情: {anomaly_details}")


def main():
    (X_train, _, _), (X_val, y_main_val, _), (X_test, y_main_test, _) = load_data()
    classifier = load_classifier()
    baseline = load_baseline()

    with open(DATA_DIR / 'tuned_thresholds.json', encoding='utf-8') as f:
        tuned = json.load(f)
    names = FEATURE_NAMES
    thresh_b = [tuned['baseline_thresholds'][n] for n in names]
    thresh_e = [tuned['ensemble_thresholds'][n] for n in names]

    # baseline 现配置阈值（0.66/0.60）
    from src.config import FEATURE_THRESHOLD_READ, FEATURE_THRESHOLD_RUN
    init_thresh = [FEATURE_THRESHOLD_READ if i < 4 else FEATURE_THRESHOLD_RUN for i in range(11)]

    print("baseline + 现配置阈值:")
    diagnose(classifier, baseline, X_val, y_main_val, init_thresh, "验证集")
    diagnose(classifier, baseline, X_test, y_main_test, init_thresh, "测试集")

    print("\nbaseline + tuned阈值:")
    diagnose(classifier, baseline, X_val, y_main_val, thresh_b, "验证集")
    diagnose(classifier, baseline, X_test, y_main_test, thresh_b, "测试集")


if __name__ == '__main__':
    main()
