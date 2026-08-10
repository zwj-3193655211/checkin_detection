"""
每特征独立阈值搜索 + 三支决策对比

核心思想: 11独立MLP 的优势在于每个特征可独立设置判定阈值
（baseline 只能统一 晨读0.66/晨跑0.60）。

搜索算法: 坐标下降（每轮对11个特征依次在 [0.30, 0.90] 网格搜索最优阈值）
目标函数: review_rate + 100 * miss_rate（零漏检优先，然后最小审核率）

用法（checkin_detection 环境）:
    python tune_feature_thresholds.py
"""
import json
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

from train_features_11mlp import (
    MLPEnsembleFeatures, load_data, FEATURE_INDEX, FEATURE_NAMES, CLASS_FEATURES,
)
from src.models.mlp import MLPClassifier
from src.models.mlp_features_optimized import MLPFeaturesOptimized
from src.config import (
    CLASSIFIER_TEMPERATURE, ALPHA_AUTO_PASS, ALPHA_REVIEW,
    MIN_FEATURES, RUN_FEATURE_THRESH, READ_FEATURE_THRESH,
)


def load_classifier(path=DATA_DIR / "mlp_classifier.pt"):
    model = MLPClassifier()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def load_baseline(path=DATA_DIR / "mlp_features_optimized.pt"):
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


def decide(logits, feat_probs, y_main, thresholds):
    """基于主分类器logits + 特征概率 + 每特征阈值的三支决策"""
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

        decision = 0
        if r1 or r2:
            decision = 0
        elif r3 or r4:
            decision = 1

        if true_label == -1 and decision == 0:
            miss_count += 1
        if decision == 1:
            review_count += 1
            if true_label != -1:
                review_normal += 1

    miss_rate = miss_count / anomaly_total * 100 if anomaly_total > 0 else 0
    review_rate = review_count / len(feat_probs) * 100
    pass_rate = (normal_total - review_normal) / len(feat_probs) * 100 if len(feat_probs) > 0 else 0
    return {'miss_rate': miss_rate, 'review_rate': review_rate,
            'pass_rate': pass_rate, 'miss_count': miss_count}


def coordinate_search(logits, feat_probs, y_main, thresholds0, rounds=3):
    """坐标下降搜索每特征最优阈值"""
    grid = [round(0.30 + 0.02 * i, 2) for i in range(31)]  # 0.30~0.90
    thresh = list(thresholds0)

    for r in range(rounds):
        improved = False
        for idx in range(11):
            old = thresh[idx]
            best_t, best_s = old, None
            for t in grid:
                thresh[idx] = t
                m = decide(logits, feat_probs, y_main, thresh)
                score = m['review_rate'] + 100 * m['miss_rate']
                if best_s is None or score < best_s:
                    best_s, best_t = score, t
            thresh[idx] = best_t
            if best_t != old:
                improved = True
        m = decide(logits, feat_probs, y_main, thresh)
        print(f"  第{r + 1}轮: score={m['review_rate'] + 100 * m['miss_rate']:.2f} "
              f"review={m['review_rate']:.2f}% miss={m['miss_rate']:.2f}%")
        if not improved:
            break
    return thresh


def main():
    print("=" * 70)
    print("每特征独立阈值搜索 + 对比评估")
    print("=" * 70)

    (X_train, _, yf_train), (X_val, y_main_val, yf_val), \
        (X_test, y_main_test, yf_test) = load_data()

    classifier = load_classifier()
    baseline = load_baseline()
    ensemble = load_ensemble()

    # 预计算
    print("\n预计算概率...")
    with torch.no_grad():
        prob_b_val = baseline(X_val.float(), inference=True)
        prob_e_val = ensemble(X_val.float(), inference=True)
        logits_val = classifier(X_val.float())
        prob_b_test = baseline(X_test.float(), inference=True)
        prob_e_test = ensemble(X_test.float(), inference=True)
        logits_test = classifier(X_test.float())

    from src.config import FEATURE_THRESHOLD_READ, FEATURE_THRESHOLD_RUN
    init_thresh = [FEATURE_THRESHOLD_READ if i < 4 else FEATURE_THRESHOLD_RUN for i in range(11)]
    print(f"\n初始阈值(现配置): {init_thresh}")

    m0b = decide(logits_val, prob_b_val, y_main_val, init_thresh)
    m0e = decide(logits_val, prob_e_val, y_main_val, init_thresh)
    print(f"\n初始三支决策:")
    print(f"  baseline: review={m0b['review_rate']:.2f}% miss={m0b['miss_rate']:.2f}%")
    print(f"  11-MLP  : review={m0e['review_rate']:.2f}% miss={m0e['miss_rate']:.2f}%")

    print("\n--- 搜索 baseline 阈值 ---")
    thresh_b = coordinate_search(logits_val, prob_b_val, y_main_val, init_thresh)

    print("\n--- 搜索 11-MLP 阈值 ---")
    thresh_e = coordinate_search(logits_val, prob_e_val, y_main_val, init_thresh)

    print("\n最优阈值:")
    print("  baseline:", {FEATURE_NAMES[i]: thresh_b[i] for i in range(11)})
    print("  11-MLP  :", {FEATURE_NAMES[i]: thresh_e[i] for i in range(11)})

    print("\n" + "=" * 70)
    print("【最终对比（各自最优阈值）】")
    print("=" * 70)
    results = {}
    for name, logits_t, pb, pe, y_main in [
        ("验证集", logits_val, prob_b_val, prob_e_val, y_main_val),
        ("测试集", logits_test, prob_b_test, prob_e_test, y_main_test),
    ]:
        print(f"\n  --- {name} ---")
        mb = decide(logits_t, pb, y_main, thresh_b)
        me = decide(logits_t, pe, y_main, thresh_e)
        print(f"  baseline: review={mb['review_rate']:.2f}% pass={mb['pass_rate']:.2f}% miss={mb['miss_rate']:.2f}%")
        print(f"  11-MLP  : review={me['review_rate']:.2f}% pass={me['pass_rate']:.2f}% miss={me['miss_rate']:.2f}%")
        diff = mb['review_rate'] - me['review_rate']
        verdict = '11-MLP更优' if diff > 0.5 else ('baseline更优' if diff < -0.5 else '持平')
        print(f"  审核率差异: {diff:.2f}pp ({verdict})")
        results[name] = {'baseline': mb, 'ensemble': me}

    # 保存
    out = {
        'baseline_thresholds': {FEATURE_NAMES[i]: thresh_b[i] for i in range(11)},
        'ensemble_thresholds': {FEATURE_NAMES[i]: thresh_e[i] for i in range(11)},
        'results': {k: {kk: vv for kk, vv in v.items()}
                    for k, v in results.items()},
    }
    with open(DATA_DIR / 'tuned_thresholds.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n阈值与结果已保存: data/tuned_thresholds.json")


if __name__ == '__main__':
    main()
