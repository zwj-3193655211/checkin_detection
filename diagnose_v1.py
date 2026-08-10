"""
诊断 11-MLP v1：低审核率(15.89%)的真相
检查: 特征误报率 / 自动通过样本的特征预测一致性 / 临界异常样本(196)的匹配数
"""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_features_11mlp import (
    MLPEnsembleFeatures, load_data, FEATURE_INDEX, FEATURE_NAMES, CLASS_FEATURES,
)
from src.models.mlp import MLPClassifier
from src.models.mlp_features_optimized import MLPFeaturesOptimized
from src.config import (
    CLASSIFIER_TEMPERATURE, ALPHA_AUTO_PASS, ALPHA_REVIEW,
    MIN_FEATURES, RUN_FEATURE_THRESH, READ_FEATURE_THRESH,
    FEATURE_THRESHOLD_READ, FEATURE_THRESHOLD_RUN,
)


def load_classifier(path=PROJECT_ROOT / "data/mlp_classifier.pt"):
    m = MLPClassifier()
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m


def load_baseline(path=PROJECT_ROOT / "data/mlp_features_optimized.pt"):
    m = MLPFeaturesOptimized()
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m


def load_v1(path=PROJECT_ROOT / "data/mlp_features_11x.pt"):
    sd = torch.load(path, map_location='cpu')
    m = MLPEnsembleFeatures(num_features=11, hidden_dim=128)  # v1 用 128 隐层
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
    m.eval()
    return m


def matched_features(feat_probs, pred_label, thresholds):
    matched = []
    for f, fidx in FEATURE_INDEX.items():
        if feat_probs[fidx].item() > thresholds[fidx] and f in CLASS_FEATURES[pred_label]:
            matched.append(f)
    return matched


def main():
    (_, _, _), (X_val, y_main_val, yf_val), (X_test, y_main_test, yf_test) = load_data()
    clf = load_classifier()
    base = load_baseline()
    v1 = load_v1()

    with torch.no_grad():
        logits = clf(X_test.float())
        scaled = torch.softmax(logits / CLASSIFIER_TEMPERATURE, dim=1)
        confs, preds = scaled.max(dim=1)
        pb = base(X_test.float(), inference=True)
        pv = v1(X_test.float(), inference=True)

    thresh = [FEATURE_THRESHOLD_READ if i < 4 else FEATURE_THRESHOLD_RUN for i in range(11)]

    print("=" * 70)
    print("1) 各特征误报率(FP)与漏报率(FN)对比（测试集，阈值=判定阈值）")
    print("=" * 70)
    print(f"{'特征':<8} {'baseline FP':<12} {'v1 FP':<12} {'baseline FN':<12} {'v1 FN':<12}")
    for name, idx in FEATURE_INDEX.items():
        tb = thresh[idx]
        y = yf_test[:, idx]
        fp_b = ((pb[:, idx] > tb) & (y == 0)).float().mean().item() * 100
        fp_v = ((pv[:, idx] > tb) & (y == 0)).float().mean().item() * 100
        fn_b = ((pb[:, idx] <= tb) & (y == 1)).float().mean().item() * 100
        fn_v = ((pv[:, idx] <= tb) & (y == 1)).float().mean().item() * 100
        print(f"{name:<8} {fp_b:<12.2f} {fp_v:<12.2f} {fn_b:<12.2f} {fn_v:<12.2f}")

    print("\n" + "=" * 70)
    print("2) 自动通过样本中『特征预测与真实标签不一致』的比例")
    print("   即：匹配特征数达标但该样本真实场景特征很少（说明通过依赖错误特征预测）")
    print("=" * 70)
    for tag, probs in [("baseline", pb), ("11-MLP v1", pv)]:
        inconsistent = 0
        auto_pass = 0
        for i in range(len(X_test)):
            if y_main_test[i].item() == -1:
                continue
            pred_label = '晨读' if preds[i] == 0 else '晨跑'
            conf = confs[i].item()
            matched = matched_features(probs[i], pred_label, thresh)
            r1 = pred_label == '晨跑' and conf >= ALPHA_AUTO_PASS and len(matched) >= RUN_FEATURE_THRESH
            r2 = pred_label == '晨读' and conf >= ALPHA_AUTO_PASS and len(matched) >= READ_FEATURE_THRESH
            if r1 or r2:
                auto_pass += 1
                # 该样本真实标注的特征数（仅统计预测标签对应类别的特征）
                true_feats = []
                yf = yf_test[i]
                for f, fidx in FEATURE_INDEX.items():
                    if f in CLASS_FEATURES[pred_label] and yf[fidx].item() == 1:
                        true_feats.append(f)
                if len(true_feats) < (READ_FEATURE_THRESH if pred_label == '晨读' else RUN_FEATURE_THRESH):
                    inconsistent += 1
        print(f"  {tag}: 自动通过 {auto_pass} 个，其中真实特征不足门槛 {inconsistent} 个 "
              f"({inconsistent / auto_pass * 100:.1f}% 依赖错误特征预测)")

    print("\n" + "=" * 70)
    print("3) 临界异常样本(测试集 idx=196) 的匹配特征与决策")
    print("=" * 70)
    for tag, probs in [("baseline", pb), ("11-MLP v1", pv)]:
        i = 196
        pred_label = '晨读' if preds[i] == 0 else '晨跑'
        conf = confs[i].item()
        matched = matched_features(probs[i], pred_label, thresh)
        matched2 = [f for f, fidx in FEATURE_INDEX.items()
                    if probs[i, fidx].item() > thresh[fidx] and f in CLASS_FEATURES[pred_label]]
        mcount = len(matched2)
        r1 = pred_label == '晨跑' and conf >= ALPHA_AUTO_PASS and mcount >= RUN_FEATURE_THRESH
        r2 = pred_label == '晨读' and conf >= ALPHA_AUTO_PASS and mcount >= READ_FEATURE_THRESH
        r3 = mcount < MIN_FEATURES
        r4 = conf < ALPHA_REVIEW
        decision = '审核' if (r3 or r4) else '自动通过'
        print(f"  {tag}: conf={conf:.3f} pred={pred_label} matched={matched} ({mcount}) → {decision}")

    print("\n" + "=" * 70)
    print("4) v1 对全部 24 个异常样本的匹配特征数（val+test）")
    print("=" * 70)
    with torch.no_grad():
        logits_val = clf(X_val.float())
        scaled_v = torch.softmax(logits_val / CLASSIFIER_TEMPERATURE, dim=1)
        confs_v, preds_v = scaled_v.max(dim=1)
        pv_val = v1(X_val.float(), inference=True)
        pb_val = base(X_val.float(), inference=True)

    for tag, plist in [("baseline(测试集)", (pb, X_test, y_main_test, confs, preds)),
                       ("v1(测试集)", (pv, X_test, y_main_test, confs, preds)),
                       ("v1(验证集)", (pv_val, X_val, y_main_val, confs_v, preds_v))]:
        probs, X, y_main, conf, pred = plist
        counts = []
        for i in range(len(X)):
            if y_main[i].item() == -1:
                pred_label = '晨读' if pred[i] == 0 else '晨跑'
                mcount = len(matched_features(probs[i], pred_label, thresh))
                counts.append((round(conf[i].item(), 3), mcount))
        print(f"  {tag}: {sorted(counts)}")


if __name__ == '__main__':
    main()
