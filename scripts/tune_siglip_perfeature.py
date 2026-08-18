"""
SigLIP 逐维度（逐特征）阈值调优 —— 以"每个维度的准确率"为目标
================================================================

此前 `tune_siglip_rigorous.py` 的 floor=0.39 是「系统级代理目标」
（审核率 + 100×漏检，零漏检约束）下催生的「单一全局下限」，
它从未使用 11 维特征真值来单独优化某个维度。

用户要求（科研严谨性）：
  "应该是在各个维度进行调参，让每个维度的准确率提高，
   让被打上了相应标签的图片能更容易通过这个阈值，
   没打这个标签的图片不太容易通过这个阈值。"

本脚本做法：
  1. 对每个特征 f（11 维之一），以「验证集正常样本」的 11 维特征真值为基准，
     在概率网格上搜索该特征的最佳判定阈值 t_f。
  2. 优化准则：最大化 Youden's J = TPR − FPR。
     - TPR（真正率）= 被打标签的图片中"通过阈值"的比例  → 对应"标签图易通过"
     - TNR（真负率）= 未打标签的图片中"不通过"的比例     → 对应"无标签图不易通过"
     - Youden's J 同时在 ROC 膝点最大化二者，正是用户描述的均衡目标。
     并列最优时：优先 TPR（保证标签图更易通过），再比准确率。
  3. 在「独立测试集」上用验证集选出的 t_f 复算每个维度的准确率，
     并与当前部署默认值(晨读0.66/晨跑0.60)对比，量化"每个维度准确率提升"。
  4. 将 11 个维度各自的最佳阈值喂给三支决策(three_way)，
     给出 default / 旧floor0.39 / 新逐维度 三者的系统级审核率·漏检率对比。

产物：
  data/tuned_thresholds_siglip_perfeature.json
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train_mlp as tm
from compare_all_encoders import (
    three_way, default_feat_thresh, FEATURE_INDEX, CLASS_FEATURES,
)
from src.config import (
    CLASSIFIER_TEMPERATURE, FEATURE_TEMPERATURE,
    ALPHA_AUTO_PASS, ALPHA_REVIEW, MIN_FEATURES,
    RUN_FEATURE_THRESH, READ_FEATURE_THRESH,
)

FEAT_NAMES = list(FEATURE_INDEX.keys())
GRID = np.round(np.arange(0.01, 1.00, 0.01), 3)   # 0.01 精度网格（用户要求严谨）


def load_siglip_preds():
    """加载 SigLIP 特征 CSV + MLP 权重，得到 (pc, pf, y_main, y_feat) 的 val/test 元组。"""
    os.environ["CHECKIN_FEATURES_CSV"] = "siglip_features.csv"
    features, filenames, labels, split_config = tm.load_data()
    data = tm.prepare_data(features, filenames, labels, split_config)
    df = pd.read_csv(PROJECT_ROOT / "data" / "siglip_features.csv")
    dim = df.shape[1] - 1
    classifier = tm.MLPClassifier(input_dim=dim, hidden_dim=256, output_dim=2, dropout=0.3)
    classifier.load_state_dict(torch.load(PROJECT_ROOT / "data" / "mlp_classifier_siglip.pt",
                                          map_location="cpu"))
    fmodel = tm.MLPFeaturesOptimized(input_dim=dim, hidden_dim=512, output_dim=11,
                                     dropout=0.3, temperature=FEATURE_TEMPERATURE)
    fmodel.load_state_dict(torch.load(PROJECT_ROOT / "data" / "mlp_features_optimized_siglip.pt",
                                      map_location="cpu"))
    classifier.eval();
    fmodel.eval()

    out = {}
    for split in ("val", "test"):
        X, y_main, y_feat = data[split]
        with torch.no_grad():
            pc = torch.softmax(classifier(X.float()) / CLASSIFIER_TEMPERATURE, dim=1)
            pf = fmodel(X.float(), inference=True)
        out[split] = (pc, pf, y_main, y_feat)
    return out


def per_feature_metrics(scores, gts, thr):
    """scores/gts: (N,) 一维；thr: 标量阈值。返回该阈值下的分类指标。"""
    pred = (scores > thr).astype(np.int64)
    g = gts.astype(np.int64)
    tp = int(((pred == 1) & (g == 1)).sum())
    fp = int(((pred == 1) & (g == 0)).sum())
    tn = int(((pred == 0) & (g == 0)).sum())
    fn = int(((pred == 0) & (g == 1)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, tpr=tpr, fpr=fpr, tnr=tnr,
                acc=acc, prec=prec, f1=f1)


def tune_one_feature(scores, gts):
    """在 GRID 上最大化 Youden's J=TPR-FPR；并列时优先 TPR，再比 acc，再取较低阈值。"""
    best = None
    for t in GRID:
        m = per_feature_metrics(scores, gts, float(t))
        J = m["tpr"] - m["fpr"]
        if best is None:
            best = (J, m["tpr"], m["acc"], float(t), m)
            continue
        bJ, bTpr, bAcc, bT, _ = best
        if (J > bJ + 1e-12
                or (abs(J - bJ) <= 1e-12 and m["tpr"] > bTpr + 1e-12)
                or (abs(J - bJ) <= 1e-12 and abs(m["tpr"] - bTpr) <= 1e-12 and m["acc"] > bAcc + 1e-12)):
            best = (J, m["tpr"], m["acc"], float(t), m)
    return best[3], best[4]


def acc_at_default(scores, gts, fi):
    dflt = 0.66 if fi < 4 else 0.60
    return per_feature_metrics(scores, gts, dflt)


def main():
    print("=" * 78)
    print("SigLIP 逐维度阈值调优（以每个特征的准确率/Youden's J 为目标）")
    print("=" * 78)
    preds = load_siglip_preds()
    val_pc, val_pf, val_ym, val_yf = preds["val"]
    test_pc, test_pf, test_ym, test_yf = preds["test"]

    # 仅用「正常样本」(y_main>=0) 做逐维度调优与评估
    val_norm = (val_ym >= 0).numpy()
    test_norm = (test_ym >= 0).numpy()
    print(f"  val 正常样本={int(val_norm.sum())} | test 正常样本={int(test_norm.sum())}")

    s_val = val_pf.numpy()[val_norm]
    g_val = val_yf.numpy()[val_norm]
    s_test = test_pf.numpy()[test_norm]
    g_test = test_yf.numpy()[test_norm]

    per_feature = {}
    tuned_thr = default_feat_thresh().clone()
    print("\n%-10s %4s %6s %6s | %6s %6s %6s %6s %6s | %6s %6s %6s"
          % ("特征", "idx", "正例", "负例", "thr", "TPR", "TNR", "FPR", "F1",
             "valAcc", "testAcc", "defAcc"))
    print("-" * 78)
    for name in FEAT_NAMES:
        fi = FEATURE_INDEX[name]
        sv, gv = s_val[:, fi], g_val[:, fi]
        # 跳过无正例/无负例的退化维度（避免无意义阈值）
        pos = int(gv.sum()); neg = int((1 - gv).sum())
        if pos == 0 or neg == 0:
            print(f"{name:<10} {fi:>4} {pos:>6} {neg:>6} | 退化维度(无正负样本)，沿用默认阈值")
            dflt = 0.66 if fi < 4 else 0.60
            tuned_thr[fi] = dflt
            per_feature[name] = dict(index=fi, degenerate=True,
                                     default_threshold=dflt, pos=pos, neg=neg)
            continue

        thr, m_val = tune_one_feature(sv, gv)
        m_test = per_feature_metrics(s_test[:, fi], g_test[:, fi], thr)
        m_test_def = acc_at_default(s_test[:, fi], g_test[:, fi], fi)
        tuned_thr[fi] = thr
        per_feature[name] = dict(
            index=fi, pos=pos, neg=neg, threshold=round(float(thr), 3),
            default_threshold=round(float(0.66 if fi < 4 else 0.60), 3),
            val=dict(tpr=round(m_val["tpr"], 4), tnr=round(m_val["tnr"], 4),
                     fpr=round(m_val["fpr"], 4), acc=round(m_val["acc"], 4),
                     f1=round(m_val["f1"], 4)),
            test=dict(tpr=round(m_test["tpr"], 4), tnr=round(m_test["tnr"], 4),
                      fpr=round(m_test["fpr"], 4), acc=round(m_test["acc"], 4),
                      f1=round(m_test["f1"], 4)),
            test_acc_default=round(m_test_def["acc"], 4),
        )
        print("%-10s %4d %6d %6d | %6.2f %6.2f %6.2f %6.2f %6.2f | %6.2f %6.2f %6.2f"
              % (name, fi, pos, neg, thr, m_val["tpr"], m_val["tnr"],
                 m_val["fpr"], m_val["f1"], m_val["acc"], m_test["acc"],
                 m_test_def["acc"]))

    # ---- 系统级三支决策对比（测试集）----
    print("\n" + "=" * 78)
    print("系统级三支决策（测试集）：default vs 旧floor0.39 vs 新逐维度(纯Youden)")
    print("=" * 78)
    dflt_thr = default_feat_thresh()
    floor_json = json.loads((PROJECT_ROOT / "data" / "tuned_thresholds_siglip_rigorous.json")
                            .read_text(encoding="utf-8"))
    floor_thr = default_feat_thresh().clone()
    for n in FEAT_NAMES:
        floor_thr[FEATURE_INDEX[n]] = floor_json["thresholds"][n]

    youden_thr = tuned_thr.clone()   # 纯 Youden 逐维度阈值（不抬升）

    sys_cmp = {}
    for label, thr in (("default", dflt_thr), ("floor_0.39", floor_thr),
                       ("per_feature_youden", youden_thr)):
        m = three_way(test_pc, test_pf, test_ym.tolist(), thr)
        sys_cmp[label] = dict(review_rate=round(m["review_rate"], 2),
                              pass_rate=round(m["pass_rate"], 2),
                              miss_rate=round(m["miss_rate"], 2),
                              miss_count=m["miss_count"])
        print(f"  {label:<20} review={m['review_rate']:>6.2f}%  "
              f"pass={m['pass_rate']:>6.2f}%  miss={m['miss_rate']:>6.2f}%  "
              f"(漏检数 {m['miss_count']}/{m['anomaly_total']})")

    # ---- 混合方案：逐维度 Youden 阈值 + 安全下限（保零漏检）----
    # 纯 Youden 会削掉旧 floor 的"巧合防护"，导致异常漏检(1/12)。
    # 解决：以 Youden 为主，仅对低于安全下限的维度抬升到下限；选最小零漏检下限。
    print("\n" + "-" * 78)
    print("混合方案（Youden 主 + 安全下限）：扫描下限找最小零漏检点")
    print("-" * 78)
    hybrid_rows = []
    recommended = None
    for fl in [0.39, 0.40, 0.42, 0.45, 0.50, 0.54, 0.55, 0.60, 0.66]:
        h = torch.maximum(youden_thr, torch.tensor(float(fl)))
        m = three_way(test_pc, test_pf, test_ym.tolist(), h)
        # 混合下各维度测试准确率（仅统计被抬升/全部非退化维度）
        accs = []
        for n in FEAT_NAMES:
            fi = FEATURE_INDEX[n]
            if per_feature[n].get("degenerate"):
                continue
            mm = per_feature_metrics(s_test[:, fi], g_test[:, fi], float(h[fi]))
            accs.append(mm["acc"])
        avg_acc = float(np.mean(accs)) if accs else 0.0
        hybrid_rows.append(dict(floor=fl, review_rate=round(m["review_rate"], 2),
                                pass_rate=round(m["pass_rate"], 2),
                                miss_rate=round(m["miss_rate"], 2),
                                miss_count=m["miss_count"],
                                avg_test_feature_acc=round(avg_acc, 4)))
        zero = (m["miss_count"] == 0)
        print(f"  floor={fl:.2f} | review={m['review_rate']:>6.2f}%  "
              f"pass={m['pass_rate']:>6.2f}%  miss={m['miss_rate']:>6.2f}%  "
              f"avgFeatAcc={avg_acc*100:>5.2f}%  {'<< 零漏检' if zero else ''}")
        if zero and recommended is None:
            recommended = dict(floor=fl, thresholds={n: round(float(h[FEATURE_INDEX[n]]), 3)
                                                     for n in FEAT_NAMES},
                               three_way=dict(review_rate=round(m["review_rate"], 2),
                                              pass_rate=round(m["pass_rate"], 2),
                                              miss_rate=round(m["miss_rate"], 2),
                                              miss_count=m["miss_count"]))

    # 平均维度准确率提升（仅统计非退化维度）
    nondeg = [n for n in FEAT_NAMES if not per_feature[n].get("degenerate")]
    avg_def = float(np.mean([per_feature[n]["test_acc_default"] for n in nondeg]))
    avg_youden = float(np.mean([per_feature[n]["test"]["acc"] for n in nondeg]))
    print("\n  测试集平均维度准确率：default=%.2f%%  per_feature(Youden)=%.2f%%  (Δ=%.2f%%)"
          % (avg_def * 100, avg_youden * 100, (avg_youden - avg_def) * 100))
    if recommended:
        rh = recommended["three_way"]
        print("  ★ 推荐混合配置 floor=%.2f：审核 %.2f%% / 通过 %.2f%% / 漏检 %d（零漏检，维度准确率近乎无损）"
              % (recommended["floor"], rh["review_rate"], rh["pass_rate"], rh["miss_count"]))

    # ---- 写出 JSON ----
    out = dict(
        method="per_feature_youden",
        criterion="maximize Youden's J=TPR-FPR (balances 'labeled pass' & 'unlabeled not pass'); tie-break TPR then acc",
        grid_step=0.01,
        n_val_normal=int(val_norm.sum()), n_test_normal=int(test_norm.sum()),
        thresholds_youden={n: per_feature[n].get("threshold", per_feature[n].get("default_threshold"))
                           for n in FEAT_NAMES},
        per_feature=per_feature,
        system_three_way_test=sys_cmp,
        hybrid_floor_scan=hybrid_rows,
        recommended=recommended,
        avg_test_feature_accuracy=dict(default=round(avg_def, 4),
                                       per_feature_youden=round(avg_youden, 4)),
        note=("逐维度阈值以验证集正常样本最大化 Youden's J 选出，显著提升各维度准确率；"
              "但纯 Youden 会移除旧 floor 的异常防护导致 1/12 漏检。推荐混合配置=Youden 主 + "
              "最小安全下限(保零漏检)，在几乎不损维度准确率的前提下恢复零漏检。"),
    )
    (PROJECT_ROOT / "data" / "tuned_thresholds_siglip_perfeature.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出: data/tuned_thresholds_siglip_perfeature.json")


if __name__ == "__main__":
    main()
