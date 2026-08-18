"""
对比：SigLIP baseline（单头 11 维共享 MLP） vs SigLIP 拆分（11 独立头）
=========================================================================
两个架构在 SigLIP 768 维特征上各自独立做 0.01 严格阈值调优（验证集坐标下降），
再在测试集上用统一三支决策比较 漏检率/审核率/通过率 + 各特征准确率。

复用 compare_all_encoders 的向量化 three_way、default_feat_thresh、FEATURE_INDEX。

用法：python scripts/compare_siglip_architectures.py
产物：
  data/siglip_arch_compare.json   # 结构化对比
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import compare_all_encoders as C
import train_features_11mlp_siglip as S

FEATURE_INDEX = C.FEATURE_INDEX
FEATURE_NAMES = list(FEATURE_INDEX.keys())


def tune_feat_thresh_floor(prob_conf, prob_feat, y_val, floor, grid_step=0.01, max_rounds=80):
    idx = FEATURE_INDEX
    cur = {n: C.default_feat_thresh()[idx[n]].item() for n in FEATURE_NAMES}
    grid = [float(t) for t in np.arange(0.30, 0.705, grid_step) if t >= floor - 1e-9]
    if not grid:
        grid = [floor]
    ov = C.default_feat_thresh().clone()

    def obj():
        for n in FEATURE_NAMES:
            ov[idx[n]] = cur[n]
        m = C.three_way(prob_conf, prob_feat, y_val, ov)
        return m["review_rate"] + 100.0 * m["miss_rate"]

    improved, rounds = True, 0
    while improved and rounds < max_rounds:
        improved = False; rounds += 1
        for n in FEATURE_NAMES:
            best_t, best_o = cur[n], obj()
            for t in grid:
                cur[n] = t; o = obj()
                if o < best_o - 1e-9:
                    best_o, best_t = o, t
            if best_t != cur[n]:
                improved = True
            cur[n] = best_t
    tuned = C.default_feat_thresh().clone()
    for n in FEATURE_NAMES:
        tuned[idx[n]] = cur[n]
    return tuned, {n: round(float(cur[n]), 3) for n in FEATURE_NAMES}


def feat_acc(pf, yf):
    preds = (pf > 0.5).float()
    return {n: (preds[:, FEATURE_INDEX[n]] == yf[:, FEATURE_INDEX[n]]).float().mean().item()
            for n in FEATURE_NAMES}


def get_preds_single():
    """baseline 单头：复用 compare_all_encoders 的 loader"""
    return C.get_mlp_preds("siglip_features.csv",
                           "data/mlp_classifier_siglip.pt",
                           "data/mlp_features_optimized_siglip.pt")


def get_preds_split():
    """拆分 11 头：SigLIP 分类器 + 11 独立头"""
    from src.models.mlp import MLPClassifier
    cls = MLPClassifier(input_dim=768, hidden_dim=256, output_dim=2, dropout=0.3)
    cls.load_state_dict(torch.load(PROJECT_ROOT / "data" / "mlp_classifier_siglip.pt", map_location='cpu'))
    cls.eval()
    ens = S.load_ensemble()
    feats = torch.tensor(
        __import__('pandas').read_csv(PROJECT_ROOT / "data" / "siglip_features.csv")
        .drop('filename', axis=1).values, dtype=torch.float32)
    fnames = __import__('pandas').read_csv(PROJECT_ROOT / "data" / "siglip_features.csv")['filename'].tolist()
    with open(PROJECT_ROOT / "data" / "labels.json", encoding='utf-8') as f:
        labels = json.load(f).get('labels', json.load(open(PROJECT_ROOT / "data" / "labels.json", encoding='utf-8')))
    with open(PROJECT_ROOT / "data" / "split_config.json", encoding='utf-8') as f:
        split = json.load(f)
    fd = {fn: feats[i] for i, fn in enumerate(fnames)}
    out = {}
    for split_name, files in [("val", split['val_files']), ("test", split['test_files'])]:
        X, ym = [], []
        for fn in files:
            if fn not in fd:
                continue
            info = labels.get(fn, {})
            lab = info.get('label', '未知')
            ym.append(-1 if lab == '异常' else (1 if lab == '晨跑' else 0))
            X.append(fd[fn])
        X = torch.stack(X); ym = torch.tensor(ym, dtype=torch.long)
        with torch.no_grad():
            pc = torch.softmax(cls(X.float()) / C.CLASSIFIER_TEMPERATURE, dim=1)
            pf = ens(X.float(), inference=True)
        out[split_name] = (pc, pf, ym)
    return out


def main():
    print("=" * 78)
    print("SigLIP：baseline 单头 vs 拆分 11 头（各自独立 0.01 严格调优）")
    print("=" * 78)
    single = get_preds_single()
    split = get_preds_split()
    # 测试集 11 维特征真值（与上面的预测按相同 test_files 顺序对齐）
    _, _, (_, _, yf_test) = S.load_data_siglip()

    results = {}
    for arch_name, preds in [("baseline_single_head", single), ("split_11_head", split)]:
        vpc, vpf, vy = preds["val"]
        tpc, tpf, ty = preds["test"]
        # 在验证集上做 0.01 严格 floor 扫描，取最小零漏检 floor
        best = None
        for fl in [round(float(f), 2) for f in np.arange(0.30, 0.551, 0.01)]:
            tuned, _ = tune_feat_thresh_floor(vpc, vpf, vy, fl)
            mt = C.three_way(tpc, tpf, ty, tuned)
            if mt["miss_count"] == 0:
                if best is None or (fl, mt["review_rate"]) < (best[0], best[1]["review_rate"]):
                    best = (fl, mt, tuned)
        fl, mt, tuned = best
        thr = {n: round(float(tuned[FEATURE_INDEX[n]]), 3) for n in FEATURE_NAMES}
        acc = feat_acc(tpf, yf_test)
        results[arch_name] = dict(
            minimal_zero_miss_floor=fl,
            test_review_rate=mt["review_rate"], test_pass_rate=mt["pass_rate"],
            test_miss_count=mt["miss_count"], test_miss_rate=mt["miss_rate"],
            thresholds=thr,
            feat_acc={n: round(acc[n] * 100, 2) for n in FEATURE_NAMES},
        )
        print(f"\n[{arch_name}] 最小零漏检 floor={fl:.2f} | 审核率={mt['review_rate']:.2f}% "
              f"| 通过率={mt['pass_rate']:.2f}% | 漏检={mt['miss_count']}")

    # 逐特征准确率对比
    print("\n" + "=" * 78)
    print("各特征准确率（测试集, %）")
    print(f"{'特征':<10}{'baseline':>12}{'split':>12}{'Δ':>10}")
    base_acc = results["baseline_single_head"]["feat_acc"]
    split_acc = results["split_11_head"]["feat_acc"]
    for n in FEATURE_NAMES:
        d = split_acc[n] - base_acc[n]
        print(f"{n:<10}{base_acc[n]:>12.2f}{split_acc[n]:>12.2f}{d:>+10.2f}")

    (PROJECT_ROOT / "data" / "siglip_arch_compare.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出: data/siglip_arch_compare.json")


if __name__ == '__main__':
    main()
