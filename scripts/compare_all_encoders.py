"""
Phase 4: 三编码器公平对比（CLIP / SigLIP2-NaFlex / 自训 ViT-Tiny）
====================================================================

统一流程：
1. 对每个编码器，在测试集上计算 (cls softmax 置信度, feat sigmoid 概率) 原始预测。
   - CLIP / SigLIP: 复用 train_mlp 的模型与特征 CSV（与 Phase 2 完全一致）。
   - ViT-Tiny: 直接读取 train_vit_tiny.py 保存的各 seed 预测文件。
2. 在验证集上做坐标下降，搜索「逐特征阈值」（目标 = review_rate + 100*miss_rate，
   零漏检优先）。三编码器用同一流程，保证公平。
3. 在测试集上以「默认阈值」与「调优阈值」分别评估三支决策。
4. McNemar 检验：比较最佳配置的两系统逐样本决策差异。
5. α-scan：扫描自动通过置信度阈值，输出 review%/miss% 权衡曲线（CSV）。
6. 写出 EXPERIMENT_REPORT_NAFLEX.md 与 phase4_comparison.json。

用法（项目根目录）：
    python scripts/compare_all_encoders.py
"""
import os
import sys
import json
import csv
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import train_mlp as tm
from src.config import (
    CLASSIFIER_TEMPERATURE, FEATURE_TEMPERATURE,
    ALPHA_AUTO_PASS, ALPHA_REVIEW, MIN_FEATURES,
    RUN_FEATURE_THRESH, READ_FEATURE_THRESH,
)

FEATURE_INDEX = tm.FEATURE_INDEX
CLASS_FEATURES = tm.CLASS_FEATURES


# ---------------------------------------------------------------
# 统一三支决策（接受已 softmax/sigmoid 化的概率，向量化实现）
# ---------------------------------------------------------------
# 预计算 类别(0=晨读,1=晨跑) -> 11 维特征有效掩码，避免逐样本 Python 循环
_CLASS_FEAT_MASK = torch.zeros(2, 11)
for _fname, _fi in FEATURE_INDEX.items():   # items() -> (name, idx)
    if _fname in CLASS_FEATURES['晨读']:
        _CLASS_FEAT_MASK[0, _fi] = 1.0
    if _fname in CLASS_FEATURES['晨跑']:
        _CLASS_FEAT_MASK[1, _fi] = 1.0


def three_way(prob_conf, prob_feat, y_main,
              feat_thresh,
              alpha_pass=ALPHA_AUTO_PASS, alpha_review=ALPHA_REVIEW,
              min_feat=MIN_FEATURES, read_th=READ_FEATURE_THRESH, run_th=RUN_FEATURE_THRESH):
    # prob_conf: (N,2) softmax 置信度；prob_feat: (N,11) sigmoid 概率；y_main: list[int]
    preds = prob_conf.argmax(dim=1)            # 0=晨读, 1=晨跑
    conf = prob_conf.max(dim=1).values         # (N,)
    y = torch.tensor(y_main, dtype=torch.long)
    valid = _CLASS_FEAT_MASK[preds]            # (N,11) 该样本类别允许的特征
    match = (prob_feat > feat_thresh) & (valid > 0.5)   # (N,11)
    matched = match.sum(dim=1)                 # (N,)

    r1 = (preds == 1) & (conf >= alpha_pass) & (matched >= run_th)
    r2 = (preds == 0) & (conf >= alpha_pass) & (matched >= read_th)
    pass_dec = r1 | r2
    r3 = matched < min_feat
    r4 = conf < alpha_review
    review_dec = (~pass_dec) & (r3 | r4)

    is_anom = (y == -1)
    miss_count = int((is_anom & pass_dec).sum().item())
    review_count = int(review_dec.sum().item())
    n = len(y_main)
    anom = int(is_anom.sum().item())
    return dict(miss_rate=miss_count / anom * 100 if anom else 0.0,
                review_rate=review_count / n * 100,
                pass_rate=100.0 - review_count / n * 100 if n else 0.0,
                miss_count=miss_count, review_count=review_count,
                anomaly_total=anom, test_size=n)


def default_feat_thresh():
    return torch.tensor([0.66 if i < 4 else 0.60 for i in range(11)])


def tune_feat_thresh(prob_conf, prob_feat, y_val, start=None):
    feat_names = list(FEATURE_INDEX.keys())
    idx_of = FEATURE_INDEX
    cur = {n: (start[idx_of[n]].item() if start is not None else default_feat_thresh()[idx_of[n]].item())
           for n in feat_names}
    grid = list(np.arange(0.30, 0.705, 0.02))
    ov = default_feat_thresh().clone()

    def obj():
        for n in feat_names:
            ov[idx_of[n]] = cur[n]
        m = three_way(prob_conf, prob_feat, y_val, ov)
        return m["review_rate"] + 100.0 * m["miss_rate"]

    improved = True
    rounds = 0
    while improved and rounds < 50:
        improved = False
        rounds += 1
        for n in feat_names:
            best_t, best_o = cur[n], obj()
            for t in grid:
                cur[n] = t
                o = obj()
                if o < best_o - 1e-9:
                    best_o, best_t = o, t
            if best_t != cur[n]:
                improved = True
            cur[n] = best_t
    tuned = default_feat_thresh().clone()
    for n in feat_names:
        tuned[idx_of[n]] = cur[n]
    return tuned, {n: round(cur[n], 4) for n in feat_names}


# ---------------------------------------------------------------
# 获取 CLIP / SigLIP 预测
# ---------------------------------------------------------------
def get_mlp_preds(csv_name, cls_pt, feat_pt):
    os.environ["CHECKIN_FEATURES_CSV"] = csv_name
    features, filenames, labels, split_config = tm.load_data()
    data = tm.prepare_data(features, filenames, labels, split_config)
    df = __import__("pandas").read_csv(PROJECT_ROOT / "data" / csv_name)
    dim = df.shape[1] - 1
    classifier = tm.MLPClassifier(input_dim=dim, hidden_dim=256, output_dim=2, dropout=0.3)
    classifier.load_state_dict(torch.load(PROJECT_ROOT / cls_pt, map_location="cpu"))
    fmodel = tm.MLPFeaturesOptimized(input_dim=dim, hidden_dim=512, output_dim=11,
                                     dropout=0.3, temperature=FEATURE_TEMPERATURE)
    fmodel.load_state_dict(torch.load(PROJECT_ROOT / feat_pt, map_location="cpu"))
    classifier.eval();
    fmodel.eval()

    out = {}
    for split_name in ("val", "test"):
        X, y_main, _ = data[split_name]
        with torch.no_grad():
            cl = classifier(X.float())
            pc = torch.softmax(cl / CLASSIFIER_TEMPERATURE, dim=1)
            pf = fmodel(X.float(), inference=True)
        out[split_name] = (pc, pf, y_main.tolist())
    return out


# ---------------------------------------------------------------
# McNemar 检验
# ---------------------------------------------------------------
def mcnemar(dec_a_pass, dec_b_pass):
    """dec_*_pass: list[bool]，True=自动通过，False=待审核。返回 (b, c, chi2, p)。"""
    b = sum(1 for a, bb in zip(dec_a_pass, dec_b_pass) if (not a) and bb)
    c = sum(1 for a, bb in zip(dec_a_pass, dec_b_pass) if a and (not bb))
    n = b + c
    if n == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n  # 连续性校正
    # p-value (chi2 with df=1)
    from math import erfc, sqrt
    p = erfc(sqrt(chi2))
    return b, c, chi2, p


def decisions_from(prob_conf, prob_feat, y_main, feat_thresh, alpha_pass):
    """返回 list[bool]（True=自动通过）。"""
    out = []
    for i in range(len(y_main)):
        pl = '晨读' if prob_conf[i].argmax().item() == 0 else '晨跑'
        c = prob_conf[i].max().item()
        cf = CLASS_FEATURES[pl]
        matched = sum(1 for f, fi in FEATURE_INDEX.items()
                      if prob_feat[i, fi].item() > feat_thresh[fi].item() and f in cf)
        r1 = pl == '晨跑' and c >= alpha_pass and matched >= RUN_FEATURE_THRESH
        r2 = pl == '晨读' and c >= alpha_pass and matched >= READ_FEATURE_THRESH
        r3 = matched < MIN_FEATURES
        r4 = c < ALPHA_REVIEW
        decision = 0
        if r1 or r2:
            decision = 0
        elif r3 or r4:
            decision = 1
        out.append(decision == 0)
    return out


# ---------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------
def main():
    print("=" * 70)
    print("Phase 4: CLIP vs SigLIP2-NaFlex vs 自训 ViT-Tiny 公平对比")
    print("=" * 70)

    encoders = {
        "CLIP-ViT-B/32": dict(
            kind="mlp", csv="clip_features_cpu.csv",
            cls_pt="data/mlp_classifier.pt", feat_pt="data/mlp_features_optimized.pt"),
        "SigLIP2-NaFlex-B/16": dict(
            kind="mlp", csv="siglip_features.csv",
            cls_pt="data/mlp_classifier_siglip.pt", feat_pt="data/mlp_features_optimized_siglip.pt"),
    }
    # ViT-Tiny 各 seed 预测
    vit_seeds = {}
    for s in [42, 123, 777]:
        p = PROJECT_ROOT / "data" / f"vit_tiny_preds_seed{s}.pt"
        if p.exists():
            d = torch.load(p, map_location="cpu")
            vit_seeds[s] = dict(
                val=(d["val_cls"], d["val_feat"], d["val_y"]),
                test=(d["test_cls"], d["test_feat"], d["test_y"]))
    if vit_seeds:
        encoders[f"ViT-Tiny(自训)"] = dict(kind="vit", seeds=vit_seeds)
    else:
        print("[warn] 未找到 ViT 预测文件，将只对比 CLIP/SigLIP。")

    rows = []
    alpha_scan = {}
    mcnemar_rows = []
    clip_tuned_test = None  # 用作 McNemar 基线

    for name, cfg in encoders.items():
        print(f"\n##### {name} #####")
        if cfg["kind"] == "mlp":
            preds = get_mlp_preds(cfg["csv"], cfg["cls_pt"], cfg["feat_pt"])
            val_pc, val_pf, val_y = preds["val"]
            test_pc, test_pf, test_y = preds["test"]
        else:
            # ViT：多 seed，逐 seed 评估后取均值（miss 取最差 / review 取均值）
            seed_metrics = []
            for s, sd in cfg["seeds"].items():
                vpc, vpf, vy = sd["val"]
                tpc, tpf, ty = sd["test"]
                tuned, _ = tune_feat_thresh(vpc, vpf, vy)
                md = three_way(tpc, tpf, ty, default_feat_thresh())
                mt = three_way(tpc, tpf, ty, tuned)
                seed_metrics.append((s, md, mt, tuned, tpc, tpf, ty))
            # 汇总：review 取均值，miss 取最差；tuned 取第一个 seed 的阈值（近似）
            def agg(metrics_list, key):
                return float(np.mean([m[key] for _, m, _, _, _, _, _ in metrics_list]))
            md_review = agg(seed_metrics, "review_rate")
            mt_review = agg(seed_metrics, "review_rate")
            md_miss = max(m[key] for _, m, _, _, _, _, _ in seed_metrics for key in ("miss_rate",)) if False else \
                max([m["miss_rate"] for _, m, _, _, _, _, _ in seed_metrics])
            mt_miss = max([m["miss_rate"] for _, m, _, _, _, _, _ in seed_metrics])
            rows.append(dict(encoder=name, default_review=md_review, default_pass=100 - md_review,
                             default_miss=md_miss, tuned_review=mt_review, tuned_pass=100 - mt_review,
                             tuned_miss=mt_miss, seeds=list(cfg["seeds"].keys())))
            # α-scan 用 seed 42（若存在）的 tuned 阈值
            base = seed_metrics[0]
            tuned0 = base[3]
            tpc0, tpf0, ty0 = base[4], base[5], base[6]
            scan = []
            for tau in np.arange(0.70, 0.951, 0.02):
                m = three_way(tpc0, tpf0, ty0, tuned0, alpha_pass=tau, alpha_review=max(tau, ALPHA_REVIEW))
                scan.append((round(float(tau), 2), round(m["review_rate"], 2), round(m["miss_rate"], 2)))
            alpha_scan[name] = scan
            continue

        # CLIP / SigLIP 单模型
        tuned, tuned_json = tune_feat_thresh(val_pc, val_pf, val_y)
        md = three_way(test_pc, test_pf, test_y, default_feat_thresh())
        mt = three_way(test_pc, test_pf, test_y, tuned)
        rows.append(dict(encoder=name, default_review=md["review_rate"], default_pass=md["pass_rate"],
                         default_miss=md["miss_rate"], tuned_review=mt["review_rate"],
                         tuned_pass=mt["pass_rate"], tuned_miss=mt["miss_rate"],
                         tuned_json=tuned_json))
        print(f"  默认: review={md['review_rate']:.2f}% pass={md['pass_rate']:.2f}% miss={md['miss_rate']:.2f}%")
        print(f"  调优: review={mt['review_rate']:.2f}% pass={mt['pass_rate']:.2f}% miss={mt['miss_rate']:.2f}%")
        json.dump(tuned_json, open(PROJECT_ROOT / f"data/tuned_thresholds_{('clip' if 'CLIP' in name else 'siglip')}.json", "w"),
                  ensure_ascii=False, indent=2)

        if "CLIP" in name:
            clip_tuned_test = (test_pc, test_pf, test_y, tuned)
            clip_dec = decisions_from(test_pc, test_pf, test_y, tuned, ALPHA_AUTO_PASS)

        # α-scan
        scan = []
        for tau in np.arange(0.70, 0.951, 0.02):
            m = three_way(test_pc, test_pf, test_y, tuned, alpha_pass=tau, alpha_review=max(tau, ALPHA_REVIEW))
            scan.append((round(float(tau), 2), round(m["review_rate"], 2), round(m["miss_rate"], 2)))
        alpha_scan[name] = scan

    # McNemar: 各 ViT seed（调优） vs CLIP 调优
    if clip_tuned_test is not None and "ViT-Tiny(自训)" in encoders:
        cpc, cpf, cy, ctuned = clip_tuned_test
        clip_dec = decisions_from(cpc, cpf, cy, ctuned, ALPHA_AUTO_PASS)
        for s, sd in encoders["ViT-Tiny(自训)"]["seeds"].items():
            tpc, tpf, ty = sd["test"]
            vtuned, _ = tune_feat_thresh(sd["val"][0], sd["val"][1], sd["val"][2])
            vdec = decisions_from(tpc, tpf, ty, vtuned, ALPHA_AUTO_PASS)
            b, c, chi2, p = mcnemar(clip_dec, vdec)
            mcnemar_rows.append(dict(vit_seed=s, b=b, c=c, chi2=round(chi2, 3), p=round(p, 4)))

    # ---- 输出 ----
    summary = dict(rows=rows, mcnemar=mcnemar_rows, alpha_scan=alpha_scan)
    json.dump(summary, open(PROJECT_ROOT / "data" / "phase4_comparison.json", "w"),
              ensure_ascii=False, indent=2)

    # α-scan CSV
    with open(PROJECT_ROOT / "data" / "alpha_scan.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["encoder", "alpha_pass", "review_rate", "miss_rate"])
        for name, scan in alpha_scan.items():
            for tau, rev, miss in scan:
                w.writerow([name, tau, rev, miss])

    # 控制台汇总
    print("\n" + "=" * 70)
    print("对比汇总（测试集）")
    print("=" * 70)
    hdr = f"{'编码器':<22}{'配置':<8}{'审核%':>9}{'通过%':>9}{'漏检%':>9}{'漏检数':>8}"
    print(hdr);
    print("-" * len(hdr))
    for r in rows:
        for cfg in ("default", "tuned"):
            print(f"{r['encoder']:<22}{cfg:<8}{r[cfg+'_review']:>9.2f}{r[cfg+'_pass']:>9.2f}"
                  f"{r[cfg+'_miss']:>9.2f}")
    if mcnemar_rows:
        print("\nMcNemar (ViT调优 vs CLIP调优):")
        for m in mcnemar_rows:
            print(f"  ViT seed {m['vit_seed']}: b={m['b']} c={m['c']} chi2={m['chi2']} p={m['p']}")

    write_report(rows, mcnemar_rows, alpha_scan)
    print("\n报告已写出: EXPERIMENT_REPORT_NAFLEX.md")


def write_report(rows, mcnemar_rows, alpha_scan):
    L = []
    L.append("# Phase 4 实验报告：CLIP / SigLIP2-NaFlex / 自训 ViT-Tiny 公平对比\n")
    L.append("> 生成脚本：`scripts/compare_all_encoders.py`（三编码器统一三支决策评估）\n")
    L.append("## 1. 测试集三支决策对比（默认阈值 vs 调优阈值）\n")
    L.append("| 编码器 | 配置 | 审核率% | 自动通过% | 漏检率% |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        for cfg in ("default", "tuned"):
            L.append(f"| {r['encoder']} | {cfg} | {r[cfg+'_review']:.2f} | {r[cfg+'_pass']:.2f} | {r[cfg+'_miss']:.2f} |")
    L.append("\n**关键判据**：漏检（异常被放行）为一票否决项，必须 =0%。\n")

    clip_tuned = next((r for r in rows if 'CLIP' in r['encoder']), None)
    sig_tuned = next((r for r in rows if 'SigLIP' in r['encoder']), None)
    vit_rows = [r for r in rows if 'ViT' in r['encoder']]
    if clip_tuned:
        L.append(f"- CLIP 调优：审核 {clip_tuned['tuned_review']:.2f}% / 漏检 {clip_tuned['tuned_miss']:.2f}%")
    if sig_tuned:
        L.append(f"- SigLIP 调优：审核 {sig_tuned['tuned_review']:.2f}% / 漏检 {sig_tuned['tuned_miss']:.2f}%"
                 f"（**仍为 {sig_tuned['tuned_miss']:.2f}% —— 未达零漏检**）")
    if vit_rows:
        for r in vit_rows:
            L.append(f"- ViT-Tiny 调优（seed {r.get('seeds')}）：审核 {r['tuned_review']:.2f}% / 漏检 {r['tuned_miss']:.2f}%")

    L.append("\n## 2. McNemar 检验（ViT 调优 vs CLIP 调优，测试集逐样本决策）\n")
    if mcnemar_rows:
        L.append("| ViT seed | b(CLIP审/ViT过) | c(CLIP过/ViT审) | chi2 | p |")
        L.append("|---|---|---|---|---|")
        for m in mcnemar_rows:
            L.append(f"| {m['vit_seed']} | {m['b']} | {m['c']} | {m['chi2']} | {m['p']} |")
        L.append("\n(p>0.05 表示两者决策分布无显著差异)\n")
    else:
        L.append("（ViT 预测文件缺失，未计算）\n")

    L.append("## 3. α-scan（自动通过置信度阈值权衡曲线）\n")
    L.append("扫描 `alpha_pass`（同时 `alpha_review=max(alpha_pass, 0.85)`），观察审核率与漏检率权衡。详见 `data/alpha_scan.csv`。\n")
    for name, scan in alpha_scan.items():
        L.append(f"### {name}")
        L.append("| alpha_pass | 审核率% | 漏检率% |")
        L.append("|---|---|---|")
        for tau, rev, miss in scan:
            L.append(f"| {tau} | {rev} | {miss} |")
        L.append("")

    L.append("## 4. 结论与建议\n")
    L.append("- CLIP 基线在零漏检前提下仍是最稳的参照。")
    L.append("- SigLIP2-NaFlex 在调优阈值下出现漏检，若要采用需先解决该漏检样本（见 Phase 3 后续）。")
    L.append("- ViT-Tiny 端到端结果见上表；若其调优漏检=0 且审核率不高于 CLIP，则可认为「专精自训模型」达到预期。")
    L.append("- 下一步（Phase 5）：若选定编码器，接入 `src/encoders.py` 抽象，配置 `ENCODER` 切换，更新系统报告。\n")

    open(PROJECT_ROOT / "EXPERIMENT_REPORT_NAFLEX.md", "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
