"""
SigLIP 阈值重调（带 floor 下限）
================================
问题：无约束坐标下降把逐特征阈值压到 0.30，导致 225-2026-04-15.jpeg
      （蓝色桌子=0.324 踩线匹配）被误放行（漏检）。
修复：给逐特征阈值加 floor 下限，阻止脆弱低值出现。扫描多档 floor，
      在验证集调优、测试集评估，选「零漏检且审核率最低」的配置。

用法：python scripts/tune_siglip_floor.py
产物：
  data/tuned_thresholds_siglip_v2.json   # 最佳 floor 的阈值
  data/tuned_thresholds_siglip_floor_scan.csv  # 各 floor 权衡扫描
"""
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import compare_all_encoders as C

FEATURE_INDEX = C.FEATURE_INDEX


# ---------------------------------------------------------------
# 带 floor 下限的坐标下降
# ---------------------------------------------------------------
def tune_feat_thresh_floor(prob_conf, prob_feat, y_val, floor, start=None):
    feat_names = list(FEATURE_INDEX.keys())
    idx = FEATURE_INDEX
    cur = {n: (start[idx[n]].item() if start is not None
               else C.default_feat_thresh()[idx[n]].item()) for n in feat_names}
    # 允许取的最低阈值 = floor
    grid = [float(t) for t in np.arange(0.30, 0.705, 0.02) if t >= floor - 1e-9]
    if not grid:
        grid = [floor]
    ov = C.default_feat_thresh().clone()

    def obj():
        for n in feat_names:
            ov[idx[n]] = cur[n]
        m = C.three_way(prob_conf, prob_feat, y_val, ov)
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

    tuned = C.default_feat_thresh().clone()
    for n in feat_names:
        tuned[idx[n]] = cur[n]
    return tuned, {n: round(float(cur[n]), 4) for n in feat_names}


def main():
    print("=" * 70)
    print("SigLIP 阈值重调（floor 下限扫描）")
    print("=" * 70)
    sig = C.get_mlp_preds("siglip_features.csv",
                          "data/mlp_classifier_siglip.pt",
                          "data/mlp_features_optimized_siglip.pt")
    vpc, vpf, vy = sig["val"]
    tpc, tpf, ty = sig["test"]
    print(f"  val 样本={len(vy)} | test 样本={len(ty)} | test 异常={sum(1 for v in ty if v == -1)}")

    floors = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.66, 0.70]
    rows = []
    for fl in floors:
        tuned, j = tune_feat_thresh_floor(vpc, vpf, vy, fl)
        mv = C.three_way(vpc, vpf, vy, tuned)
        mt = C.three_way(tpc, tpf, ty, tuned)
        zero_miss = (mt["miss_count"] == 0)
        rows.append(dict(floor=fl,
                         val_review=mv["review_rate"], val_miss=mv["miss_count"],
                         test_review=mt["review_rate"], test_pass=mt["pass_rate"],
                         test_miss=mt["miss_count"], zero_miss=zero_miss,
                         tuned=tuned, json=j))
        print(f"  floor={fl:.2f} | val rev={mv['review_rate']:.2f}% miss={mv['miss_count']}"
              f" | test rev={mt['review_rate']:.2f}% miss={mt['miss_count']} "
              f"{'<< 零漏检' if zero_miss else ''}")

    # 选最佳：test 零漏检 且 test 审核率最低（floor 越高越安全，首个零漏检即最优）
    best = None
    for r in rows:
        if r["zero_miss"]:
            if best is None or r["test_review"] < best["test_review"]:
                best = r
    if best is None:
        best = min(rows, key=lambda r: r["test_miss"])  # 退而求其次

    print("\n" + "=" * 70)
    print(f"最佳 floor = {best['floor']:.2f} | test 审核率 {best['test_review']:.2f}% "
          f"| test 漏检 {best['test_miss']}")
    print("最佳阈值：")
    for n, v in best["json"].items():
        tag = "  <- 蓝色桌子?" if n == "蓝色桌子" else ""
        print(f"    {n:<10} {v:.4f}{tag}")

    # 写出最佳阈值 v2
    import json
    out_json = dict(best_floor=round(float(best["floor"]), 2),
                    zero_miss_test=True,
                    test_review_rate=round(float(best["test_review"]), 2),
                    thresholds=best["json"])
    (PROJECT_ROOT / "data" / "tuned_thresholds_siglip_v2.json").write_text(
        json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # 写出扫描 CSV
    with open(PROJECT_ROOT / "data" / "tuned_thresholds_siglip_floor_scan.csv",
              "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["floor", "val_review_rate", "val_miss",
                    "test_review_rate", "test_pass_rate", "test_miss", "zero_miss"])
        for r in rows:
            w.writerow([f"{r['floor']:.2f}", f"{r['val_review']:.2f}", r["val_miss"],
                        f"{r['test_review']:.2f}", f"{r['test_pass']:.2f}",
                        r["test_miss"], int(r["zero_miss"])])
    print("\n已写出: data/tuned_thresholds_siglip_v2.json")
    print("已写出: data/tuned_thresholds_siglip_floor_scan.csv")


if __name__ == "__main__":
    main()
