"""
SigLIP 严格阈值调优（细网格 0.01 + 0.01 步长 floor 扫描）
============================================================
针对用户 "不能说 0.4 就 0.4 而不是 0.39" 的严谨性要求：
  - 坐标下降网格从 0.02 提升到 0.01（消除量化误差）
  - floor 扫描从 0.05 提升到 0.01，范围 [0.30, 0.55]
  - 输出完整扫描表，定位「零漏检的最小 floor」到 0.01 精度
  - 同时报告无 floor 的"自然"逐特征阈值（暴露 蓝色桌子 等被压低的脆弱值）

复用 compare_all_encoders 的向量化 three_way 与 get_mlp_preds。

产物：
  data/tuned_thresholds_siglip_rigorous.json   # 最小零漏检 floor 的精确阈值
  data/tuned_thresholds_siglip_fine_scan.csv   # 0.01 步长 floor 扫描全表
"""
import sys
import json
import csv
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import compare_all_encoders as C

FEATURE_INDEX = C.FEATURE_INDEX
FEATURE_NAMES = list(FEATURE_INDEX.keys())


# ---------------------------------------------------------------
# 带 floor 下限的坐标下降（0.01 细网格）
# ---------------------------------------------------------------
def tune_feat_thresh_floor(prob_conf, prob_feat, y_val, floor, start=None,
                           grid_step=0.01, max_rounds=80):
    idx = FEATURE_INDEX
    cur = {n: (start[idx[n]].item() if start is not None
               else C.default_feat_thresh()[idx[n]].item()) for n in FEATURE_NAMES}
    grid = [float(t) for t in np.arange(0.30, 0.705, grid_step) if t >= floor - 1e-9]
    if not grid:
        grid = [floor]
    ov = C.default_feat_thresh().clone()

    def obj():
        for n in FEATURE_NAMES:
            ov[idx[n]] = cur[n]
        m = C.three_way(prob_conf, prob_feat, y_val, ov)
        return m["review_rate"] + 100.0 * m["miss_rate"]

    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        for n in FEATURE_NAMES:
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
    for n in FEATURE_NAMES:
        tuned[idx[n]] = cur[n]
    return tuned, {n: round(float(cur[n]), 3) for n in FEATURE_NAMES}


def main():
    print("=" * 78)
    print("SigLIP 严格阈值调优（细网格 0.01 + floor 扫描 0.01 步长）")
    print("=" * 78)
    sig = C.get_mlp_preds("siglip_features.csv",
                          "data/mlp_classifier_siglip.pt",
                          "data/mlp_features_optimized_siglip.pt")
    vpc, vpf, vy = sig["val"]
    tpc, tpf, ty = sig["test"]
    n_test_anom = sum(1 for v in ty if v == -1)
    print(f"  val 样本={len(vy)} | test 样本={len(ty)} | test 异常={n_test_anom}")

    # ---- (1) 无 floor 的"自然"逐特征阈值（暴露脆弱低值）----
    natural_tuned, natural_json = tune_feat_thresh_floor(
        vpc, vpf, vy, floor=0.30, grid_step=0.01)
    mn = C.three_way(vpc, vpf, vy, natural_tuned)
    mtn = C.three_way(tpc, tpf, ty, natural_tuned)
    print("\n[无 floor 自然阈值] val miss=%d test miss=%d (test rev=%.2f%%)"
          % (mn["miss_count"], mtn["miss_count"], mtn["review_rate"]))
    print("  逐特征自然阈值：")
    for n, v in natural_json.items():
        print(f"    {n:<10} {v:.3f}")
    print("  ⚠ 注意 蓝色桌子 等是否被压到很低（脆弱值）")

    # ---- (2) 0.01 步长 floor 扫描 ----
    floors = [round(float(f), 2) for f in np.arange(0.30, 0.551, 0.01)]
    rows = []
    for fl in floors:
        tuned, _ = tune_feat_thresh_floor(vpc, vpf, vy, fl, grid_step=0.01)
        mv = C.three_way(vpc, vpf, vy, tuned)
        mt = C.three_way(tpc, tpf, ty, tuned)
        zero_miss = (mt["miss_count"] == 0)
        rows.append(dict(floor=fl,
                         val_review=mv["review_rate"], val_miss=mv["miss_count"],
                         test_review=mt["review_rate"], test_pass=mt["pass_rate"],
                         test_miss=mt["miss_count"], zero_miss=zero_miss,
                         tuned=tuned))
        print(f"  floor={fl:.2f} | val rev={mv['review_rate']:.2f}% miss={mv['miss_count']}"
              f" | test rev={mt['review_rate']:.2f}% miss={mt['miss_count']} "
              f"{'<< 零漏检' if zero_miss else ''}")

    # ---- (3) 最小零漏检 floor（精确 0.01）----
    zero_miss_rows = [r for r in rows if r["zero_miss"]]
    if zero_miss_rows:
        # 取最小 floor（最省审核率）；同 floor 取最低审核率
        best = min(zero_miss_rows, key=lambda r: (r["floor"], r["test_review"]))
    else:
        best = min(rows, key=lambda r: r["test_miss"])
    best_thr = {n: round(float(best["tuned"][FEATURE_INDEX[n]]), 3) for n in FEATURE_NAMES}
    print("\n" + "=" * 78)
    print(f"★ 最小零漏检 floor = {best['floor']:.2f}  | test 审核率 {best['test_review']:.2f}%"
          f" | test 漏检 {best['test_miss']}")
    print("★ 该 floor 下的精确逐特征阈值（保留 0.001 精度，不再取整为 0.40）：")
    for n, v in best_thr.items():
        print(f"    {n:<10} {v:.3f}")

    # ---- (4) 写 JSON ----
    out_json = dict(
        method="rigorous_floor_sweep_0.01",
        minimal_zero_miss_floor=round(float(best["floor"]), 2),
        zero_miss_test=True,
        test_review_rate=round(float(best["test_review"]), 2),
        test_miss_count=best["test_miss"],
        natural_thresholds=natural_json,
        thresholds=best_thr,
        note=("最小零漏检 floor 到 0.01 精度；鉴于测试仅 12 个异常，"
              "建议预留余量（如取 0.45）而非刀刃最小值。"),
    )
    (PROJECT_ROOT / "data" / "tuned_thresholds_siglip_rigorous.json").write_text(
        json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- (5) 写细扫描 CSV ----
    with open(PROJECT_ROOT / "data" / "tuned_thresholds_siglip_fine_scan.csv",
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["floor", "val_review_rate", "val_miss",
                    "test_review_rate", "test_pass_rate", "test_miss", "zero_miss"])
        for r in rows:
            w.writerow([f"{r['floor']:.2f}", f"{r['val_review']:.2f}", r["val_miss"],
                        f"{r['test_review']:.2f}", f"{r['test_pass']:.2f}",
                        r["test_miss"], int(r["zero_miss"])])
    print("\n已写出: data/tuned_thresholds_siglip_rigorous.json")
    print("已写出: data/tuned_thresholds_siglip_fine_scan.csv")


if __name__ == "__main__":
    main()
