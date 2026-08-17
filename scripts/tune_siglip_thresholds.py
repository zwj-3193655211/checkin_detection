"""
Phase 2 公平性对比脚本：CLIP vs SigLIP2-NaFlex 三支决策对比
============================================================

功能：
1. 为两个编码器（CLIP / SigLIP2-NaFlex）分别加载特征 CSV 与对应 MLP 模型。
2. 在验证集上做坐标下降（coordinate descent）搜索「逐特征阈值」，
   目标函数 = review_rate + 100 * miss_rate（零漏检优先）。
3. 在测试集上用「默认阈值」与「调优阈值」两种配置分别跑三支决策，
   输出 review / pass / miss 三类指标，做公平对比。

用法：
    python scripts/tune_siglip_thresholds.py
（从项目根目录运行；自动把根目录与 scripts/ 加入 sys.path）

注意：本脚本复用 train_mlp.py 的 prepare_data / validate_with_three_way_decision，
以保证与训练管线完全一致的三支决策逻辑。
"""

import os
import sys
import json
import copy
from pathlib import Path

import torch
import numpy as np

# ---- 路径设置：让 `import train_mlp` 与 `from src...` 都能解析 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train_mlp as tm  # 复用 prepare_data / validate_with_three_way_decision / FEATURE_INDEX 等

# ---------------------------------------------------------------
# 默认「逐特征阈值」：晨读特征(idx 0-3)=0.66，晨跑特征(idx 4-10)=0.60
# ---------------------------------------------------------------
def default_thresh(idx):
    return tm.FEATURE_THRESHOLD_READ if idx < 4 else tm.FEATURE_THRESHOLD_RUN


# ---------------------------------------------------------------
# 全局阈值覆盖表（idx -> threshold），通过 monkeypatch _get_feature_threshold 生效
# ---------------------------------------------------------------
_OVERRIDE = {}

def _patched_get_feature_threshold(idx):
    return _OVERRIDE.get(idx, default_thresh(idx))

tm._get_feature_threshold = _patched_get_feature_threshold


# ===============================================================
# 编码器配置
# ===============================================================
ENCODERS = [
    {
        "name": "CLIP-ViT-B/32",
        "csv": "clip_features_cpu.csv",
        "cls_pt": "data/mlp_classifier.pt",
        "feat_pt": "data/mlp_features_optimized.pt",
        "tuned_json": "data/tuned_thresholds.json",
    },
    {
        "name": "SigLIP2-NaFlex-B/16",
        "csv": "siglip_features.csv",
        "cls_pt": "data/mlp_classifier_siglip.pt",
        "feat_pt": "data/mlp_features_optimized_siglip.pt",
        "tuned_json": "data/tuned_thresholds_siglip.json",
    },
]


def load_models(csv_name, cls_pt, feat_pt):
    """根据特征 CSV 维度重建并加载 MLP 模型。"""
    df = __import__("pandas").read_csv(PROJECT_ROOT / "data" / csv_name)
    dim = df.shape[1] - 1
    classifier = tm.MLPClassifier(input_dim=dim, hidden_dim=256, output_dim=2, dropout=0.3)
    classifier.load_state_dict(torch.load(PROJECT_ROOT / cls_pt, map_location="cpu"))
    features_model = tm.MLPFeaturesOptimized(input_dim=dim, hidden_dim=512, output_dim=11, dropout=0.3, temperature=tm.FEATURE_TEMPERATURE)
    features_model.load_state_dict(torch.load(PROJECT_ROOT / feat_pt, map_location="cpu"))
    classifier.eval(); features_model.eval()
    return classifier, features_model, dim


def objective_on_val(classifier, features_model, val):
    """目标函数 = review_rate + 100 * miss_rate（零漏检优先）。"""
    m = tm.validate_with_three_way_decision(
        classifier, features_model, val[0], val[1], val[2]
    )
    return m["review_rate"] + 100.0 * m["miss_rate"], m


def tune_feature_thresholds(classifier, features_model, val):
    """坐标下降搜索逐特征阈值（在验证集上）。返回 {idx: threshold} 与每特征名映射。"""
    feat_names = list(tm.FEATURE_INDEX.keys())
    idx_of = tm.FEATURE_INDEX
    # 当前解从默认阈值开始
    cur = {name: default_thresh(idx_of[name]) for name in feat_names}
    grid = list(np.arange(0.30, 0.705, 0.02))  # 0.30 ~ 0.70

    def apply_and_obj():
        _OVERRIDE.clear()
        for name in feat_names:
            _OVERRIDE[idx_of[name]] = cur[name]
        return objective_on_val(classifier, features_model, val)[0]

    improved = True
    rounds = 0
    while improved and rounds < 30:
        improved = False
        rounds += 1
        for name in feat_names:
            best_t = cur[name]
            best_obj = apply_and_obj()
            for t in grid:
                cur[name] = t
                obj = apply_and_obj()
                if obj < best_obj - 1e-9:
                    best_obj = obj
                    best_t = t
            if best_t != cur[name]:
                improved = True
            cur[name] = best_t
    return cur, feat_names, idx_of


def evaluate_test(classifier, features_model, test, override):
    """在测试集上用给定「逐特征阈值覆盖」跑三支决策。"""
    _OVERRIDE.clear()
    _OVERRIDE.update(override)
    m = tm.validate_with_three_way_decision(
        classifier, features_model, test[0], test[1], test[2], is_final=True
    )
    return m


def main():
    print("=" * 70)
    print("Phase 2 公平性对比：CLIP vs SigLIP2-NaFlex（三支决策）")
    print("=" * 70)

    results = {}
    for enc in ENCODERS:
        print("\n" + "#" * 70)
        print(f"# 编码器: {enc['name']}  (特征={enc['csv']})")
        print("#" * 70)

        # 加载数据（prepare_data 内部用 CHECKIN_FEATURES_CSV 选 CSV）
        os.environ["CHECKIN_FEATURES_CSV"] = enc["csv"]
        features, filenames, labels, split_config = tm.load_data()
        data = tm.prepare_data(features, filenames, labels, split_config)
        val = data["val"]
        test = data["test"]
        print(f"  验证集样本={len(val[0])}  测试集样本={len(test[0])}  "
              f"(异常={(test[1] == -1).sum().item()})")

        classifier, features_model, dim = load_models(enc["csv"], enc["cls_pt"], enc["feat_pt"])
        print(f"  模型维度 input_dim={dim}")

        # 1) 默认阈值
        default_override = {tm.FEATURE_INDEX[n]: default_thresh(tm.FEATURE_INDEX[n])
                            for n in tm.FEATURE_INDEX}
        m_default = evaluate_test(classifier, features_model, test, default_override)

        # 2) 调优阈值（坐标下降）
        print("  [坐标下降调优逐特征阈值 on VAL ...]")
        tuned, feat_names, idx_of = tune_feature_thresholds(classifier, features_model, val)
        # 保存调优结果
        tuned_json = {name: round(tuned[name], 4) for name in feat_names}
        with open(PROJECT_ROOT / enc["tuned_json"], "w", encoding="utf-8") as f:
            json.dump(tuned_json, f, ensure_ascii=False, indent=2)
        print(f"  调优阈值已写入 {enc['tuned_json']}: {tuned_json}")
        m_tuned = evaluate_test(classifier, features_model, test,
                                {idx_of[n]: tuned[n] for n in feat_names})

        results[enc["name"]] = {
            "default": m_default,
            "tuned": m_tuned,
            "dim": dim,
            "test_size": len(test[0]),
            "test_anomaly": int((test[1] == -1).sum().item()),
        }

    # ---------------- 汇总对比表 ----------------
    print("\n" + "=" * 70)
    print("对比汇总（测试集）")
    print("=" * 70)
    header = f"{'编码器':<22}{'阈值配置':<12}{'审核率%':>10}{'自动通过%':>12}{'漏检率%':>10}{'漏检数':>8}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        for mode in ("default", "tuned"):
            m = r[mode]
            review = m["review_rate"]
            passr = 100.0 - review
            miss = m["miss_rate"]
            miss_n = m["miss_count"]
            cfg = "默认" if mode == "default" else "调优"
            print(f"{name:<22}{cfg:<12}{review:>10.2f}{passr:>12.2f}{miss:>10.2f}{miss_n:>8}")
    print("-" * len(header))
    print("注：自动通过率 = 100% - 审核率；漏检 = 异常样本被放行（一票否决项）。")


if __name__ == "__main__":
    main()
