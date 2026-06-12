"""
特征/分类器温度调优脚本
功能：展示选定参数附近的效果表，为参数选择提供支持依据
搜索4个参数:
  T_CLS   — 主分类器 softmax 温度 (src/config.py CLASSIFIER_TEMPERATURE)
  T_FEAT  — 特征 sigmoid 温度 (src/config.py FEATURE_TEMPERATURE)
            晨读与晨跑共用同一个温度 (特征模型本身只有一个temperature)
  FT_READ — 晨读特征得分门槛 (src/config.py FEATURE_THRESHOLD_READ)
  FT_RUN  — 晨跑特征得分门槛 (src/config.py FEATURE_THRESHOLD_RUN)
通过率计算:
    决策命中 (matched >= READ_TH/RUN_TH) 且 匹配项在标签里全为True -> 正确
"""
import sys, json, numpy as np, torch, pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "data"
sys.path.insert(0, str(BASE))

from src.models.mlp import MLPClassifier
from src.models.mlp_features_optimized import MLPFeaturesOptimized

# ===== 加载模型 =====
cls_model = MLPClassifier()
cls_model.load_state_dict(torch.load(DATA / "mlp_classifier.pt", map_location="cpu"))
cls_model.eval()

feat_model = MLPFeaturesOptimized(temperature=1.8)
feat_model.load_state_dict(torch.load(DATA / "mlp_features_optimized.pt", map_location="cpu"))
feat_model.eval()

# ===== 加载数据 =====
df = pd.read_csv(DATA / "clip_features_cpu.csv")
filenames_all = df["filename"].tolist()
feats = torch.tensor(df.drop("filename", axis=1).values, dtype=torch.float32)
label_map = json.load(open(DATA / "labels.json", encoding="utf-8")).get("labels", {})
split = json.load(open(DATA / "split_config.json", encoding="utf-8"))

all_files = set(split.get("train_files", [])) | set(split.get("val_files", [])) | set(split.get("test_files", []))
valid_idx = [i for i, fn in enumerate(filenames_all) if fn in all_files]
filenames = [filenames_all[i] for i in valid_idx]
feats = feats[valid_idx]

# 解析标签和特征
labels, true_neg, file_features = [], [], []
for fn in filenames:
    info = label_map.get(fn, {})
    lbl = info.get("label", "未知")
    feat_dict = info.get("features", {})
    labels.append(lbl)
    true_neg.append(lbl == "异常")
    file_features.append(feat_dict)

# 特征定义
FEAT_ALL  = ["人脸", "蓝色桌子", "教室", "投影幕布", "跑道", "天空", "绿地", "树木", "旗杆", "号码布", "主席台"]
FEAT_READ = ["人脸", "蓝色桌子", "教室", "投影幕布"]
FEAT_RUN  = ["人脸", "跑道", "天空", "绿地", "树木", "旗杆", "号码布", "主席台"]

# 决策阈值 (与生产一致)
RUN_TH, READ_TH = 5, 3
ALPHA_PASS = 0.80
ALPHA_REVIEW = 0.85

anomaly_total = sum(true_neg)
print(f"数据: {len(labels)}张 (正常{sum(1 for x in true_neg if not x)} 异常{anomaly_total})")
print("=" * 80)

# ===== 预计算 raw logits (与温度无关) =====
with torch.no_grad():
    cls_logits = cls_model(feats).numpy()
    raw_feat_logits = feat_model(feats, inference=False).numpy()

# ===== 调试: 用生产参考参数 (T_CLS=5.0, T_FEAT=1.8) 算一次 =====
def softmax_np(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def sigmoid_np(z):
    return 1.0 / (1.0 + np.exp(-z))

ref_cls_probs = softmax_np(cls_logits / 5.0, axis=1)
ref_pred_labels = ["晨读" if p == 0 else "晨跑" for p in ref_cls_probs.argmax(axis=1)]
ref_confidences = ref_cls_probs.max(axis=1)
probs_debug = np.clip(sigmoid_np(raw_feat_logits / 1.8), 0, 0.95)

print("\n【调试信息 - 参考参数 T_CLS=5.0 T_FEAT=1.8】")
pred_read_count = sum(1 for pl in ref_pred_labels if pl == "晨读")
pred_run_count  = sum(1 for pl in ref_pred_labels if pl == "晨跑")
true_read_count = sum(1 for lbl in labels if lbl == "晨读")
true_run_count  = sum(1 for lbl in labels if lbl == "晨跑")
print(f"参考参数下 pred: 晨读 {pred_read_count} 张, 晨跑 {pred_run_count} 张")
print(f"真实标签:        晨读 {true_read_count} 张, 晨跑 {true_run_count} 张")

# 晨读样本特征得分分布
read_idx_ref = [i for i, pl in enumerate(ref_pred_labels) if pl == "晨读"]
read_scores_ref = probs_debug[read_idx_ref][:, [FEAT_ALL.index(f) for f in FEAT_READ]]
print(f"\n晨读样本特征得分 (生产路径 sigmoid(logit/1.8)):")
print(f"  均值: {read_scores_ref.mean():.4f}  最小: {read_scores_ref.min():.4f}  最大: {read_scores_ref.max():.4f}")
print(f"  > 0.66 比例: {(read_scores_ref > 0.66).mean()*100:.1f}%")
print(f"  任意3个 > 0.66 (生产决策线): {((read_scores_ref > 0.66).sum(axis=1) >= 3).mean()*100:.1f}%")
print(f"  任意4个 > 0.66: {((read_scores_ref > 0.66).sum(axis=1) >= 4).mean()*100:.1f}%")

# 按特征查看
print(f"\n晨读各特征得分 > 0.66 比例:")
for j, f in enumerate(FEAT_READ):
    feat_idx = FEAT_ALL.index(f)
    read_feat_scores = probs_debug[read_idx_ref][:, feat_idx]
    print(f"  {f}: {((read_feat_scores > 0.66).sum())}/{len(read_feat_scores)} ({(read_feat_scores > 0.66).mean()*100:.1f}%)")
print("=" * 80)


# ===== 参数组合范围 =====
T_CLS_range   = [3.0, 4.0, 5.0, 6.0, 7.0]
T_FEAT_range  = [1.4, 1.6, 1.8, 2.0, 2.2]
FT_READ_range = [0.62, 0.64, 0.66, 0.68, 0.70]
FT_RUN_range  = [0.56, 0.58, 0.60, 0.62, 0.64]


def evaluate(T_cls, T_feat, FT_read, FT_run):
    """评估给定参数组合"""
    cls_probs = softmax_np(cls_logits / T_cls, axis=1)
    pred_labels = ["晨读" if p == 0 else "晨跑" for p in cls_probs.argmax(axis=1)]
    confidences = cls_probs.max(axis=1)

    probs = np.clip(sigmoid_np(raw_feat_logits / T_feat), 0, 0.95)

    miss = review = auto_pass = 0
    read_correct = read_total = 0
    run_correct  = run_total  = 0

    for i, pl in enumerate(pred_labels):
        feats_info = file_features[i]

        if pl == "晨读":
            FEAT, FT, MIN_M = FEAT_READ, FT_read, READ_TH
        else:
            FEAT, FT, MIN_M = FEAT_RUN, FT_run, RUN_TH

        matched_features = [f for f in FEAT if probs[i][FEAT_ALL.index(f)] > FT]
        matched = len(matched_features)

        if pl == "晨读" and matched >= READ_TH:
            if all(feats_info.get(f) is True for f in matched_features):
                read_correct += 1
            read_total += 1
        elif pl == "晨跑" and matched >= RUN_TH:
            if all(feats_info.get(f) is True for f in matched_features):
                run_correct += 1
            run_total += 1

        dec = "通过" if (confidences[i] >= ALPHA_PASS and matched >= MIN_M) else "审核"

        if true_neg[i] and dec == "通过":
            miss += 1
        elif dec == "审核":
            review += 1
        else:
            auto_pass += 1

    total = len(pred_labels)

    is_user_sel = (abs(T_cls - 5.0) < 0.01 and abs(T_feat - 1.8) < 0.01
                   and abs(FT_read - 0.66) < 0.01 and abs(FT_run - 0.60) < 0.01)
    if is_user_sel:
        print(f"\n【当前选定参数调试 T_CLS={T_cls}, T_FEAT={T_feat}, FT_READ={FT_read}, FT_RUN={FT_run}】")
        print(f"  read_correct={read_correct}, read_total={read_total},  rate={read_correct/max(read_total,1)*100:.1f}%")
        print(f"  run_correct={run_correct}, run_total={run_total},  rate={run_correct/max(run_total,1)*100:.1f}%")
        for i, pl in enumerate(pred_labels):
            if pl == "晨读":
                feats_info = file_features[i]
                print(f"\n  第一个晨读样本 {i}:")
                for f in FEAT_READ:
                    feat_idx = FEAT_ALL.index(f)
                    label_val = feats_info.get(f)
                    score = probs[i][feat_idx]
                    over_ft = score > FT_read
                    print(f"    {f}: score={score:.4f}, label={label_val}, >FT={over_ft}")
                break

    return {
        "miss": miss / max(anomaly_total, 1) * 100,
        "review": review / total * 100,
        "pass": auto_pass / total * 100,
        "read_rate": read_correct / max(read_total, 1),
        "run_rate": run_correct / max(run_total, 1),
    }


# ===== 生成所有组合的效果表 =====
print("\n【4参数组合效果表】")
print("=" * 110)
print(f"{'T_CLS':>6} {'T_FEAT':>7} {'FT_READ':>7} {'FT_RUN':>6} | {'漏检%':>6} {'通过%':>6} {'审核%':>6} | {'晨读%':>6} {'晨跑%':>6} {'差异':>6}")
print("-" * 110)

results = []
debug_lines = []

import io
old_stdout = sys.stdout
for T_cls in T_CLS_range:
    for T_feat in T_FEAT_range:
        for FT_read in FT_READ_range:
            for FT_run in FT_RUN_range:
                string_buffer = io.StringIO()
                sys.stdout = string_buffer
                r = evaluate(T_cls, T_feat, FT_read, FT_run)
                debug_output = string_buffer.getvalue()
                if debug_output:
                    debug_lines.append(debug_output)
                sys.stdout = old_stdout

                results.append({
                    "T_cls": T_cls, "T_feat": T_feat,
                    "FT_read": FT_read, "FT_run": FT_run,
                    **r
                })

results.sort(key=lambda x: (x["miss"], abs(x["read_rate"] - x["run_rate"])))

lines = []
lines.append("【4参数组合效果表】")
lines.append("=" * 110)
lines.append(f"{'T_CLS':>6} {'T_FEAT':>7} {'FT_READ':>7} {'FT_RUN':>6} | {'漏检%':>6} {'通过%':>6} {'审核%':>6} | {'晨读%':>6} {'晨跑%':>6} {'差异':>6}")
lines.append("-" * 110)

for r in results:
    mark = ""
    if (abs(r["T_cls"] - 5.0) < 0.01 and abs(r["T_feat"] - 1.8) < 0.01
        and abs(r["FT_read"] - 0.66) < 0.01 and abs(r["FT_run"] - 0.60) < 0.01):
        mark = " <-当前选定"
    line = (f"{r['T_cls']:>6.2f} {r['T_feat']:>7.2f} {r['FT_read']:>7.2f} {r['FT_run']:>6.2f} | "
            f"{r['miss']:>6.1f} {r['pass']:>6.1f} {r['review']:>6.1f} | "
            f"{r['read_rate']*100:>6.1f} {r['run_rate']*100:>6.1f} {abs(r['read_rate']-r['run_rate'])*100:>6.1f}{mark}")
    lines.append(line)
    print(line)

output_file = BASE / "tune_balance_results.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))
    if debug_lines:
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("【调试信息】\n")
        f.write("\n".join(debug_lines))
print(f"\n结果已保存到: {output_file}")
