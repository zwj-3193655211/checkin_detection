"""定位 SigLIP2-NaFlex-B/16 在测试集「调优阈值」下漏检的异常图片。
复现 compare_all_encoders 的 SigLIP 推理 + 验证集坐标下降调阈值，然后逐样本判定，
找出 y_main==-1 且被判定为「自动通过」的样本。"""
import os, sys, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import torch
import train_mlp as tm
import compare_all_encoders as C

# 1) 复现 SigLIP 推理（与 compare 完全一致）
preds = C.get_mlp_preds("siglip_features.csv",
                         "data/mlp_classifier_siglip.pt",
                         "data/mlp_features_optimized_siglip.pt")
val_pc, val_pf, val_y = preds["val"]
test_pc, test_pf, test_y = preds["test"]

# 2) 测试集文件名对齐（prepare_data 丢弃了文件名，按相同顺序重建）
features, filenames, labels, split = tm.load_data()
feature_dict = {f: i for i, f in enumerate(filenames)}
test_fnames = [f for f in split["test_files"] if f in feature_dict]

# 3) 验证集坐标下降调阈值（确定性，等价于 tuned_thresholds_siglip.json）
tuned, tuned_json = C.tune_feat_thresh(val_pc, val_pf, val_y)

# 4) 逐样本三支决策（与 C.three_way 同逻辑，但要拿到索引）
preds_idx = test_pc.argmax(dim=1)
conf = test_pc.max(dim=1).values
valid = C._CLASS_FEAT_MASK[preds_idx]
match = (test_pf > tuned) & (valid > 0.5)
matched = match.sum(dim=1)
r1 = (preds_idx == 1) & (conf >= C.ALPHA_AUTO_PASS) & (matched >= C.RUN_FEATURE_THRESH)
r2 = (preds_idx == 0) & (conf >= C.ALPHA_AUTO_PASS) & (matched >= C.READ_FEATURE_THRESH)
pass_dec = r1 | r2
y = torch.tensor(test_y, dtype=torch.long)
is_anom = (y == -1)
miss_mask = is_anom & pass_dec

print("=== 配置阈值 ===")
print("ALPHA_AUTO_PASS =", C.ALPHA_AUTO_PASS, "| ALPHA_REVIEW =", C.ALPHA_REVIEW,
      "| MIN_FEATURES =", C.MIN_FEATURES,
      "| READ_FEATURE_THRESH =", C.READ_FEATURE_THRESH, "| RUN_FEATURE_THRESH =", C.RUN_FEATURE_THRESH)
print("调优逐特征阈值:", json.dumps(tuned_json, ensure_ascii=False))
print("测试集: 样本数=%d  异常数=%d" % (len(test_y), int(is_anom.sum().item())))
print("==> 调优下漏检数:", int(miss_mask.sum().item()))

# 诊断：列出所有异常样本的决策，便于对照为什么只有这张被漏
print("\n=== 测试集所有异常样本决策 ===")
for i in range(len(test_y)):
    if test_y[i] == -1:
        pl = "晨读" if preds_idx[i] == 0 else "晨跑"
        req = C.READ_FEATURE_THRESH if pl == "晨读" else C.RUN_FEATURE_THRESH
        dec = "自动通过(漏!)" if pass_dec[i] else "待审核"
        print("  %-22s pred=%s conf=%.4f matched=%d/%d -> %s"
              % (test_fnames[i], pl, conf[i].item(), int(matched[i].item()), req, dec))

print("\n=== 漏检图片明细 ===")
miss_idx = torch.where(miss_mask)[0].tolist()
for i in miss_idx:
    print("文件:", test_fnames[i])
    print("路径: data/raw/" + test_fnames[i])
    print("真值: 异常(-1) | 预测类别:", "晨读" if preds_idx[i] == 0 else "晨跑",
          "| 分类置信度: %.4f" % conf[i].item(),
          "| 匹配特征数: %d" % int(matched[i].item()))
    # 该预测类别允许的特征及其概率
    pl = "晨读" if preds_idx[i] == 0 else "晨跑"
    cf = tm.CLASS_FEATURES[pl]
    print("预测类别允许特征及概率(>阈值视为匹配):")
    for f in cf:
        fi = tm.FEATURE_INDEX[f]
        thr = tuned_json[f]
        p = test_pf[i, fi].item()
        mark = "✓匹配" if (p > tuned[fi].item() and f in cf) else "✗"
        print("   %-6s p=%.3f thr=%.2f %s" % (f, p, thr, mark))
