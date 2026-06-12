"""
消融实验脚本
功能：通过控制变量法验证各参数对系统性能的影响
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

# 解析标签
labels, true_neg = [], []
for fn in filenames:
    info = label_map.get(fn, {})
    lbl = info.get("label", "未知")
    labels.append(lbl)
    true_neg.append(lbl == "异常")

anomaly_total = sum(true_neg)
normal_total = sum(1 for x in true_neg if not x)

print(f"数据: {len(labels)}张 (正常{normal_total} 异常{anomaly_total})")
print("=" * 80)

# ===== 预计算 raw logits =====
with torch.no_grad():
    cls_logits = cls_model(feats).numpy()
    raw_feat_logits = feat_model(feats, inference=False).numpy()

# ===== 辅助函数 =====
def softmax_np(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def sigmoid_np(z):
    return 1.0 / (1.0 + np.exp(-z))

def evaluate_system(T_cls, T_feat, FT_read, FT_run, 
                     ALPHA_AUTO_PASS, ALPHA_REVIEW, 
                     READ_TH, RUN_TH, MIN_FEATURES,
                     use_rules=True, enhance_number_bib=True):
    """
    评估系统性能
    
    参数:
        T_cls: 主分类器温度
        T_feat: 特征预测器温度
        FT_read: 晨读特征阈值
        FT_run: 晨跑特征阈值
        ALPHA_AUTO_PASS: 自动通过置信度阈值
        ALPHA_REVIEW: 待审核置信度阈值
        READ_TH: 晨读最少匹配特征数
        RUN_TH: 晨跑最少匹配特征数
        MIN_FEATURES: 最少总特征数
        use_rules: 是否使用三支决策规则
        enhance_number_bib: 是否使用号码布增强
    """
    FEAT_ALL = ["人脸", "蓝色桌子", "教室", "投影幕布", "跑道", "天空", "绿地", "树木", "旗杆", "号码布", "主席台"]
    FEAT_READ = ["人脸", "蓝色桌子", "教室", "投影幕布"]
    FEAT_RUN = ["人脸", "跑道", "天空", "绿地", "树木", "旗杆", "号码布", "主席台"]
    
    # 预测
    cls_probs = softmax_np(cls_logits / T_cls, axis=1)
    pred_labels = ["晨读" if p == 0 else "晨跑" for p in cls_probs.argmax(axis=1)]
    confidences = cls_probs.max(axis=1)
    
    probs = np.clip(sigmoid_np(raw_feat_logits / T_feat), 0, 0.95)
    
    # 号码布增强
    if enhance_number_bib:
        probs[:, 9] = np.clip(probs[:, 9] * 2, 0, 0.95)
    
    # 统计
    miss = review = auto_pass = 0
    rule1 = rule2 = rule3 = rule4 = default = 0
    
    # 新增统计指标
    read_pass = run_pass = 0
    read_total = run_total = 0
    number_bib_scores = []
    
    for i, pl in enumerate(pred_labels):
        if pl == "晨读":
            matched = sum(1 for f in FEAT_READ if probs[i][FEAT_ALL.index(f)] > FT_read)
            read_total += 1
            if matched >= READ_TH:
                read_pass += 1
        else:
            matched = sum(1 for f in FEAT_RUN if probs[i][FEAT_ALL.index(f)] > FT_run)
            run_total += 1
            if matched >= RUN_TH:
                run_pass += 1
        
        # 号码布得分
        number_bib_scores.append(probs[i][9])
        
        if use_rules:
            # 使用三支决策规则
            if pl == "晨跑" and confidences[i] >= ALPHA_AUTO_PASS and matched >= RUN_TH:
                dec = "通过"
                rule1 += 1
            elif pl == "晨读" and confidences[i] >= ALPHA_AUTO_PASS and matched >= READ_TH:
                dec = "通过"
                rule2 += 1
            elif matched < MIN_FEATURES:
                dec = "审核"
                rule3 += 1
            elif confidences[i] < ALPHA_REVIEW:
                dec = "审核"
                rule4 += 1
            else:
                dec = "通过"
                default += 1
        else:
            # 只用置信度判断
            if confidences[i] >= ALPHA_REVIEW:
                dec = "通过"
            else:
                dec = "审核"
        
        if true_neg[i] and dec == "通过":
            miss += 1
        elif dec == "审核":
            review += 1
        else:
            auto_pass += 1
    
    total = len(pred_labels)
    
    # 计算额外指标
    number_bib_mean = np.mean(number_bib_scores)
    
    return {
        "miss": miss,
        "miss_rate": miss / max(anomaly_total, 1) * 100,
        "review": review,
        "review_rate": review / total * 100,
        "pass": auto_pass,
        "pass_rate": auto_pass / total * 100,
        "rule1": rule1, "rule2": rule2, "rule3": rule3, "rule4": rule4, "default": default,
        "read_pass_rate": read_pass / max(read_total, 1) * 100 if read_total > 0 else 0,
        "run_pass_rate": run_pass / max(run_total, 1) * 100 if run_total > 0 else 0,
        "number_bib_mean": number_bib_mean,
        "read_total": read_total,
        "run_total": run_total,
    }


def print_result(name, result, expected=""):
    """打印实验结果"""
    print(f"\n{'='*60}")
    print(f"【实验】{name}")
    print(f"{'='*60}")
    print(f"  漏检: {result['miss']} ({result['miss_rate']:.1f}%)")
    print(f"  通过: {result['pass']} ({result['pass_rate']:.1f}%)")
    print(f"  待审核: {result['review']} ({result['review_rate']:.1f}%)")
    if result['rule1'] + result['rule2'] + result['rule3'] + result['rule4'] + result['default'] > 0:
        print(f"  规则触发: 规则1={result['rule1']}, 规则2={result['rule2']}, 规则3={result['rule3']}, 规则4={result['rule4']}, 默认={result['default']}")
    # 新增指标
    if 'read_pass_rate' in result:
        diff = abs(result['read_pass_rate'] - result['run_pass_rate'])
        print(f"  晨读通过率: {result['read_pass_rate']:.1f}% ({result['read_total']}个样本)")
        print(f"  晨跑通过率: {result['run_pass_rate']:.1f}% ({result['run_total']}个样本)")
        print(f"  通过率差异: {diff:.1f}%")
    if 'number_bib_mean' in result:
        print(f"  号码布平均得分: {result['number_bib_mean']:.3f} (阈值0.60)")
    if expected:
        print(f"  预期结果: {expected}")
    return result


# ===== 基准配置 =====
BASELINE = {
    "T_cls": 5.0, "T_feat": 1.8,
    "FT_read": 0.66, "FT_run": 0.60,
    "ALPHA_AUTO_PASS": 0.80, "ALPHA_REVIEW": 0.85,
    "READ_TH": 3, "RUN_TH": 5, "MIN_FEATURES": 3,
    "use_rules": True, "enhance_number_bib": True
}

# ===== 实验一：温度参数设为1 =====
print("\n" + "="*80)
print("实验一：温度参数设为1（模型过于自信）")
print("="*80)
exp1 = evaluate_system(T_cls=1.0, T_feat=1.0, FT_read=0.66, FT_run=0.60,
                       ALPHA_AUTO_PASS=0.80, ALPHA_REVIEW=0.85,
                       READ_TH=3, RUN_TH=5, MIN_FEATURES=3,
                       use_rules=True, enhance_number_bib=True)
print_result("温度T=1.0", exp1, 
            "预期：模型过于自信，晨读晨跑特征得分差异大，晨读普遍通过，晨跑普遍待审核")

# ===== 实验二：不使用规则约束 =====
print("\n" + "="*80)
print("实验二：不使用三支决策规则，只用置信度判断")
print("="*80)
exp2 = evaluate_system(T_cls=5.0, T_feat=1.8, FT_read=0.66, FT_run=0.60,
                       ALPHA_AUTO_PASS=0.80, ALPHA_REVIEW=0.85,
                       READ_TH=3, RUN_TH=5, MIN_FEATURES=3,
                       use_rules=False, enhance_number_bib=True)
print_result("不使用规则", exp2, 
            "预期：漏检率上升，人工审核率变化")

# ===== 实验三：统一特征阈值 =====
print("\n" + "="*80)
print("实验三：晨读晨跑使用统一特征阈值（0.60）")
print("="*80)
exp3 = evaluate_system(T_cls=5.0, T_feat=1.8, FT_read=0.60, FT_run=0.60,
                       ALPHA_AUTO_PASS=0.80, ALPHA_REVIEW=0.85,
                       READ_TH=3, RUN_TH=5, MIN_FEATURES=3,
                       use_rules=True, enhance_number_bib=True)
print_result("统一阈值FT=0.60", exp3, 
            "预期：晨读晨跑通过率差异大")

# ===== 实验四：不使用号码布增强 =====
print("\n" + "="*80)
print("实验四：不使用号码布增强")
print("="*80)
exp4 = evaluate_system(T_cls=5.0, T_feat=1.8, FT_read=0.66, FT_run=0.60,
                       ALPHA_AUTO_PASS=0.80, ALPHA_REVIEW=0.85,
                       READ_TH=3, RUN_TH=5, MIN_FEATURES=3,
                       use_rules=True, enhance_number_bib=False)
print_result("无号码布增强", exp4, 
            "预期：号码布得分显著低于0.60")

# ===== 基准对比 =====
print("\n" + "="*80)
print("【基准配置】")
print("="*80)
baseline = evaluate_system(**BASELINE)
print_result("基准配置", baseline)

# ===== 汇总对比表 =====
print("\n\n" + "="*120)
print("【消融实验汇总对比】")
print("="*120)
print(f"{'实验':<22} | {'漏检%':<6} {'通过%':<6} {'审核%':<6} | {'晨读%':<6} {'晨跑%':<6} {'差异%':<6} | {'号码布':<8}")
print("-"*120)

all_results = [
    ("基准配置", baseline, baseline),
    ("实验一：T=1.0", exp1, baseline),
    ("实验二：无规则约束", exp2, baseline),
    ("实验三：统一阈值FT=0.60", exp3, baseline),
    ("实验四：无号码布增强", exp4, baseline),
]

for name, result, base in all_results:
    diff = abs(result['read_pass_rate'] - result['run_pass_rate']) if 'read_pass_rate' in result else 0
    number_bib = result.get('number_bib_mean', 0)
    print(f"{name:<22} | {result['miss_rate']:>5.1f}% {result['pass_rate']:>5.1f}% {result['review_rate']:>5.1f}% | {result.get('read_pass_rate', 0):>5.1f}% {result.get('run_pass_rate', 0):>5.1f}% {diff:>5.1f}% | {number_bib:>7.3f}")

# ===== 保存结果 =====
output_file = BASE / "ablation_results.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("消融实验汇总对比\n")
    f.write("="*120 + "\n")
    f.write(f"{'实验':<22} | {'漏检%':<6} {'通过%':<6} {'审核%':<6} | {'晨读%':<6} {'晨跑%':<6} {'差异%':<6} | {'号码布':<8}\n")
    f.write("-"*120 + "\n")
    for name, result, base in all_results:
        diff = abs(result['read_pass_rate'] - result['run_pass_rate']) if 'read_pass_rate' in result else 0
        number_bib = result.get('number_bib_mean', 0)
        f.write(f"{name:<22} | {result['miss_rate']:>5.1f}% {result['pass_rate']:>5.1f}% {result['review_rate']:>5.1f}% | {result.get('read_pass_rate', 0):>5.1f}% {result.get('run_pass_rate', 0):>5.1f}% {diff:>5.1f}% | {number_bib:>7.3f}\n")

print(f"\n结果已保存到: {output_file}")