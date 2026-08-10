"""
消融实验脚本

功能：通过控制变量法验证各参数对系统性能的影响

消融实验（Ablation Study）是一种深度学习模型分析方法，通过逐一移除或改变
模型中的某个组件/参数，观察性能变化，从而判断该组件/参数的重要性。

本脚本通过消融实验验证以下参数的作用：
1. 温度参数(Temperature)：模型过度自信问题
2. 三支决策规则：特征匹配的作用
3. 统一特征阈值 vs 分类别阈值
4. 号码布增强：关键特征的单独加权

实验结果将输出各配置下的：
- 漏检率：异常样本被错误通过的比例
- 通过率：自动通过的比例
- 待审核率：需要人工审核的比例
- 晨读/晨跑通过率差异：评估阈值设置是否平衡
"""

# ==================== 标准库 ====================
import sys, json, numpy as np, torch, pandas as pd
from pathlib import Path

# ==================== 项目路径设置 ====================
BASE = Path(__file__).parent
DATA = BASE / "data"
sys.path.insert(0, str(BASE))

# ==================== 模型导入 ====================
from src.models.mlp import MLPClassifier
from src.models.mlp_features_optimized import MLPFeaturesOptimized

# ==================== 加载预训练模型 ====================
# 主分类器：二分类（晨读/晨跑）
cls_model = MLPClassifier()
cls_model.load_state_dict(torch.load(DATA / "mlp_classifier.pt", map_location="cpu"))
cls_model.eval()

# 特征预测器：11维特征预测（带温度参数）
feat_model = MLPFeaturesOptimized(temperature=1.8)
feat_model.load_state_dict(torch.load(DATA / "mlp_features_optimized.pt", map_location="cpu"))
feat_model.eval()

# ==================== 加载数据集 ====================
# CLIP特征数据：包含所有图片的512维CLIP特征向量
df = pd.read_csv(DATA / "clip_features_cpu.csv")
filenames_all = df["filename"].tolist()
feats = torch.tensor(df.drop("filename", axis=1).values, dtype=torch.float32)

# 标签数据：从labels.json读取每张图片的真实标签和特征标注
label_map = json.load(open(DATA / "labels.json", encoding="utf-8")).get("labels", {})

# 数据集划分配置：train/val/test
split = json.load(open(DATA / "split_config.json", encoding="utf-8"))

# 合并所有数据集的图片（用于评估）
all_files = set(split.get("train_files", [])) | set(split.get("val_files", [])) | set(split.get("test_files", []))
valid_idx = [i for i, fn in enumerate(filenames_all) if fn in all_files]
filenames = [filenames_all[i] for i in valid_idx]
feats = feats[valid_idx]

# ==================== 解析标签 ====================
# 构建标签列表和异常标记
labels, true_neg = [], []
for fn in filenames:
    info = label_map.get(fn, {})
    lbl = info.get("label", "未知")
    labels.append(lbl)
    # 异常标记：标签为"异常"的样本
    true_neg.append(lbl == "异常")

anomaly_total = sum(true_neg)    # 异常样本总数
normal_total = sum(1 for x in true_neg if not x)  # 正常样本总数

print(f"数据: {len(labels)}张 (正常{normal_total} 异常{anomaly_total})")
print("=" * 80)

# ==================== 预计算模型输出 ====================
# 在评估前预先计算所有logits，避免重复计算
with torch.no_grad():
    cls_logits = cls_model(feats).numpy()                    # 主分类器logits
    raw_feat_logits = feat_model(feats, inference=False).numpy()  # 特征预测器logits（未缩放）

# ==================== 辅助函数 ====================
# NumPy版本的softmax：数值稳定的实现
def softmax_np(z, axis=-1):
    """
    数值稳定的softmax实现
    
    Softmax公式：p_i = exp(z_i) / sum(exp(z_j))
    数值稳定技巧：先减去每行的最大值再计算exp
    """
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

# NumPy版本的sigmoid
def sigmoid_np(z):
    """Sigmoid函数：σ(z) = 1 / (1 + exp(-z))"""
    return 1.0 / (1.0 + np.exp(-z))

def evaluate_system(T_cls, T_feat, FT_read, FT_run, 
                     ALPHA_AUTO_PASS, ALPHA_REVIEW, 
                     READ_TH, RUN_TH, MIN_FEATURES,
                     use_rules=True, enhance_number_bib=True):
    """
    评估系统在给定参数配置下的性能
    
    这是消融实验的核心函数，通过传入不同的参数组合，
    模拟系统在"假设"配置下的表现。
    
    参数说明:
        T_cls: 主分类器的温度参数
            - T=1.0: 原始softmax，模型可能过度自信
            - T>1 (如5.0): 降低置信度，让概率分布更平滑
            
        T_feat: 特征预测器的温度参数
            - 影响特征预测的置信度
            
        FT_read: 晨读特征阈值 (0~1)
            - 特征预测概率超过此阈值才算"检测到"该特征
            - 晨读特征使用0.66（较高，因为晨读特征较少）
            
        FT_run: 晨跑特征阈值 (0~1)
            - 晨跑特征使用0.60（稍低，提高检出率）
            
        ALPHA_AUTO_PASS: 自动通过置信度阈值
            - 置信度 >= 此值时，模型才有资格自动通过
            
        ALPHA_REVIEW: 待审核置信度阈值
            - 置信度 < 此值时，必须人工审核
            - 通常 ALPHA_REVIEW > ALPHA_AUTO_PASS
            
        READ_TH: 晨读最少匹配特征数
            - 晨读图片至少需要匹配上几个特征才能自动通过
            
        RUN_TH: 晨跑最少匹配特征数
            - 晨跑图片至少需要匹配上几个特征才能自动通过
            - 通常 RUN_TH > READ_TH（因为晨跑特征更多）
            
        MIN_FEATURES: 最少总特征数
            - 任何图片匹配特征数 < 此值都必须审核
            
        use_rules: 是否启用三支决策规则
            - True: 使用完整的规则体系
            - False: 只用置信度判断（简化版）
            
        enhance_number_bib: 是否启用号码布增强
            - True: 号码布得分乘以2，提高晨跑检出率
            - False: 不增强
    
    返回:
        dict: 包含各项指标的字典
            - miss/miss_rate: 漏检数/漏检率
            - review/review_rate: 待审核数/待审核率
            - pass/pass_rate: 自动通过数/通过率
            - rule1~4/default: 各规则触发次数
            - read_pass_rate/run_pass_rate: 晨读/晨跑通过率
            - number_bib_mean: 号码布平均得分
    """
    # 特征名称列表（11维特征）
    FEAT_ALL = ["人脸", "蓝色桌子", "教室", "投影幕布", "跑道", "天空", "绿地", "树木", "旗杆", "号码布", "主席台"]
    # 晨读专属特征（4个）
    FEAT_READ = ["人脸", "蓝色桌子", "教室", "投影幕布"]
    # 晨跑专属特征（8个，不含公共的人脸）
    FEAT_RUN = ["人脸", "跑道", "天空", "绿地", "树木", "旗杆", "号码布", "主席台"]
    
    # ========== 1. 预测阶段 ==========
    # 主分类器预测 + 温度缩放
    cls_probs = softmax_np(cls_logits / T_cls, axis=1)
    pred_labels = ["晨读" if p == 0 else "晨跑" for p in cls_probs.argmax(axis=1)]
    confidences = cls_probs.max(axis=1)
    
    # 特征预测 + 温度缩放 + 置信度裁剪（防止过度自信）
    probs = np.clip(sigmoid_np(raw_feat_logits / T_feat), 0, 0.95)
    
    # 号码布增强
    if enhance_number_bib:
        probs[:, 9] = np.clip(probs[:, 9] * 2, 0, 0.95)
    
    # ========== 2. 统计阶段 ==========
    miss = review = auto_pass = 0  # 漏检/待审核/自动通过计数
    rule1 = rule2 = rule3 = rule4 = default = 0  # 各规则触发计数
    
    # 额外统计指标
    read_pass = run_pass = 0  # 晨读/晨跑通过数
    read_total = run_total = 0  # 晨读/晨跑总数
    number_bib_scores = []  # 号码布得分记录
    
    for i, pl in enumerate(pred_labels):
        # ========== 2.1 特征匹配统计 ==========
        if pl == "晨读":
            # 晨读：统计匹配的特征数（需超过晨读阈值）
            matched = sum(1 for f in FEAT_READ if probs[i][FEAT_ALL.index(f)] > FT_read)
            read_total += 1
            if matched >= READ_TH:
                read_pass += 1
        else:
            # 晨跑：统计匹配的特征数（需超过晨跑阈值）
            matched = sum(1 for f in FEAT_RUN if probs[i][FEAT_ALL.index(f)] > FT_run)
            run_total += 1
            if matched >= RUN_TH:
                run_pass += 1
        
        # 记录号码布得分（用于分析）
        number_bib_scores.append(probs[i][9])
        
        # ========== 2.2 三支决策判断 ==========
        if use_rules:
            # 使用完整的三支决策规则
            # 规则1: 晨跑 + 高置信度 + 足够特征 → 自动通过
            if pl == "晨跑" and confidences[i] >= ALPHA_AUTO_PASS and matched >= RUN_TH:
                dec = "通过"
                rule1 += 1
            # 规则2: 晨读 + 高置信度 + 足够特征 → 自动通过
            elif pl == "晨读" and confidences[i] >= ALPHA_AUTO_PASS and matched >= READ_TH:
                dec = "通过"
                rule2 += 1
            # 规则3: 特征太少 → 待审核
            elif matched < MIN_FEATURES:
                dec = "审核"
                rule3 += 1
            # 规则4: 置信度太低 → 待审核
            elif confidences[i] < ALPHA_REVIEW:
                dec = "审核"
                rule4 += 1
            # 默认：满足基本条件就通过
            else:
                dec = "通过"
                default += 1
        else:
            # 简化版：只用置信度判断
            if confidences[i] >= ALPHA_REVIEW:
                dec = "通过"
            else:
                dec = "审核"
        
        # ========== 2.3 统计漏检 ==========
        # 漏检：真实标签是"异常"，但决策是"通过"（放行了不该放行的）
        if true_neg[i] and dec == "通过":
            miss += 1
        elif dec == "审核":
            review += 1
        else:
            auto_pass += 1
    
    total = len(pred_labels)
    
    # 计算号码布平均得分
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
    """
    格式化打印实验结果
    
    Args:
        name: 实验名称
        result: evaluate_system返回的结果字典
        expected: 预期结果说明（可选）
    """
    print(f"\n{'='*60}")
    print(f"【实验】{name}")
    print(f"{'='*60}")
    print(f"  漏检: {result['miss']} ({result['miss_rate']:.1f}%)")
    print(f"  通过: {result['pass']} ({result['pass_rate']:.1f}%)")
    print(f"  待审核: {result['review']} ({result['review_rate']:.1f}%)")
    
    # 打印各规则触发次数
    if result['rule1'] + result['rule2'] + result['rule3'] + result['rule4'] + result['default'] > 0:
        print(f"  规则触发: 规则1={result['rule1']}, 规则2={result['rule2']}, "
              f"规则3={result['rule3']}, 规则4={result['rule4']}, 默认={result['default']}")
    
    # 打印晨读/晨跑通过率
    if 'read_pass_rate' in result:
        diff = abs(result['read_pass_rate'] - result['run_pass_rate'])
        print(f"  晨读通过率: {result['read_pass_rate']:.1f}% ({result['read_total']}个样本)")
        print(f"  晨跑通过率: {result['run_pass_rate']:.1f}% ({result['run_total']}个样本)")
        print(f"  通过率差异: {diff:.1f}%")
    
    # 打印号码布得分
    if 'number_bib_mean' in result:
        print(f"  号码布平均得分: {result['number_bib_mean']:.3f} (阈值0.60)")
    
    if expected:
        print(f"  预期结果: {expected}")
    return result


# ==================== 基准配置 ====================
# 这是系统当前的默认配置
BASELINE = {
    "T_cls": 5.0, "T_feat": 1.8,        # 温度参数
    "FT_read": 0.66, "FT_run": 0.60,    # 特征阈值
    "ALPHA_AUTO_PASS": 0.80,             # 自动通过阈值
    "ALPHA_REVIEW": 0.85,               # 待审核阈值
    "READ_TH": 3, "RUN_TH": 5,          # 特征匹配阈值
    "MIN_FEATURES": 3,                   # 最少特征数
    "use_rules": True,                    # 启用规则
    "enhance_number_bib": True           # 号码布增强
}

# ==================== 实验一：温度参数设为1 ====================
# 验证温度参数的作用
# T=1.0时，softmax不做平滑，模型输出会更"自信"
# 预期：模型过于自信，晨读晨跑特征得分差异大
print("\n" + "="*80)
print("实验一：温度参数设为1（模型过于自信）")
print("="*80)
exp1 = evaluate_system(T_cls=1.0, T_feat=1.0, FT_read=0.66, FT_run=0.60,
                       ALPHA_AUTO_PASS=0.80, ALPHA_REVIEW=0.85,
                       READ_TH=3, RUN_TH=5, MIN_FEATURES=3,
                       use_rules=True, enhance_number_bib=True)
print_result("温度T=1.0", exp1, 
            "预期：模型过于自信，晨读晨跑特征得分差异大，晨读普遍通过，晨跑普遍待审核")

# ==================== 实验二：不使用规则约束 ====================
# 验证三支决策规则的作用
# 不用规则时，只用置信度判断
# 预期：漏检率上升，人工审核率变化
print("\n" + "="*80)
print("实验二：不使用三支决策规则，只用置信度判断")
print("="*80)
exp2 = evaluate_system(T_cls=5.0, T_feat=1.8, FT_read=0.66, FT_run=0.60,
                       ALPHA_AUTO_PASS=0.80, ALPHA_REVIEW=0.85,
                       READ_TH=3, RUN_TH=5, MIN_FEATURES=3,
                       use_rules=False, enhance_number_bib=True)
print_result("不使用规则", exp2, 
            "预期：漏检率上升，人工审核率变化")

# ==================== 实验三：统一特征阈值 ====================
# 验证分类别阈值是否必要
# 晨读晨跑使用相同的特征阈值(0.60)
# 预期：晨读晨跑通过率差异大（因为特征数量不同）
print("\n" + "="*80)
print("实验三：晨读晨跑使用统一特征阈值（0.60）")
print("="*80)
exp3 = evaluate_system(T_cls=5.0, T_feat=1.8, FT_read=0.60, FT_run=0.60,
                       ALPHA_AUTO_PASS=0.80, ALPHA_REVIEW=0.85,
                       READ_TH=3, RUN_TH=5, MIN_FEATURES=3,
                       use_rules=True, enhance_number_bib=True)
print_result("统一阈值FT=0.60", exp3, 
            "预期：晨读晨跑通过率差异大")

# ==================== 实验四：不使用号码布增强 ====================
# 验证号码布增强的作用
# 号码布是晨跑的标志性特征，增强它可以提高晨跑检出率
# 预期：号码布得分显著低于0.60（需要增强才能达到阈值）
print("\n" + "="*80)
print("实验四：不使用号码布增强")
print("="*80)
exp4 = evaluate_system(T_cls=5.0, T_feat=1.8, FT_read=0.66, FT_run=0.60,
                       ALPHA_AUTO_PASS=0.80, ALPHA_REVIEW=0.85,
                       READ_TH=3, RUN_TH=5, MIN_FEATURES=3,
                       use_rules=True, enhance_number_bib=False)
print_result("无号码布增强", exp4, 
            "预期：号码布得分显著低于0.60")

# ==================== 基准对比 ====================
print("\n" + "="*80)
print("【基准配置】")
print("="*80)
baseline = evaluate_system(**BASELINE)
print_result("基准配置", baseline)

# ==================== 汇总对比表 ====================
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
    print(f"{name:<22} | {result['miss_rate']:>5.1f}% {result['pass_rate']:>5.1f}% {result['review_rate']:>5.1f}% | "
          f"{result.get('read_pass_rate', 0):>5.1f}% {result.get('run_pass_rate', 0):>5.1f}% {diff:>5.1f}% | {number_bib:>7.3f}")

# ==================== 保存结果 ====================
output_file = BASE / "ablation_results.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("消融实验汇总对比\n")
    f.write("="*120 + "\n")
    f.write(f"{'实验':<22} | {'漏检%':<6} {'通过%':<6} {'审核%':<6} | {'晨读%':<6} {'晨跑%':<6} {'差异%':<6} | {'号码布':<8}\n")
    f.write("-"*120 + "\n")
    for name, result, base in all_results:
        diff = abs(result['read_pass_rate'] - result['run_pass_rate']) if 'read_pass_rate' in result else 0
        number_bib = result.get('number_bib_mean', 0)
        f.write(f"{name:<22} | {result['miss_rate']:>5.1f}% {result['pass_rate']:>5.1f}% {result['review_rate']:>5.1f}% | "
                f"{result.get('read_pass_rate', 0):>5.1f}% {result.get('run_pass_rate', 0):>5.1f}% {diff:>5.1f}% | {number_bib:>7.3f}\n")

print(f"\n结果已保存到: {output_file}")