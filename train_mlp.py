"""
统一训练脚本 - 训练双MLP模型

功能:
1. 训练二分类器MLPClassifier（晨读/晨跑）
2. 训练优化版特征预测器MLPFeaturesOptimized（11维特征）

核心改进:
- 特征预测器使用ConfidenceRegularizedLoss防止过度自信
- 推理时使用温度缩放降低置信度
- 使用Focal Loss处理类别不平衡
- 特征级数据增强（高斯噪声）
- 早停机制防止过拟合
- 三支决策端到端验证

训练流程:
1. 加载CLIP特征和标注数据
2. 划分训练集/验证集/测试集
3. 训练主分类器（二分类）
4. 训练特征预测器（11维）
5. 在测试集上评估模型性能

作者: AI Assistant
"""

# ==================== PyTorch和数值计算库 ====================
import os
import torch
import torch.nn as nn
import numpy as np

# ==================== 数据处理 ====================
import pandas as pd
import json
import os
from pathlib import Path

# ==================== 项目内部模块 ====================
import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.models.mlp import MLPClassifier
from src.models.mlp_features_optimized import MLPFeaturesOptimized, ConfidenceRegularizedLoss

# 从config导入训练参数
from src.config import (
    CLASSIFIER_TEMPERATURE as TEMPERATURE,  # 主分类器温度参数
    FEATURE_TEMPERATURE,                     # 特征预测器温度参数
    FEATURE_THRESHOLD_READ,                  # 晨读特征阈值
    FEATURE_THRESHOLD_RUN,                   # 晨跑特征阈值
    ALPHA_AUTO_PASS,                         # 自动通过置信度阈值
    ALPHA_REVIEW,                            # 待审核置信度阈值
    MIN_FEATURES,                            # 最少特征数
    RUN_FEATURE_THRESH,                      # 晨跑特征匹配阈值
    READ_FEATURE_THRESH,                     # 晨读特征匹配阈值
)

# ==================== 随机种子 ====================
# 设置随机种子确保实验可复现（可用 CHECKIN_SEED 环境变量覆盖，多seed实验用）
SEED = int(os.environ.get('CHECKIN_SEED', 42))
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==================== 特征索引定义 ====================
# 特征索引将特征名称映射到输出向量的位置
FEATURE_INDEX = {
    "人脸": 0, "蓝色桌子": 1, "教室": 2, "投影幕布": 3,
    "跑道": 4, "天空": 5, "绿地": 6, "树木": 7,
    "旗杆": 8, "号码布": 9, "主席台": 10,
}

# 类别与特征对应关系
CLASS_FEATURES = {
    '晨读': ['人脸', '蓝色桌子', '教室', '投影幕布'],
    '晨跑': ['人脸', '跑道', '天空', '绿地', '树木', '旗杆', '号码布', '主席台']
}

# ====== 参数已统一从 src/config.py 导入，修改 config.py 即可全局生效 ======


def _get_feature_threshold(idx):
    """
    获取特征阈值：晨读特征(idx 0-3)用0.66，晨跑特征(idx 4-10)用0.60
    
    晨读场景特征较少（4个），使用较高阈值确保准确性
    晨跑场景特征较多（8个），使用稍低阈值提高检出率
    
    Args:
        idx: 特征索引 (0-10)
    
    Returns:
        float: 对应特征的阈值
    """
    return FEATURE_THRESHOLD_READ if idx < 4 else FEATURE_THRESHOLD_RUN


def validate_with_three_way_decision(classifier, features_model, X_val, y_main_val, y_features_val, epoch=0, is_final=False):
    """
    使用三支决策进行端到端验证
    
    三支决策将预测分为三类：
    - 自动通过：模型高置信度 + 特征匹配良好
    - 待审核：模型置信度不足或特征匹配差
    - 拒绝/异常：明确判定为异常
    
    该函数模拟完整的三支决策流程，统计：
    - 漏检率：异常样本被错误放行的比例
    - 审核率：需要人工审核的比例
    - 自动通过率：正常样本被正确放行的比例
    
    Args:
        classifier: 主分类器模型
        features_model: 特征预测器模型
        X_val: 验证集特征
        y_main_val: 验证集主标签（0=晨读，1=晨跑，-1=异常）
        y_features_val: 验证集特征标签（11维one-hot）
        epoch: 当前训练轮次（用于打印）
        is_final: 是否为最终评估（打印完整信息）
    
    Returns:
        dict: 验证指标字典，包含漏检率、审核率、各规则触发次数等
    """
    classifier.eval()
    features_model.eval()

    with torch.no_grad():
        # 主分类器预测
        logits = classifier(X_val.float())
        scaled_probs = torch.softmax(logits / TEMPERATURE, dim=1)
        confidences, preds = scaled_probs.max(dim=1)

        # 特征预测
        feat_probs = features_model(X_val.float(), inference=True)

    # 统计计数
    normal_mask = (y_main_val >= 0)   # 正常样本掩码（晨读或晨跑）
    anomaly_mask = (y_main_val == -1)  # 异常样本掩码
    normal_total = normal_mask.sum().item()
    anomaly_total = anomaly_mask.sum().item()

    # 各规则触发计数
    rule1_count = 0  # 晨跑高置信+足够特征 -> 自动通过
    rule2_count = 0  # 晨读高置信+足够特征 -> 自动通过
    rule3_count = 0  # 特征太少 -> 待审核
    rule4_count = 0  # 置信度太低 -> 待审核
    review_count = 0  # 待审核总数
    miss_count = 0   # 漏检数（异常被放行）
    review_normal = 0  # 待审核中的正常样本数

    for i in range(len(X_val)):
        pred_label = '晨读' if preds[i] == 0 else '晨跑'
        confidence = confidences[i].item()
        true_label = y_main_val[i].item()

        # 统计该样本匹配的特征数
        class_feats = CLASS_FEATURES[pred_label]
        matched = [f for f, fidx in FEATURE_INDEX.items()
                   if feat_probs[i, fidx].item() > _get_feature_threshold(fidx) and f in class_feats]
        matched_count = len(matched)

        # 三支决策规则判断
        r1 = pred_label == '晨跑' and confidence >= ALPHA_AUTO_PASS and matched_count >= RUN_FEATURE_THRESH
        r2 = pred_label == '晨读' and confidence >= ALPHA_AUTO_PASS and matched_count >= READ_FEATURE_THRESH
        r3 = matched_count < MIN_FEATURES
        r4 = confidence < ALPHA_REVIEW

        if r1:
            decision = 0
            rule1_count += 1
        elif r2:
            decision = 0
            rule2_count += 1
        elif r3:
            decision = 1
            rule3_count += 1
        elif r4:
            decision = 1
            rule4_count += 1
        else:
            decision = 0

        is_anomaly = (true_label == -1)
        is_review = (decision == 1)

        # 漏检：异常样本被放行
        if is_anomaly and not is_review:
            miss_count += 1
        # 待审核计数
        if is_review:
            review_count += 1
            if not is_anomaly:
                review_normal += 1

    # 计算各项指标
    miss_rate = miss_count / anomaly_total * 100 if anomaly_total > 0 else 0
    review_rate = review_count / len(X_val) * 100
    normal_pass = normal_total - review_normal
    normal_pass_rate = normal_pass / len(X_val) * 100 if len(X_val) > 0 else 0

    # 定期打印或最终打印详细信息
    if is_final or (epoch % 10 == 0 and epoch > 0):
        print(f"\n  [三支决策验证]")
        print(f"    漏检率: {miss_rate:.2f}% (漏检{miss_count}/{anomaly_total})")
        print(f"    审核率: {review_rate:.2f}% (待审核{review_count}/{len(X_val)})")
        print(f"    自动通过率: {normal_pass_rate:.2f}% (通过{normal_pass}/{len(X_val)})")
        print(f"    规则触发: 规则1={rule1_count}, 规则2={rule2_count}, "
              f"规则3={rule3_count}, 规则4={rule4_count}")

    return {
        'miss_rate': miss_rate,
        'review_rate': review_rate,
        'normal_pass_rate': normal_pass_rate,
        'rule1': rule1_count,
        'rule2': rule2_count,
        'rule3': rule3_count,
        'rule4': rule4_count,
        'miss_count': miss_count
    }


def load_data():
    """
    加载特征和标签数据
    
    数据来源:
    - clip_features_cpu.csv: 所有图片的CLIP特征向量
    - labels.json: 图片的标注信息（标签+特征）
    - split_config.json: 数据集划分配置
    
    Returns:
        tuple: (特征张量, 文件名列表, 标签字典, 划分配置字典)
    """
    project_root = Path(__file__).parent

    # 加载特征（默认CLIP缓存，可用 CHECKIN_FEATURES_CSV 切换到其他编码器的特征）
    csv_name = os.environ.get('CHECKIN_FEATURES_CSV', 'clip_features_cpu.csv')
    df = pd.read_csv(project_root / 'data' / csv_name)
    print(f"[load_data] 特征文件: {csv_name} (dim={df.shape[1]-1})")
    filenames = df['filename'].tolist()
    features = torch.tensor(df.drop('filename', axis=1).values, dtype=torch.float32)

    # 加载标签数据
    with open(project_root / 'data' / 'labels.json', 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
        labels = labels_data.get('labels', labels_data)

    # 加载划分配置
    with open(project_root / 'data' / 'split_config.json', 'r', encoding='utf-8') as f:
        split_config = json.load(f)

    return features, filenames, labels, split_config


def prepare_data(features, filenames, labels, split_config):
    """
    准备训练数据
    
    根据split_config将数据划分为训练集、验证集、测试集，
    并将标签信息转换为模型需要的格式。
    
    Args:
        features: 所有图片的特征张量 (N, 512)
        filenames: 文件名列表
        labels: 标签字典 {文件名: {label: str, features: dict}}
        split_config: 划分配置 {train_files: [], val_files: [], test_files: []}
    
    Returns:
        dict: 包含train/val/test三个键的字典，每个键对应一个元组
              (X, y_main, y_features, filenames)
    """
    # 构建特征字典：文件名 -> 特征向量
    feature_dict = {fname: features[i] for i, fname in enumerate(filenames)}

    def prepare_split(file_list, is_train=False):
        """
        准备单个数据划分
        
        Args:
            file_list: 该划分包含的文件名列表
            is_train: 是否为训练集（训练集会过滤掉异常样本）
        """
        X, y_main, y_features, filenames_list = [], [], [], []

        for fname in file_list:
            if fname in feature_dict:
                label_info = labels.get(fname, {})
                label_name = label_info.get('label', '未知')

                # 训练集只包含晨读和晨跑，不包含异常
                if is_train and label_name not in ['晨读', '晨跑']:
                    continue

                # 主标签编码：0=晨读，1=晨跑，-1=异常
                if label_name == '晨读':
                    main_label = 0
                elif label_name == '晨跑':
                    main_label = 1
                else:
                    main_label = -1

                # 构建11维特征标签（one-hot形式）
                feature_label = [0] * 11
                saved_features = label_info.get('features', {})

                for feature_key, feature_idx in FEATURE_INDEX.items():
                    if saved_features.get(feature_key, False):
                        feature_label[feature_idx] = 1

                # 如果没有任何标注的特征，根据主标签推断
                if sum(feature_label) == 0:
                    if saved_features.get('人脸', False):
                        feature_label[0] = 1
                    if label_name == '晨读':
                        for key, idx in [('蓝色桌子', 1), ('教室', 2), ('投影幕布', 3)]:
                            if saved_features.get(key, False):
                                feature_label[idx] = 1
                    elif label_name == '晨跑':
                        for key, idx in [('跑道', 4), ('天空', 5), ('绿地', 6), ('树木', 7),
                                        ('旗杆', 8), ('号码布', 9), ('主席台', 10)]:
                            if saved_features.get(key, False):
                                feature_label[idx] = 1

                X.append(feature_dict[fname])
                y_main.append(main_label)
                y_features.append(feature_label)
                filenames_list.append(fname)

        X = torch.stack(X)
        y_main = torch.tensor(y_main, dtype=torch.long)
        y_features = torch.tensor(y_features, dtype=torch.float32)

        return X, y_main, y_features, filenames_list

    # 准备三个数据划分
    X_train, y_main_train, y_features_train, _ = prepare_split(split_config['train_files'], is_train=True)
    X_val, y_main_val, y_features_val, _ = prepare_split(split_config['val_files'], is_train=False)
    X_test, y_main_test, y_features_test, _ = prepare_split(split_config['test_files'], is_train=False)

    # 打印数据统计
    print(f"\n训练集: {len(X_train)} 样本")
    print(f"  晨读: {(y_main_train == 0).sum().item()}, 晨跑: {(y_main_train == 1).sum().item()}")

    print(f"验证集: {len(X_val)} 样本")
    val_normal = (y_main_val >= 0).sum().item()
    val_anomaly = (y_main_val == -1).sum().item()
    print(f"  正常: {val_normal}, 异常: {val_anomaly}")

    print(f"测试集: {len(X_test)} 样本")
    test_normal = (y_main_test >= 0).sum().item()
    test_anomaly = (y_main_test == -1).sum().item()
    print(f"  正常: {test_normal}, 异常: {test_anomaly}")

    return {
        'train': (X_train, y_main_train, y_features_train),
        'val': (X_val, y_main_val, y_features_val),
        'test': (X_test, y_main_test, y_features_test)
    }


def augment_features(X, noise_std=0.02):
    """
    对CLIP特征添加高斯噪声作为数据增强
    
    数据增强可以提高模型泛化能力，通过人为增加训练数据的多样性。
    高斯噪声是一种常用的特征级增强方法。
    
    Args:
        X: 输入特征张量
        noise_std: 噪声标准差（默认为0.02）
    
    Returns:
        张量: 添加噪声后的特征
    """
    if noise_std > 0:
        noise = torch.randn_like(X) * noise_std
        return X + noise
    return X


def train_classifier(X_train, y_train, X_val, y_val, output_path='data/mlp_classifier.pt'):
    """
    训练二分类器MLP（晨读/晨跑）
    
    使用标准交叉熵损失训练，验证集准确率作为模型选择标准。
    
    训练配置:
    - 优化器: Adam (lr=0.001, weight_decay=1e-4)
    - 学习率调度: StepLR (每20轮降低50%)
    - 早停: 验证准确率不再提升时保存最佳模型
    
    Args:
        X_train, y_train: 训练集特征和标签
        X_val, y_val: 验证集特征和标签
        output_path: 模型保存路径
    
    Returns:
        训练好的模型
    """
    print("\n" + "=" * 60)
    print("训练二分类器（晨读/晨跑）")
    print("=" * 60)

    # 过滤训练集和验证集中的正常样本（只保留晨读和晨跑）
    train_mask = (y_train >= 0)
    X_train_filtered = X_train[train_mask]
    y_train_filtered = y_train[train_mask]

    val_mask = (y_val >= 0)
    X_val_filtered = X_val[val_mask]
    y_val_filtered = y_val[val_mask]

    # 创建模型（input_dim 从特征维度自动推导，兼容512/768等）
    model = MLPClassifier(input_dim=X_train.shape[1], hidden_dim=256, output_dim=2, dropout=0.3)
    print(f"\n模型结构:\n{model}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    epochs = 100
    batch_size = 32
    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        indices = torch.randperm(len(X_train_filtered))

        # 小批量训练
        for i in range(0, len(X_train_filtered), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_filtered[batch_idx]
            y_batch = y_train_filtered[batch_idx]

            outputs = model(X_batch.float())
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_filtered.float())
            val_pred = val_outputs.argmax(dim=1)
            val_acc = (val_pred == y_val_filtered).sum().item() / len(y_val_filtered)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(X_train_filtered)*batch_size:.4f} "
                  f"Val Acc: {val_acc*100:.2f}%")

    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), output_path)
    print(f"\n[OK] 二分类器已保存: {output_path}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")

    return model


def train_features(X_train, y_train, X_val, y_val, y_main_val, classifier, output_path='data/mlp_features_optimized.pt'):
    """
    训练优化版特征预测器
    
    特征预测器是一个多标签分类模型，输出11维特征的存在概率。
    使用ConfidenceRegularizedLoss防止模型过度自信。
    
    训练策略:
    - 损失函数: ConfidenceRegularizedLoss = Focal Loss - α×熵正则化
    - 优化器: AdamW (lr=1e-3, weight_decay=1e-4)
    - 学习率调度: CosineAnnealingWarmRestarts (T_0=15, T_mult=2)
    - 早停: 验证损失25轮不下降则停止
    - 梯度裁剪: 最大范数5.0防止梯度爆炸
    - 数据增强: 高斯噪声 (std=0.02)
    
    Args:
        X_train, y_train: 训练集特征和特征标签(11维one-hot)
        X_val, y_val: 验证集特征和特征标签
        y_main_val: 验证集主标签（用于三支决策验证）
        classifier: 已训练的主分类器（用于三支决策验证）
        output_path: 模型保存路径
    
    Returns:
        训练好的特征预测器模型
    """
    print("\n" + "=" * 60)
    print("训练优化版特征预测器")
    print("目标: 提高准确性，降低过高置信度")
    print("=" * 60)

    # 创建模型（input_dim 从特征维度自动推导，兼容512/768等）
    model = MLPFeaturesOptimized(
        input_dim=X_train.shape[1],
        hidden_dim=512,
        output_dim=11,
        dropout=0.25,
        temperature=FEATURE_TEMPERATURE
    )
    print(f"\n模型结构:\n{model}")

    # 使用带置信度正则化的损失函数
    # Focal Loss处理类别不平衡，熵正则化防止过度自信
    criterion = ConfidenceRegularizedLoss(alpha=0.15, gamma=2.0)

    # AdamW优化器 + 余弦退火学习率调度
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    # 早停配置
    patience = 25          # 容忍25轮验证损失不下降
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_miss_rate = float('inf')

    # 训练配置
    epochs = 250
    batch_size = 32
    noise_std = 0.02      # 数据增强：添加高斯噪声

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        indices = torch.randperm(len(X_train))

        # 小批量训练
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            # 数据增强：添加高斯噪声
            X_batch = augment_features(X_train[batch_idx], noise_std)
            y_batch = y_train[batch_idx]

            # 前向传播
            outputs = model(X_batch.float(), inference=False)
            loss = criterion(outputs, y_batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.float(), inference=False)
            val_loss = criterion(val_outputs, y_val).item()

            val_probs = model(X_val.float(), inference=True)
            val_preds = (val_probs > 0.5).float()
            val_acc = (val_preds == y_val).float().mean().item()

        # 早停检查：验证损失是否改善
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_miss_rate = float('inf')
            patience_counter = 0
        else:
            patience_counter += 1

        # 三支决策验证（每10轮或最后一轮）
        val_metrics = validate_with_three_way_decision(classifier, model, X_val, y_main_val, y_val, epoch=epoch+1)

        # 记录最佳漏检率对应的模型
        if val_metrics['miss_rate'] < best_miss_rate:
            best_miss_rate = val_metrics['miss_rate']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

        # 定期打印训练信息
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            print(f"  训练损失: {total_loss/len(X_train)*batch_size:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  验证准确率: {val_acc*100:.2f}%")
            print(f"  当前学习率: {scheduler.get_last_lr()[0]:.6f}")

        # 早停判断
        if patience_counter >= patience:
            print(f"\n早停触发，当前轮次: {epoch+1}")
            break

    # 加载最佳模型并保存
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), output_path)
    print(f"\n[OK] 优化版特征预测器已保存: {output_path}")

    return model


def evaluate_model(classifier, features_model, X_test, y_main_test, y_features_test):
    """
    在测试集上评估模型性能
    
    评估内容:
    1. 正常样本（晨读+晨跑）分类准确率
    2. 异常样本检测统计
    3. 各特征预测准确率和平均置信度
    4. 三支决策端到端验证结果
    
    Args:
        classifier: 主分类器
        features_model: 特征预测器
        X_test, y_main_test: 测试集主标签
        y_features_test: 测试集特征标签
    """
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)

    classifier.eval()
    features_model.eval()

    with torch.no_grad():
        # 主分类器预测
        logits = classifier(X_test.float())
        probs = torch.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)

        # 特征预测
        feat_probs = features_model(X_test.float(), inference=True)
        feat_preds = (feat_probs > 0.5).float()

    # 分离正常和异常样本
    normal_mask = (y_main_test >= 0)
    anomaly_mask = (y_main_test == -1)

    # 正常样本准确率
    if normal_mask.sum() > 0:
        normal_acc = (preds[normal_mask] == y_main_test[normal_mask]).float().mean().item()
        print(f"\n正常样本（晨读+晨跑）准确率: {normal_acc*100:.2f}%")

    # 异常样本检测统计
    if anomaly_mask.sum() > 0:
        anomaly_confs = confidences[anomaly_mask]
        print(f"\n异常样本检测:")
        print(f"  异常样本数: {len(anomaly_confs)}")
        print(f"  平均置信度: {anomaly_confs.mean().item():.4f}")
        print(f"  置信度范围: [{anomaly_confs.min().item():.4f}, {anomaly_confs.max().item():.4f}]")

    # 特征预测准确率
    print(f"\n特征预测准确率: {(feat_preds == y_features_test).float().mean().item()*100:.2f}%")

    print(f"\n各特征准确率:")
    for feature_name, feature_idx in FEATURE_INDEX.items():
        feat_acc = (feat_preds[:, feature_idx] == y_features_test[:, feature_idx]).float().mean().item()
        avg_prob = feat_probs[:, feature_idx].mean().item()
        print(f"  {feature_name}: 准确率={feat_acc*100:.2f}%, 平均置信度={avg_prob:.4f}")

    # 三支决策端到端验证
    print("\n" + "=" * 60)
    print("【三支决策端到端验证】")
    print("=" * 60)
    final_metrics = validate_with_three_way_decision(classifier, features_model, X_test, y_main_test, y_features_test, is_final=True)
    print(f"\n  规则说明:")
    print(f"    规则1: 晨跑 且 置信度 >= {ALPHA_AUTO_PASS} 且 特征数 >= {RUN_FEATURE_THRESH} → 自动通过")
    print(f"    规则2: 晨读 且 置信度 >= {ALPHA_AUTO_PASS} 且 特征数 >= {READ_FEATURE_THRESH} → 自动通过")
    print(f"    规则3: 特征数 < {MIN_FEATURES} → 待审核")
    print(f"    规则4: 置信度 < {ALPHA_REVIEW} → 待审核")


def main():
    """
    主训练流程
    
    执行步骤:
    1. 加载数据和模型配置
    2. 准备训练/验证/测试集
    3. 训练主分类器（二分类）
    4. 训练特征预测器（11维）
    5. 在测试集上评估模型
    """
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("=" * 60)
    print("统一训练脚本 - 训练双MLP模型")
    print("=" * 60)

    # 1. 加载数据
    features, filenames, labels, split_config = load_data()
    
    # 2. 准备数据集
    dataset = prepare_data(features, filenames, labels, split_config)

    X_train, y_main_train, y_features_train = dataset['train']
    X_val, y_main_val, y_features_val = dataset['val']
    X_test, y_main_test, y_features_test = dataset['test']

    # 3. 训练主分类器（输出文件名可用 CHECKIN_MODEL_SUFFIX 加后缀，避免覆盖基线）
    suffix = os.environ.get('CHECKIN_MODEL_SUFFIX', '')
    classifier = train_classifier(X_train, y_main_train, X_val, y_main_val,
                                  output_path=f'data/mlp_classifier{suffix}.pt')

    # 4. 训练特征预测器
    features_model = train_features(X_train, y_features_train, X_val, y_features_val, y_main_val, classifier,
                                    output_path=f'data/mlp_features_optimized{suffix}.pt')

    # 5. 评估模型
    evaluate_model(classifier, features_model, X_test, y_main_test, y_features_test)

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
