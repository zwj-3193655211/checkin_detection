"""
数据预处理脚本

功能:
1. 图像质量检查 - 检查图片尺寸、亮度、格式等
2. 数据集统计分析 - 统计学生数、日期分布、质量问题等
3. CLIP特征预提取 - 使用CLIP模型提取图片特征向量
4. 数据集划分 - 按类别分层采样划分训练集/验证集/测试集
5. 特征分布统计 - 统计各特征在数据集中的分布情况
"""

# ============================================================
# 导入必要的库
# ============================================================
import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# ============================================================
# 路径配置
# ============================================================
# 项目根目录 = scripts的父目录
PROJECT_ROOT = Path(__file__).parent.parent
# 将src目录添加到Python路径，以便导入项目模块
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 数据目录配置
DATA_DIR = PROJECT_ROOT / "data"          # 数据根目录
RAW_DIR = DATA_DIR / "raw"                # 原始图片目录
CACHE_DIR = DATA_DIR / "cache" / "features"  # CLIP特征缓存目录
PICTURE_DIR = DATA_DIR / "raw"            # 图片目录（与RAW_DIR相同）

# ============================================================
# 特征定义
# ============================================================
# 11维特征名称列表（与标注文件中的特征名对应）
FEATURE_NAMES = [
    '人脸',      # 0  - 检测是否有人脸
    '蓝色课桌',  # 1  - 检测蓝色课桌（晨读特征）
    '投影幕布',  # 2  - 检测投影幕布（晨读特征）
    '讲台',      # 3  - 检测讲台（晨读特征）
    '书架/书籍', # 4  - 检测书架或书籍（晨读特征）
    '教室环境',  # 5  - 检测整体教室环境（晨读特征）
    '跑道',      # 6  - 检测跑道（晨跑特征）
    '运动场地',  # 7  - 检测运动场地（晨跑特征）
    '号码布',    # 8  - 检测号码布（晨跑特征）
    '运动鞋',    # 9  - 检测运动鞋（晨跑特征）
    '户外环境',  # 10 - 检测户外环境（晨跑特征）
]


# ============================================================
# 函数1: 检查单张图片质量
# ============================================================
def check_image_quality(image_path: str) -> dict:
    """
    检查单张图片的质量指标

    质量指标包括:
    - 图片尺寸（宽、高、宽高比）
    - 图片格式（JPEG、PNG等）
    - 亮度（平均像素值）
    - 文件大小

    Args:
        image_path: 图片文件的完整路径

    Returns:
        dict: 包含各项质量指标的字典
              valid: 图片是否可正常打开
              width/height: 图片尺寸
              mean_brightness: 平均亮度
              is_too_dark/is_too_bright: 亮度是否异常
              is_reasonable_size: 尺寸是否合理
              error: 如果打开失败，返回错误信息
    """
    try:
        # 打开图片
        img = Image.open(image_path)

        # 获取图片基本信息
        width, height = img.size                              # 图片宽高
        aspect_ratio = width / height                         # 宽高比
        format_type = img.format                              # 图片格式（JPEG/PNG等）

        # 将图片转换为RGB模式（某些图片可能是灰度或RGBA模式）
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 将图片转换为numpy数组，计算像素统计信息
        img_array = np.array(img)
        # 计算平均亮度（所有像素的均值，范围0-255）
        mean_brightness = np.mean(img_array)

        # 判断亮度是否异常
        # 亮度 < 30: 过暗，可能曝光不足或夜间拍摄
        # 亮度 > 225: 过亮，可能过度曝光
        is_too_dark = mean_brightness < 30
        is_too_bright = mean_brightness > 225

        # 检查图片尺寸是否合理
        # 最小边 < 100px: 太小，细节可能不清晰
        # 最大边 > 5000px: 太大，可能影响处理速度
        min_dimension = min(width, height)
        max_dimension = max(width, height)
        is_reasonable_size = min_dimension >= 100 and max_dimension <= 5000

        # 获取文件大小（字节）
        file_size = os.path.getsize(image_path)

        # 返回质量检查结果
        return {
            'valid': True,                    # 图片可正常打开
            'width': width,                   # 宽度（像素）
            'height': height,                 # 高度（像素）
            'aspect_ratio': aspect_ratio,     # 宽高比
            'format': format_type,            # 图片格式
            'mode': img.mode,                 # 颜色模式
            'mean_brightness': float(mean_brightness),  # 平均亮度
            'is_too_dark': is_too_dark,       # 是否过暗
            'is_too_bright': is_too_bright,   # 是否过亮
            'is_reasonable_size': is_reasonable_size,  # 尺寸是否合理
            'file_size': file_size,           # 文件大小（字节）
            'error': None                     # 无错误
        }
    except Exception as e:
        # 图片打开失败时返回错误信息
        return {
            'valid': False,
            'error': str(e),
            'width': 0,
            'height': 0,
        }


def analyze_dataset(output_file: str = None) -> dict:
    """
    分析整个数据集的统计信息

    该函数遍历数据集中的所有图片，进行以下统计:
    1. 基本信息: 图片总数、学生数、日期分布
    2. 质量检查: 损坏图片、过暗/过亮、尺寸异常等
    3. 尺寸统计: 宽高分布、宽高比分布
    4. 亮度统计: 平均亮度分布

    Args:
        output_file: 可选，统计结果保存的JSON文件名

    Returns:
        dict: 包含所有统计信息的字典
    """
    # 定义支持的图片格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    # 获取所有图片文件列表
    image_files = [
        f for f in os.listdir(RAW_DIR)
        if f.lower().endswith(image_extensions)
    ]

    print(f"Analyzing {len(image_files)} images...")

    # 初始化统计结果字典
    stats = {
        'total': len(image_files),        # 图片总数
        'by_student': defaultdict(int),  # 按学号统计（学号 -> 图片数）
        'by_date': defaultdict(int),      # 按日期统计（日期 -> 图片数）
        'by_size': defaultdict(int),      # 按尺寸统计（尺寸 -> 图片数）
        'quality_issues': {               # 质量问题列表
            'corrupted': [],              # 损坏的图片
            'too_dark': [],               # 过暗的图片
            'too_bright': [],            # 过亮的图片
            'too_small': [],             # 尺寸过小的图片
            'too_large': [],             # 尺寸过大的图片
        },
        'size_distribution': [],           # 尺寸分布列表
        'brightness_distribution': [],      # 亮度分布列表
        'students': set(),                # 学生学号集合
        'dates': set(),                   # 日期集合
    }

    # 遍历每张图片进行质量检查和统计
    for filename in tqdm(image_files, desc="Analyzing"):
        # --------------------------------------------------------
        # 1. 解析文件名，提取学号和日期
        # 文件名格式: {学号}-{日期}.jpeg
        # 例如: 36-2026-04-22.jpeg
        # --------------------------------------------------------
        # 去除扩展名，按'-'分割
        parts = filename.replace('.jpeg', '').replace('.jpg', '').split('-')
        if len(parts) >= 3:
            student_id = parts[0]                     # 学号（第1部分）
            date = '-'.join(parts[1:3])               # 日期（第2-3部分，如2026-04-22）
            stats['students'].add(student_id)        # 添加到学生集合
            stats['dates'].add(date)                   # 添加到日期集合
            stats['by_student'][student_id] += 1      # 该学号的图片数+1

        # --------------------------------------------------------
        # 2. 检查图片质量
        # --------------------------------------------------------
        image_path = os.path.join(PICTURE_DIR, filename)
        quality = check_image_quality(image_path)

        if quality['valid']:
            # 图片有效，记录尺寸和亮度分布
            stats['size_distribution'].append({
                'width': quality['width'],
                'height': quality['height'],
                'aspect_ratio': quality['aspect_ratio']
            })
            stats['brightness_distribution'].append(quality['mean_brightness'])

            # 检查并记录质量问题
            if quality['is_too_dark']:
                stats['quality_issues']['too_dark'].append(filename)
            if quality['is_too_bright']:
                stats['quality_issues']['too_bright'].append(filename)
            if not quality['is_reasonable_size']:
                # 区分尺寸过小和尺寸过大
                if quality['width'] < 100 or quality['height'] < 100:
                    stats['quality_issues']['too_small'].append(filename)
                else:
                    stats['quality_issues']['too_large'].append(filename)
        else:
            # 图片损坏，记录错误
            stats['quality_issues']['corrupted'].append(filename)

    # --------------------------------------------------------
    # 3. 转换数据类型（set -> list）以便JSON序列化
    # --------------------------------------------------------
    stats['by_student'] = dict(stats['by_student'])
    stats['by_date'] = dict(stats['by_date'])
    stats['students'] = list(stats['students'])
    stats['dates'] = sorted(list(stats['dates']))

    # --------------------------------------------------------
    # 4. 计算统计摘要
    # --------------------------------------------------------
    if stats['size_distribution']:
        # 提取宽高数组
        widths = [s['width'] for s in stats['size_distribution']]
        heights = [s['height'] for s in stats['size_distribution']]

        # 计算宽高和亮度的统计指标（均值、标准差、最小、最大）
        stats['summary'] = {
            'width_mean': float(np.mean(widths)),      # 平均宽度
            'width_std': float(np.std(widths)),        # 宽度标准差
            'width_min': int(np.min(widths)),          # 最小宽度
            'width_max': int(np.max(widths)),          # 最大宽度
            'height_mean': float(np.mean(heights)),    # 平均高度
            'height_std': float(np.std(heights)),      # 高度标准差
            'height_min': int(np.min(heights)),        # 最小高度
            'height_max': int(np.max(heights)),       # 最大高度
            'brightness_mean': float(np.mean(stats['brightness_distribution'])),  # 平均亮度
            'brightness_std': float(np.std(stats['brightness_distribution'])),    # 亮度标准差
            'num_students': len(stats['students']),   # 学生人数
            'num_dates': len(stats['dates']),          # 打卡天数
            'valid_images': len(stats['size_distribution']),      # 有效图片数
            'corrupted_images': len(stats['quality_issues']['corrupted']),  # 损坏图片数
        }
    else:
        stats['summary'] = {}

    # --------------------------------------------------------
    # 5. 保存统计结果到JSON文件
    # --------------------------------------------------------
    if output_file:
        output_path = DATA_DIR / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Statistics saved to {output_path}")

    return stats


def extract_clip_features(
    batch_size: int = 32,
    device: str = None,
    force_recompute: bool = False
) -> str:
    """
    使用CLIP模型提取图片特征向量并缓存

    CLIP (Contrastive Language-Image Pre-training) 是OpenAI提出的视觉-语言预训练模型。
    该函数使用CLIP ViT-B/32模型将图片转换为512维的特征向量，供后续MLP分类器使用。

    处理流程:
    1. 加载CLIP模型
    2. 读取所有图片并预处理（缩放、裁剪、归一化）
    3. 分批提取特征向量
    4. 保存到缓存文件

    Args:
        batch_size: 每批处理的图片数量，默认32
        device: 运行设备，'cuda'（GPU）或'cpu'，默认自动选择
        force_recompute: 是否强制重新计算（忽略缓存），默认False

    Returns:
        str: 缓存文件的完整路径

    注意:
        缓存文件格式: data/cache/features/clip_features_{device}.pt
        包含: features (torch.Tensor), filenames (list), model (str), device (str)
    """
    # 导入CLIP库和PyTorch数据加载相关模块
    import clip
    from torch.utils.data import Dataset, DataLoader

    # 自动选择设备：有GPU用GPU，没有用CPU
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Extracting CLIP features on {device}")

    # --------------------------------------------------------
    # 1. 创建缓存目录和缓存文件路径
    # --------------------------------------------------------
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"clip_features_{device}.pt"

    # --------------------------------------------------------
    # 2. 检查缓存是否存在
    # --------------------------------------------------------
    if cache_file.exists() and not force_recompute:
        print(f"Cache exists at {cache_file}, skipping...")
        return str(cache_file)

    # --------------------------------------------------------
    # 3. 加载CLIP模型
    # --------------------------------------------------------
    # ViT-B/32: Vision Transformer Base模型，patch size为32
    # 返回: model (CLIP模型), preprocess (图片预处理函数)
    model, preprocess = clip.load('ViT-B/32', device=device)
    model.eval()  # 设置为评估模式（禁用dropout等）

    # --------------------------------------------------------
    # 4. 获取所有图片文件列表
    # --------------------------------------------------------
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = sorted([
        f for f in os.listdir(PICTURE_DIR)
        if f.lower().endswith(image_extensions)
    ])

    print(f"Processing {len(image_files)} images...")

    # 存储所有特征和文件名
    all_features = []
    all_filenames = []

    # --------------------------------------------------------
    # 5. 分批处理图片
    # --------------------------------------------------------
    for i in tqdm(range(0, len(image_files), batch_size), desc="Extracting features"):
        # 获取当前批次的文件列表
        batch_files = image_files[i:i+batch_size]
        batch_images = []

        # 加载并预处理当前批次的图片
        for filename in batch_files:
            image_path = os.path.join(PICTURE_DIR, filename)
            try:
                # 打开图片并转换为RGB
                img = Image.open(image_path).convert('RGB')
                # 使用CLIP的预处理函数（Resize + CenterCrop + Normalize）
                img_tensor = preprocess(img)
                batch_images.append(img_tensor)
                all_filenames.append(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

        # --------------------------------------------------------
        # 6. 批量提取特征
        # --------------------------------------------------------
        if batch_images:
            # 将图片张量堆叠成批次 (batch_size, 3, 224, 224)
            batch_tensor = torch.stack(batch_images).to(device)

            # 禁用梯度计算（节省显存和计算时间）
            with torch.no_grad():
                # 使用CLIP的encode_image提取特征
                # 输出形状: (batch_size, 512)
                features = model.encode_image(batch_tensor)
                # 将特征移到CPU保存
                all_features.append(features.cpu())

    # --------------------------------------------------------
    # 7. 合并所有批次的特征
    # 最终形状: (total_images, 512)
    # --------------------------------------------------------
    all_features = torch.cat(all_features, dim=0)

    # --------------------------------------------------------
    # 8. 保存缓存
    # --------------------------------------------------------
    cache_data = {
        'features': all_features,         # 特征张量 (N, 512)
        'filenames': all_filenames,         # 文件名列表
        'model': 'ViT-B/32',              # 模型名称
        'device': device,                  # 运行设备
    }
    torch.save(cache_data, cache_file)
    print(f"Features saved to {cache_file}")
    print(f"Shape: {all_features.shape}")  # 打印特征维度

    return str(cache_file)


def split_dataset(
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    output_file: str = "split_config.json"
) -> dict:
    """
    划分训练集/验证集/测试集（按类别分层采样）

    使用分层采样策略，确保每个集合中各类别的比例与整体数据集一致。

    设计考量:
    - 正常类别（晨读/晨跑）：按比例划分到训练集、验证集、测试集
    - 异常类别：全部放入验证集和测试集（训练集不包含），用于测试模型对异常的识别能力

    Args:
        train_ratio: 训练集比例，默认0.7（70%）
        val_ratio: 验证集比例，默认0.15（15%）
        test_ratio: 测试集比例，默认0.15（15%）
        output_file: 输出配置文件名，默认"split_config.json"

    Returns:
        dict: 包含划分信息的字典
              train_files: 训练集文件名列表
              val_files: 验证集文件名列表
              test_files: 测试集文件名列表
              statistics: 各类别数量统计
              ratios: 划分比例

    注意:
        随机种子固定为42，确保划分结果可复现
    """
    import random

    # --------------------------------------------------------
    # 1. 获取所有图片文件列表
    # --------------------------------------------------------
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = sorted([
        f for f in os.listdir(PICTURE_DIR)
        if f.lower().endswith(image_extensions)
    ])

    # --------------------------------------------------------
    # 2. 加载标注文件获取类别信息
    # --------------------------------------------------------
    labels_file = DATA_DIR / "labels.json"
    labels = {}
    if labels_file.exists():
        with open(labels_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            labels = data.get('labels', data)  # 兼容两种格式

    print(f"已加载 {len(labels)} 个标注")

    # --------------------------------------------------------
    # 3. 按类别分组
    # --------------------------------------------------------
    category_files = {
        '晨读': [],     # 晨读图片
        '晨跑': [],     # 晨跑图片
        '异常': [],     # 异常图片
        '未知': []      # 未标注的图片
    }

    for filename in image_files:
        # 从标注文件中获取该图片的类别
        label_info = labels.get(filename, {})
        category = label_info.get('label', '未知')  # 默认标记为"未知"
        if category not in category_files:
            category = '未知'
        category_files[category].append(filename)

    # 输出各类别数量
    print("\n各类别样本数:")
    for cat, files in category_files.items():
        print(f"  {cat}: {len(files)}")

    # --------------------------------------------------------
    # 4. 分层划分
    # --------------------------------------------------------
    train_files = []  # 训练集
    val_files = []    # 验证集
    test_files = []   # 测试集

    # 固定随机种子，确保结果可复现
    random.seed(42)

    for category, files in category_files.items():
        # 随机打乱顺序
        random.shuffle(files)
        n = len(files)

        # --------------------------------------------------------
        # 4.1 异常/未知类别：只划分到验证集和测试集
        # 设计原因：训练时不学习异常样本，让模型专注于正常特征
        # --------------------------------------------------------
        if category == '异常' or category == '未知':
            # 计算验证集和测试集的分割点
            # 比例: val_ratio : test_ratio = val : (val+test)
            n_val = int(n * val_ratio / (val_ratio + test_ratio))
            n_test = n - n_val  # 剩余的放入测试集

            val_files.extend(files[:n_val])    # 前n_val个放入验证集
            test_files.extend(files[n_val:])   # 剩余的放入测试集

            print(f"\n{category} 划分（训练集不包含）:")
            print(f"  训练: 0, 验证: {n_val}, 测试: {n_test}")

        # --------------------------------------------------------
        # 4.2 正常类别（晨读/晨跑）：按比例划分到三个集合
        # --------------------------------------------------------
        else:
            # 计算各集合的大小
            n_train = int(n * train_ratio)  # 训练集大小
            n_val = int(n * val_ratio)      # 验证集大小
            # 测试集 = 剩余的
            n_test = len(files) - n_train - n_val

            # 按顺序分割
            train_files.extend(files[:n_train])                        # 前n_train个 -> 训练集
            val_files.extend(files[n_train:n_train+n_val])             # 中间n_val个 -> 验证集
            test_files.extend(files[n_train+n_val:])                    # 剩余 -> 测试集

            print(f"\n{category} 划分:")
            print(f"  训练: {n_train}, 验证: {n_val}, 测试: {n_test}")

    # --------------------------------------------------------
    # 5. 打乱最终列表（让各类别混合在一起）
    # --------------------------------------------------------
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    # --------------------------------------------------------
    # 6. 统计各集合中的类别分布
    # --------------------------------------------------------
    def count_categories(file_list):
        """统计文件列表中各类别的数量"""
        counts = {'晨读': 0, '晨跑': 0, '异常': 0, '未知': 0}
        for f in file_list:
            label_info = labels.get(f, {})
            cat = label_info.get('label', '未知')
            if cat not in counts:
                cat = '未知'
            counts[cat] += 1
        return counts

    train_counts = count_categories(train_files)
    val_counts = count_categories(val_files)
    test_counts = count_categories(test_files)

    # --------------------------------------------------------
    # 7. 构建输出结果
    # --------------------------------------------------------
    split_config = {
        'train_files': train_files,        # 训练集文件列表
        'val_files': val_files,            # 验证集文件列表
        'test_files': test_files,          # 测试集文件列表
        'statistics': {                    # 统计信息
            'train': len(train_files),    # 训练集大小
            'val': len(val_files),        # 验证集大小
            'test': len(test_files),      # 测试集大小
            'total': len(image_files),    # 总大小
            'train_categories': train_counts,     # 训练集类别分布
            'val_categories': val_counts,         # 验证集类别分布
            'test_categories': test_counts,        # 测试集类别分布
        },
        'ratios': {                      # 划分比例
            'train': train_ratio,         # 训练集比例
            'val': val_ratio,             # 验证集比例
            'test': test_ratio,           # 测试集比例
        }
    }

    # --------------------------------------------------------
    # 8. 保存划分结果到JSON文件
    # --------------------------------------------------------
    output_path = DATA_DIR / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(split_config, f, ensure_ascii=False, indent=2)

    # 输出统计信息
    print(f"\nDataset split saved to {output_path}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    print(f"\n训练集类别分布: {train_counts}")
    print(f"验证集类别分布: {val_counts}")
    print(f"测试集类别分布: {test_counts}")

    return split_config


def analyze_feature_distribution(output_file: str = None) -> dict:
    """
    统计各特征在数据集中的分布情况（基于标注文件）

    该函数从标注文件中读取所有图片的特征标注，统计:
    1. 类别分布：晨读、晨跑、异常各有几张
    2. 场景分布：morning_reading、morning_running、abnormal各有几张
    3. 特征分布：各特征（人脸、教室、跑道等）在数据集中出现的次数

    Args:
        output_file: 可选，统计结果保存的JSON文件名

    Returns:
        dict: 包含各类别、场景、特征统计信息的字典
    """
    # --------------------------------------------------------
    # 1. 加载标注文件
    # --------------------------------------------------------
    labels_file = DATA_DIR / "labels.json"
    if not labels_file.exists():
        print(f"标注文件不存在: {labels_file}")
        return {}

    with open(labels_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    labels = data.get('labels', {})  # 获取标注字典
    print(f"已加载 {len(labels)} 个标注")

    # --------------------------------------------------------
    # 2. 初始化统计变量
    # --------------------------------------------------------
    total_images = len(labels)  # 总图片数

    # 特征统计：每个特征的 有/无/未标注 数量
    feature_stats = defaultdict(lambda: {'true_count': 0, 'false_count': 0, 'missing_count': 0})

    # 类别统计：晨读/晨跑/异常 数量
    label_stats = defaultdict(int)

    # 场景统计：morning_reading/morning_running/abnormal 数量
    scene_stats = defaultdict(int)

    # --------------------------------------------------------
    # 3. 遍历所有标注，统计各类别、场景、特征
    # --------------------------------------------------------
    for filename, label_info in labels.items():
        # 统计类别（如"晨读"、"晨跑"、"异常"）
        label = label_info.get('label', '未知')
        label_stats[label] += 1

        # 统计场景（如"morning_reading"、"morning_running"）
        scene = label_info.get('scene', '未知')
        scene_stats[scene] += 1

        # 统计特征（遍历所有可能的特征名）
        features = label_info.get('features', {})
        all_feature_names = [
            '人脸', '蓝色桌子', '教室', '投影幕布', '跑道',
            '天空', '绿地', '树木', '旗杆', '号码布', '主席台'
        ]

        for feature_name in all_feature_names:
            if feature_name in features:
                # 特征有标注
                if features[feature_name]:
                    feature_stats[feature_name]['true_count'] += 1   # 有该特征
                else:
                    feature_stats[feature_name]['false_count'] += 1  # 无该特征
            else:
                # 特征未标注
                feature_stats[feature_name]['missing_count'] += 1

    # --------------------------------------------------------
    # 4. 构建输出结果
    # --------------------------------------------------------
    results = {
        'total_images': total_images,
        'label_distribution': {},     # 类别分布
        'scene_distribution': {},     # 场景分布
        'feature_distribution': {}    # 特征分布
    }

    # --------------------------------------------------------
    # 5. 输出统计结果
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("数据集特征分布统计")
    print("=" * 60)

    # 5.1 类别分布
    print("\n【类别分布】")
    print("-" * 40)
    for label, count in sorted(label_stats.items(), key=lambda x: -x[1]):
        percentage = count / total_images * 100
        results['label_distribution'][label] = {'count': count, 'percentage': percentage}
        print(f"  {label}: {count} 张 ({percentage:.1f}%)")

    # 5.2 场景分布
    print("\n【场景分布】")
    print("-" * 40)
    for scene, count in sorted(scene_stats.items(), key=lambda x: -x[1]):
        percentage = count / total_images * 100
        results['scene_distribution'][scene] = {'count': count, 'percentage': percentage}
        print(f"  {scene}: {count} 张 ({percentage:.1f}%)")

    # 5.3 特征分布
    print("\n【特征分布】")
    print("-" * 40)
    # 打印表头
    print(f"{'特征名称':<12} {'有':<8} {'无':<8} {'未标注':<8} {'有(%)':<8}")
    print("-" * 60)

    for feature_name in ['人脸', '蓝色桌子', '教室', '投影幕布', '跑道', '天空', '绿地', '树木', '旗杆', '号码布', '主席台']:
        stats = feature_stats[feature_name]
        true_count = stats['true_count']      # 有该特征的图片数
        false_count = stats['false_count']   # 无该特征的图片数
        missing_count = stats['missing_count']  # 未标注的图片数
        true_pct = true_count / total_images * 100 if total_images > 0 else 0  # 有该特征的比例

        results['feature_distribution'][feature_name] = {
            'true_count': true_count,
            'false_count': false_count,
            'missing_count': missing_count,
            'percentage': true_pct
        }

        print(f"  {feature_name:<10} {true_count:<8} {false_count:<8} {missing_count:<8} {true_pct:.1f}%")

    print("\n" + "=" * 60)

    # --------------------------------------------------------
    # 6. 保存结果到JSON文件
    # --------------------------------------------------------
    if output_file:
        output_path = DATA_DIR / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"统计结果已保存到 {output_path}")

    return results


def main():
    """
    主函数：命令行入口

    支持的运行模式:
    - analyze: 仅分析数据集
    - extract_features: 仅提取CLIP特征
    - split: 仅划分数据集
    - features: 仅统计特征分布
    - all: 执行所有步骤（默认）

    命令行参数:
    --mode: 运行模式 (默认: all)
    --batch_size: CLIP特征提取的批次大小 (默认: 32)
    --force: 是否强制重新计算（忽略缓存）(默认: False)
    --device: 运行设备，cuda/cpu (默认: 自动选择)

    使用示例:
    python preprocessing.py --mode all                    # 执行所有步骤
    python preprocessing.py --mode features              # 仅统计特征
    python preprocessing.py --mode extract --force       # 强制重新提取特征
    """
    # --------------------------------------------------------
    # 1. 定义命令行参数
    # --------------------------------------------------------
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument('--mode', type=str, default='all',
                        choices=['analyze', 'extract_features', 'split', 'features', 'all'],
                        help='运行模式: analyze(分析数据集), extract_features(提取特征), split(划分数据), features(特征统计), all(全部)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='CLIP特征提取的批次大小，默认32')
    parser.add_argument('--force', action='store_true',
                        help='强制重新计算（忽略缓存）')
    parser.add_argument('--device', type=str, default=None,
                        help='运行设备，如cuda或cpu，默认自动选择')

    # 解析命令行参数
    args = parser.parse_args()

    # --------------------------------------------------------
    # 2. 根据模式执行相应的功能
    # --------------------------------------------------------
    if args.mode == 'analyze' or args.mode == 'all':
        print("\n=== Step 1: 分析数据集 ===")
        analyze_dataset('dataset_stats.json')

    if args.mode == 'extract_features' or args.mode == 'all':
        print("\n=== Step 2: 提取CLIP特征 ===")
        extract_clip_features(
            batch_size=args.batch_size,
            device=args.device,
            force_recompute=args.force
        )

    if args.mode == 'split' or args.mode == 'all':
        print("\n=== Step 3: 划分数据集 ===")
        split_dataset()

    if args.mode == 'features' or args.mode == 'all':
        print("\n=== Step 4: 统计特征分布 ===")
        analyze_feature_distribution('feature_stats.json')

    print("\n=== 完成 ===")


# ============================================================
# 程序入口
# ============================================================
# 当直接运行此脚本时，调用main函数
# 当作为模块导入时，不会自动执行
if __name__ == '__main__':
    main()
