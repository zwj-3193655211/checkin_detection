"""
数据预处理脚本

功能:
1. 图像质量检查
2. 数据集统计分析
3. CLIP特征预提取
4. 数据集划分
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache" / "features"


def check_image_quality(image_path: str) -> dict:
    """
    检查单张图片质量

    Returns:
        dict: 包含质量指标
    """
    try:
        img = Image.open(image_path)

        # 基本信息
        width, height = img.size
        aspect_ratio = width / height
        format_type = img.format

        # 转换为RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 计算图像统计信息
        img_array = np.array(img)
        mean_brightness = np.mean(img_array)

        # 检查是否有异常（过暗或过亮）
        is_too_dark = mean_brightness < 30
        is_too_bright = mean_brightness > 225

        # 检查尺寸是否合理
        min_dimension = min(width, height)
        max_dimension = max(width, height)
        is_reasonable_size = min_dimension >= 100 and max_dimension <= 5000

        # 文件大小
        file_size = os.path.getsize(image_path)

        return {
            'valid': True,
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'format': format_type,
            'mode': img.mode,
            'mean_brightness': float(mean_brightness),
            'is_too_dark': is_too_dark,
            'is_too_bright': is_too_bright,
            'is_reasonable_size': is_reasonable_size,
            'file_size': file_size,
            'error': None
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'width': 0,
            'height': 0,
        }


def analyze_dataset(output_file: str = None) -> dict:
    """
    分析整个数据集

    Returns:
        dict: 统计信息
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    image_files = [
        f for f in os.listdir(RAW_DIR)
        if f.lower().endswith(image_extensions)
    ]

    print(f"Analyzing {len(image_files)} images...")

    stats = {
        'total': len(image_files),
        'by_student': defaultdict(int),
        'by_date': defaultdict(int),
        'by_size': defaultdict(int),
        'quality_issues': {
            'corrupted': [],
            'too_dark': [],
            'too_bright': [],
            'too_small': [],
            'too_large': [],
        },
        'size_distribution': [],
        'brightness_distribution': [],
        'students': set(),
        'dates': set(),
    }

    for filename in tqdm(image_files, desc="Analyzing"):
        # 解析学号和日期
        # 格式: {学号}-{日期}.jpeg
        parts = filename.replace('.jpeg', '').replace('.jpg', '').split('-')
        if len(parts) >= 3:
            student_id = parts[0]
            date = '-'.join(parts[1:3])  # YYYY-MM-DD
            stats['students'].add(student_id)
            stats['dates'].add(date)
            stats['by_student'][student_id] += 1

        image_path = os.path.join(PICTURE_DIR, filename)
        quality = check_image_quality(image_path)

        if quality['valid']:
            stats['size_distribution'].append({
                'width': quality['width'],
                'height': quality['height'],
                'aspect_ratio': quality['aspect_ratio']
            })
            stats['brightness_distribution'].append(quality['mean_brightness'])

            if quality['is_too_dark']:
                stats['quality_issues']['too_dark'].append(filename)
            if quality['is_too_bright']:
                stats['quality_issues']['too_bright'].append(filename)
            if not quality['is_reasonable_size']:
                if quality['width'] < 100 or quality['height'] < 100:
                    stats['quality_issues']['too_small'].append(filename)
                else:
                    stats['quality_issues']['too_large'].append(filename)
        else:
            stats['quality_issues']['corrupted'].append(filename)

    # 转换为普通dict
    stats['by_student'] = dict(stats['by_student'])
    stats['by_date'] = dict(stats['by_date'])
    stats['students'] = list(stats['students'])
    stats['dates'] = sorted(list(stats['dates']))

    # 计算统计摘要
    if stats['size_distribution']:
        widths = [s['width'] for s in stats['size_distribution']]
        heights = [s['height'] for s in stats['size_distribution']]
        stats['summary'] = {
            'width_mean': float(np.mean(widths)),
            'width_std': float(np.std(widths)),
            'width_min': int(np.min(widths)),
            'width_max': int(np.max(widths)),
            'height_mean': float(np.mean(heights)),
            'height_std': float(np.std(heights)),
            'height_min': int(np.min(heights)),
            'height_max': int(np.max(heights)),
            'brightness_mean': float(np.mean(stats['brightness_distribution'])),
            'brightness_std': float(np.std(stats['brightness_distribution'])),
            'num_students': len(stats['students']),
            'num_dates': len(stats['dates']),
            'valid_images': len(stats['size_distribution']),
            'corrupted_images': len(stats['quality_issues']['corrupted']),
        }
    else:
        stats['summary'] = {}

    # 保存结果
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
    预提取CLIP特征到缓存

    Args:
        batch_size: 批处理大小
        device: 设备类型
        force_recompute: 是否强制重新计算

    Returns:
        str: 缓存文件路径
    """
    import clip
    from torch.utils.data import Dataset, DataLoader

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Extracting CLIP features on {device}")

    # 创建缓存目录
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"clip_features_{device}.pt"

    # 检查缓存
    if cache_file.exists() and not force_recompute:
        print(f"Cache exists at {cache_file}, skipping...")
        return str(cache_file)

    # 加载CLIP模型
    model, preprocess = clip.load('ViT-B/32', device=device)
    model.eval()

    # 获取所有图片
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = sorted([
        f for f in os.listdir(PICTURE_DIR)
        if f.lower().endswith(image_extensions)
    ])

    print(f"Processing {len(image_files)} images...")

    all_features = []
    all_filenames = []

    # 分批处理
    for i in tqdm(range(0, len(image_files), batch_size), desc="Extracting features"):
        batch_files = image_files[i:i+batch_size]
        batch_images = []

        for filename in batch_files:
            image_path = os.path.join(PICTURE_DIR, filename)
            try:
                img = Image.open(image_path).convert('RGB')
                img_tensor = preprocess(img)
                batch_images.append(img_tensor)
                all_filenames.append(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

        if batch_images:
            batch_tensor = torch.stack(batch_images).to(device)

            with torch.no_grad():
                features = model.encode_image(batch_tensor)
                all_features.append(features.cpu())

    # 合并所有特征
    all_features = torch.cat(all_features, dim=0)

    # 保存缓存
    cache_data = {
        'features': all_features,
        'filenames': all_filenames,
        'model': 'ViT-B/32',
        'device': device,
    }
    torch.save(cache_data, cache_file)
    print(f"Features saved to {cache_file}")
    print(f"Shape: {all_features.shape}")

    return str(cache_file)


def split_dataset(
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    output_file: str = "split_config.json"
) -> dict:
    """
    划分训练集/验证集/测试集

    Args:
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        output_file: 输出配置文件

    Returns:
        dict: 划分信息
    """
    import random

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = sorted([
        f for f in os.listdir(PICTURE_DIR)
        if f.lower().endswith(image_extensions)
    ])

    # 按学号分组，保持各学号的数据分布
    student_images = defaultdict(list)
    for filename in image_files:
        parts = filename.replace('.jpeg', '').replace('.jpg', '').split('-')
        if len(parts) >= 1:
            student_id = parts[0]
            student_images[student_id].append(filename)

    # 分层划分
    train_files = []
    val_files = []
    test_files = []

    random.seed(42)

    for student_id, files in student_images.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train+n_val])
        test_files.extend(files[n_train+n_val:])

    split_config = {
        'train_files': train_files,
        'val_files': val_files,
        'test_files': test_files,
        'statistics': {
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files),
            'total': len(image_files),
        },
        'ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio,
        }
    }

    # 保存
    output_path = DATA_DIR / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(split_config, f, ensure_ascii=False, indent=2)

    print(f"Dataset split saved to {output_path}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    return split_config


def main():
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument('--mode', type=str, default='all',
                        choices=['analyze', 'extract_features', 'split', 'all'],
                        help='运行模式')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--force', action='store_true', help='强制重新计算')
    parser.add_argument('--device', type=str, default=None, help='设备类型')

    args = parser.parse_args()

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

    print("\n=== 完成 ===")


if __name__ == '__main__':
    main()
