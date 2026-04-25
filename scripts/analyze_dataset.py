import os
import sys
import json
from pathlib import Path
from PIL import Image
import torch
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def analyze_image(image_path: str) -> dict:
    """分析单张图片的基本信息"""
    img = Image.open(image_path)
    width, height = img.size
    mode = img.mode

    file_size = os.path.getsize(image_path) / 1024  # KB

    return {
        'filename': os.path.basename(image_path),
        'width': width,
        'height': height,
        'mode': mode,
        'size_kb': round(file_size, 2),
        'aspect_ratio': round(width / height, 3) if height > 0 else 0,
    }


def analyze_dataset(data_dir: str) -> dict:
    """分析整个数据集"""
    from tqdm import tqdm

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(image_extensions)
    ]

    stats = {
        'total_images': len(image_files),
        'widths': [],
        'heights': [],
        'sizes_kb': [],
        'aspect_ratios': [],
        'samples': [],
    }

    print(f"Analyzing {len(image_files)} images...")

    for filename in tqdm(image_files):
        image_path = os.path.join(data_dir, filename)
        try:
            info = analyze_image(image_path)
            stats['widths'].append(info['width'])
            stats['heights'].append(info['height'])
            stats['sizes_kb'].append(info['size_kb'])
            stats['aspect_ratios'].append(info['aspect_ratio'])
            if len(stats['samples']) < 10:
                stats['samples'].append(info)
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")

    # 计算统计信息
    stats['width_mean'] = np.mean(stats['widths']) if stats['widths'] else 0
    stats['width_std'] = np.std(stats['widths']) if stats['widths'] else 0
    stats['width_min'] = min(stats['widths']) if stats['widths'] else 0
    stats['width_max'] = max(stats['widths']) if stats['widths'] else 0

    stats['height_mean'] = np.mean(stats['heights']) if stats['heights'] else 0
    stats['height_std'] = np.std(stats['heights']) if stats['heights'] else 0
    stats['height_min'] = min(stats['heights']) if stats['heights'] else 0
    stats['height_max'] = max(stats['heights']) if stats['heights'] else 0

    stats['size_mean'] = np.mean(stats['sizes_kb']) if stats['sizes_kb'] else 0
    stats['size_std'] = np.std(stats['sizes_kb']) if stats['sizes_kb'] else 0

    return stats


def print_stats(stats: dict):
    """打印统计信息"""
    print("\n" + "="*60)
    print("数据集统计分析")
    print("="*60)

    print(f"\n总图片数: {stats['total_images']}")

    print(f"\n图像尺寸:")
    print(f"  宽度: {stats['width_min']:.0f} - {stats['width_max']:.0f} px")
    print(f"  平均: {stats['width_mean']:.1f} ± {stats['width_std']:.1f} px")
    print(f"  高度: {stats['height_min']:.0f} - {stats['height_max']:.0f} px")
    print(f"  平均: {stats['height_mean']:.1f} ± {stats['height_std']:.1f} px")

    print(f"\n文件大小:")
    print(f"  平均: {stats['size_mean']:.1f} ± {stats['size_std']:.1f} KB")

    print(f"\n样本示例:")
    for sample in stats['samples']:
        print(f"  {sample['filename']}: {sample['width']}x{sample['height']}, {sample['size_kb']}KB")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="分析签到图片数据集")
    parser.add_argument('--data_dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--output', type=str, default=None, help='输出JSON文件')

    args = parser.parse_args()

    stats = analyze_dataset(args.data_dir)
    print_stats(stats)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n统计结果已保存到: {args.output}")


if __name__ == '__main__':
    main()