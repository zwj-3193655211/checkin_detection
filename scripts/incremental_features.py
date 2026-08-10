"""
增量提取CLIP特征：只处理缓存中缺失的图片，合并后保存并更新CSV。

用法（checkin_detection 环境）:
    python scripts/incremental_features.py
"""
import os
import sys
from pathlib import Path

import torch
from PIL import Image

import clip

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache" / "features"
CACHE_FILE = CACHE_DIR / "clip_features_cpu.pt"
CSV_FILE = DATA_DIR / "clip_features_cpu.csv"

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 加载旧缓存
    if CACHE_FILE.exists():
        cache = torch.load(CACHE_FILE, map_location='cpu')
        old_features = cache['features']
        old_filenames = cache['filenames']
        print(f"旧缓存: {len(old_filenames)} 张")
    else:
        old_features = torch.empty(0, 512)
        old_filenames = []

    # 找出缺失的图片
    all_images = sorted(
        f for f in os.listdir(RAW_DIR)
        if f.lower().endswith(IMAGE_EXTS)
    )
    have = set(old_filenames)
    new_images = [f for f in all_images if f not in have]
    print(f"需要增量提取: {len(new_images)} 张 (总共 {len(all_images)})")

    if not new_images:
        print("没有新图，跳过。")
        return

    # 加载CLIP
    model, preprocess = clip.load('ViT-B/32', device=device)
    model.eval()

    new_features = []
    new_fnames = []
    with torch.no_grad():
        for i, fname in enumerate(new_images):
            path = RAW_DIR / fname
            try:
                img = Image.open(path).convert('RGB')
                x = preprocess(img).unsqueeze(0).to(device)
                feat = model.encode_image(x).cpu()
                new_features.append(feat)
                new_fnames.append(fname)
            except Exception as e:
                print(f"失败: {fname}: {e}")
            if (i + 1) % 5 == 0:
                print(f"  进度 {i + 1}/{len(new_images)}")

    if new_features:
        new_features = torch.cat(new_features, dim=0)
        all_features = torch.cat([old_features, new_features], dim=0)
        all_filenames = old_filenames + new_fnames
        print(f"合并后: {all_features.shape} 共 {len(all_filenames)} 张")

        # 保存缓存
        cache_data = {
            'features': all_features,
            'filenames': all_filenames,
            'model': 'ViT-B/32',
            'device': 'cpu',
        }
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(cache_data, CACHE_FILE)
        print(f"缓存已保存: {CACHE_FILE}")

        # 更新CSV
        import pandas as pd
        df = pd.DataFrame(
            all_features.numpy(),
            columns=[f'feat_{i}' for i in range(512)]
        )
        df.insert(0, 'filename', all_filenames)
        df.to_csv(CSV_FILE, index=False)
        print(f"CSV已更新: {CSV_FILE} 共 {len(df)} 行")
    else:
        print("没有成功提取任何特征")


if __name__ == '__main__':
    main()
