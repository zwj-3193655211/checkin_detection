"""
PT转CSV脚本

将 clip_features_cpu.pt 转换为 clip_features_cpu.csv
"""
import torch
import pandas as pd
import os
from pathlib import Path

# 设置路径
DATA_DIR = Path(__file__).parent
PT_FILE = DATA_DIR / "cache" / "features" / "clip_features_cpu.pt"
CSV_FILE = DATA_DIR / "clip_features_cpu.csv"


def convert_pt_to_csv(pt_file=None, csv_file=None):
    """将PT文件转换为CSV文件"""
    pt_file = Path(pt_file) if pt_file else PT_FILE
    csv_file = Path(csv_file) if csv_file else CSV_FILE

    if not pt_file.exists():
        print(f"错误: PT文件不存在: {pt_file}")
        return False

    print(f"加载PT文件: {pt_file}")
    data = torch.load(pt_file)
    features = data['features']
    filenames = data['filenames']

    print(f"PT文件包含 {len(filenames)} 个特征")

    # 转换为DataFrame
    df = pd.DataFrame(
        features.numpy(),
        columns=[f'feat_{i}' for i in range(512)]
    )
    df.insert(0, 'filename', filenames)

    # 保存CSV
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_file, index=False)

    print(f"CSV保存完成: {csv_file}")
    print(f"共 {len(df)} 行")

    return True


if __name__ == "__main__":
    convert_pt_to_csv()
