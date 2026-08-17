# -*- coding: utf-8 -*-
"""Phase 2: 用 SigLIP2-NaFlex 批量重提全部图片特征（零裁切）

与 CLIP 版 (scripts/preprocessing.py) 的区别:
- 编码器: SigLIP2-NaFlex-B/16 (768维, 原生宽高比, 不裁切)
- 运行环境: VidPic venv (torch 2.11+cu128), GPU
- 输出: data/siglip_features.csv (filename + feat_0..feat_767), 不触碰旧CSV

用法:
  "D:\\VidPic Studio\\.venv\\Scripts\\python.exe" scripts/extract_siglip_features.py
  可选环境变量:
    NAFLEX_MAX_PATCHES  token预算上限, 默认 256 (B/16支持最高512)
"""
import os
import sys
import time
from pathlib import Path

import torch
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "data" / "models" / "siglip2-base-patch16-naflex"
RAW_DIR = ROOT / "data" / "raw"
MAX_PATCHES = int(os.environ.get("NAFLEX_MAX_PATCHES", 256))
OUT_NAME = os.environ.get("NAFLEX_OUT_CSV", "siglip_features.csv")
BATCH = 32

def main():
    import transformers
    assert transformers.__version__ == "4.57.6", "transformers 版本漂移"
    assert torch.cuda.is_available(), "需要 GPU"

    from transformers import AutoModel, AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(MODEL_DIR, max_num_patches=MAX_PATCHES)
    model = AutoModel.from_pretrained(MODEL_DIR, dtype=torch.float16).cuda().eval()

    out_csv = ROOT / "data" / OUT_NAME
    if out_csv.exists():
        done = set(pd.read_csv(out_csv, usecols=["filename"])["filename"])
        print(f"[resume] 已有 {len(done)} 条，增量模式")
    else:
        done = set()

    files = sorted([p for p in RAW_DIR.glob("*.jpeg") if p.name not in done])
    print(f"待提取: {len(files)} 张 | token上限: {MAX_PATCHES} | batch: {BATCH}")

    rows, names = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(files), BATCH):
            batch_paths = files[i:i + BATCH]
            imgs = []
            for p in batch_paths:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception as e:
                    print(f"  [skip] {p.name}: {e}")
            if not imgs:
                continue
            inputs = proc(images=imgs, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items() if torch.is_tensor(v)}
            feats = model.get_image_features(**inputs)  # (B, 768) fp16
            feats = feats.float().cpu()
            for p, f in zip(batch_paths, feats):
                rows.append(f.numpy())
                names.append(p.name)
            if (i // BATCH) % 10 == 0:
                dt = time.time() - t0
                print(f"  {i + len(batch_paths)}/{len(files)}  ({dt:.0f}s)")

    if names:
        df_new = pd.DataFrame(rows, columns=[f"feat_{j}" for j in range(rows[0].shape[0])])
        df_new.insert(0, "filename", names)
        if out_csv.exists():
            df_old = pd.read_csv(out_csv)
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        df_new.to_csv(out_csv, index=False)
        print(f"完成: {len(df_new)} 条特征 -> {out_csv.name} (dim={df_new.shape[1]-1})")
    else:
        print("无新图片需要提取")

if __name__ == "__main__":
    main()
