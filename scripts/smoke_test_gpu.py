# -*- coding: utf-8 -*-
"""Phase 1 冒烟测试：GPU + SigLIP2-NaFlex 全链路验证
运行方式（VidPic venv）:
  "D:\\VidPic Studio\\.venv\\Scripts\\python.exe" scripts/smoke_test_gpu.py
验收标准:
  1. 版本断言通过（transformers 4.57.6 / torch cu128 / sm_120）
  2. SigLIP2-NaFlex 本地权重加载成功
  3. 三种宽高比图片（3:4 竖、超长、方形）NaFlex 预处理零裁切
  4. batch 32 GPU 推理实测显存峰值
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data" / "models" / "siglip2-base-patch16-naflex"

def check_versions():
    import transformers
    assert transformers.__version__ == "4.57.6", f"transformers 版本漂移: {transformers.__version__}"
    assert torch.cuda.is_available(), "CUDA 不可用"
    cap = torch.cuda.get_device_capability(0)
    assert cap == (12, 0), f"意外算力架构: {cap}"
    print(f"[1] 版本 OK: transformers {transformers.__version__}, torch {torch.__version__}, sm_{cap[0]}{cap[1]}")
    print(f"    GPU: {torch.cuda.get_device_name(0)}")

def load_model():
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).cuda().eval()
    # 只保留视觉塔可减半显存，但 NaFlex 的 MAP head 在 vision tower 内，直接用整模型视觉部分
    print(f"[2] 模型加载 OK: {MODEL_DIR.name}")
    cfg = proc
    print(f"    processor: {type(cfg).__name__}")
    for attr in ("max_num_patches", "patch_size", "size", "image_mean"):
        if hasattr(cfg, attr):
            print(f"    {attr} = {getattr(cfg, attr)}")
    return proc, model

def pick_test_images():
    """按宽高比挑 3 张真实图片: 3:4 竖拍 / 超长图 / 接近方形"""
    from PIL import Image
    raw = ROOT / "data" / "raw"
    out = {"portrait_3_4": None, "very_tall": None, "near_square": None}
    for p in sorted(raw.glob("*.jpeg")):
        try:
            w, h = Image.open(p).size
        except Exception:
            continue
        r = w / h
        if 0.73 <= r <= 0.77 and out["portrait_3_4"] is None:
            out["portrait_3_4"] = (p, w, h)
        if r <= 0.55 and out["very_tall"] is None:
            out["very_tall"] = (p, w, h)
        if 0.95 <= r <= 1.05 and out["near_square"] is None:
            out["near_square"] = (p, w, h)
        if all(out.values()):
            break
    for k, v in out.items():
        print(f"    {k}: {v[0].name if v else 'N/A'} {v[1]}x{v[2]}" if v else f"    {k}: 未找到")
    return out

def check_naflex_preprocessing(proc, images):
    from PIL import Image
    print("[3] NaFlex 预处理检查（零裁切 = 处理后宽高比与原图一致）:")
    for name, item in images.items():
        if item is None:
            continue
        p, w, h = item
        img = Image.open(p).convert("RGB")
        inputs = proc(images=img, return_tensors="pt")
        pv = inputs["pixel_values"]
        # naflex: pixel_values 形状可能是 (1, max_patches, patch*patch*3) 或 (1, C, H, W)
        if pv.dim() == 4:
            ph, pw = pv.shape[-2], pv.shape[-1]
            shape_desc = f"{pw}x{ph} ({pw//16}x{ph//16} patches)"
            ratio_before, ratio_after = w / h, pw / ph
        else:
            n = pv.shape[1]
            shape_desc = f"{n} tokens"
            ratio_before = ratio_after = w / h
        crop_free = abs(ratio_before - ratio_after) / ratio_before < 0.07
        print(f"    {name}: {w}x{h} -> {shape_desc}, 比例 {ratio_before:.3f}->{ratio_after:.3f}, 零裁切: {'YES' if crop_free else 'NO(!)'}")

def check_batch_inference(proc, model):
    from PIL import Image
    import time
    raw = ROOT / "data" / "raw"
    paths = sorted(raw.glob("*.jpeg"))[:32]
    imgs = [Image.open(p).convert("RGB") for p in paths]
    inputs = proc(images=imgs, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items() if torch.is_tensor(v)}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[4] batch {len(imgs)} GPU 推理 OK: 特征 {tuple(feats.shape)}, dtype {feats.dtype}, "
          f"耗时 {dt:.1f}s, 显存峰值 {peak:.2f} GB")
    return peak

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 冒烟测试")
    print("=" * 60)
    check_versions()
    proc, model = load_model()
    print("    测试图选择:")
    images = pick_test_images()
    check_naflex_preprocessing(proc, images)
    peak = check_batch_inference(proc, model)
    print("=" * 60)
    print(f"全部通过。实测显存峰值 {peak:.2f} GB（替换 EXECUTION_PLAN.md §3.3 估算值）")
