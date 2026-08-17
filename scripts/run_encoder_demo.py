# -*- coding: utf-8 -*-
"""Phase 5 验证：src/encoders 抽象端到端跑通（直接加载权重，非读 CSV）

演示 + 自检：
  1. 列出当前环境可用编码器（CLIP 依赖 openai/clip 包，缺失则跳过）。
  2. 对每个可用编码器，在测试集样本上直接 predict()，校验
     (cls_softmax (B,2), feat_sigmoid (B,11)) 形状与分类头精度。
  3. 演示 src/config.ENCODER 切换：改一行即可换编码器。

用法（项目根目录）：
    python scripts/run_encoder_demo.py            # 全部可用编码器
    python scripts/run_encoder_demo.py --encoder SigLIP
    python scripts/run_encoder_demo.py --n 50
"""
import os
import sys
import json
import argparse
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.encoders import get_encoder, available_encoders


def build_test_files(labels, split):
    out = []
    for fn in split.get("test_files", []):
        li = labels.get(fn, {})
        name = li.get("label", "未知")
        if name in ("晨读", "晨跑"):
            out.append((fn, 0 if name == "晨读" else 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=str, default=None,
                    help="指定单一编码器；默认跑全部可用")
    ap.add_argument("--n", type=int, default=60, help="测试集样本数（取前 N 张）")
    a = ap.parse_args()

    labels = json.load(open(ROOT / "data" / "labels.json", encoding="utf-8"))
    labels = labels.get("labels", labels)
    split = json.load(open(ROOT / "data" / "split_config.json", encoding="utf-8"))
    test_files = build_test_files(labels, split)[: a.n]
    print(f"测试样本: {len(test_files)} 张 | 设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    avail = available_encoders()
    print(f"当前环境可用编码器: {avail}")
    if a.encoder:
        if a.encoder not in avail:
            print(f"[skip] 指定编码器 {a.encoder} 不可用（依赖缺失）")
            return
        targets = [a.encoder]
    else:
        targets = avail

    raw_dir = ROOT / "data" / "raw"
    for name in targets:
        print("\n" + "=" * 64)
        print(f"编码器: {name}")
        print("=" * 64)
        try:
            if name == "ViT-Tiny":
                enc = get_encoder(name, device="cpu",
                                  seed=int(os.environ.get("VIT_TINY_SEED", 42)))
            else:
                enc = get_encoder(name, device="cpu")
        except Exception as e:
            print(f"  [fail] 实例化失败: {repr(e)[:200]}")
            continue

        imgs, gts = [], []
        for fn, gt in test_files:
            p = raw_dir / fn
            if not p.exists():
                continue
            try:
                imgs.append(Image.open(p).convert("RGB"))
                gts.append(gt)
            except Exception:
                pass
        if not imgs:
            print("  [skip] 无可用图片")
            continue

        cls, feat = enc.predict(imgs)
        print(f"  输出形状: cls={tuple(cls.shape)} feat={tuple(feat.shape)}")
        assert cls.shape == (len(imgs), 2), "cls 形状错误"
        assert feat.shape == (len(imgs), 11), "feat 形状错误"

        pred = cls.argmax(dim=1).tolist()
        correct = sum(1 for p, g in zip(pred, gts) if p == g)
        acc = correct / len(gts) * 100 if gts else 0.0
        conf = cls.max(dim=1).values.mean().item()
        fm = feat.mean().item()
        print(f"  分类头精度(样本内): {acc:.1f}% ({correct}/{len(gts)})")
        print(f"  平均置信度: {conf:.3f} | 平均特征概率: {fm:.3f}")
        print(f"  [ok] {name} 抽象接口跑通")


if __name__ == "__main__":
    main()
