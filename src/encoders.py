# -*- coding: utf-8 -*-
"""src/encoders.py — 统一编码器抽象（Phase 5 系统接入核心）
====================================================================

把三条「图片 -> (二分类 softmax, 11 维特征 sigmoid)」管线统一到一个接口，
配合 src/config.ENCODER 切换，解决：
  (R1) 摆脱对通用 CLIP 的强依赖，可换用专精自训模型；
  (R2) 原生宽高比零裁切（SigLIP NaFlex / ViT-Tiny letterbox）；
  (R3) 让对比实验中胜出的编码器可直接接入部署系统。

接口约定：
    encoder.predict(images) -> (cls_softmax (B,2), feat_sigmoid (B,11))
      images: list[PIL.Image] 或单个 PIL.Image（均会被转 RGB）
    返回**已温度缩放**后的概率，与 train_mlp / Phase 4 公平对比完全一致。
    注意：特征概率不再额外 sigmoid（MLPFeaturesOptimized.inference=True 已含）。

CLIP 编码器必须与产出 clip_features_cpu.csv 的 OpenAI CLIP 完全一致，
否则双 MLP 头权重不匹配；因此 CLIPEncoder 仍走 openai/clip 包
（部署环境自带，本实验 venv 可能未装，工厂/演示会优雅跳过）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple, Union

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.config import CLASSIFIER_TEMPERATURE, FEATURE_TEMPERATURE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEncoder:
    name = "base"

    def predict(self, images) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (cls_softmax (B,2), feat_sigmoid (B,11))，均在 CPU 上。"""
        raise NotImplementedError

    @staticmethod
    def _as_list(images) -> List[Image.Image]:
        if isinstance(images, (list, tuple)):
            return list(images)
        return [images]


class CLIPEncoder(BaseEncoder):
    """OpenAI CLIP ViT-B/32 (512维) + 双 MLP 头。需 openai/clip 包。"""

    name = "CLIP-ViT-B/32"

    def __init__(self, device: str = DEVICE, cls_pt=None, feat_pt=None):
        import clip  # 仅在此处导入，缺失时不影响其他编码器
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()

        from src.models.mlp import MLPClassifier
        from src.models.mlp_features_optimized import MLPFeaturesOptimized
        d = ROOT / "data"
        self.cls = MLPClassifier(input_dim=512, hidden_dim=256, output_dim=2, dropout=0.3)
        self.cls.load_state_dict(torch.load(cls_pt or d / "mlp_classifier.pt", map_location=device))
        self.feat = MLPFeaturesOptimized(input_dim=512, hidden_dim=512, output_dim=11,
                                         dropout=0.3, temperature=FEATURE_TEMPERATURE)
        self.feat.load_state_dict(torch.load(feat_pt or d / "mlp_features_optimized.pt", map_location=device))
        self.cls.eval()
        self.feat.eval()

    def predict(self, images) -> Tuple[torch.Tensor, torch.Tensor]:
        imgs = self._as_list(images)
        batch = torch.stack([self.preprocess(im.convert("RGB")) for im in imgs]).to(self.device)
        with torch.no_grad():
            feats = self.clip_model.encode_image(batch).float()
            cl = torch.softmax(self.cls(feats) / CLASSIFIER_TEMPERATURE, dim=1)
            ft = self.feat(feats, inference=True)  # 已含 sigmoid + 温度缩放
        return cl.cpu(), ft.cpu()


class SigLIPEncoder(BaseEncoder):
    """SigLIP2-NaFlex-B/16 (768维, 原生宽高比零裁切) + 双 MLP 头(_siglip)。"""

    name = "SigLIP2-NaFlex-B/16"

    def __init__(self, device: str = DEVICE, model_dir=None, cls_pt=None, feat_pt=None,
                 max_patches: int = 256):
        from transformers import AutoModel, AutoImageProcessor
        self.device = device
        md = Path(model_dir or ROOT / "data" / "models" / "siglip2-base-patch16-naflex")
        self.proc = AutoImageProcessor.from_pretrained(md, max_num_patches=max_patches)
        self.model = AutoModel.from_pretrained(md, dtype=torch.float16).to(device).eval()

        from src.models.mlp import MLPClassifier
        from src.models.mlp_features_optimized import MLPFeaturesOptimized
        d = ROOT / "data"
        self.cls = MLPClassifier(input_dim=768, hidden_dim=256, output_dim=2, dropout=0.3)
        self.cls.load_state_dict(torch.load(cls_pt or d / "mlp_classifier_siglip.pt", map_location=device))
        self.feat = MLPFeaturesOptimized(input_dim=768, hidden_dim=512, output_dim=11,
                                         dropout=0.3, temperature=FEATURE_TEMPERATURE)
        self.feat.load_state_dict(torch.load(feat_pt or d / "mlp_features_optimized_siglip.pt", map_location=device))
        self.cls.eval()
        self.feat.eval()

    def predict(self, images) -> Tuple[torch.Tensor, torch.Tensor]:
        imgs = self._as_list(images)
        inputs = self.proc(images=imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs).float().cpu()  # (B,768) fp16->fp32
            cl = torch.softmax(self.cls(feats) / CLASSIFIER_TEMPERATURE, dim=1)
            ft = self.feat(feats, inference=True)
        return cl.cpu(), ft.cpu()


class ViTTinyEncoder(BaseEncoder):
    """自训 ViT-Tiny 端到端（letterbox 零裁切 + 双头）。需 timm + safetensors。"""

    name = "ViT-Tiny(自训)"
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, device: str = DEVICE, ckpt=None, weights=None, seed: int = 42):
        from train_vit_tiny import ViTCheckin, letterbox, load_backbone
        self.device = device
        self.letterbox = letterbox
        backbone = load_backbone()
        self.model = ViTCheckin(backbone).to(device)
        ck = Path(ckpt or ROOT / "data" / f"vit_tiny_seed{seed}.pt")
        self.model.load_state_dict(torch.load(ck, map_location=device))
        self.model.eval()

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        img = self.letterbox(img.convert("RGB"))
        t = torch.from_numpy(__import__("numpy").asarray(img).copy()).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor(self.MEAN).view(3, 1, 1)
        std = torch.tensor(self.STD).view(3, 1, 1)
        return (t - mean) / std

    def predict(self, images) -> Tuple[torch.Tensor, torch.Tensor]:
        imgs = self._as_list(images)
        x = torch.stack([self._preprocess(im) for im in imgs]).to(self.device)
        with torch.no_grad():
            cl_logits, fh_logits = self.model(x)
            cl = torch.softmax(cl_logits / CLASSIFIER_TEMPERATURE, dim=1)
            ft = torch.sigmoid(fh_logits / FEATURE_TEMPERATURE)
        return cl.cpu(), ft.cpu()


# ---------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------
_ENCODERS = {
    "CLIP": CLIPEncoder,
    "SigLIP": SigLIPEncoder,
    "ViT-Tiny": ViTTinyEncoder,
}


def get_encoder(name: str = "CLIP", **kwargs) -> BaseEncoder:
    """按名称构造编码器。name ∈ {CLIP, SigLIP, ViT-Tiny}。"""
    if name not in _ENCODERS:
        raise ValueError(f"未知编码器 {name!r}，可选: {list(_ENCODERS)}")
    return _ENCODERS[name](**kwargs)


def available_encoders() -> List[str]:
    """探测当前环境可实例化的编码器（CLIP 依赖 openai/clip 包）。"""
    avail = []
    for n, cls in _ENCODERS.items():
        try:
            # 仅探测依赖是否可导入，不真正加载权重
            if n == "CLIP":
                import clip  # noqa
            elif n == "SigLIP":
                import transformers  # noqa
            elif n == "ViT-Tiny":
                import timm  # noqa
            avail.append(n)
        except Exception:
            pass
    return avail
