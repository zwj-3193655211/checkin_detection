import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np


class CLIPFeatureExtractor:
    """
    CLIP特征提取器

    使用OpenAI CLIP模型提取图像特征
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            model_name: CLIP模型名称
            device: 设备类型
            cache_dir: 模型缓存目录
        """
        try:
            import clip
        except ImportError:
            raise ImportError(
                "CLIP not installed. Please install with: pip install git+https://github.com/openai/CLIP.git"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        # 加载模型
        self.model, self.preprocess = clip.load(model_name, device=self.device, download_root=cache_dir)
        self.model.eval()

    @torch.no_grad()
    def extract_image_features(self, image) -> torch.Tensor:
        """
        提取单张图像的特征

        Args:
            image: PIL Image或torch.Tensor

        Returns:
            features: 特征向量 [512] (ViT-B/32)
        """
        if isinstance(image, np.ndarray):
            from PIL import Image
            image = Image.fromarray(image)

        if isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        features = self.model.encode_image(image)
        return features.squeeze(0)

    @torch.no_grad()
    def extract_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        批量提取特征

        Args:
            images: 图像张量 [batch_size, C, H, W]

        Returns:
            features: 特征矩阵 [batch_size, feature_dim]
        """
        if images.device != self.device:
            images = images.to(self.device)

        features = self.model.encode_image(images)
        return features

    @torch.no_grad()
    def extract_text_features(self, text: str) -> torch.Tensor:
        """
        提取文本特征

        Args:
            text: 文本字符串

        Returns:
            features: 特征向量 [512]
        """
        text_tokens = clip.tokenize([text]).to(self.device)
        features = self.model.encode_text(text_tokens)
        return features.squeeze(0)

    @torch.no_grad()
    def extract_text_batch(self, texts: list) -> torch.Tensor:
        """
        批量提取文本特征

        Args:
            texts: 文本列表

        Returns:
            features: 特征矩阵 [batch_size, feature_dim]
        """
        text_tokens = clip.tokenize(texts).to(self.device)
        features = self.model.encode_text(text_tokens)
        return features

    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return 512 if "ViT-B" in self.model_name else 768

    def zero_shot_classify(
        self,
        image: torch.Tensor,
        class_names: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        零样本分类

        Args:
            image: 图像张量 [C, H, W] 或 [1, C, H, W]
            class_names: 类别名称列表

        Returns:
            probs: 各类别的概率
            features: 图像特征
        """
        if isinstance(image, np.ndarray):
            from PIL import Image
            image = Image.fromarray(image)

        if isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        # 提取图像特征
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # 提取文本特征
        text_descriptions = [f"a photo of {name}" for name in class_names]
        text_tokens = clip.tokenize(text_descriptions).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 计算相似度
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        return similarity.squeeze(0), image_features.squeeze(0)


class FeaturePreprocessor:
    """特征预处理器"""

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """处理特征"""
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        return features


class FeatureCache:
    """特征缓存管理器"""

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir
        self.cache = {}

    def save_features(
        self,
        filenames: list,
        features: torch.Tensor,
        path: str,
    ):
        """保存特征到缓存"""
        data = {
            'filenames': filenames,
            'features': features.cpu().numpy(),
        }
        torch.save(data, path)

    def load_features(self, path: str) -> Tuple[list, torch.Tensor]:
        """加载特征缓存"""
        data = torch.load(path)
        return data['filenames'], torch.from_numpy(data['features'])

    def get_feature(self, filename: str) -> Optional[torch.Tensor]:
        """获取单个特征"""
        return self.cache.get(filename)

    def add_feature(self, filename: str, feature: torch.Tensor):
        """添加特征到缓存"""
        self.cache[filename] = feature