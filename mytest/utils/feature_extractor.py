import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import cv2

from utils import cosine_similarity


class FeatureExtractor:
    """
    OSNet特征提取器 - 专门用于main.py调用

    功能:
        - 加载OSNet模型
        - 从图像数组提取特征
        - 从图像文件提取特征
    """

    def __init__(self, model_name='osnet_x0_25', model_path=None, device=None):
        """
        初始化特征提取器

        参数:
            model_name: 模型名称 (默认: 'osnet_x0_25')
            model_path: 模型权重文件路径 (可选)
            device: 运行设备 ('cuda' 或 'cpu', 自动检测)
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = None
        self.transform = None
        self._initialize()

    def _initialize(self):
        """初始化torchreid的FeatureExtractor"""
        print(f"初始化OSNet特征提取器 (model={self.model_name}, device={self.device})...")

        # 导入torchreid
        try:
            import torchreid
            from torchreid.reid.utils import FeatureExtractor
        except ImportError:
            raise ImportError("请安装torchreid: pip install torchreid")

        # 构建参数
        kwargs = {
            'model_name': self.model_name,
            'device': self.device,
            'verbose': True
        }

        # 如果提供了模型路径且文件存在，使用指定模型
        if self.model_path and os.path.exists(self.model_path):
            kwargs['model_path'] = self.model_path
            print(f"使用自定义模型: {self.model_path}")
        else:
            print(f"使用预训练模型: {self.model_name}")

        # 创建提取器
        self.extractor = FeatureExtractor(**kwargs)

        # 定义图像预处理变换（与FeatureExtractor内部保持一致）
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print("OSNet特征提取器初始化完成!")
        
    def to(self, device):
        """将模型移动到指定设备"""
        if self.extractor and hasattr(self.extractor, 'model'):
            self.extractor.model.to(device)
            self.device = device
        return self

    def extract_from_array(self, image_array):
        """
        从OpenCV图像数组提取特征 (BGR格式)

        参数:
            image_array: numpy数组 (H, W, 3), BGR格式

        返回:
            特征向量 (torch.Tensor, shape: [1, feature_dim])
        """
        # BGR -> RGB
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # numpy -> PIL Image
        image = Image.fromarray(image_array)

        # 预处理
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # 提取特征
        self.extractor.model.eval()
        with torch.no_grad():
            features = self.extractor.model(image_tensor)

        return features.cpu()

    def extract_from_file(self, image_path):
        """
        从图像文件提取特征

        参数:
            image_path: 图像文件路径

        返回:
            特征向量 (torch.Tensor)
        """
        features = self.extractor(image_path)
        return features[0]

    def extract_batch_from_files(self, image_paths):
        """
        批量从图像文件提取特征

        参数:
            image_paths: 图像文件路径列表

        返回:
            特征向量列表
        """
        return self.extractor(image_paths)

    def __call__(self, image_input):
        """
        便捷调用方式 - 自动判断输入类型

        参数:
            image_input: 图像路径(str) 或 图像数组(numpy.ndarray)

        返回:
            特征向量
        """
        if isinstance(image_input, str):
            return self.extract_from_file(image_input)
        elif isinstance(image_input, np.ndarray):
            return self.extract_from_array(image_input)
        else:
            raise ValueError("输入必须是图像路径(str)或图像数组(numpy.ndarray)")

    def extract_features_from_image_array(self, image_array):
        """
        兼容方法 - 保持与旧代码接口一致 (给PanoramaSlicer调用)

        参数:
            image_array: numpy数组格式的图像

        返回:
            特征向量列表 (为了兼容，返回 [feature_tensor])
        """
        feature = self.extract_from_array(image_array)
        return [feature]

    def extract_batch_arrays(self, image_arrays: list) -> list:
        """
        批量从多个 numpy 图像数组提取特征（一次 GPU forward pass）。

        参数:
            image_arrays: BGR numpy 数组列表

        返回:
            特征向量列表，每个元素 shape [1, feat_dim]
        """
        if not image_arrays:
            return []

        tensors = []
        for img in image_arrays:
            if img is None or img.size == 0:
                # 空图像用零张量占位
                tensors.append(torch.zeros(3, 256, 128))
                continue
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            tensors.append(self.transform(pil_img))

        # 始终用 FP32；OSNet 含 BatchNorm，不做 half 转换
        batch = torch.stack(tensors).float().to(self.device)   # (N, 3, 256, 128)
        self.extractor.model.eval()
        with torch.no_grad():
            feats = self.extractor.model(batch).float()  # (N, feat_dim)

        feats_cpu = feats.cpu()
        return [feats_cpu[i : i + 1] for i in range(feats_cpu.shape[0])]

    def extract_batch_gpu_crops(self, gpu_crops: list) -> list:
        """
        直接接受 GPU 上的 [3, H, W] float 0-1 RGB 张量列表。
        在 GPU 上完成 resize + normalize，跳过 CPU numpy/PIL/transform 路径，
        比 extract_batch_arrays 少约 5ms（省去 BGR→RGB、PIL、torchvision transforms）。

        参数:
            gpu_crops: [3, H, W] float 0-1 RGB CUDA tensor 列表

        返回:
            特征向量列表，每个元素 shape [1, feat_dim]（CPU tensor）
        """
        if not gpu_crops:
            return []

        device = next(self.extractor.model.parameters()).device
        # 归一化常数直接放 GPU，只分配一次
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

        tensors = []
        for t in gpu_crops:
            if t is None or t.numel() == 0:
                tensors.append(torch.zeros(3, 256, 128, device=device))
                continue
            # GPU 上双线性插值到 OSNet 标准输入尺寸，无需经过 PIL
            r = F.interpolate(
                t.unsqueeze(0).float(), size=(256, 128),
                mode='bilinear', align_corners=False
            )[0]
            tensors.append((r - mean) / std)

        batch = torch.stack(tensors)  # [N, 3, 256, 128]，已在 GPU
        self.extractor.model.eval()
        with torch.no_grad():
            feats = self.extractor.model(batch).float()

        # 一次 GPU→CPU 传输（一次 CUDA sync），再在 CPU 上切片，避免 N 次独立 sync
        feats_cpu = feats.cpu()
        return [feats_cpu[i : i + 1] for i in range(feats_cpu.shape[0])]

    def extract_features(self, image_paths):
        """
        兼容方法 - 批量从文件提取特征

        参数:
            image_paths: 图像路径列表

        返回:
            特征向量列表
        """
        return self.extract_batch_from_files(image_paths)
