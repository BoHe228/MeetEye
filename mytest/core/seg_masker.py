"""
Seg-guided ReID 特征提取器
流程：Pose 检测框 → YOLO-Seg 精细分割 → 背景置黑 → OSNet 特征重提取
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple


class SegMasker:
    """
    用 YOLO-Seg 为每个 Pose 检测框生成人体前景掩码，再批量重提取 ReID 特征。
    仅处理 class 0（person），分割失败时回退到原始 crop，保证特征不丢失。
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.25,
                 device: str = 'cuda'):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        # 缓存上一帧分割结果，供 draw_last_masks 使用
        self._last_masks: List[Tuple[int, int, int, int, Optional[np.ndarray]]] = []
        print(f"[SegMasker] 分割模型已加载: {model_path}  device={device}")

    # ── 内部：批量 YOLO-Seg 推理，返回每个 crop 对应的二值掩码（或 None）──
    def _batch_masks(self, crops: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        对 crop 列表做一次批量 predict，返回与 crops 等长的掩码列表。
        每个掩码为与对应 crop 同尺寸的 uint8 数组（1=前景, 0=背景）；
        若该 crop 无有效检测则对应位置返回 None。
        """
        if not crops:
            return []

        results = self.model.predict(
            crops,
            conf=self.conf_threshold,
            classes=[0],
            verbose=False,
            device=self.device,
        )

        masks_out: List[Optional[np.ndarray]] = []
        for result, crop in zip(results, crops):
            if result.masks is None or len(result.boxes.conf) == 0:
                masks_out.append(None)
                continue

            best_idx = int(result.boxes.conf.argmax())
            mask_raw = result.masks.data[best_idx].cpu().numpy()  # H × W, float
            h, w = crop.shape[:2]
            if mask_raw.shape != (h, w):
                mask_raw = cv2.resize(mask_raw, (w, h),
                                      interpolation=cv2.INTER_NEAREST)
            masks_out.append((mask_raw > 0.5).astype(np.uint8))

        return masks_out

    # ── 公开接口 ─────────────────────────────────────────────────────────
    def reextract_features(self, panorama: np.ndarray, detections: list,
                           feature_extractor) -> None:
        """
        对 detections 中的所有检测框执行：
          1. 从全景图裁剪 crop
          2. 批量 YOLO-Seg → 人体掩码（同时缓存到 self._last_masks）
          3. 背景置黑（分割失败时保留原 crop）
          4. 批量 OSNet 特征重提取，更新 det['feature']

        直接原地修改 detections，无返回值。
        """
        self._last_masks = []

        if not detections or feature_extractor is None:
            return

        ph, pw = panorama.shape[:2]

        # 收集有效 crop 及其在 detections 中的下标
        crops: List[np.ndarray] = []
        det_indices: List[int] = []
        bboxes: List[Tuple[int, int, int, int]] = []

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(pw, x2), min(ph, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(panorama[y1:y2, x1:x2].copy())
            det_indices.append(i)
            bboxes.append((x1, y1, x2, y2))

        if not crops:
            return

        # 批量分割
        masks = self._batch_masks(crops)

        # 缓存本帧所有 (bbox, mask) 供可视化使用
        self._last_masks = [(x1, y1, x2, y2, mask)
                            for (x1, y1, x2, y2), mask in zip(bboxes, masks)]

        # 应用掩码：前景保留，背景置黑；失败时保留原 crop
        masked_crops: List[np.ndarray] = []
        for crop, mask in zip(crops, masks):
            if mask is not None:
                mc = crop.copy()
                mc[mask == 0] = 0
                masked_crops.append(mc)
            else:
                masked_crops.append(crop)

        # 批量 ReID 特征重提取（一次 GPU forward pass）
        features = feature_extractor.extract_batch_arrays(masked_crops)
        for det_i, feat in zip(det_indices, features):
            detections[det_i]['feature'] = feat

    def draw_last_masks(self, image: np.ndarray,
                        color: Tuple[int, int, int] = (0, 255, 128),
                        alpha: float = 0.40) -> np.ndarray:
        """
        将上一帧缓存的分割掩码以半透明彩色叠加绘制到 image 上。
        分割失败（mask=None）的框不绘制，只绘制成功的前景区域。

        参数:
            image : 待绘制的 BGR 图像（会被原地修改并返回）
            color : 前景叠加颜色，BGR，默认青绿色 (0, 255, 128)
            alpha : 叠加透明度，0=不叠加，1=完全覆盖，默认 0.40
        返回:
            绘制后的图像（与输入同一对象）
        """
        if not self._last_masks:
            return image

        overlay = image.copy()
        for x1, y1, x2, y2, mask in self._last_masks:
            if mask is None:
                continue
            roi = overlay[y1:y2, x1:x2]
            roi[mask == 1] = color

        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)
        return image
