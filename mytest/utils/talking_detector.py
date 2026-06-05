"""
说话检测模块：基于 MediaPipe FaceMesh 的嘴部开合比（MAR）判断人员是否正在说话

使用前需安装：pip install mediapipe
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple

try:
    import mediapipe as mp
    _MP_FACE_MESH = mp.solutions.face_mesh
    _MP_AVAILABLE = True
except (ImportError, AttributeError):
    _MP_AVAILABLE = False

# MediaPipe FaceMesh 嘴部关键点索引（478 点模型）
_UPPER_LIP = [13, 312, 82, 80]   # 上唇内缘
_LOWER_LIP = [14, 317, 87, 84]   # 下唇内缘
_MOUTH_LEFT  = 61
_MOUTH_RIGHT = 291


class TalkingDetector:
    """
    基于 MediaPipe FaceMesh 的说话检测器。

    流程：
    1. 用 YOLO 头部关键点（COCO 0-4：鼻、左眼、右眼、左耳、右耳）估算人脸区域并裁剪
    2. 在裁剪区域内跑 FaceMesh，取嘴部 478 点
    3. 计算嘴巴纵横比（MAR = 垂直开口 / 水平宽度）
    4. 多帧滑动平均后与阈值比较，输出说话/静默
    """

    def __init__(self, mar_threshold: float = 0.035, smooth_frames: int = 3):
        """
        Args:
            mar_threshold:  MAR 超过此值判定为说话（推荐 0.03-0.06，需按场景调参）
            smooth_frames:  滑动平均帧数，抑制单帧抖动
        """
        if not _MP_AVAILABLE:
            raise RuntimeError(
                "mediapipe 未安装，请运行：pip install mediapipe\n"
                "安装后重新启动程序以启用说话检测功能。"
            )

        self.mar_threshold = mar_threshold
        self.smooth_frames = smooth_frames

        self._face_mesh = _MP_FACE_MESH.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,          # 启用 478 点（含虹膜），嘴部点更密集
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # track_id → 最近 N 帧 MAR 值
        self._mar_history: dict = {}

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _crop_face(
        self,
        frame: np.ndarray,
        keypoints: List,
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """
        根据 YOLO 头部关键点裁剪人脸区域。

        Returns:
            (face_bgr, (x1, y1)) ；关键点不足时返回 (None, (0, 0))
        """
        h, w = frame.shape[:2]

        visible = [
            (float(kp[0]), float(kp[1]))
            for kp in keypoints[:5]
            if len(kp) >= 3 and float(kp[2]) > 0.3
        ]
        if len(visible) < 2:
            return None, (0, 0)

        xs = [p[0] for p in visible]
        ys = [p[1] for p in visible]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        pad = span * 0.9

        x1 = max(0, int(min(xs) - pad))
        y1 = max(0, int(min(ys) - pad))
        x2 = min(w, int(max(xs) + pad))
        y2 = min(h, int(max(ys) + pad * 1.6))   # 向下多留空间覆盖嘴巴

        if x2 - x1 < 24 or y2 - y1 < 24:
            return None, (0, 0)

        return frame[y1:y2, x1:x2].copy(), (x1, y1)

    def _compute_mar(self, landmarks, img_w: int, img_h: int) -> float:
        """计算嘴巴纵横比（MAR）。"""
        def pt(idx: int) -> np.ndarray:
            lm = landmarks[idx]
            return np.array([lm.x * img_w, lm.y * img_h])

        upper = np.mean([pt(i) for i in _UPPER_LIP], axis=0)
        lower = np.mean([pt(i) for i in _LOWER_LIP], axis=0)
        left  = pt(_MOUTH_LEFT)
        right = pt(_MOUTH_RIGHT)

        vertical   = float(np.linalg.norm(upper - lower))
        horizontal = float(np.linalg.norm(left  - right))
        return vertical / (horizontal + 1e-6)

    # ── 公开方法 ──────────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        keypoints: List,
        track_id: int,
    ) -> bool:
        """
        判断指定目标是否正在说话。

        Args:
            frame:      完整全景帧（BGR）
            keypoints:  YOLO COCO-17 关键点，每项 [x, y, conf]
            track_id:   跟踪 ID（用于时序平滑历史）

        Returns:
            True = 说话中，False = 静默
        """
        face_crop, _ = self._crop_face(frame, keypoints)
        if face_crop is None:
            return False

        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return False

        fh, fw = face_crop.shape[:2]
        mar = self._compute_mar(result.multi_face_landmarks[0].landmark, fw, fh)

        history = self._mar_history.setdefault(track_id, [])
        history.append(mar)
        if len(history) > self.smooth_frames:
            history.pop(0)

        avg_mar = sum(history) / len(history)
        return avg_mar > self.mar_threshold

    def cleanup_track(self, track_id: int) -> None:
        """清理已消失目标的历史数据，防止内存泄漏。"""
        self._mar_history.pop(track_id, None)

    def close(self) -> None:
        """释放 FaceMesh 资源。"""
        self._face_mesh.close()
