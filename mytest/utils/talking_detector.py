"""
说话检测模块：基于 MediaPipe FaceLandmarker（Tasks API）的嘴部开合比（MAR）
判断人员是否正在说话。

适用于 mediapipe >= 0.10（旧版 solutions.face_mesh 已移除）。
首次运行时自动下载模型文件（~12 MB）。
"""
import os
import urllib.request
import cv2
import numpy as np
from typing import List, Optional, Tuple

try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_tasks
    from mediapipe.tasks.python import vision as _mp_vision
    _MP_AVAILABLE = True
except (ImportError, AttributeError):
    _MP_AVAILABLE = False

# MediaPipe Face Mesh 478 点中嘴部关键点索引
_UPPER_LIP   = [13, 312, 82, 80]   # 上唇内缘
_LOWER_LIP   = [14, 317, 87, 84]   # 下唇内缘
_MOUTH_LEFT  = 61
_MOUTH_RIGHT = 291

# 模型文件默认存放路径（与 main.py 同级目录的 models/ 子目录）
_DEFAULT_MODEL_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_MODEL_FILENAME     = "face_landmarker.task"
_MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_model(model_path: str) -> str:
    """若模型文件不存在则自动下载，返回实际路径。"""
    if os.path.isfile(model_path):
        return model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"[TalkingDetector] 首次使用，下载模型到 {model_path} …")
    try:
        urllib.request.urlretrieve(_MODEL_DOWNLOAD_URL, model_path)
        print("[TalkingDetector] 模型下载完成。")
    except Exception as e:
        raise RuntimeError(
            f"模型下载失败：{e}\n"
            f"请手动下载并放置到 {model_path}：\n"
            f"  {_MODEL_DOWNLOAD_URL}"
        ) from e
    return model_path


class TalkingDetector:
    """
    基于 MediaPipe FaceLandmarker（Tasks API）的说话检测器。

    流程：
    1. 用 YOLO 头部关键点（COCO 0-4：鼻、左眼、右眼、左耳、右耳）估算人脸区域并裁剪
    2. 在裁剪区域内跑 FaceLandmarker，取嘴部 478 点
    3. 计算嘴巴纵横比（MAR = 垂直开口 / 水平宽度）
    4. 多帧滑动平均后与阈值比较，输出说话/静默
    """

    def __init__(
        self,
        mar_threshold: float = 0.035,
        smooth_frames: int = 3,
        model_path: Optional[str] = None,
    ):
        """
        Args:
            mar_threshold:  MAR 超过此值判定为说话（推荐 0.03-0.06，需按场景调参）
            smooth_frames:  滑动平均帧数，抑制单帧抖动
            model_path:     face_landmarker.task 路径；None 时使用默认路径并自动下载
        """
        if not _MP_AVAILABLE:
            raise RuntimeError(
                "mediapipe 导入失败，请确认已安装：pip install mediapipe"
            )

        self.mar_threshold = mar_threshold
        self.smooth_frames = smooth_frames

        if model_path is None:
            model_path = os.path.join(_DEFAULT_MODEL_DIR, _MODEL_FILENAME)
        model_path = _ensure_model(model_path)

        options = _mp_vision.FaceLandmarkerOptions(
            base_options=_mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=_mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_score=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = _mp_vision.FaceLandmarker.create_from_options(options)

        # track_id → 最近 N 帧 MAR 值
        self._mar_history: dict = {}

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _crop_face(
        self,
        frame: np.ndarray,
        keypoints: List,
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """根据 YOLO 头部关键点（索引 0-4）裁剪人脸区域。"""
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
        y2 = min(h, int(max(ys) + pad * 1.6))  # 向下多留空间覆盖嘴巴

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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return False

        fh, fw = face_crop.shape[:2]
        mar = self._compute_mar(result.face_landmarks[0], fw, fh)

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
        """释放 FaceLandmarker 资源。"""
        self._landmarker.close()
