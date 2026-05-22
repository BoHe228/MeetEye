"""
基于人脸关键点的头部姿态距离估算器

核心思路：
  1. 利用鼻尖相对于左右眼中点的横向偏移，通过简化透视模型估算头部偏转角（Yaw）。
  2. 将"斜向观测"到的表观眼距修正回正面等效眼距：Dreal = Dapp / cos(yaw)。
  3. 通过经验公式将修正后的像素眼距映射为物理距离（米）。

偏转角估算模型：
  当头部偏转 yaw 角时（绕竖直轴旋转）：
    - 双眼像素距离  Dapp = Dreal_px * cos(yaw)
    - 鼻尖横向偏移  nose_lateral = D_nose * sin(yaw)
  其中 D_nose 是鼻尖相对于面部中心的深度（单位与眼距相同）。
  两式相除：
    nose_lateral / (Dapp/2) ≈ (2*D_nose/W) * tan(yaw) = _NOSE_SCALE * tan(yaw)
  因此：
    tan(yaw) = (nose_lateral / (Dapp/2)) / _NOSE_SCALE
    cos(yaw) = 1 / sqrt(1 + tan(yaw)^2)
    Dreal   = Dapp / cos(yaw) = Dapp * sqrt(1 + tan(yaw)^2)
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# ── 经验标定公式参数（与 main_GPU_webui.py 保持一致）──────────────────
_EYE_K1: float = 0.024030
_EYE_K2: float = 0.044812

# 头部偏转角熔断阈值（度）：超过此值视为透视畸变过严，丢弃当帧结果
# 正脸 yaw≈0°，轻微偏转 yaw<25°，超过该值则帧间沿用上一有效距离
_YAW_FUSE_DEG: float = 45.0

# 鼻深比例因子：≈ 2 * D_nose/W（鼻尖深度 / 眼距）的经验估计
# 典型人脸：D_nose ≈ 20-30mm，W（IPD）≈ 63mm → 2*D_nose/W ≈ 0.6
_NOSE_SCALE: float = 0.6

# 人均瞳距参考值（米），用于 calibration_scale 计算
_MEAN_IPD_M: float = 0.063


def estimate_distance_from_eyes(eye_pixel_dist: float) -> Optional[float]:
    """
    双眼像素距离 → 物理距离（m）。

    标定公式：distance = 1 / (K1 * D + K2)
    此函数与 main_GPU_webui.py 中的公式完全对应，集中定义以避免重复。
    """
    denom = _EYE_K1 * eye_pixel_dist + _EYE_K2
    return 1.0 / denom if denom > 0 else None


class HeadPoseDistanceEstimator:
    """
    基于头部姿态修正的用户-摄像头距离估算器。

    工作流程（每帧）：
      Dapp (表观像素眼距) + nose_lateral (鼻尖横向偏移)
        → tan(yaw) = nose_lateral / (Dapp/2) / _NOSE_SCALE
          → 角度熔断（yaw > yaw_fuse_deg 则沿用上帧距离）
            → Dreal = Dapp / cos(yaw)（透视修正）
              → estimate_distance_from_eyes(Dreal) → 物理距离（m）

    关键点：
      - 正脸时：nose_lateral ≈ 0，yaw ≈ 0°，Dreal ≈ Dapp（无修正）
      - 偏转时：nose_lateral 增大，yaw 增大，Dreal > Dapp（补偿缩短的眼距）
      - 严重侧脸：yaw > yaw_fuse_deg，沿用上帧距离避免大误差

    Parameters
    ----------
    calibration_scale : float, optional
        标定比例因子（m/px），为未来扩展保留，当前未参与计算。
        可由 HeadPoseDistanceEstimator.compute_calibration_scale() 从正脸帧获取。
    yaw_fuse_deg : float
        偏转角熔断阈值（度），默认 45°。
    """

    def __init__(
        self,
        calibration_scale: float = 0.0,
        yaw_fuse_deg: float = _YAW_FUSE_DEG,
    ) -> None:
        self.calibration_scale: float = calibration_scale
        self.yaw_fuse_deg: float = yaw_fuse_deg
        self._last_valid_distance: Optional[float] = None

    # ------------------------------------------------------------------ #
    # 初始化辅助：从正脸帧计算 calibration_scale                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_calibration_scale(
        frontal_left_eye: Tuple[float, float],
        frontal_right_eye: Tuple[float, float],
        real_ipd_m: float = _MEAN_IPD_M,
    ) -> Optional[float]:
        """
        从第一帧正脸数据计算标定比例因子。

        Parameters
        ----------
        frontal_left_eye  : (x, y) 正脸帧中左眼像素坐标
        frontal_right_eye : (x, y) 正脸帧中右眼像素坐标
        real_ipd_m        : 真实眼距（米），默认 0.063 m

        Returns
        -------
        calibration_scale (m/px)，若眼距为 0 则返回 None。
        """
        le = np.array(frontal_left_eye, dtype=np.float64)
        re = np.array(frontal_right_eye, dtype=np.float64)
        d_frontal = float(np.linalg.norm(re - le))
        if d_frontal < 1e-6:
            return None
        return real_ipd_m / d_frontal

    # ------------------------------------------------------------------ #
    # 关键点提取辅助                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_keypoints(
        keypoints,
        conf_threshold: float = 0.1,
    ) -> Tuple[Optional[Tuple[float, float]],
               Optional[Tuple[float, float]],
               Optional[Tuple[float, float]]]:
        """
        从 YOLO-Pose 关键点数组中提取 (left_eye, right_eye, nose)。

        COCO 关键点顺序：kpt[0]=鼻尖, kpt[1]=左眼, kpt[2]=右眼
        每行格式：[x, y] 或 [x, y, conf]

        Returns
        -------
        (left_eye, right_eye, nose) — 各为 (x, y) 元组。
        若关键点不足或置信度低于阈值，返回 (None, None, None)。
        """
        if keypoints is None:
            return None, None, None
        kpts = np.array(keypoints)
        if kpts.shape[0] < 3 or kpts.shape[1] < 2:
            return None, None, None
        has_conf = kpts.shape[1] >= 3
        if has_conf:
            for idx in (0, 1, 2):
                if float(kpts[idx, 2]) < conf_threshold:
                    return None, None, None
        nose  = (float(kpts[0, 0]), float(kpts[0, 1]))
        l_eye = (float(kpts[1, 0]), float(kpts[1, 1]))
        r_eye = (float(kpts[2, 0]), float(kpts[2, 1]))
        return l_eye, r_eye, nose

    # ------------------------------------------------------------------ #
    # 核心计算                                                              #
    # ------------------------------------------------------------------ #

    def compute_distance(
        self,
        left_eye:  Tuple[float, float],
        right_eye: Tuple[float, float],
        nose:      Tuple[float, float],
    ) -> Optional[float]:
        """
        根据三个关键点估算用户与摄像头之间的物理距离。

        Parameters
        ----------
        left_eye  : (x_l, y_l) 左眼像素坐标
        right_eye : (x_r, y_r) 右眼像素坐标
        nose      : (x_n, y_n) 鼻尖像素坐标

        Returns
        -------
        物理距离（m）。当前帧被熔断或出现数值异常时，
        返回上一帧有效距离（若尚无历史帧则返回 None）。
        """
        le = np.array(left_eye,  dtype=np.float64)
        re = np.array(right_eye, dtype=np.float64)
        no = np.array(nose,      dtype=np.float64)

        # ── Step 1: 表观像素眼距 Dapp ──────────────────────────────────
        Dapp = float(np.linalg.norm(re - le))
        if Dapp < 1e-6:
            return self._last_valid_distance

        # ── Step 2: 鼻尖横向偏移（相对于眼睛中点）──────────────────────
        # 正脸时 nose_lateral ≈ 0；头部偏转时，鼻尖向旋转方向移动
        eye_mid_x = (le[0] + re[0]) / 2.0
        nose_lateral = float(no[0] - eye_mid_x)

        # 归一化：以半眼距为单位，量纲消除，结果对分辨率不敏感
        nose_norm = nose_lateral / (Dapp / 2.0)

        # ── Step 3: 估算偏转角 yaw ──────────────────────────────────────
        # 在简化透视模型下：nose_norm ≈ _NOSE_SCALE * tan(yaw)
        tan_yaw = abs(nose_norm) / _NOSE_SCALE
        yaw_deg = float(np.degrees(np.arctan(tan_yaw)))

        # ── Step 4: 角度熔断机制 ────────────────────────────────────────
        # 偏转角超过阈值时，透视修正误差过大，沿用上帧有效距离
        if yaw_deg > self.yaw_fuse_deg:
            return self._last_valid_distance

        # ── Step 5: 还原真实像素眼距 Dreal ──────────────────────────────
        # 投影使斜向观测到的眼距缩短：Dapp = Dreal * cos(yaw)
        # 因此：Dreal = Dapp / cos(yaw) = Dapp * sqrt(1 + tan_yaw^2)
        cos_yaw = 1.0 / float(np.sqrt(1.0 + tan_yaw ** 2))
        Dreal = Dapp / cos_yaw  # Dreal ≥ Dapp，修正后眼距更大，距离估计更准确

        # ── Step 6: 像素眼距 → 物理距离 ─────────────────────────────────
        distance = estimate_distance_from_eyes(Dreal)
        if distance is None:
            return self._last_valid_distance

        self._last_valid_distance = distance
        return distance

    # ── 便捷属性与工具方法 ───────────────────────────────────────────────
    @property
    def last_valid_distance(self) -> Optional[float]:
        """上一帧有效物理距离（m），供外部查询。"""
        return self._last_valid_distance

    def reset(self) -> None:
        """清除历史帧缓存（换人或场景切换时调用）。"""
        self._last_valid_distance = None


# ══════════════════════════════════════════════════════════════════════ #
#  Example Usage                                                         #
# ══════════════════════════════════════════════════════════════════════ #
if __name__ == "__main__":
    # ── 1. 用正脸数据初始化（calibration_scale 为未来扩展保留）───────────
    FRONTAL_LEFT_EYE  = (410.0, 300.0)
    FRONTAL_RIGHT_EYE = (510.0, 300.0)   # 像素眼距 = 100 px
    FRONTAL_NOSE      = (460.0, 360.0)   # 鼻尖正好在眼睛中点正下方

    calib_scale = HeadPoseDistanceEstimator.compute_calibration_scale(
        FRONTAL_LEFT_EYE, FRONTAL_RIGHT_EYE, real_ipd_m=0.063,
    )
    print(f"[Init] calibration_scale = {calib_scale:.6f} m/px")

    estimator = HeadPoseDistanceEstimator(
        calibration_scale=calib_scale,
        yaw_fuse_deg=45.0,
    )

    # ── 2. 第一帧（正脸）───────────────────────────────────────────────
    # nose_lateral = 460 - (410+510)/2 = 0，yaw=0°，Dreal=Dapp=100px
    # distance = 1/(0.024030*100 + 0.044812) = 1/2.4478 ≈ 0.409m
    d = estimator.compute_distance(FRONTAL_LEFT_EYE, FRONTAL_RIGHT_EYE, FRONTAL_NOSE)
    print(f"[Frame 0 | 正脸]   yaw≈0°,  distance = {d:.3f} m" if d else "[Frame 0] 无有效结果")

    # ── 3. 后续帧：不同偏转程度 ─────────────────────────────────────────
    test_frames = [
        # 轻微偏转：鼻子向右移了约 8px → nose_norm≈0.18，yaw≈17°
        ("轻微偏转 ~17°", (415.0, 301.0), (505.0, 299.0), (472.0, 361.0)),
        # 中度偏转：鼻子向右移了约 20px，眼距缩短 → nose_norm≈0.52，yaw≈31°
        ("中度偏转 ~31°", (430.0, 302.0), (495.0, 298.0), (480.0, 365.0)),
        # 严重侧脸：鼻子大幅右移，眼距很小 → yaw>45°，触发熔断
        ("严重侧脸（熔断）", (455.0, 305.0), (485.0, 300.0), (484.0, 370.0)),
    ]

    for desc, le, re, no in test_frames:
        d_new = estimator.compute_distance(le, re, no)
        # 计算并显示 nose_lateral 以验证逻辑
        Dapp_debug = ((re[0]-le[0])**2 + (re[1]-le[1])**2)**0.5
        eye_mid_x = (le[0] + re[0]) / 2.0
        nose_norm_debug = (no[0] - eye_mid_x) / (Dapp_debug / 2.0)
        yaw_debug = np.degrees(np.arctan(abs(nose_norm_debug) / _NOSE_SCALE))
        fused = (d_new == estimator.last_valid_distance and yaw_debug > 45)
        tag = "（沿用上帧）" if fused else ""
        print(f"[{desc:14s}] Dapp={Dapp_debug:5.1f}px  nose_norm={nose_norm_debug:+.2f}"
              f"  yaw≈{yaw_debug:.1f}°  distance={d_new:.3f}m {tag}"
              if d_new else f"[{desc}] 暂无有效距离")
