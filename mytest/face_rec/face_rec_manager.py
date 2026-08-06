"""
Minimal AdaFace manager used by MeetEye.

The wider AdaFace project is intentionally not imported wholesale here. This
module keeps only the pieces MeetEye needs:
  - load an IR-18 AdaFace checkpoint
  - load a directory of .npy identity features
  - load a directory of known face photos, using the filename stem as identity
  - align a face from YOLO pose keypoints
  - extract and match a 512D L2-normalized feature
  - cache recognition attempts by TrackID
"""
from __future__ import annotations

import os
import sys
import time
import json
import math
import base64
from functools import lru_cache
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch


_DIR = os.path.dirname(os.path.abspath(__file__))
_ADAFACE_DIR = os.path.join(_DIR, "AdaFace")
if _ADAFACE_DIR not in sys.path:
    sys.path.insert(0, _ADAFACE_DIR)

from net import build_model  # noqa: E402


_POSE_NOSE = 0
_POSE_LEFT_EYE = 1
_POSE_RIGHT_EYE = 2

# Ultralytics yolov8n-face 5-point order:
#   0=left eye, 1=right eye, 2=nose, 3=left mouth, 4=right mouth
_FACE_LEFT_EYE = 0
_FACE_RIGHT_EYE = 1
_FACE_NOSE = 2
_FACE_LEFT_MOUTH = 3
_FACE_RIGHT_MOUTH = 4
_NOSE_SCALE = 0.6
_KNOWN_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_ALIGNED_KNOWN_FACE_SUFFIXES = (
    "_aligned",
    "_adaface112",
    "_crop112",
    "_facecrop",
)

_ARCFACE_TEMPLATE_112 = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _cv2():
    import cv2
    return cv2


def _as_xyc(keypoints, idx: int) -> Optional[Tuple[float, float, float]]:
    if keypoints is None or idx >= len(keypoints):
        return None
    kp = keypoints[idx]
    if kp is None or len(kp) < 2:
        return None
    x, y = float(kp[0]), float(kp[1])
    conf = float(kp[2]) if len(kp) >= 3 else 1.0
    if (x == 0 and y == 0) or conf <= 0:
        return None
    return x, y, conf


def _pick_eye_nose_points(keypoints) -> Tuple[
    Optional[Tuple[float, float, float]],
    Optional[Tuple[float, float, float]],
    Optional[Tuple[float, float, float]],
]:
    """
    Return (left_eye, right_eye, nose) for supported keypoint layouts.

    MeetEye may run either a COCO pose model (17 keypoints) or a face model
    (5 keypoints). The index order is different, so using one fixed mapping
    silently crops the wrong area and makes recognition look "dead".
    """
    n = len(keypoints) if keypoints is not None else 0
    if n == 5:
        return (
            _as_xyc(keypoints, _FACE_LEFT_EYE),
            _as_xyc(keypoints, _FACE_RIGHT_EYE),
            _as_xyc(keypoints, _FACE_NOSE),
        )
    return (
        _as_xyc(keypoints, _POSE_LEFT_EYE),
        _as_xyc(keypoints, _POSE_RIGHT_EYE),
        _as_xyc(keypoints, _POSE_NOSE),
    )


def _normalize_align_mode(mode: Optional[str]) -> str:
    value = str(mode or "eye").strip().lower().replace("_", "-")
    if value in {"eye", "eyes"}:
        return "eye"
    if value in {"five-point", "5point", "fivepoint", "arcface"}:
        return "five-point"
    raise ValueError("face_rec align_mode must be one of: eye, five-point")


def _pick_face_five_points(keypoints) -> Optional[np.ndarray]:
    if keypoints is None or len(keypoints) != 5:
        return None
    points = []
    for idx in (
        _FACE_LEFT_EYE,
        _FACE_RIGHT_EYE,
        _FACE_NOSE,
        _FACE_LEFT_MOUTH,
        _FACE_RIGHT_MOUTH,
    ):
        kp = _as_xyc(keypoints, idx)
        if kp is None:
            return None
        x, y, conf = kp
        if conf < 0.1:
            return None
        points.append([x, y])
    return np.asarray(points, dtype=np.float32)


class FaceRecManager:
    """Small AdaFace wrapper for track-aware face recognition."""

    def __init__(
        self,
        model_path: str,
        library_dir: str,
        known_face_dir: Optional[str] = None,
        threshold: float = 0.35,
        frontal_yaw_thresh: float = 0.35,
        cooldown_frames: int = 30,
        device: str = "cpu",
        dynamic_library: bool = False,
        dynamic_library_dir: Optional[str] = None,
        dynamic_match_interval: Optional[int] = None,
        dynamic_max_samples_per_id: int = 5,
        dynamic_update_similarity: Optional[float] = None,
        dynamic_min_sample_diversity: float = 0.015,
        dynamic_primary_max_yaw_deg: float = 20.0,
        dynamic_min_keypoint_conf: float = 0.6,
        dynamic_update_min_face_height: Optional[int] = None,
        dynamic_update_min_keypoint_conf: float = 0.7,
        dynamic_update_min_score: float = 0.5,
        dynamic_primary_min_face_height: Optional[int] = None,
        dynamic_primary_min_keypoint_conf: float = 0.8,
        dynamic_primary_min_score: float = 0.6,
        dynamic_pose_sample_similarity: Optional[float] = None,
        dynamic_pose_library_similarity: Optional[float] = None,
        dynamic_pose_library_max_similarity: float = 0.90,
        dynamic_pose_sample_interval: int = 30,
        dynamic_pending_seed_samples: int = 2,
        dynamic_locked_sample_similarity: float = 0.35,
        dynamic_primary_ema_alpha: float = 0.1,
        dynamic_enroll_confirm_frames: int = 3,
        dynamic_binding_ttl_frames: int = 900,
        dynamic_supplement_fallback_threshold: Optional[float] = None,
        dynamic_match_margin: float = 0.08,
        dynamic_ambiguous_keep_bound: bool = True,
        dynamic_ambiguous_keep_min_score: Optional[float] = None,
        dynamic_global_assignment: bool = True,
        dynamic_auto_alias: bool = True,
        dynamic_alias_threshold: Optional[float] = None,
        dynamic_alias_min_samples: int = 2,
        dynamic_alias_min_hits: int = 2,
        dynamic_alias_margin: float = 0.03,
        dynamic_alias_probe_samples: int = 30,
        dynamic_switch_margin: float = 0.15,
        dynamic_min_face_height: int = 64,
        dynamic_enroll_max_yaw_deg: float = 30.0,
        dynamic_binding_mismatch_threshold: Optional[float] = None,
        dynamic_lock_to_track: bool = True,
        align_mode: str = "eye",
        five_point_scale: float = 1.20,
        known_feature_library: bool = False,
        known_feature_dir: Optional[str] = None,
        known_feature_update_threshold: float = 0.65,
        known_feature_pose_threshold: float = 0.60,
        known_feature_update_margin: float = 0.08,
        known_feature_primary_ema_alpha: float = 0.1,
        known_feature_max_samples_per_id: int = 12,
        debug_dump_dir: Optional[str] = None,
        debug_dump_every: int = 1,
        debug_dump_max: int = 0,
        observer_enabled: bool = False,
        observer_max_tracks: int = 4,
        observer_jpeg_size: int = 96,
        observer_pending_ttl_frames: int = 90,
    ):
        self.model_path = model_path
        self.library_dir = library_dir
        self.known_face_dir = known_face_dir
        self.threshold = float(threshold)
        self.frontal_yaw_thresh = float(frontal_yaw_thresh)
        self.cooldown_frames = max(1, int(cooldown_frames))
        self.dynamic_library = bool(dynamic_library)
        self.dynamic_match_interval = max(
            1, int(dynamic_match_interval or self.cooldown_frames)
        )
        self.dynamic_max_samples_per_id = max(1, int(dynamic_max_samples_per_id))
        self.dynamic_update_similarity = float(
            dynamic_update_similarity
            if dynamic_update_similarity is not None
            else max(0.15, self.threshold - 0.05)
        )
        self.dynamic_min_sample_diversity = float(dynamic_min_sample_diversity)
        self.dynamic_primary_max_yaw_deg = float(dynamic_primary_max_yaw_deg)
        self.dynamic_min_keypoint_conf = float(dynamic_min_keypoint_conf)
        self.dynamic_update_min_face_height = max(
            1, int(dynamic_update_min_face_height or dynamic_min_face_height)
        )
        self.dynamic_update_min_keypoint_conf = float(dynamic_update_min_keypoint_conf)
        self.dynamic_update_min_score = float(dynamic_update_min_score)
        self.dynamic_primary_min_face_height = max(
            1, int(dynamic_primary_min_face_height or self.dynamic_update_min_face_height)
        )
        self.dynamic_primary_min_keypoint_conf = float(dynamic_primary_min_keypoint_conf)
        self.dynamic_primary_min_score = float(dynamic_primary_min_score)
        self.dynamic_pose_sample_similarity = float(
            dynamic_pose_sample_similarity
            if dynamic_pose_sample_similarity is not None
            else self.threshold
        )
        self.dynamic_pose_library_similarity = float(
            dynamic_pose_library_similarity
            if dynamic_pose_library_similarity is not None
            else self.dynamic_pose_sample_similarity
        )
        self.dynamic_pose_library_max_similarity = float(dynamic_pose_library_max_similarity)
        self.dynamic_pose_sample_interval = max(0, int(dynamic_pose_sample_interval or 0))
        self.dynamic_pending_seed_samples = max(0, int(dynamic_pending_seed_samples or 0))
        self.dynamic_locked_sample_similarity = float(dynamic_locked_sample_similarity)
        self.dynamic_primary_ema_alpha = min(
            1.0, max(0.0, float(dynamic_primary_ema_alpha))
        )
        self.dynamic_enroll_confirm_frames = max(1, int(dynamic_enroll_confirm_frames))
        self.dynamic_binding_ttl_frames = max(1, int(dynamic_binding_ttl_frames))
        self.dynamic_supplement_fallback_threshold = float(
            dynamic_supplement_fallback_threshold
            if dynamic_supplement_fallback_threshold is not None
            else max(self.threshold, self.threshold + 0.10)
        )
        self.dynamic_match_margin = max(0.0, float(dynamic_match_margin))
        self.dynamic_ambiguous_keep_bound = bool(dynamic_ambiguous_keep_bound)
        self.dynamic_ambiguous_keep_min_score = float(
            dynamic_ambiguous_keep_min_score
            if dynamic_ambiguous_keep_min_score is not None
            else self.threshold
        )
        self.dynamic_global_assignment = bool(dynamic_global_assignment)
        self.dynamic_auto_alias = bool(dynamic_auto_alias)
        self.dynamic_alias_threshold = float(
            dynamic_alias_threshold
            if dynamic_alias_threshold is not None
            else max(0.50, self.threshold + 0.20)
        )
        self.dynamic_alias_min_samples = max(1, int(dynamic_alias_min_samples))
        self.dynamic_alias_min_hits = max(1, int(dynamic_alias_min_hits))
        self.dynamic_alias_margin = max(0.0, float(dynamic_alias_margin))
        self.dynamic_alias_probe_samples = max(1, int(dynamic_alias_probe_samples))
        self.dynamic_switch_margin = float(dynamic_switch_margin)
        self.dynamic_min_face_height = max(1, int(dynamic_min_face_height))
        self.dynamic_enroll_max_yaw_deg = float(dynamic_enroll_max_yaw_deg)
        self.dynamic_binding_mismatch_threshold = float(
            dynamic_binding_mismatch_threshold
            if dynamic_binding_mismatch_threshold is not None
            else max(0.10, self.threshold - 0.12)
        )
        self.dynamic_lock_to_track = bool(dynamic_lock_to_track)
        self.align_mode = _normalize_align_mode(align_mode)
        self.five_point_scale = min(1.60, max(0.70, float(five_point_scale)))
        self.known_feature_library = bool(known_feature_library)
        self.known_feature_dir = known_feature_dir or os.path.join(
            _DIR, "face_photo_features"
        )
        self.known_feature_update_threshold = float(known_feature_update_threshold)
        self.known_feature_pose_threshold = float(known_feature_pose_threshold)
        self.known_feature_update_margin = max(0.0, float(known_feature_update_margin))
        self.known_feature_primary_ema_alpha = min(
            1.0, max(0.0, float(known_feature_primary_ema_alpha))
        )
        self.known_feature_max_samples_per_id = max(1, int(known_feature_max_samples_per_id))
        self.dynamic_library_dir = dynamic_library_dir or os.path.join(
            "face_library_dynamic",
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}",
        )
        self.device = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.debug_dump_dir = debug_dump_dir
        self.debug_dump_every = max(1, int(debug_dump_every or 1))
        self.debug_dump_max = max(0, int(debug_dump_max or 0))
        self._debug_dump_count = 0
        self._debug_seen_features = 0
        self.observer_enabled = bool(observer_enabled)
        self.observer_max_tracks = max(1, int(observer_max_tracks or 4))
        self.observer_jpeg_size = max(48, min(160, int(observer_jpeg_size or 96)))
        self.observer_pending_ttl_frames = max(0, int(observer_pending_ttl_frames or 0))
        self._observer_latest: Dict[str, dict] = {}
        self._track_face_debug: Dict[int, dict] = {}
        if self.debug_dump_dir:
            os.makedirs(self.debug_dump_dir, exist_ok=True)
            os.makedirs(os.path.join(self.debug_dump_dir, "aligned_faces"), exist_ok=True)
            os.makedirs(os.path.join(self.debug_dump_dir, "bbox_crops"), exist_ok=True)
            os.makedirs(os.path.join(self.debug_dump_dir, "features"), exist_ok=True)
            print(f"[FaceRecDebug] dump enabled: {self.debug_dump_dir}")

        self.model = self._load_model(model_path)
        self.known_names, self.known_matrix = [], None
        self._known_name_set = set()
        self._known_feature_layout_loaded = set()
        self._known_feature_update_debug_seen = set()
        if self.dynamic_library:
            os.makedirs(self.dynamic_library_dir, exist_ok=True)
            os.makedirs(os.path.join(self.dynamic_library_dir, "_preview"), exist_ok=True)
            if library_dir and os.path.isdir(library_dir):
                self.known_names, self.known_matrix = self._load_library(library_dir)
                self._known_name_set = set(self.known_names)
            self.lib_names, self.lib_matrix = [], None
            self._dynamic_id_features: Dict[str, Dict[str, list]] = {}
            self._dynamic_anchor_by_id: Dict[str, dict] = {}
            self._dynamic_track_bindings: Dict[int, str] = {}
            # FaceID is a global identity. TrackID is only a temporary carrier.
            # Same-frame assignments are kept separately to prevent two visible
            # targets from drawing the same faceid in one frame.
            self._dynamic_frame_id: Optional[int] = None
            self._dynamic_frame_assignments: Dict[str, int] = {}
            self._dynamic_aliases: Dict[str, str] = {}
            self._dynamic_alias_votes: Dict[Tuple[str, str], int] = {}
            self._dynamic_alias_probe_features: Dict[str, list] = {}
            self._dynamic_identity_labels: Dict[str, dict] = {}
            self._known_bind_consecutive: Dict[str, dict] = {}
            self._dynamic_pending_enroll: Dict[int, dict] = {}
            self._dynamic_last_seen_frame: Dict[int, int] = {}
            self._dynamic_last_pose_sample_frame: Dict[str, int] = {}
            self._dynamic_next_id = 1
            print(f"[FaceRec] dynamic library enabled: {self.dynamic_library_dir}")
            print("[FaceRec] dynamic library starts empty; previous sessions are not loaded")
        else:
            self.lib_names, self.lib_matrix = self._load_library(library_dir)
            self.known_names = list(self.lib_names)
            self.known_matrix = self.lib_matrix
            self._known_name_set = set(self.known_names)
        self._last_attempt_frame: Dict[int, int] = {}

    # ------------------------------------------------------------------
    def _load_model(self, model_path: str) -> torch.nn.Module:
        model = build_model("ir_18")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"AdaFace checkpoint not found: {model_path}")

        ckpt = torch.load(model_path, map_location="cpu")
        state = ckpt
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]
                    break

        cleaned = {}
        for key, value in state.items():
            new_key = key
            for prefix in ("module.", "model."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            cleaned[new_key] = value

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[FaceRec] checkpoint missing keys: {len(missing)}")
        if unexpected:
            print(f"[FaceRec] checkpoint unexpected keys: {len(unexpected)}")

        model.to(self.device)
        model.eval()
        print(f"[FaceRec] AdaFace IR-18 loaded on {self.device}: {model_path}")
        return model

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_feature(feature: np.ndarray) -> np.ndarray:
        vec = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return vec
        return vec / norm

    def _load_library(self, library_dir: str) -> Tuple[list, Optional[np.ndarray]]:
        if not os.path.isdir(library_dir):
            print(f"[FaceRec] feature library not found: {library_dir}")
            return [], None

        names, feats = [], []
        for fname in sorted(os.listdir(library_dir)):
            if not fname.endswith(".npy"):
                continue
            path = os.path.join(library_dir, fname)
            try:
                arr = np.load(path)
            except Exception as exc:
                print(f"[FaceRec] skip invalid feature {path}: {exc}")
                continue

            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            for idx, row in enumerate(arr):
                feat = self._normalize_feature(row)
                if feat.size != 512 or np.linalg.norm(feat) <= 1e-6:
                    continue
                stem = os.path.splitext(fname)[0]
                names.append(stem if arr.shape[0] == 1 else f"{stem}#{idx + 1}")
                feats.append(feat)

        if not feats:
            print(f"[FaceRec] no usable .npy features in: {library_dir}")
            return [], None

        matrix = np.stack(feats).astype(np.float32)
        print(f"[FaceRec] loaded {len(names)} face feature(s) from {library_dir}")
        return names, matrix

    def _append_known_features(self, names: list, feats: list) -> int:
        valid_names, valid_feats = [], []
        for name, feat in zip(names, feats):
            feat = self._normalize_feature(feat)
            if not name or feat.size != 512 or np.linalg.norm(feat) <= 1e-6:
                continue
            valid_names.append(str(name))
            valid_feats.append(feat.astype(np.float32))
        if not valid_feats:
            return 0

        if self.known_matrix is None or not self.known_names:
            self.known_names = valid_names
            self.known_matrix = np.stack(valid_feats).astype(np.float32)
        else:
            self.known_names.extend(valid_names)
            self.known_matrix = np.concatenate(
                [self.known_matrix, np.stack(valid_feats).astype(np.float32)],
                axis=0,
            )
        self._known_name_set = set(self.known_names)
        return len(valid_feats)

    def _known_features_for_name(self, name: str) -> list:
        if self.known_matrix is None or not self.known_names:
            return []
        return [
            self._normalize_feature(feature)
            for known_name, feature in zip(self.known_names, self.known_matrix)
            if known_name == name
        ]

    def _rebuild_known_features_for_name(self, name: str, feats: list) -> None:
        keep_names, keep_feats = [], []
        if self.known_matrix is not None and self.known_names:
            for known_name, feature in zip(self.known_names, self.known_matrix):
                if known_name == name:
                    continue
                keep_names.append(known_name)
                keep_feats.append(self._normalize_feature(feature))
        for feature in feats:
            keep_names.append(name)
            keep_feats.append(self._normalize_feature(feature))
        self.known_names = keep_names
        self.known_matrix = (
            np.stack(keep_feats).astype(np.float32) if keep_feats else None
        )
        self._known_name_set = set(self.known_names)

    def _known_feature_path(self, name: str) -> str:
        return os.path.join(self.known_feature_dir, f"{name}.npy")

    def _known_feature_meta_path(self, name: str) -> str:
        return os.path.join(self.known_feature_dir, f"{name}.json")

    def _known_feature_has_layout_file(self, name: str) -> bool:
        path = self._known_feature_meta_path(name)
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return False
        return meta.get("layout") == "image_primary_pose_v1"

    def _known_feature_has_layout_meta(self, name: str) -> bool:
        return name in self._known_feature_layout_loaded or self._known_feature_has_layout_file(name)

    def _save_known_features_for_name(self, name: str) -> None:
        if not self.known_feature_library:
            return
        feats = self._known_features_for_name(name)
        if not feats:
            return
        os.makedirs(self.known_feature_dir, exist_ok=True)
        np.save(self._known_feature_path(name), np.stack(feats).astype(np.float32))
        meta = {
            "layout": "image_primary_pose_v1",
            "image_index": 0,
            "primary_index": 1 if len(feats) >= 2 else None,
            "pose_start_index": 2,
            "max_pose_samples": int(self.known_feature_max_samples_per_id),
            "primary_ema_alpha": float(self.known_feature_primary_ema_alpha),
        }
        with open(self._known_feature_meta_path(name), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self._known_feature_layout_loaded.add(name)

    def _known_feature_layout_for_name(
        self, name: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], list]:
        rows = self._known_features_for_name(name)
        if not rows:
            return None, None, []
        image_feature = rows[0]
        has_meta = self._known_feature_has_layout_meta(name)
        if has_meta and len(rows) >= 2:
            primary_feature = rows[1]
            pose_features = rows[2:]
        else:
            # Legacy files were flat [photo, sample1, sample2, ...]. Treat only
            # row0 as the immutable image feature and rebuild primary from it.
            primary_feature = image_feature
            pose_features = rows[1:]
        return (
            self._normalize_feature(image_feature),
            self._normalize_feature(primary_feature),
            [self._normalize_feature(row) for row in pose_features[:self.known_feature_max_samples_per_id]],
        )

    def _rebuild_known_feature_layout_for_name(
        self,
        name: str,
        image_feature: np.ndarray,
        primary_feature: np.ndarray,
        pose_features: list,
    ) -> None:
        rows = [
            self._normalize_feature(image_feature),
            self._normalize_feature(primary_feature),
        ]
        rows.extend(
            self._normalize_feature(row)
            for row in pose_features[:self.known_feature_max_samples_per_id]
        )
        self._rebuild_known_features_for_name(name, rows)

    def _set_known_image_feature_for_name(
        self,
        name: str,
        image_feature: np.ndarray,
    ) -> None:
        image_feature = self._normalize_feature(image_feature)
        old_image, old_primary, old_poses = self._known_feature_layout_for_name(name)
        if old_image is None or not self._known_feature_has_layout_file(name):
            primary_feature = image_feature
        else:
            primary_feature = old_primary if old_primary is not None else image_feature
        self._rebuild_known_feature_layout_for_name(
            name,
            image_feature,
            primary_feature,
            old_poses,
        )
        self._save_known_features_for_name(name)

    def _load_known_feature_library(self, allowed_names: set) -> int:
        if not self.known_feature_library or not self.known_feature_dir:
            return 0
        if not os.path.isdir(self.known_feature_dir):
            return 0

        names, feats = [], []
        for fname in sorted(os.listdir(self.known_feature_dir)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() != ".npy" or stem not in allowed_names:
                continue
            path = os.path.join(self.known_feature_dir, fname)
            try:
                arr = np.load(path)
            except Exception as exc:
                print(f"[FaceRec] skip invalid known feature {path}: {exc}")
                continue
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            rows = []
            for row in arr:
                feat = self._normalize_feature(row)
                if feat.size != 512 or np.linalg.norm(feat) <= 1e-6:
                    continue
                rows.append(feat)
            if not rows:
                continue
            if self._known_feature_has_layout_file(stem) and len(rows) >= 2:
                layout_rows = rows[:2 + self.known_feature_max_samples_per_id]
            else:
                layout_rows = [rows[0], rows[0]]
                layout_rows.extend(rows[1:1 + self.known_feature_max_samples_per_id])
            self._known_feature_layout_loaded.add(stem)
            names.extend([stem] * len(layout_rows))
            feats.extend(layout_rows)
        added = self._append_known_features(names, feats)
        if added:
            print(f"[FaceRec] loaded {added} known face feature(s) from {self.known_feature_dir}")
        return added

    def _append_known_feature_sample(self, name: str, feature: np.ndarray) -> bool:
        if not self.known_feature_library or not name:
            return False
        feat = self._normalize_feature(feature)
        image_feature, primary_feature, pose_features = self._known_feature_layout_for_name(name)
        if image_feature is None:
            image_feature = feat
            primary_feature = feat
            pose_features = []
        existing = [image_feature, primary_feature] + pose_features
        if existing:
            best_existing = max(float(np.dot(old, feat)) for old in existing)
            if best_existing >= 1.0 - self.dynamic_min_sample_diversity:
                return False

        pose_features.append(feat)
        pose_features = pose_features[-self.known_feature_max_samples_per_id:]
        self._rebuild_known_feature_layout_for_name(
            name, image_feature, primary_feature, pose_features
        )
        self._save_known_features_for_name(name)
        return True

    def _set_known_update_debug(
        self,
        quality: Optional[dict],
        *,
        reason: str,
        name: Optional[str] = None,
        matched_name: Optional[str] = None,
        score: Optional[float] = None,
        second_score: Optional[float] = None,
        best_existing: Optional[float] = None,
        sample_count: Optional[int] = None,
        image_score: Optional[float] = None,
        primary_score: Optional[float] = None,
        pose_score: Optional[float] = None,
        updated_role: Optional[str] = None,
        updated: bool = False,
    ) -> None:
        if quality is None:
            return
        quality["known_update_reason"] = reason
        quality["known_update_name"] = name
        quality["known_update_matched_name"] = matched_name
        quality["known_update_score"] = None if score is None else float(score)
        quality["known_update_second_score"] = (
            None if second_score is None else float(second_score)
        )
        quality["known_update_threshold"] = float(self.known_feature_update_threshold)
        quality["known_update_pose_threshold"] = float(self.known_feature_pose_threshold)
        quality["known_update_match_threshold"] = float(self.threshold)
        quality["known_update_margin"] = float(self.known_feature_update_margin)
        quality["known_update_best_existing"] = (
            None if best_existing is None else float(best_existing)
        )
        quality["known_update_image_score"] = (
            None if image_score is None else float(image_score)
        )
        quality["known_update_primary_score"] = (
            None if primary_score is None else float(primary_score)
        )
        quality["known_update_pose_score"] = (
            None if pose_score is None else float(pose_score)
        )
        quality["known_update_duplicate_threshold"] = float(
            1.0 - self.dynamic_min_sample_diversity
        )
        quality["known_update_primary_ema_alpha"] = float(
            self.known_feature_primary_ema_alpha
        )
        quality["known_update_sample_count"] = sample_count
        quality["known_update_role"] = updated_role
        quality["known_update_updated"] = bool(updated)

    @staticmethod
    def _score_debug_value(score: Optional[float]) -> Optional[float]:
        return None if score is None else round(float(score), 6)

    def _set_known_bind_debug(
        self,
        quality: Optional[dict],
        *,
        reason: str,
        canonical: Optional[str] = None,
        feature_count: Optional[int] = None,
        min_hits: Optional[int] = None,
        sample_scores: Optional[list] = None,
        votes: Optional[Dict[str, dict]] = None,
        selected_name: Optional[str] = None,
        selected_score: Optional[float] = None,
        selected_hits: Optional[int] = None,
        current_name: Optional[str] = None,
        current_score: Optional[float] = None,
        current_hits: Optional[int] = None,
        result_name: Optional[str] = None,
    ) -> None:
        if quality is None:
            return
        quality["known_bind_reason"] = reason
        quality["known_bind_dynamic_id"] = canonical
        quality["known_bind_feature_count"] = feature_count
        quality["known_bind_min_hits"] = min_hits
        quality["known_bind_selected_name"] = selected_name
        quality["known_bind_selected_score"] = self._score_debug_value(selected_score)
        quality["known_bind_selected_hits"] = selected_hits
        quality["known_bind_current_name"] = current_name
        quality["known_bind_current_score"] = self._score_debug_value(current_score)
        quality["known_bind_current_hits"] = current_hits
        quality["known_bind_result_name"] = result_name

        if sample_scores is not None:
            quality["known_bind_scores"] = sample_scores
        if votes is not None:
            vote_rows = []
            for name, info in votes.items():
                vote_rows.append({
                    "name": name,
                    "hits": int(info.get("hits") or 0),
                    "best": self._score_debug_value(info.get("best")),
                })
            vote_rows.sort(key=lambda row: (row["hits"], row["best"] or 0.0), reverse=True)
            quality["known_bind_votes"] = vote_rows

    def _known_sample_scores_debug(self, scores: list, sample_index: int) -> dict:
        name, score, second_score = self._match_from_scores(scores)
        second_name = scores[1][0] if len(scores) >= 2 else None
        accepted_vote = (
            name is not None
            and score >= second_score + self.dynamic_match_margin
        )
        return {
            "sample": int(sample_index),
            "winner": name,
            "score": self._score_debug_value(score),
            "second_name": second_name,
            "second_score": self._score_debug_value(second_score),
            "margin": self._score_debug_value(score - second_score),
            "required_margin": self._score_debug_value(self.dynamic_match_margin),
            "accepted_vote": bool(accepted_vote),
            "top": [
                {"name": top_name, "score": self._score_debug_value(top_score)}
                for top_name, top_score in scores[:8]
            ],
        }

    def _print_known_update_skip_once(
        self,
        name: Optional[str],
        reason: str,
        score: Optional[float],
        second_score: Optional[float],
        best_existing: Optional[float],
    ) -> None:
        if reason == "updated":
            return
        key = (str(name or "null"), str(reason))
        if key in self._known_feature_update_debug_seen:
            return
        self._known_feature_update_debug_seen.add(key)
        score_text = "N/A" if score is None else f"{score:.3f}"
        second_text = "N/A" if second_score is None else f"{second_score:.3f}"
        existing_text = "N/A" if best_existing is None else f"{best_existing:.3f}"
        print(
            f"[FaceRec] known feature skip {name or 'null'}: {reason} "
            f"(score={score_text}, second={second_text}, existing={existing_text})"
        )

    def _maybe_update_known_feature_library(
        self,
        name: Optional[str],
        feature: Optional[np.ndarray],
        quality: Optional[dict] = None,
    ) -> bool:
        if not self.known_feature_library:
            self._set_known_update_debug(quality, reason="disabled", name=name)
            return False
        if not name:
            self._set_known_update_debug(quality, reason="missing_name", name=name)
            return False
        if feature is None:
            self._set_known_update_debug(quality, reason="missing_feature", name=name)
            return False
        if self.known_matrix is None or not self.known_names:
            self._set_known_update_debug(quality, reason="empty_known_library", name=name)
            self._print_known_update_skip_once(name, "empty_known_library", None, None, None)
            return False

        known_scores = self._dedup_scores_by_name(self._known_identity_scores(feature))
        matched_name, score, second_score = self._match_from_scores(known_scores)
        top_name = known_scores[0][0] if known_scores else None
        if score < self.threshold:
            self._set_known_update_debug(
                quality,
                reason="known_match_below_threshold",
                name=name,
                matched_name=top_name,
                score=score,
                second_score=second_score,
            )
            self._print_known_update_skip_once(
                name, "known_match_below_threshold", score, second_score, None
            )
            return False
        if matched_name is None:
            self._set_known_update_debug(
                quality,
                reason="known_match_ambiguous",
                name=name,
                matched_name=top_name,
                score=score,
                second_score=second_score,
            )
            self._print_known_update_skip_once(
                name, "known_match_ambiguous", score, second_score, None
            )
            return False
        if matched_name != name:
            self._set_known_update_debug(
                quality,
                reason="matched_other_name",
                name=name,
                matched_name=matched_name,
                score=score,
                second_score=second_score,
            )
            self._print_known_update_skip_once(
                name, "matched_other_name", score, second_score, None
            )
            return False
        if score < self.known_feature_update_threshold:
            self._set_known_update_debug(
                quality,
                reason="low_similarity",
                name=name,
                matched_name=matched_name,
                score=score,
                second_score=second_score,
            )
            self._print_known_update_skip_once(
                name, "low_similarity", score, second_score, None
            )
            return False
        if score < second_score + self.known_feature_update_margin:
            self._set_known_update_debug(
                quality,
                reason="ambiguous_margin",
                name=name,
                matched_name=matched_name,
                score=score,
                second_score=second_score,
            )
            self._print_known_update_skip_once(
                name, "ambiguous_margin", score, second_score, None
            )
            return False

        feat = self._normalize_feature(feature)
        image_feature, primary_feature, pose_features = self._known_feature_layout_for_name(name)
        if image_feature is None or primary_feature is None:
            self._set_known_update_debug(
                quality,
                reason="missing_image_feature",
                name=name,
                matched_name=matched_name,
                score=score,
                second_score=second_score,
            )
            self._print_known_update_skip_once(
                name, "missing_image_feature", score, second_score, None
            )
            return False

        image_score = float(np.dot(image_feature, feat))
        primary_score = float(np.dot(primary_feature, feat))
        pose_score = (
            max(float(np.dot(row, feat)) for row in pose_features)
            if pose_features else None
        )
        existing = [image_feature, primary_feature] + pose_features
        best_existing = max(float(np.dot(old, feat)) for old in existing)
        sample_count = len(existing)
        duplicate_threshold = 1.0 - self.dynamic_min_sample_diversity

        if image_score >= self.known_feature_update_threshold:
            if primary_score >= duplicate_threshold:
                self._set_known_update_debug(
                    quality,
                    reason="primary_too_similar",
                    name=name,
                    matched_name=matched_name,
                    score=score,
                    second_score=second_score,
                    best_existing=best_existing,
                    sample_count=sample_count,
                    image_score=image_score,
                    primary_score=primary_score,
                    pose_score=pose_score,
                    updated_role="primary",
                )
                self._print_known_update_skip_once(
                    name, "primary_too_similar", score, second_score, best_existing
                )
                return False
            alpha = self.known_feature_primary_ema_alpha
            new_primary = self._normalize_feature(
                (1.0 - alpha) * primary_feature + alpha * feat
            )
            self._rebuild_known_feature_layout_for_name(
                name, image_feature, new_primary, pose_features
            )
            self._save_known_features_for_name(name)
            count = len(self._known_features_for_name(name))
            self._set_known_update_debug(
                quality,
                reason="primary_ema_updated",
                name=name,
                matched_name=matched_name,
                score=score,
                second_score=second_score,
                best_existing=best_existing,
                sample_count=count,
                image_score=image_score,
                primary_score=primary_score,
                pose_score=pose_score,
                updated_role="primary",
                updated=True,
            )
            print(
                f"[FaceRec] known feature update {name}: primary EMA, "
                f"{count} sample(s), image_score={image_score:.3f}"
            )
            return True

        if image_score < self.known_feature_pose_threshold:
            self._set_known_update_debug(
                quality,
                reason="low_image_similarity",
                name=name,
                matched_name=matched_name,
                score=score,
                second_score=second_score,
                best_existing=best_existing,
                sample_count=sample_count,
                image_score=image_score,
                primary_score=primary_score,
                pose_score=pose_score,
            )
            self._print_known_update_skip_once(
                name, "low_image_similarity", score, second_score, best_existing
            )
            return False

        if best_existing >= duplicate_threshold:
            self._set_known_update_debug(
                quality,
                reason="pose_too_similar",
                name=name,
                matched_name=matched_name,
                score=score,
                second_score=second_score,
                best_existing=best_existing,
                sample_count=sample_count,
                image_score=image_score,
                primary_score=primary_score,
                pose_score=pose_score,
                updated_role="pose",
            )
            self._print_known_update_skip_once(
                name, "pose_too_similar", score, second_score, best_existing
            )
            return False

        pose_features.append(feat)
        pose_features = pose_features[-self.known_feature_max_samples_per_id:]
        self._rebuild_known_feature_layout_for_name(
            name, image_feature, primary_feature, pose_features
        )
        self._save_known_features_for_name(name)
        count = len(self._known_features_for_name(name))
        self._set_known_update_debug(
            quality,
            reason="pose_sample_updated",
            name=name,
            matched_name=matched_name,
            score=score,
            second_score=second_score,
            best_existing=best_existing,
            sample_count=count,
            image_score=image_score,
            primary_score=primary_score,
            pose_score=pose_score,
            updated_role="pose",
            updated=True,
        )
        print(
            f"[FaceRec] known feature update {name}: pose sample, "
            f"{count} sample(s), image_score={image_score:.3f}"
        )
        return True

    def _read_image_bgr(self, path: str) -> Optional[np.ndarray]:
        cv2 = _cv2()
        try:
            data = np.fromfile(path, dtype=np.uint8)
        except Exception:
            return None
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img if img is not None and img.size else None

    def _center_reference_face(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        cv2 = _cv2()
        if image_bgr is None or image_bgr.size == 0:
            return None
        h, w = image_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return None
        side = min(h, w)
        x1 = max(0, (w - side) // 2)
        y1 = max(0, (h - side) // 2)
        crop = image_bgr[y1:y1 + side, x1:x1 + side]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _known_face_identity_from_stem(stem: str) -> Tuple[str, bool]:
        lower = stem.lower()
        for suffix in _ALIGNED_KNOWN_FACE_SUFFIXES:
            if lower.endswith(suffix) and len(stem) > len(suffix):
                return stem[:-len(suffix)], True
        return stem, False

    def _select_known_face_image_files(self, image_dir: str) -> List[Tuple[str, str, bool]]:
        selected: Dict[str, Tuple[str, bool]] = {}
        for fname in sorted(os.listdir(image_dir)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in _KNOWN_IMAGE_EXTENSIONS:
                continue
            name, aligned_crop = self._known_face_identity_from_stem(stem)
            if not name:
                continue
            current = selected.get(name)
            if current is None or (aligned_crop and not current[1]):
                selected[name] = (fname, aligned_crop)
        return [(fname, name, aligned_crop) for name, (fname, aligned_crop) in selected.items()]

    def _aligned_reference_face(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        cv2 = _cv2()
        if image_bgr is None or image_bgr.size == 0:
            return None
        return cv2.resize(image_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)

    def _detect_reference_face(
        self,
        image_bgr: np.ndarray,
        detector=None,
        imgsz: Optional[int] = None,
        conf: float = 0.1,
        iou: float = 0.99,
    ) -> Tuple[Optional[np.ndarray], str]:
        if detector is None or image_bgr is None or image_bgr.size == 0:
            return self._center_reference_face(image_bgr), "center_crop"
        # Prefer the public Ultralytics YOLO.predict API. YOLOPoseDetector
        # wrappers store that object in .model, while a raw YOLO instance's
        # .model is the lower-level BaseModel and does not accept imgsz/conf.
        model = detector if hasattr(detector, "predict") else getattr(detector, "model", detector)
        predict_kwargs = {
            "imgsz": int(imgsz or 864),
            "conf": float(conf),
            "iou": float(iou),
            "verbose": False,
        }
        if torch.cuda.is_available():
            predict_kwargs.update({"half": True, "device": 0})
        try:
            results = model.predict(image_bgr, **predict_kwargs)
        except AssertionError as exc:
            msg = str(exc)
            if "input size" not in msg or "max model size" not in msg:
                print(f"[FaceRec] known face detector failed, fallback center crop: {exc}")
                return self._center_reference_face(image_bgr), "center_crop"
            try:
                batch_size = 3
                results = model.predict(
                    [image_bgr, image_bgr.copy(), image_bgr.copy()],
                    **predict_kwargs,
                )
                results = list(results or [])[:1]
                print(
                    "[FaceRec] known face detector retried with "
                    f"duplicated batch={batch_size}"
                )
            except Exception as retry_exc:
                print(
                    "[FaceRec] known face detector batch retry failed, "
                    f"fallback center crop: {retry_exc}"
                )
                return self._center_reference_face(image_bgr), "center_crop"
        except Exception as exc:
            print(f"[FaceRec] known face detector failed, fallback center crop: {exc}")
            return self._center_reference_face(image_bgr), "center_crop"

        best = None
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            boxes_data = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            keypoints_data = []
            if hasattr(result, "keypoints") and result.keypoints is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()
            for idx, (box, score) in enumerate(zip(boxes_data, confidences)):
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                rank = area * max(0.01, float(score))
                kps = keypoints_data[idx].tolist() if idx < len(keypoints_data) else None
                if best is None or rank > best[0]:
                    best = (rank, [x1, y1, x2, y2], kps)
        if best is None:
            return self._center_reference_face(image_bgr), "center_crop"

        _rank, bbox, keypoints = best
        if keypoints is not None:
            face = self.align_face(image_bgr, keypoints)
            if face is not None:
                return face, "detector_align"

        safe = self._safe_bbox(bbox, image_bgr.shape[1], image_bgr.shape[0])
        if safe is None:
            return self._center_reference_face(image_bgr), "center_crop"
        x1, y1, x2, y2 = safe
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return self._center_reference_face(image_bgr), "center_crop"
        return _cv2().resize(crop, (112, 112), interpolation=_cv2().INTER_LINEAR), "detector_bbox"

    def load_known_face_images(
        self,
        image_dir: Optional[str] = None,
        *,
        detector=None,
        imgsz: Optional[int] = None,
        conf: float = 0.1,
        iou: float = 0.99,
    ) -> int:
        """Load known people from photos. The filename stem is the displayed identity."""
        image_dir = image_dir or self.known_face_dir
        if not image_dir:
            return 0
        if not os.path.isdir(image_dir):
            print(f"[FaceRec] known face image dir not found: {image_dir}")
            return 0

        image_files = self._select_known_face_image_files(image_dir)
        image_names = {name for _fname, name, _aligned_crop in image_files}
        if self.known_feature_library:
            os.makedirs(self.known_feature_dir, exist_ok=True)
            self._load_known_feature_library(image_names)

        names, feats = [], []
        align_counts: Dict[str, int] = {}
        skipped_existing = 0
        refreshed_existing = 0
        for fname, name, aligned_crop in image_files:
            path = os.path.join(image_dir, fname)
            img = self._read_image_bgr(path)
            if img is None:
                print(f"[FaceRec] skip invalid known face image: {path}")
                continue
            if aligned_crop:
                face, mode = self._aligned_reference_face(img), "aligned_crop"
            else:
                face, mode = self._detect_reference_face(
                    img, detector=detector, imgsz=imgsz, conf=conf, iou=iou
                )
            if face is None:
                print(f"[FaceRec] skip known face image without usable face: {path}")
                continue
            try:
                feat = self.extract_feature(face)
            except Exception as exc:
                print(f"[FaceRec] known face feature failed {path}: {exc}")
                continue
            if self.known_feature_library and name in self._known_name_set:
                self._set_known_image_feature_for_name(name, feat)
                refreshed_existing += 1
                align_counts[mode] = align_counts.get(mode, 0) + 1
                continue
            names.append(name)
            feats.append(feat)
            align_counts[mode] = align_counts.get(mode, 0) + 1

        added = self._append_known_features(names, feats)
        if self.known_feature_library:
            for name in names:
                self._set_known_image_feature_for_name(
                    name, self._known_features_for_name(name)[0]
                )
                self._save_known_features_for_name(name)
        if added:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(align_counts.items()))
            print(f"[FaceRec] loaded {added} known face image(s) from {image_dir}"
                  + (f" ({detail})" if detail else ""))
            if self.known_feature_library:
                print(f"[FaceRec] known feature library initialized/filled in {self.known_feature_dir}")
        elif skipped_existing or refreshed_existing:
            print(
                "[FaceRec] known face images mapped to existing feature library: "
                f"{skipped_existing + refreshed_existing}"
            )
        elif any(os.path.splitext(f)[1].lower() in _KNOWN_IMAGE_EXTENSIONS
                 for f in os.listdir(image_dir)):
            print(f"[FaceRec] no usable known face image in: {image_dir}")
        return added

    # ------------------------------------------------------------------
    def is_frontal(self, keypoints) -> bool:
        yaw_proxy = self._yaw_proxy(keypoints)
        if yaw_proxy is None:
            return False
        return yaw_proxy <= self.frontal_yaw_thresh

    def _yaw_deg(self, keypoints) -> Optional[float]:
        yaw_proxy = self._yaw_proxy(keypoints)
        if yaw_proxy is None:
            return None
        tan_yaw = yaw_proxy / _NOSE_SCALE
        return float(np.degrees(np.arctan(tan_yaw)))

    def _yaw_proxy(self, keypoints) -> Optional[float]:
        left_eye, right_eye, nose = _pick_eye_nose_points(keypoints)
        if nose is None or left_eye is None or right_eye is None:
            return None

        lx, ly, lc = left_eye
        rx, ry, rc = right_eye
        nx, _ny, nc = nose
        if min(lc, rc, nc) < 0.1:
            return None

        eye_dist = float(np.hypot(rx - lx, ry - ly))
        if eye_dist < 3.0:
            return None

        eye_mid_x = (lx + rx) * 0.5
        return abs(nx - eye_mid_x) / eye_dist

    @staticmethod
    def _face_height(face_bgr: np.ndarray) -> int:
        return int(face_bgr.shape[0]) if face_bgr is not None and face_bgr.ndim >= 2 else 0

    def _face_to_data_uri(self, face_bgr: Optional[np.ndarray]) -> Optional[str]:
        if not self.observer_enabled:
            return None
        if face_bgr is None or not isinstance(face_bgr, np.ndarray) or face_bgr.size == 0:
            return None
        cv2 = _cv2()
        img = face_bgr
        if img.shape[0] != self.observer_jpeg_size or img.shape[1] != self.observer_jpeg_size:
            img = cv2.resize(
                img,
                (self.observer_jpeg_size, self.observer_jpeg_size),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        data = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{data}"

    def _observer_quality_payload(self, quality: Optional[dict]) -> dict:
        quality = quality or {}
        return {
            "reason": quality.get("reason"),
            "sample_reason": quality.get("sample_reason"),
            "primary_reason": quality.get("primary_reason"),
            "trigger_ok": bool(quality.get("trigger_ok", False)),
            "sample_ok": bool(quality.get("sample_ok", False)),
            "primary_ok": bool(quality.get("primary_ok", False)),
            "locked_pose_quality_ok": bool(quality.get("locked_pose_quality_ok", False)),
            "pose_sample_quality_ok": bool(quality.get("pose_sample_quality_ok", False)),
            "min_side": None if quality.get("min_side") is None else round(float(quality.get("min_side")), 2),
            "min_keypoint_conf": None if quality.get("min_keypoint_conf") is None else round(float(quality.get("min_keypoint_conf")), 3),
            "det_score": None if quality.get("det_score") is None else round(float(quality.get("det_score")), 3),
            "yaw_valid": quality.get("yaw_deg") is not None,
            "yaw_deg": None if quality.get("yaw_deg") is None else round(float(quality.get("yaw_deg")), 2),
            "quality_score": None if quality.get("quality_score") is None else round(float(quality.get("quality_score")), 3),
            "update_reason": quality.get("update_reason"),
            "best_existing": None if quality.get("best_existing") is None else round(float(quality.get("best_existing")), 3),
            "best_other": None if quality.get("best_other") is None else round(float(quality.get("best_other")), 3),
            "sample_similarity": None if quality.get("sample_similarity") is None else round(float(quality.get("sample_similarity")), 3),
            "pose_library_similarity": None if quality.get("pose_library_similarity") is None else round(float(quality.get("pose_library_similarity")), 3),
            "pose_library_threshold": None if quality.get("pose_library_threshold") is None else round(float(quality.get("pose_library_threshold")), 3),
            "pose_library_max_similarity": None if quality.get("pose_library_max_similarity") is None else round(float(quality.get("pose_library_max_similarity")), 3),
            "pose_sample_interval": quality.get("pose_sample_interval"),
            "pose_sample_missing_frames": quality.get("pose_sample_missing_frames"),
            "known_update_reason": quality.get("known_update_reason"),
            "known_update_name": quality.get("known_update_name"),
            "known_update_matched_name": quality.get("known_update_matched_name"),
            "known_update_score": None if quality.get("known_update_score") is None else round(float(quality.get("known_update_score")), 3),
            "known_update_second_score": None if quality.get("known_update_second_score") is None else round(float(quality.get("known_update_second_score")), 3),
            "known_update_threshold": None if quality.get("known_update_threshold") is None else round(float(quality.get("known_update_threshold")), 3),
            "known_update_pose_threshold": None if quality.get("known_update_pose_threshold") is None else round(float(quality.get("known_update_pose_threshold")), 3),
            "known_update_match_threshold": None if quality.get("known_update_match_threshold") is None else round(float(quality.get("known_update_match_threshold")), 3),
            "known_update_margin": None if quality.get("known_update_margin") is None else round(float(quality.get("known_update_margin")), 3),
            "known_update_best_existing": None if quality.get("known_update_best_existing") is None else round(float(quality.get("known_update_best_existing")), 3),
            "known_update_image_score": None if quality.get("known_update_image_score") is None else round(float(quality.get("known_update_image_score")), 3),
            "known_update_primary_score": None if quality.get("known_update_primary_score") is None else round(float(quality.get("known_update_primary_score")), 3),
            "known_update_pose_score": None if quality.get("known_update_pose_score") is None else round(float(quality.get("known_update_pose_score")), 3),
            "known_update_duplicate_threshold": None if quality.get("known_update_duplicate_threshold") is None else round(float(quality.get("known_update_duplicate_threshold")), 3),
            "known_update_primary_ema_alpha": None if quality.get("known_update_primary_ema_alpha") is None else round(float(quality.get("known_update_primary_ema_alpha")), 3),
            "known_update_sample_count": quality.get("known_update_sample_count"),
            "known_update_role": quality.get("known_update_role"),
            "known_update_updated": bool(quality.get("known_update_updated", False)),
            "known_bind_reason": quality.get("known_bind_reason"),
            "known_bind_dynamic_id": quality.get("known_bind_dynamic_id"),
            "known_bind_feature_count": quality.get("known_bind_feature_count"),
            "known_bind_min_hits": quality.get("known_bind_min_hits"),
            "known_bind_selected_name": quality.get("known_bind_selected_name"),
            "known_bind_selected_score": quality.get("known_bind_selected_score"),
            "known_bind_selected_hits": quality.get("known_bind_selected_hits"),
            "known_bind_current_name": quality.get("known_bind_current_name"),
            "known_bind_current_score": quality.get("known_bind_current_score"),
            "known_bind_current_hits": quality.get("known_bind_current_hits"),
            "known_bind_result_name": quality.get("known_bind_result_name"),
            "known_bind_votes": quality.get("known_bind_votes"),
            "known_bind_scores": quality.get("known_bind_scores"),
        }

    def _remember_track_face_debug(
        self,
        *,
        track_id: int,
        frame_id: int,
        quality: Optional[dict],
        face_id: Optional[str],
        raw_face_id: Optional[str],
        event: str,
        score: Optional[float],
        second_score: Optional[float],
    ) -> None:
        self._track_face_debug[int(track_id)] = {
            "track_id": int(track_id),
            "frame_id": int(frame_id),
            "face_id": face_id,
            "raw_face_id": raw_face_id,
            "event": event,
            "score": None if score is None else round(float(score), 3),
            "second_score": None if second_score is None else round(float(second_score), 3),
            "quality": self._observer_quality_payload(quality),
        }

    def get_track_face_debug(self, track_id: int) -> Optional[dict]:
        item = self._track_face_debug.get(int(track_id))
        return dict(item) if item else None

    def _observer_feature_similarity(self, feature: np.ndarray, ref: np.ndarray) -> float:
        return float(np.dot(self._normalize_feature(feature), self._normalize_feature(ref)))

    def _observer_identity_samples(self, name: Optional[str], feature: np.ndarray) -> dict:
        canonical = self._resolve_dynamic_alias(name) if self.dynamic_library else name
        if not canonical or not self.dynamic_library:
            return {"face_id": canonical, "primary": [], "supplement": []}
        groups = self._dynamic_id_features.get(canonical, {})
        primary_feats = groups.get("primary", [])
        supplement_feats = groups.get("supplement", [])
        primary_faces = groups.get("primary_faces", [])
        supplement_faces = groups.get("supplement_faces", [])

        primary = []
        for idx, ref in enumerate(primary_feats[:1]):
            primary.append({
                "index": idx + 1,
                "similarity": round(self._observer_feature_similarity(feature, ref), 3),
                "crop": self._face_to_data_uri(primary_faces[idx] if idx < len(primary_faces) else None),
            })

        supplement = []
        for idx, ref in enumerate(supplement_feats[:self.dynamic_max_samples_per_id]):
            supplement.append({
                "index": idx + 1,
                "similarity": round(self._observer_feature_similarity(feature, ref), 3),
                "crop": self._face_to_data_uri(supplement_faces[idx] if idx < len(supplement_faces) else None),
            })
        return {
            "face_id": canonical,
            "primary": primary,
            "supplement": supplement,
        }

    def _observer_anchor_payload(self, name: Optional[str], feature: np.ndarray) -> Optional[dict]:
        canonical = self._resolve_dynamic_alias(name) if self.dynamic_library else name
        if not canonical:
            return None
        anchor = self._dynamic_anchor_by_id.get(canonical)
        if not anchor:
            return None
        anchor_feature = anchor.get("feature")
        if anchor_feature is None:
            return None
        return {
            "face_id": canonical,
            "similarity": round(self._observer_feature_similarity(feature, anchor_feature), 3),
            "crop": anchor.get("crop"),
            "quality": self._observer_quality_payload(anchor.get("quality")),
        }

    def _observer_top_scores(self, feature: np.ndarray, limit: int = 5) -> list:
        rows = []
        for name, score in self._identity_scores(feature)[:max(1, int(limit))]:
            rows.append({
                "face_id": name,
                "score": round(float(score), 3),
            })
        return rows

    def _update_observer_anchor(
        self,
        name: str,
        feature: np.ndarray,
        face_bgr: np.ndarray,
        quality: Optional[dict],
    ) -> None:
        if not self.observer_enabled or not self.dynamic_library or not name:
            return
        quality = quality or {}
        if not quality.get("primary_ok"):
            return
        canonical = self._resolve_dynamic_alias(name) or name
        yaw = quality.get("yaw_deg")
        q_score = float(quality.get("quality_score") or 0.0)
        old = self._dynamic_anchor_by_id.get(canonical)
        old_quality = old.get("quality", {}) if old else {}
        old_yaw = old_quality.get("yaw_deg") if old_quality else None
        old_score = float(old_quality.get("quality_score") or -1.0)
        better = old is None
        if old is not None:
            if yaw is not None and old_yaw is not None:
                better = (float(yaw), -q_score) < (float(old_yaw), -old_score)
            elif yaw is not None and old_yaw is None:
                better = True
            elif yaw is None and old_yaw is None:
                better = q_score > old_score
        if better:
            self._dynamic_anchor_by_id[canonical] = {
                "feature": self._normalize_feature(feature).copy(),
                "crop": self._face_to_data_uri(face_bgr),
                "quality": dict(quality),
            }

    def _update_observer_observation(
        self,
        *,
        track_id: int,
        frame_id: int,
        feature: np.ndarray,
        face_bgr: np.ndarray,
        quality: Optional[dict],
        face_id: Optional[str],
        raw_face_id: Optional[str],
        event: str,
        score: Optional[float],
        second_score: Optional[float],
    ) -> None:
        self._remember_track_face_debug(
            track_id=track_id,
            frame_id=frame_id,
            quality=quality,
            face_id=face_id,
            raw_face_id=raw_face_id,
            event=event,
            score=score,
            second_score=second_score,
        )
        if not self.observer_enabled or feature is None or face_bgr is None:
            return
        target_name = raw_face_id or face_id
        canonical = self._resolve_dynamic_alias(target_name) if target_name else None
        observer_key = f"face:{canonical}" if canonical else f"track:{int(track_id)}"
        if canonical:
            self._observer_latest.pop(f"track:{int(track_id)}", None)
        anchor = self._observer_anchor_payload(canonical, feature)
        gallery = self._observer_identity_samples(canonical, feature)
        self._observer_latest[observer_key] = {
            "observer_key": observer_key,
            "track_id": int(track_id),
            "frame_id": int(frame_id),
            "face_id": face_id,
            "raw_face_id": raw_face_id,
            "target_face_id": canonical,
            "event": event,
            "score": None if score is None else round(float(score), 3),
            "second_score": None if second_score is None else round(float(second_score), 3),
            "quality": self._observer_quality_payload(quality),
            "current_crop": self._face_to_data_uri(face_bgr),
            "frontal_anchor": anchor,
            "gallery": gallery,
            "top_scores": self._observer_top_scores(feature),
        }
        if len(self._observer_latest) > self.observer_max_tracks * 2:
            old_items = sorted(
                self._observer_latest.items(),
                key=lambda item: int(item[1].get("frame_id", -1)),
                reverse=True,
            )
            self._observer_latest = dict(old_items[:self.observer_max_tracks])

    def get_debug_snapshot(self) -> dict:
        if not self.observer_enabled:
            return {"enabled": False}
        latest_frame = max(
            (int(item.get("frame_id", -1)) for item in self._observer_latest.values()),
            default=-1,
        )
        if self.observer_pending_ttl_frames > 0 and latest_frame >= 0:
            for key, item in list(self._observer_latest.items()):
                if not str(key).startswith("track:"):
                    continue
                frame_id = int(item.get("frame_id", -1))
                if frame_id >= 0 and latest_frame - frame_id > self.observer_pending_ttl_frames:
                    self._observer_latest.pop(key, None)
        observations = sorted(
            self._observer_latest.values(),
            key=lambda item: int(item.get("frame_id", -1)),
            reverse=True,
        )[:self.observer_max_tracks]
        identities = []
        if self.dynamic_library:
            for name in sorted(self._dynamic_id_features, key=self._dynamic_id_number):
                canonical = self._resolve_dynamic_alias(name) or name
                if canonical != name:
                    continue
                groups = self._dynamic_id_features.get(name, {})
                anchor = self._dynamic_anchor_by_id.get(name)
                identities.append({
                    "face_id": name,
                    "primary_count": len(groups.get("primary", [])),
                    "supplement_count": len(groups.get("supplement", [])),
                    "anchor_crop": anchor.get("crop") if anchor else None,
                    "anchor_quality": self._observer_quality_payload(anchor.get("quality")) if anchor else None,
                })
        return {
            "enabled": True,
            "mode": "frontal_anchor_and_gallery",
            "max_tracks": self.observer_max_tracks,
            "tracks": observations,
            "identities": identities,
        }

    @staticmethod
    def _safe_bbox(bbox, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        if bbox is None or len(bbox) < 4:
            return None
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _bbox_too_small(self, bbox, image_bgr: np.ndarray) -> bool:
        if bbox is None or image_bgr is None or image_bgr.ndim < 2:
            return False
        safe_bbox = self._safe_bbox(
            bbox,
            width=int(image_bgr.shape[1]),
            height=int(image_bgr.shape[0]),
        )
        if safe_bbox is None:
            return True
        x1, y1, x2, y2 = safe_bbox
        return min(x2 - x1, y2 - y1) < self.dynamic_min_face_height

    @staticmethod
    def _det_score_value(confidence: Optional[float]) -> float:
        if confidence is None:
            return 0.0
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            return 0.0
        return value if np.isfinite(value) else 0.0

    @staticmethod
    def _quality_score(min_side: float, min_kp_conf: float, det_score: float,
                       primary_min_side: float) -> float:
        side_den = max(1.0, float(primary_min_side))
        side_score = min(1.0, max(0.0, float(min_side)) / side_den)
        return float(
            0.45 * side_score +
            0.35 * max(0.0, min(1.0, float(min_kp_conf))) +
            0.20 * max(0.0, min(1.0, float(det_score)))
        )

    def _keypoint_min_conf(self, keypoints) -> float:
        left_eye, right_eye, nose = _pick_eye_nose_points(keypoints)
        if left_eye is None or right_eye is None or nose is None:
            return 0.0
        return min(float(left_eye[2]), float(right_eye[2]), float(nose[2]))

    def _dynamic_feature_quality(
        self,
        bbox,
        keypoints,
        confidence: Optional[float],
        image_bgr: np.ndarray,
    ) -> dict:
        min_side = 0.0
        reason = ""
        if image_bgr is None or image_bgr.ndim < 2:
            reason = "invalid_image"
        else:
            safe_bbox = self._safe_bbox(
                bbox,
                width=int(image_bgr.shape[1]),
                height=int(image_bgr.shape[0]),
            )
            if safe_bbox is None:
                reason = "invalid_bbox"
            else:
                x1, y1, x2, y2 = safe_bbox
                min_side = float(min(x2 - x1, y2 - y1))
                if min_side < self.dynamic_min_face_height:
                    reason = "small_bbox"

        min_kp_conf = self._keypoint_min_conf(keypoints)
        if not reason and min_kp_conf < self.dynamic_min_keypoint_conf:
            reason = "low_keypoint_conf"

        det_score = self._det_score_value(confidence)
        yaw_deg = self._yaw_deg(keypoints)
        sample_ok = (
            not reason and
            min_side >= self.dynamic_update_min_face_height and
            min_kp_conf >= self.dynamic_update_min_keypoint_conf and
            det_score >= self.dynamic_update_min_score
        )
        primary_ok = (
            sample_ok and
            min_side >= self.dynamic_primary_min_face_height and
            min_kp_conf >= self.dynamic_primary_min_keypoint_conf and
            det_score >= self.dynamic_primary_min_score and
            yaw_deg is not None and
            yaw_deg <= self.dynamic_primary_max_yaw_deg
        )
        if reason:
            sample_reason = reason
        elif min_side < self.dynamic_update_min_face_height:
            sample_reason = "sample_small_bbox"
        elif min_kp_conf < self.dynamic_update_min_keypoint_conf:
            sample_reason = "sample_low_keypoint_conf"
        elif det_score < self.dynamic_update_min_score:
            sample_reason = "sample_low_det_score"
        else:
            sample_reason = "ok"

        if sample_reason != "ok":
            primary_reason = sample_reason
        elif min_side < self.dynamic_primary_min_face_height:
            primary_reason = "primary_small_bbox"
        elif min_kp_conf < self.dynamic_primary_min_keypoint_conf:
            primary_reason = "primary_low_keypoint_conf"
        elif det_score < self.dynamic_primary_min_score:
            primary_reason = "primary_low_det_score"
        elif yaw_deg is None:
            primary_reason = "primary_no_yaw"
        elif yaw_deg > self.dynamic_primary_max_yaw_deg:
            primary_reason = "primary_large_yaw"
        else:
            primary_reason = "ok"
        return {
            "trigger_ok": not bool(reason),
            "reason": reason or "ok",
            "sample_reason": sample_reason,
            "primary_reason": primary_reason,
            "sample_ok": bool(sample_ok),
            "primary_ok": bool(primary_ok),
            "min_side": float(min_side),
            "min_keypoint_conf": float(min_kp_conf),
            "det_score": float(det_score),
            "yaw_deg": yaw_deg,
            "quality_score": self._quality_score(
                min_side, min_kp_conf, det_score, self.dynamic_primary_min_face_height
            ),
        }

    def _dump_debug_sample(
        self,
        *,
        panorama_bgr: np.ndarray,
        face_bgr: np.ndarray,
        feature: np.ndarray,
        track_id: int,
        frame_id: int,
        face_id: Optional[str],
        event: str,
        score: Optional[float],
        bbox=None,
        confidence: Optional[float] = None,
        yaw_deg: Optional[float] = None,
        is_primary: Optional[bool] = None,
        raw_face_id: Optional[str] = None,
        second_score: Optional[float] = None,
        quality: Optional[dict] = None,
    ) -> None:
        if not self.debug_dump_dir:
            return

        self._debug_seen_features += 1
        if (self._debug_seen_features - 1) % self.debug_dump_every != 0:
            return
        if self.debug_dump_max and self._debug_dump_count >= self.debug_dump_max:
            return

        self._debug_dump_count += 1
        sample_idx = self._debug_dump_count
        face_label = face_id or "unknown"
        stem = f"{sample_idx:06d}_f{int(frame_id):06d}_t{int(track_id)}_{face_label}"
        aligned_rel = os.path.join("aligned_faces", f"{stem}.jpg")
        feature_rel = os.path.join("features", f"{stem}.npy")

        cv2 = _cv2()
        aligned_path = os.path.join(self.debug_dump_dir, aligned_rel)
        feature_path = os.path.join(self.debug_dump_dir, feature_rel)
        cv2.imwrite(aligned_path, face_bgr)
        np.save(feature_path, self._normalize_feature(feature).astype(np.float32))

        bbox_rel = None
        safe_bbox = self._safe_bbox(
            bbox,
            width=int(panorama_bgr.shape[1]),
            height=int(panorama_bgr.shape[0]),
        )
        if safe_bbox is not None:
            x1, y1, x2, y2 = safe_bbox
            crop = panorama_bgr[y1:y2, x1:x2]
            if crop.size:
                bbox_rel = os.path.join("bbox_crops", f"{stem}.jpg")
                cv2.imwrite(os.path.join(self.debug_dump_dir, bbox_rel), crop)

        meta = {
            "sample_idx": sample_idx,
            "frame_id": int(frame_id),
            "track_id": int(track_id),
            "face_id": face_id,
            "raw_face_id": raw_face_id if raw_face_id is not None else face_id,
            "alias_target": face_id if raw_face_id is not None and raw_face_id != face_id else None,
            "event": event,
            "score": None if score is None else float(score),
            "second_score": None if second_score is None else float(second_score),
            "confidence": None if confidence is None else float(confidence),
            "yaw_deg": None if yaw_deg is None else float(yaw_deg),
            "is_primary": None if is_primary is None else bool(is_primary),
            "bbox": None if safe_bbox is None else list(safe_bbox),
            "bbox_width": None if safe_bbox is None else int(safe_bbox[2] - safe_bbox[0]),
            "bbox_height": None if safe_bbox is None else int(safe_bbox[3] - safe_bbox[1]),
            "aligned_face": aligned_rel,
            "bbox_crop": bbox_rel,
            "feature": feature_rel,
            "align_mode": self.align_mode,
            "five_point_scale": self.five_point_scale,
        }
        if quality:
            for key in (
                "trigger_ok",
                "reason",
                "sample_ok",
                "sample_reason",
                "primary_ok",
                "primary_reason",
                "locked_pose_quality_ok",
                "pose_sample_quality_ok",
                "update_reason",
                "min_side",
                "min_keypoint_conf",
                "det_score",
                "quality_score",
                "best_existing",
                "best_other",
                "sample_similarity",
                "pose_library_similarity",
                "pose_library_threshold",
                "pose_library_max_similarity",
                "pose_sample_interval",
                "pose_sample_missing_frames",
                "known_update_reason",
                "known_update_name",
                "known_update_matched_name",
                "known_update_score",
                "known_update_second_score",
                "known_update_threshold",
                "known_update_pose_threshold",
                "known_update_match_threshold",
                "known_update_margin",
                "known_update_best_existing",
                "known_update_image_score",
                "known_update_primary_score",
                "known_update_pose_score",
                "known_update_duplicate_threshold",
                "known_update_primary_ema_alpha",
                "known_update_sample_count",
                "known_update_role",
                "known_update_updated",
                "known_bind_reason",
                "known_bind_dynamic_id",
                "known_bind_feature_count",
                "known_bind_min_hits",
                "known_bind_selected_name",
                "known_bind_selected_score",
                "known_bind_selected_hits",
                "known_bind_current_name",
                "known_bind_current_score",
                "known_bind_current_hits",
                "known_bind_result_name",
                "known_bind_votes",
                "known_bind_scores",
            ):
                if key in quality:
                    value = quality.get(key)
                    if isinstance(value, np.generic):
                        value = value.item()
                    meta[key] = value
        meta_path = os.path.join(self.debug_dump_dir, "metadata.jsonl")
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    def align_face(self, image_bgr: np.ndarray, keypoints) -> Optional[np.ndarray]:
        if self.align_mode == "five-point":
            return self._align_face_five_point(image_bgr, keypoints)
        return self._align_face_eye(image_bgr, keypoints)

    def _align_face_five_point(self, image_bgr: np.ndarray, keypoints) -> Optional[np.ndarray]:
        cv2 = _cv2()
        if image_bgr is None or image_bgr.size == 0:
            return None
        src = _pick_face_five_points(keypoints)
        if src is None:
            return None

        dst = _ARCFACE_TEMPLATE_112
        mean_src = src.mean(axis=0)
        mean_dst = dst.mean(axis=0)
        src_centered = src - mean_src
        dst_centered = dst - mean_dst
        denom = float(np.sum(dst_centered[:, 0] ** 2 + dst_centered[:, 1] ** 2))
        if denom <= 1.0e-6:
            return None
        a = float(np.sum(src_centered[:, 0] * dst_centered[:, 0] +
                         src_centered[:, 1] * dst_centered[:, 1]) / denom)
        b = float(np.sum(src_centered[:, 1] * dst_centered[:, 0] -
                         src_centered[:, 0] * dst_centered[:, 1]) / denom)
        scale = math.sqrt(a * a + b * b)
        if not math.isfinite(scale) or scale <= 1.0e-3:
            return None

        tx = float(mean_src[0] - a * mean_dst[0] + b * mean_dst[1])
        ty = float(mean_src[1] - b * mean_dst[0] - a * mean_dst[1])
        center = 55.5
        crop_scale = self.five_point_scale
        sx_tx = center * (1.0 - crop_scale)
        scale_to_template = np.asarray(
            [
                [crop_scale, 0.0, sx_tx],
                [0.0, crop_scale, sx_tx],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        template_to_source = np.asarray(
            [
                [a, -b, tx],
                [b, a, ty],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        output_to_source = template_to_source @ scale_to_template
        affine = output_to_source[:2, :]
        return cv2.warpAffine(
            image_bgr,
            affine,
            (112, 112),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _align_face_eye(self, image_bgr: np.ndarray, keypoints) -> Optional[np.ndarray]:
        cv2 = _cv2()
        left_eye, right_eye, nose = _pick_eye_nose_points(keypoints)
        if left_eye is None or right_eye is None or nose is None:
            return None

        lx, ly, lc = left_eye
        rx, ry, rc = right_eye
        nx, ny, nc = nose
        if min(lc, rc, nc) < 0.1:
            return None

        eye_dist = float(np.hypot(rx - lx, ry - ly))
        if eye_dist < 6.0:
            return None

        eye_mid = np.array([(lx + rx) * 0.5, (ly + ry) * 0.5], dtype=np.float32)
        angle = np.degrees(np.arctan2(ry - ly, rx - lx))
        rot = cv2.getRotationMatrix2D(tuple(eye_mid), angle, 1.0)
        aligned = cv2.warpAffine(
            image_bgr,
            rot,
            (image_bgr.shape[1], image_bgr.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # After roll correction, crop around the eye midpoint and nose area.
        side = int(max(32.0, eye_dist * 3.2))
        center_x = float(eye_mid[0])
        center_y = float(eye_mid[1] + 0.55 * eye_dist)
        x1 = int(round(center_x - side * 0.5))
        y1 = int(round(center_y - side * 0.45))
        x2 = x1 + side
        y2 = y1 + side

        h, w = aligned.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            return None

        crop = aligned[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    def extract_feature(self, face_bgr: np.ndarray) -> np.ndarray:
        cv2 = _cv2()
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("empty face crop")

        face = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
        arr = face.astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        tensor = torch.from_numpy(arr).to(self.device)

        with torch.no_grad():
            feat, _norm = self.model(tensor)
        return self._normalize_feature(feat.detach().cpu().numpy()[0])

    def _identity_scores(self, feature: np.ndarray) -> list:
        query = self._normalize_feature(feature)
        scores = [] if self.dynamic_library else self._known_identity_scores(query)
        if self.dynamic_library and self._dynamic_id_features:
            for name, groups in self._dynamic_id_features.items():
                if self._resolve_dynamic_alias(name) != name:
                    continue
                primary = groups.get("primary", [])
                supplement = groups.get("supplement", [])
                if not primary and not supplement:
                    continue

                primary_score = 0.0
                supplement_score = 0.0
                if primary:
                    matrix = np.stack(primary).astype(np.float32)
                    primary_score = float(np.max(matrix @ query))
                if supplement and (
                    not primary
                    or primary_score < self.dynamic_supplement_fallback_threshold
                ):
                    sup_matrix = np.stack(supplement).astype(np.float32)
                    supplement_score = float(np.max(sup_matrix @ query))

                score = max(primary_score, supplement_score)
                scores.append((self._resolve_dynamic_alias(name), score))
            scores.sort(key=lambda item: item[1], reverse=True)
            return self._dedup_scores_by_name(scores)

        if self.lib_matrix is None or not self.lib_names:
            return self._dedup_scores_by_name(scores)

        sims = self.lib_matrix @ query
        for name, score in zip(self.lib_names, sims):
            scores.append((name, float(score)))
        return self._dedup_scores_by_name(scores)

    def _known_identity_scores(self, feature: np.ndarray) -> list:
        if self.known_matrix is None or not self.known_names:
            return []
        query = self._normalize_feature(feature)
        return [
            (name, float(score))
            for name, score in zip(self.known_names, self.known_matrix @ query)
        ]

    @staticmethod
    def _dedup_scores_by_name(scores: list) -> list:
        best_by_name: Dict[str, float] = {}
        for name, score in scores:
            score = float(score)
            if score > best_by_name.get(name, -1.0):
                best_by_name[name] = score
        return sorted(best_by_name.items(), key=lambda item: item[1], reverse=True)

    def _is_known_identity(self, name: Optional[str]) -> bool:
        return bool(name) and name in self._known_name_set

    def _known_match_detailed(self, feature: np.ndarray) -> Tuple[Optional[str], float, float]:
        return self._match_from_scores(
            self._dedup_scores_by_name(self._known_identity_scores(feature))
        )

    def _identity_score_for_name(self, name: str, feature: np.ndarray) -> float:
        query = self._normalize_feature(feature)
        if self._is_known_identity(name) and self.known_matrix is not None:
            best = 0.0
            for known_name, known_feature in zip(self.known_names, self.known_matrix):
                if known_name == name:
                    best = max(best, float(np.dot(known_feature, query)))
            return best
        if self.dynamic_library:
            return self._dynamic_identity_score(name, query)
        if self.lib_matrix is not None and self.lib_names:
            best = 0.0
            for lib_name, lib_feature in zip(self.lib_names, self.lib_matrix):
                if lib_name == name:
                    best = max(best, float(np.dot(lib_feature, query)))
            return best
        return 0.0

    def _match_from_scores(self, scores: list) -> Tuple[Optional[str], float, float]:
        if not scores:
            return None, 0.0, 0.0
        best_name, best_score = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else 0.0
        if best_score < self.threshold:
            return None, float(best_score), float(second_score)
        if best_score < second_score + self.dynamic_match_margin:
            return None, float(best_score), float(second_score)
        return best_name, float(best_score), float(second_score)

    def match_detailed(self, feature: np.ndarray) -> Tuple[Optional[str], float, float]:
        return self._match_from_scores(self._identity_scores(feature))

    def match(self, feature: np.ndarray) -> Tuple[Optional[str], float]:
        name, score, _second = self.match_detailed(feature)
        return name, score

    def _save_dynamic_identity(self, name: str, face_bgr: np.ndarray) -> None:
        groups = self._dynamic_id_features.get(name, {})
        all_feats = groups.get("primary", []) + groups.get("supplement", [])
        if not all_feats:
            return
        arr = np.stack(all_feats).astype(np.float32)
        np.save(os.path.join(self.dynamic_library_dir, f"{name}.npy"), arr)
        sample_idx = len(all_feats)
        preview_path = os.path.join(
            self.dynamic_library_dir, "_preview", f"{name}_{sample_idx:02d}_crop.jpg"
        )
        try:
            _cv2().imwrite(preview_path, face_bgr)
        except Exception as exc:
            print(f"[FaceRec] dynamic preview save failed for {name}: {exc}")

    def _add_dynamic_sample(
        self,
        name: str,
        feature: np.ndarray,
        face_bgr: np.ndarray,
        *,
        is_primary: bool,
        quality_score: Optional[float] = None,
        quality: Optional[dict] = None,
    ) -> bool:
        feat = self._normalize_feature(feature)
        groups = self._dynamic_id_features.setdefault(
            name, {
                "primary": [],
                "supplement": [],
                "primary_faces": [],
                "supplement_faces": [],
            }
        )
        groups.setdefault("primary_faces", [])
        groups.setdefault("supplement_faces", [])
        face_copy = face_bgr.copy() if isinstance(face_bgr, np.ndarray) else face_bgr
        if is_primary:
            bucket = groups["primary"]
            face_bucket = groups["primary_faces"]
            if bucket:
                old = np.asarray(bucket[0], dtype=np.float32)
                if old.shape == feat.shape:
                    alpha = self.dynamic_primary_ema_alpha
                    feat = self._normalize_feature((1.0 - alpha) * old + alpha * feat)
                bucket[0] = feat
            else:
                bucket.append(feat)
            if face_bucket:
                face_bucket[0] = face_copy
            else:
                face_bucket.append(face_copy)
            self._update_observer_anchor(name, feature, face_bgr, quality)
            self._save_dynamic_identity(name, face_bgr)
            self._rebuild_dynamic_matrix()
            return True

        bucket = groups["supplement"]
        all_feats = groups["primary"] + groups["supplement"]
        if all_feats:
            best_existing = max(float(np.dot(old, feat)) for old in all_feats)
            if best_existing >= 1.0 - self.dynamic_min_sample_diversity:
                return False

        bucket.append(feat)
        groups["supplement_faces"].append(face_copy)
        while len(groups["primary"]) + len(groups["supplement"]) > self.dynamic_max_samples_per_id:
            if groups["supplement"]:
                del groups["supplement"][0]
                if groups["supplement_faces"]:
                    del groups["supplement_faces"][0]
            elif len(groups["primary"]) > 1:
                del groups["primary"][0]
                if groups["primary_faces"]:
                    del groups["primary_faces"][0]
            else:
                break

        self._save_dynamic_identity(name, face_bgr)
        self._rebuild_dynamic_matrix()
        return True

    def _add_dynamic_alias_probe(self, name: str, feature: np.ndarray) -> None:
        if self._is_known_identity(name):
            return
        canonical = self._resolve_dynamic_alias(name) or name
        feat = self._normalize_feature(feature)
        bucket = self._dynamic_alias_probe_features.setdefault(canonical, [])
        if bucket:
            best_existing = max(float(np.dot(old, feat)) for old in bucket)
            if best_existing >= 1.0 - self.dynamic_min_sample_diversity:
                return
        bucket.append(feat)
        if len(bucket) > self.dynamic_alias_probe_samples:
            del bucket[0:len(bucket) - self.dynamic_alias_probe_samples]

    @staticmethod
    def _dynamic_id_number(name: str) -> int:
        if isinstance(name, str) and name.startswith("face"):
            try:
                return int(name[4:])
            except ValueError:
                pass
        return 10**9

    def _resolve_dynamic_alias(self, name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        seen = set()
        current = name
        while current in self._dynamic_aliases and current not in seen:
            seen.add(current)
            current = self._dynamic_aliases[current]
        if current != name:
            for item in seen:
                self._dynamic_aliases[item] = current
        return current

    def _dynamic_identity_features_for_alias(self, name: str) -> list:
        canonical = self._resolve_dynamic_alias(name) or name
        feats = list(self._dynamic_alias_probe_features.get(canonical, []))
        feats.extend(self._dynamic_identity_stored_features_for_alias(canonical))
        return feats

    def _dynamic_identity_stored_features_for_alias(self, name: str) -> list:
        canonical = self._resolve_dynamic_alias(name) or name
        feats = []
        for identity, groups in self._dynamic_id_features.items():
            if (self._resolve_dynamic_alias(identity) or identity) != canonical:
                continue
            feats.extend(groups.get("primary", []))
            feats.extend(groups.get("supplement", []))
        return feats

    def _dynamic_identity_similarity(self, src: str, dst: str) -> float:
        src_feats = self._dynamic_identity_features_for_alias(src)
        dst_feats = self._dynamic_identity_features_for_alias(dst)
        if not src_feats or not dst_feats:
            return 0.0
        src_matrix = np.stack(src_feats).astype(np.float32)
        dst_matrix = np.stack(dst_feats).astype(np.float32)
        return float(np.max(src_matrix @ dst_matrix.T))

    def _maybe_auto_alias_dynamic_identity(self, name: str) -> Optional[str]:
        if self._is_known_identity(name):
            return name
        if not self.dynamic_auto_alias:
            return self._resolve_dynamic_alias(name)

        canonical = self._resolve_dynamic_alias(name) or name
        groups = self._dynamic_id_features.get(canonical)
        if not groups:
            return canonical

        current_count = len(groups.get("primary", [])) + len(groups.get("supplement", []))
        if current_count < self.dynamic_alias_min_samples:
            return canonical

        current_num = self._dynamic_id_number(canonical)
        best_target, best_score = None, 0.0
        second_score = 0.0
        for other in self._dynamic_id_features:
            other_canonical = self._resolve_dynamic_alias(other) or other
            if other_canonical != other or other_canonical == canonical:
                continue
            if self._dynamic_id_number(other_canonical) >= current_num:
                continue
            score = self._dynamic_identity_similarity(canonical, other_canonical)
            if score > best_score:
                second_score = best_score
                best_target, best_score = other_canonical, score
            elif score > second_score:
                second_score = score

        if best_target is None or best_score < self.dynamic_alias_threshold:
            return canonical
        if best_score < second_score + self.dynamic_alias_margin:
            return canonical

        vote_key = (canonical, best_target)
        self._dynamic_alias_votes[vote_key] = self._dynamic_alias_votes.get(vote_key, 0) + 1
        votes = self._dynamic_alias_votes[vote_key]
        if votes < self.dynamic_alias_min_hits:
            return canonical

        self._dynamic_aliases[canonical] = best_target
        for track_id, bound in list(self._dynamic_track_bindings.items()):
            if self._resolve_dynamic_alias(bound) == best_target:
                self._dynamic_track_bindings[track_id] = best_target
        print(
            f"[FaceRec] dynamic alias {canonical} -> {best_target} "
            f"(identity_sim={best_score:.3f}, votes={votes})"
        )
        return best_target

    def _final_identity_name(self, name: str) -> str:
        if self._is_known_identity(name):
            return name
        final_name = self._maybe_auto_alias_dynamic_identity(name) or name
        return self._resolve_dynamic_alias(final_name) or final_name

    def _dynamic_identity_label_features(
        self,
        face_id: str,
        extra_feature: Optional[np.ndarray] = None,
    ) -> list:
        canonical = self._resolve_dynamic_alias(face_id) or face_id
        feats = self._dynamic_identity_stored_features_for_alias(canonical)
        if extra_feature is not None:
            feats.append(self._normalize_feature(extra_feature))
        return feats

    def _maybe_bind_known_label(
        self,
        face_id: str,
        feature: Optional[np.ndarray] = None,
        quality: Optional[dict] = None,
    ) -> Optional[str]:
        if not self.dynamic_library or self.known_matrix is None or not self.known_names:
            self._set_known_bind_debug(
                quality,
                reason="empty_known_library",
                canonical=face_id,
                result_name=None,
            )
            self._set_known_update_debug(
                quality,
                reason="empty_known_library",
                name=face_id,
            )
            return None
        canonical = self._resolve_dynamic_alias(face_id) or face_id
        if canonical not in self._dynamic_id_features:
            result_name = self.get_identity_label(canonical)
            self._set_known_bind_debug(
                quality,
                reason="missing_dynamic_identity",
                canonical=canonical,
                result_name=result_name,
            )
            self._set_known_update_debug(
                quality,
                reason="missing_dynamic_identity",
                name=canonical,
            )
            return result_name

        direct_scores = (
            self._dedup_scores_by_name(self._known_identity_scores(feature))
            if feature is not None else []
        )
        direct_debug = self._known_sample_scores_debug(direct_scores, -1) if direct_scores else None
        direct_name = direct_debug.get("winner") if direct_debug else None
        direct_score = float(direct_debug.get("score") or 0.0) if direct_debug else 0.0
        direct_second_score = (
            float(direct_debug.get("second_score") or 0.0) if direct_debug else 0.0
        )
        direct_ok = (
            direct_name is not None
            and direct_score >= self.threshold
            and direct_score >= direct_second_score + self.dynamic_match_margin
        )
        direct_state = self._known_bind_consecutive.get(canonical, {})
        if direct_ok:
            if direct_state.get("name") == direct_name:
                direct_hits = int(direct_state.get("hits") or 0) + 1
            else:
                direct_hits = 1
            direct_state = {
                "name": direct_name,
                "hits": direct_hits,
                "score": max(float(direct_state.get("score") or 0.0), direct_score)
                if direct_state.get("name") == direct_name else direct_score,
                "second_score": direct_second_score,
            }
            self._known_bind_consecutive[canonical] = direct_state
        else:
            self._known_bind_consecutive.pop(canonical, None)
            direct_hits = 0

        if direct_ok and direct_hits >= 2:
            current = self._dynamic_identity_labels.get(canonical)
            current_name = current.get("name") if current else None
            current_score = float(current.get("score") or 0.0) if current else 0.0
            current_hits = int(current.get("hits") or 0) if current else 0
            if current_name and current_name != direct_name:
                if direct_score < current_score + self.dynamic_match_margin:
                    self._set_known_bind_debug(
                        quality,
                        reason="known_bind_keep_existing_margin",
                        canonical=canonical,
                        feature_count=1,
                        min_hits=2,
                        sample_scores=[direct_debug] if direct_debug else None,
                        votes={direct_name: {"hits": direct_hits, "best": direct_score}},
                        selected_name=direct_name,
                        selected_score=direct_score,
                        selected_hits=direct_hits,
                        current_name=current_name,
                        current_score=current_score,
                        current_hits=current_hits,
                        result_name=current_name,
                    )
                    self._set_known_update_debug(
                        quality,
                        reason="known_bind_keep_existing_margin",
                        name=canonical,
                        matched_name=direct_name,
                        score=direct_score,
                        second_score=direct_second_score,
                        sample_count=direct_hits,
                    )
                    return current_name
            self._dynamic_identity_labels[canonical] = {
                "name": direct_name,
                "score": max(current_score, direct_score),
                "hits": max(current_hits, direct_hits),
            }
            self._set_known_bind_debug(
                quality,
                reason="known_bind_consecutive_selected",
                canonical=canonical,
                feature_count=1,
                min_hits=2,
                sample_scores=[direct_debug] if direct_debug else None,
                votes={direct_name: {"hits": direct_hits, "best": direct_score}},
                selected_name=direct_name,
                selected_score=direct_score,
                selected_hits=direct_hits,
                current_name=current_name,
                current_score=current_score if current else None,
                current_hits=current_hits if current else None,
                result_name=direct_name,
            )
            print(
                f"[FaceRec] bind dynamic {canonical} -> name {direct_name} "
                f"(consecutive_score={direct_score:.3f}, hits={direct_hits})"
            )
            self._maybe_update_known_feature_library(direct_name, feature, quality)
            return direct_name

        feats = self._dynamic_identity_label_features(canonical, feature)
        if not feats:
            result_name = self.get_identity_label(canonical)
            self._set_known_bind_debug(
                quality,
                reason="missing_dynamic_features",
                canonical=canonical,
                result_name=result_name,
            )
            self._set_known_update_debug(
                quality,
                reason="missing_dynamic_features",
                name=canonical,
            )
            return result_name

        votes: Dict[str, dict] = {}
        sample_scores = []
        for sample_index, feat in enumerate(feats):
            known_scores = self._dedup_scores_by_name(self._known_identity_scores(feat))
            sample_debug = self._known_sample_scores_debug(known_scores, sample_index)
            sample_scores.append(sample_debug)
            name = sample_debug.get("winner")
            score = float(sample_debug.get("score") or 0.0)
            second_score = float(sample_debug.get("second_score") or 0.0)
            if name is None:
                continue
            if score < second_score + self.dynamic_match_margin:
                continue
            bucket = votes.setdefault(name, {"hits": 0, "best": 0.0})
            bucket["hits"] += 1
            bucket["best"] = max(float(bucket["best"]), float(score))

        if not votes:
            result_name = self.get_identity_label(canonical)
            self._set_known_bind_debug(
                quality,
                reason="known_bind_no_votes",
                canonical=canonical,
                feature_count=len(feats),
                sample_scores=sample_scores,
                votes=votes,
                result_name=result_name,
            )
            self._set_known_update_debug(
                quality,
                reason="known_bind_no_votes",
                name=canonical,
            )
            return result_name

        best_name, best_info = max(
            votes.items(),
            key=lambda item: (int(item[1]["hits"]), float(item[1]["best"])),
        )
        hits = int(best_info["hits"])
        best_score = float(best_info["best"])
        min_hits = 2 if len(feats) >= 2 else 1
        if hits < min_hits:
            result_name = self.get_identity_label(canonical)
            self._set_known_bind_debug(
                quality,
                reason="known_bind_insufficient_votes",
                canonical=canonical,
                feature_count=len(feats),
                min_hits=min_hits,
                sample_scores=sample_scores,
                votes=votes,
                selected_name=best_name,
                selected_score=best_score,
                selected_hits=hits,
                result_name=result_name,
            )
            self._set_known_update_debug(
                quality,
                reason="known_bind_insufficient_votes",
                name=canonical,
                matched_name=best_name,
                score=best_score,
                sample_count=hits,
            )
            return result_name
        if len(feats) < 2 and best_score < self.threshold + 0.10:
            result_name = self.get_identity_label(canonical)
            self._set_known_bind_debug(
                quality,
                reason="known_bind_low_single_feature_score",
                canonical=canonical,
                feature_count=len(feats),
                min_hits=min_hits,
                sample_scores=sample_scores,
                votes=votes,
                selected_name=best_name,
                selected_score=best_score,
                selected_hits=hits,
                result_name=result_name,
            )
            self._set_known_update_debug(
                quality,
                reason="known_bind_low_single_feature_score",
                name=canonical,
                matched_name=best_name,
                score=best_score,
                sample_count=hits,
            )
            return result_name

        current = self._dynamic_identity_labels.get(canonical)
        if current:
            current_name = current.get("name")
            current_score = float(current.get("score") or 0.0)
            current_hits = int(current.get("hits") or 0)
            if current_name == best_name:
                if best_score > current_score or hits > current_hits:
                    current.update({"score": best_score, "hits": hits})
                self._set_known_bind_debug(
                    quality,
                    reason="known_bind_keep_same_name",
                    canonical=canonical,
                    feature_count=len(feats),
                    min_hits=min_hits,
                    sample_scores=sample_scores,
                    votes=votes,
                    selected_name=best_name,
                    selected_score=best_score,
                    selected_hits=hits,
                    current_name=current_name,
                    current_score=current_score,
                    current_hits=current_hits,
                    result_name=best_name,
                )
                self._maybe_update_known_feature_library(best_name, feature, quality)
                return best_name
            if hits < current_hits and best_score < current_score + self.dynamic_match_margin:
                self._set_known_bind_debug(
                    quality,
                    reason="known_bind_keep_existing_hits",
                    canonical=canonical,
                    feature_count=len(feats),
                    min_hits=min_hits,
                    sample_scores=sample_scores,
                    votes=votes,
                    selected_name=best_name,
                    selected_score=best_score,
                    selected_hits=hits,
                    current_name=current_name,
                    current_score=current_score,
                    current_hits=current_hits,
                    result_name=current_name,
                )
                self._set_known_update_debug(
                    quality,
                    reason="known_bind_keep_existing_hits",
                    name=canonical,
                    matched_name=best_name,
                    score=best_score,
                    sample_count=hits,
                )
                return current_name
            if best_score < current_score + self.dynamic_match_margin:
                self._set_known_bind_debug(
                    quality,
                    reason="known_bind_keep_existing_margin",
                    canonical=canonical,
                    feature_count=len(feats),
                    min_hits=min_hits,
                    sample_scores=sample_scores,
                    votes=votes,
                    selected_name=best_name,
                    selected_score=best_score,
                    selected_hits=hits,
                    current_name=current_name,
                    current_score=current_score,
                    current_hits=current_hits,
                    result_name=current_name,
                )
                self._set_known_update_debug(
                    quality,
                    reason="known_bind_keep_existing_margin",
                    name=canonical,
                    matched_name=best_name,
                    score=best_score,
                    sample_count=hits,
                )
                return current_name

        self._dynamic_identity_labels[canonical] = {
            "name": best_name,
            "score": best_score,
            "hits": hits,
        }
        self._set_known_bind_debug(
            quality,
            reason="known_bind_selected",
            canonical=canonical,
            feature_count=len(feats),
            min_hits=min_hits,
            sample_scores=sample_scores,
            votes=votes,
            selected_name=best_name,
            selected_score=best_score,
            selected_hits=hits,
            result_name=best_name,
        )
        print(
            f"[FaceRec] bind dynamic {canonical} -> name {best_name} "
            f"(known_score={best_score:.3f}, hits={hits})"
        )
        self._maybe_update_known_feature_library(best_name, feature, quality)
        return best_name

    def get_identity_label(self, face_id: Optional[str]) -> Optional[str]:
        if not face_id:
            return None
        if self._is_known_identity(face_id):
            return face_id
        canonical = self._resolve_dynamic_alias(face_id) if self.dynamic_library else face_id
        item = self._dynamic_identity_labels.get(canonical) if canonical else None
        return item.get("name") if item else None

    def get_track_label(self, track_id: int) -> Optional[str]:
        if not self.dynamic_library:
            return None
        face_id = self._dynamic_track_bindings.get(int(track_id))
        return self.get_identity_label(face_id)

    def _rebuild_dynamic_matrix(self) -> None:
        names, feats = [], []
        for name, groups in self._dynamic_id_features.items():
            for feat in groups.get("primary", []) + groups.get("supplement", []):
                names.append(name)
                feats.append(feat)
        self.lib_names = names
        self.lib_matrix = np.stack(feats).astype(np.float32) if feats else None

    def _create_dynamic_identity(
        self,
        feature: np.ndarray,
        face_bgr: np.ndarray,
        *,
        is_primary: bool,
    ) -> str:
        name = f"face{self._dynamic_next_id}"
        self._dynamic_next_id += 1
        self._dynamic_id_features[name] = {
            "primary": [],
            "supplement": [],
            "primary_faces": [],
            "supplement_faces": [],
        }
        self._add_dynamic_sample(name, feature, face_bgr, is_primary=is_primary)
        return name

    def _create_dynamic_identity_from_pending(self, pending: dict) -> Optional[str]:
        features = pending.get("features") or []
        faces = pending.get("faces") or []
        qualities = pending.get("qualities") or []
        frames = pending.get("frames") or list(range(len(features)))
        if not features:
            return None

        primary_idx = None
        best_quality = -1.0
        for idx, quality in enumerate(qualities):
            q = float(quality.get("quality_score", 0.0))
            if quality.get("primary_ok"):
                if primary_idx is None or q > best_quality:
                    primary_idx = idx
                    best_quality = q
        if primary_idx is None:
            return None

        name = f"face{self._dynamic_next_id}"
        self._dynamic_next_id += 1
        self._dynamic_id_features[name] = {
            "primary": [],
            "supplement": [],
            "primary_faces": [],
            "supplement_faces": [],
        }
        self._add_dynamic_sample(
            name,
            features[primary_idx],
            faces[primary_idx],
            is_primary=True,
            quality_score=best_quality,
            quality=qualities[primary_idx] if primary_idx < len(qualities) else None,
        )
        seed_limit = min(
            self.dynamic_pending_seed_samples,
            max(0, self.dynamic_max_samples_per_id - 1),
        )
        if seed_limit <= 0:
            return name

        primary_quality = qualities[primary_idx] if primary_idx < len(qualities) else {}
        primary_yaw = primary_quality.get("yaw_deg") if isinstance(primary_quality, dict) else None
        primary_frame = int(frames[primary_idx]) if primary_idx < len(frames) else primary_idx
        primary_feature = self._normalize_feature(features[primary_idx])

        def _pending_candidate_score(idx: int) -> Tuple[float, float, int]:
            quality = qualities[idx] if idx < len(qualities) else {}
            yaw_delta = 0.0
            if isinstance(quality, dict) and primary_yaw is not None:
                yaw = quality.get("yaw_deg")
                if yaw is not None:
                    yaw_delta = abs(float(yaw) - float(primary_yaw))
            quality_score = (
                float(quality.get("quality_score", 0.0))
                if isinstance(quality, dict) else 0.0
            )
            frame = int(frames[idx]) if idx < len(frames) else idx
            return yaw_delta, quality_score, frame

        candidates = []
        for idx, feat in enumerate(features):
            if idx == primary_idx or feat is None:
                continue
            quality = qualities[idx] if idx < len(qualities) else {}
            if isinstance(quality, dict) and not quality.get("sample_ok"):
                continue
            candidates.append(idx)
        candidates.sort(key=_pending_candidate_score, reverse=True)

        selected_frames = [primary_frame]
        selected_supplement_frames = []
        selected_features = [primary_feature]
        for idx in candidates:
            frame = int(frames[idx]) if idx < len(frames) else idx
            if (
                self.dynamic_pose_sample_interval > 0 and
                any(abs(frame - old_frame) < self.dynamic_pose_sample_interval
                    for old_frame in selected_frames)
            ):
                continue

            feat = self._normalize_feature(features[idx])
            if selected_features:
                best_pose = max(float(np.dot(old_feat, feat)) for old_feat in selected_features)
                if best_pose < self.dynamic_pose_library_similarity:
                    continue
                if best_pose > self.dynamic_pose_library_max_similarity:
                    continue

            face = faces[idx] if idx < len(faces) else faces[primary_idx]
            added = self._add_dynamic_sample(
                name,
                feat,
                face,
                is_primary=False,
                quality=qualities[idx] if idx < len(qualities) else None,
            )
            if not added:
                continue
            selected_frames.append(frame)
            selected_supplement_frames.append(frame)
            selected_features.append(feat)
            if len(selected_supplement_frames) >= seed_limit:
                break

        if selected_supplement_frames:
            self._dynamic_last_pose_sample_frame[name] = max(selected_frames)
        else:
            self._dynamic_last_pose_sample_frame[name] = primary_frame
        return name

    def _pending_dynamic_enroll_compatible(self, pending: dict,
                                           feature: np.ndarray) -> bool:
        for old_feature in pending.get("features", []):
            if old_feature is None or np.asarray(old_feature).shape != np.asarray(feature).shape:
                return False
            if float(np.dot(old_feature, feature)) < self.threshold:
                return False
        return True

    def _update_pending_dynamic_enroll(
        self,
        track_id: int,
        feature: np.ndarray,
        face_bgr: np.ndarray,
        quality: dict,
        frame_id: int,
    ) -> Tuple[Optional[str], int, bool]:
        if track_id < 0 or feature is None:
            return None, 0, False

        if self.dynamic_enroll_confirm_frames <= 1:
            name = self._create_dynamic_identity_from_pending({
                "features": [feature],
                "faces": [face_bgr],
                "qualities": [quality],
                "frames": [frame_id],
            })
            return name, 1, False

        pending = self._dynamic_pending_enroll.setdefault(
            track_id,
            {"features": [], "faces": [], "qualities": [], "frames": [], "last_frame": frame_id}
        )
        reset = False
        if not self._pending_dynamic_enroll_compatible(pending, feature):
            pending["features"] = []
            pending["faces"] = []
            pending["qualities"] = []
            pending["frames"] = []
            reset = True

        pending["features"].append(self._normalize_feature(feature))
        pending["faces"].append(face_bgr)
        pending["qualities"].append(dict(quality))
        pending["frames"].append(int(frame_id))
        pending["last_frame"] = frame_id
        hits = len(pending["features"])
        if hits < self.dynamic_enroll_confirm_frames:
            return None, hits, reset
        if not any(q.get("primary_ok") for q in pending["qualities"]):
            return None, hits, reset

        name = self._create_dynamic_identity_from_pending(pending)
        self._dynamic_pending_enroll.pop(track_id, None)
        return name, hits, reset

    def _update_dynamic_identity(
        self,
        name: str,
        feature: np.ndarray,
        face_bgr: np.ndarray,
        *,
        is_primary: bool,
    ) -> bool:
        return self._add_dynamic_sample(name, feature, face_bgr, is_primary=is_primary)

    def _best_other_dynamic_identity_score(self, name: str, feature: np.ndarray) -> float:
        query = self._normalize_feature(feature)
        best = 0.0
        target = self._resolve_dynamic_alias(name) or name
        for other in self._dynamic_id_features:
            other_canonical = self._resolve_dynamic_alias(other) or other
            if other_canonical != other or other_canonical == target:
                continue
            score = self._dynamic_identity_score(other_canonical, query)
            if score > best:
                best = score
        return float(best)

    def _maybe_update_dynamic_identity(
        self,
        name: str,
        feature: np.ndarray,
        face_bgr: np.ndarray,
        quality: dict,
        *,
        locked_track_sample: bool,
        frame_id: Optional[int] = None,
    ) -> str:
        canonical = self._resolve_dynamic_alias(name) or name
        if canonical not in self._dynamic_id_features:
            return ""

        stored_feats = self._dynamic_identity_stored_features_for_alias(canonical)
        query = self._normalize_feature(feature)
        best_existing = (
            max(float(np.dot(old, query)) for old in stored_feats)
            if stored_feats else 0.0
        )
        sample_similarity = (
            self.dynamic_locked_sample_similarity
            if locked_track_sample else self.dynamic_pose_sample_similarity
        )
        groups = self._dynamic_id_features.get(canonical, {})
        pose_feats = groups.get("supplement", [])
        pose_reference_feats = pose_feats if pose_feats else groups.get("primary", [])
        best_pose_existing = (
            max(float(np.dot(old, query)) for old in pose_reference_feats)
            if pose_reference_feats else None
        )
        quality["best_existing"] = float(best_existing)
        quality["sample_similarity"] = float(sample_similarity)
        quality["pose_library_similarity"] = best_pose_existing
        quality["pose_library_threshold"] = float(self.dynamic_pose_library_similarity)
        quality["pose_library_max_similarity"] = float(self.dynamic_pose_library_max_similarity)
        min_side = float(quality.get("min_side") or 0.0)
        det_score = float(quality.get("det_score") or 0.0)
        locked_pose_quality_ok = (
            locked_track_sample and
            bool(quality.get("trigger_ok")) and
            min_side >= self.dynamic_update_min_face_height and
            det_score >= self.dynamic_update_min_score
        )
        pose_sample_quality_ok = bool(quality.get("sample_ok")) or locked_pose_quality_ok
        quality["locked_pose_quality_ok"] = bool(locked_pose_quality_ok)
        quality["pose_sample_quality_ok"] = bool(pose_sample_quality_ok)
        last_pose_frame = self._dynamic_last_pose_sample_frame.get(canonical)
        pose_missing_frames = (
            None if frame_id is None or last_pose_frame is None
            else int(frame_id) - int(last_pose_frame)
        )
        quality["pose_sample_interval"] = int(self.dynamic_pose_sample_interval)
        quality["pose_sample_missing_frames"] = pose_missing_frames
        pose_library_ok = (
            best_pose_existing is None or
            best_pose_existing >= self.dynamic_pose_library_similarity
        )
        pose_diversity_ok = (
            best_pose_existing is None or
            best_pose_existing <= self.dynamic_pose_library_max_similarity
        )
        pose_interval_ok = (
            self.dynamic_pose_sample_interval <= 0 or
            pose_missing_frames is None or
            pose_missing_frames >= self.dynamic_pose_sample_interval
        )
        can_update_primary = (
            bool(quality.get("primary_ok")) and
            best_existing >= self.dynamic_update_similarity
        )
        can_add_sample = (
            pose_sample_quality_ok and
            best_existing >= sample_similarity and
            best_existing < 1.0 - self.dynamic_min_sample_diversity and
            pose_library_ok and
            pose_diversity_ok and
            pose_interval_ok
        )
        if not can_update_primary and not can_add_sample:
            if not pose_sample_quality_ok:
                quality["update_reason"] = quality.get("sample_reason", "sample_not_ok")
            elif best_existing < sample_similarity:
                quality["update_reason"] = "sample_low_identity_similarity"
            elif best_existing >= 1.0 - self.dynamic_min_sample_diversity:
                quality["update_reason"] = "sample_too_similar_to_existing"
            elif not pose_library_ok:
                quality["update_reason"] = "sample_low_pose_library_similarity"
            elif not pose_diversity_ok:
                quality["update_reason"] = "sample_too_similar_to_pose_library"
            elif not pose_interval_ok:
                quality["update_reason"] = "sample_interval"
            else:
                quality["update_reason"] = "primary_not_ok"
            return ""

        best_other = self._best_other_dynamic_identity_score(canonical, feature)
        quality["best_other"] = float(best_other)
        if best_existing < best_other + self.dynamic_match_margin:
            quality["update_reason"] = "sample_ambiguous_with_other_faceid"
            return ""

        primary_updated = False
        sample_updated = False
        if can_update_primary:
            primary_updated = self._add_dynamic_sample(
                canonical,
                feature,
                face_bgr,
                is_primary=True,
                quality_score=float(quality.get("quality_score", 0.0)),
                quality=quality,
            )
        if can_add_sample:
            sample_updated = self._add_dynamic_sample(
                canonical,
                feature,
                face_bgr,
                is_primary=False,
                quality=quality,
            )
        if primary_updated and sample_updated:
            if frame_id is not None:
                self._dynamic_last_pose_sample_frame[canonical] = int(frame_id)
            quality["update_reason"] = "primary_sample_updated"
            return "primary_sample"
        if primary_updated:
            quality["update_reason"] = "primary_updated"
            return "primary"
        if sample_updated:
            if frame_id is not None:
                self._dynamic_last_pose_sample_frame[canonical] = int(frame_id)
            quality["update_reason"] = "sample_updated"
            return "sample"
        quality["update_reason"] = "sample_rejected_by_diversity"
        return ""

    def _dynamic_identity_score(self, name: str, feature: np.ndarray) -> float:
        feats = self._dynamic_identity_features_for_alias(name)
        if not feats:
            return 0.0
        query = self._normalize_feature(feature)
        return float(np.max(np.stack(feats).astype(np.float32) @ query))

    def _reset_dynamic_frame_if_needed(self, frame_id: int) -> None:
        if self._dynamic_frame_id != frame_id:
            self._dynamic_frame_id = frame_id
            self._dynamic_frame_assignments = {}

    def _purge_stale_dynamic_state(self, active_track_ids: set, frame_id: int) -> None:
        if not self.dynamic_library:
            return
        for track_id, last_seen in list(self._dynamic_last_seen_frame.items()):
            if track_id in active_track_ids:
                continue
            if frame_id - last_seen <= self.dynamic_binding_ttl_frames:
                continue
            self._dynamic_last_seen_frame.pop(track_id, None)
            self._last_attempt_frame.pop(track_id, None)
            self._dynamic_track_bindings.pop(track_id, None)
            self._dynamic_pending_enroll.pop(track_id, None)
            self._track_face_debug.pop(track_id, None)

    def _is_faceid_available_this_frame(self, name: str, track_id: int) -> bool:
        owner = self._dynamic_frame_assignments.get(name)
        return owner is None or owner == track_id

    def _reserve_faceid_this_frame(self, name: str, track_id: int) -> None:
        self._dynamic_frame_assignments[name] = track_id

    def _apply_dynamic_alias_to_track(
        self,
        track_id: int,
        face_name_map: Dict[int, str],
        raw_name: str,
    ) -> str:
        final_name = self._final_identity_name(raw_name)
        self._dynamic_track_bindings[track_id] = final_name
        face_name_map[track_id] = final_name
        return final_name

    @staticmethod
    def _score_is_clear(best_score: float, second_score: float, margin: float) -> bool:
        return best_score >= second_score + margin

    def _build_dynamic_frame_record(
        self,
        panorama_bgr: np.ndarray,
        det: dict,
        frame_id: int,
        face_name_map: Dict[int, str],
    ) -> Optional[dict]:
        track_id = int(det.get("track_id", -1))
        bbox = det.get("bbox")
        confidence = det.get("confidence")
        keypoints = det.get("keypoints", [])

        if track_id < 0:
            return None

        bound_name = self._resolve_dynamic_alias(
            self._dynamic_track_bindings.get(track_id)
        )
        current_name = self._resolve_dynamic_alias(bound_name or face_name_map.get(track_id))
        last = self._last_attempt_frame.get(track_id)
        quality = self._dynamic_feature_quality(bbox, keypoints, confidence, panorama_bgr)

        record = {
            "track_id": track_id,
            "bbox": bbox,
            "confidence": confidence,
            "keypoints": keypoints,
            "bound_name": bound_name,
            "current_name": current_name,
            "face": None,
            "feature": None,
            "quality": quality,
            "yaw_deg": quality.get("yaw_deg"),
            "is_primary": bool(quality.get("primary_ok")),
            "can_enroll": bool(quality.get("sample_ok")),
            "best_score": 0.0,
            "second_score": 0.0,
            "ambiguous_match": False,
            "candidates": [],
        }

        if not quality.get("trigger_ok"):
            if current_name is not None:
                record["candidates"].append({
                    "raw": current_name,
                    "final": current_name,
                    "score": 1.0,
                    "assign_score": 2.0,
                    "second_score": 0.0,
                    "event": f"quality_carry_{quality.get('reason', 'gate')}",
                })
                return record
            face_name_map.pop(track_id, None)
            self._dynamic_pending_enroll.pop(track_id, None)
            return None

        if (current_name is not None
                and last is not None
                and frame_id - last < self.dynamic_match_interval):
            record["candidates"].append({
                "raw": current_name,
                "final": current_name,
                "score": 1.0,
                "assign_score": 2.0,
                "second_score": 0.0,
                "event": "carry",
            })
            return record

        face = self.align_face(panorama_bgr, keypoints)
        if face is None:
            if current_name is not None:
                record["candidates"].append({
                    "raw": current_name,
                    "final": current_name,
                    "score": 1.0,
                    "assign_score": 2.0,
                    "second_score": 0.0,
                    "event": "carry",
                })
                return record
            return None

        try:
            feature = self.extract_feature(face)
        except Exception as exc:
            print(f"[FaceRec] feature extraction failed for track {track_id}: {exc}")
            return None

        self._last_attempt_frame[track_id] = frame_id
        yaw_deg = quality.get("yaw_deg")
        is_primary = bool(quality.get("primary_ok"))
        can_enroll = bool(quality.get("sample_ok"))

        scores = self._identity_scores(feature)
        name, score, second_score = self._match_from_scores(scores)
        best_score = float(scores[0][1]) if scores else 0.0
        second_best = float(scores[1][1]) if len(scores) > 1 else 0.0
        ambiguous_match = (
            best_score >= self.threshold
            and not self._score_is_clear(best_score, second_best, self.dynamic_match_margin)
        )

        record.update({
            "face": face,
            "feature": feature,
            "yaw_deg": yaw_deg,
            "is_primary": is_primary,
            "can_enroll": can_enroll,
            "best_score": best_score,
            "second_score": second_best,
            "ambiguous_match": ambiguous_match,
        })

        if bound_name is not None:
            bound_score = self._identity_score_for_name(bound_name, feature)
            raw_name = bound_name
            event = "dynamic_ambiguous_keep" if ambiguous_match else "dynamic_lock_keep"
            final_name = self._final_identity_name(raw_name)
            if (not ambiguous_match
                    or (self.dynamic_ambiguous_keep_bound
                        and bound_score >= self.dynamic_ambiguous_keep_min_score)):
                record["candidates"].append({
                    "raw": raw_name,
                    "final": final_name,
                    "score": float(bound_score),
                    "assign_score": 2.0 + float(bound_score),
                    "second_score": second_score,
                    "event": event,
                })

            should_switch = (
                not self.dynamic_lock_to_track
                and name is not None
                and name != bound_name
                and bound_score < self.dynamic_binding_mismatch_threshold
                and score >= bound_score + self.dynamic_switch_margin
            )
            if should_switch:
                final_name = self._final_identity_name(name)
                record["candidates"].append({
                    "raw": name,
                    "final": final_name,
                    "score": float(score),
                    "assign_score": float(score),
                    "second_score": second_score,
                    "event": "dynamic_switch",
                })
            return record

        if name is not None:
            final_name = self._final_identity_name(name)
            record["candidates"].append({
                "raw": name,
                "final": final_name,
                "score": float(score),
                "assign_score": float(score),
                "second_score": second_score,
                "event": "dynamic_match",
            })
        return record

    @staticmethod
    def _candidate_better(left: tuple, right: tuple) -> bool:
        if left[0] > right[0] + 1e-9:
            return True
        if abs(left[0] - right[0]) <= 1e-9 and left[1] > right[1]:
            return True
        return False

    def _select_dynamic_frame_assignments(self, records: list) -> Dict[int, dict]:
        indexed = [
            (idx, record)
            for idx, record in enumerate(records)
            if record.get("candidates")
        ]
        if not indexed:
            return {}

        face_ids = sorted({
            candidate["final"]
            for _idx, record in indexed
            for candidate in record.get("candidates", [])
        })
        face_to_bit = {name: bit for bit, name in enumerate(face_ids)}

        if len(face_ids) > 20 or len(indexed) > 24:
            chosen = {}
            used = set()
            edges = []
            for idx, record in indexed:
                for cand_idx, candidate in enumerate(record.get("candidates", [])):
                    assign_score = float(candidate.get("assign_score", candidate["score"]))
                    edges.append((assign_score, idx, cand_idx, candidate))
            for _score, idx, cand_idx, candidate in sorted(edges, reverse=True):
                tid = int(records[idx]["track_id"])
                final_name = candidate["final"]
                if tid in chosen or final_name in used:
                    continue
                chosen[tid] = candidate
                used.add(final_name)
            return chosen

        @lru_cache(maxsize=None)
        def solve(pos: int, used_mask: int) -> tuple:
            if pos >= len(indexed):
                return 0.0, 0, ()

            best = solve(pos + 1, used_mask)
            record_idx, record = indexed[pos]
            for cand_idx, candidate in enumerate(record.get("candidates", [])):
                bit = 1 << face_to_bit[candidate["final"]]
                if used_mask & bit:
                    continue
                rest_score, rest_count, rest_pairs = solve(pos + 1, used_mask | bit)
                assign_score = float(candidate.get("assign_score", candidate["score"]))
                current = (
                    rest_score + assign_score,
                    rest_count + 1,
                    ((record_idx, cand_idx),) + rest_pairs,
                )
                if self._candidate_better(current, best):
                    best = current
            return best

        _score, _count, pairs = solve(0, 0)
        assignments = {}
        for record_idx, cand_idx in pairs:
            record = records[record_idx]
            candidate = record["candidates"][cand_idx]
            assignments[int(record["track_id"])] = candidate
        return assignments

    def _apply_dynamic_frame_assignment(
        self,
        panorama_bgr: np.ndarray,
        record: dict,
        candidate: dict,
        face_name_map: Dict[int, str],
        frame_id: int,
    ) -> None:
        track_id = int(record["track_id"])
        raw_name = candidate["raw"]
        final_name = self._final_identity_name(raw_name)
        if not self._is_faceid_available_this_frame(final_name, track_id):
            face_name_map.pop(track_id, None)
            feature = record.get("feature")
            face = record.get("face")
            if feature is not None and face is not None:
                self._update_observer_observation(
                    track_id=track_id,
                    frame_id=frame_id,
                    feature=feature,
                    face_bgr=face,
                    quality=record.get("quality"),
                    face_id=None,
                    raw_face_id=raw_name,
                    event="dynamic_frame_conflict",
                    score=candidate.get("score"),
                    second_score=candidate.get("second_score"),
                )
                self._dump_debug_sample(
                    panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
                    track_id=track_id, frame_id=frame_id, face_id=None,
                    event="dynamic_frame_conflict", score=candidate.get("score"),
                    bbox=record.get("bbox"), confidence=record.get("confidence"),
                    yaw_deg=record.get("yaw_deg"), is_primary=record.get("is_primary"),
                    raw_face_id=raw_name,
                    quality=record.get("quality"),
                )
            return

        previous = face_name_map.get(track_id)
        self._dynamic_track_bindings[track_id] = final_name
        face_name_map[track_id] = final_name
        self._reserve_faceid_this_frame(final_name, track_id)

        feature = record.get("feature")
        face = record.get("face")
        if feature is None or face is None:
            return

        event = candidate.get("event") or "dynamic_match"
        if event == "carry":
            event = "dynamic_lock_keep"
        score = float(candidate.get("score") or 0.0)
        self._add_dynamic_alias_probe(raw_name, feature)
        if previous != final_name:
            action = "switch" if event == "dynamic_switch" else (
                "match" if event == "dynamic_match" else (
                    "ambiguous_keep" if event == "dynamic_ambiguous_keep" else "lock_keep"
                )
            )
            print(f"[FaceRec] dynamic global {action} track {track_id} -> {final_name} "
                  f"(score={score:.3f})")

        update_kind = ""
        if event != "dynamic_ambiguous_keep":
            update_kind = self._maybe_update_dynamic_identity(
                raw_name,
                feature,
                face,
                record.get("quality") or {},
                locked_track_sample=event in {
                    "dynamic_lock_keep",
                    "carry",
                } or str(event).startswith("quality_carry"),
                frame_id=frame_id,
            )
            if update_kind:
                groups = self._dynamic_id_features.get(raw_name, {})
                n_primary = len(groups.get("primary", []))
                n_supp = len(groups.get("supplement", []))
                yaw_deg = record.get("yaw_deg")
                yaw_text = f"{yaw_deg:.1f}°" if yaw_deg is not None else "N/A"
                print(f"[FaceRec] dynamic update {raw_name}: "
                      f"{n_primary} primary, {n_supp} supplement "
                      f"(+{update_kind}, yaw={yaw_text})")
        self._maybe_bind_known_label(final_name, feature, record.get("quality"))
        observer_event = event if not update_kind else f"{event}+{update_kind}"
        self._update_observer_observation(
            track_id=track_id,
            frame_id=frame_id,
            feature=feature,
            face_bgr=face,
            quality=record.get("quality"),
            face_id=final_name,
            raw_face_id=raw_name,
            event=observer_event,
            score=score,
            second_score=candidate.get("second_score"),
        )

        self._dump_debug_sample(
            panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
            track_id=track_id, frame_id=frame_id, face_id=final_name,
            event=event, score=score, bbox=record.get("bbox"),
            confidence=record.get("confidence"), yaw_deg=record.get("yaw_deg"),
            is_primary=record.get("is_primary"), raw_face_id=raw_name,
            second_score=candidate.get("second_score"),
            quality=record.get("quality"),
        )

    def _finalize_unassigned_dynamic_record(
        self,
        panorama_bgr: np.ndarray,
        record: dict,
        face_name_map: Dict[int, str],
        frame_id: int,
    ) -> None:
        track_id = int(record["track_id"])
        face_name_map.pop(track_id, None)
        feature = record.get("feature")
        face = record.get("face")
        if feature is None or face is None:
            return

        score = float(record.get("best_score") or 0.0)

        def _observe(event_name: str, face_id: Optional[str], raw_name: Optional[str]) -> None:
            self._update_observer_observation(
                track_id=track_id,
                frame_id=frame_id,
                feature=feature,
                face_bgr=face,
                quality=record.get("quality"),
                face_id=face_id,
                raw_face_id=raw_name,
                event=event_name,
                score=score,
                second_score=record.get("second_score"),
            )

        if record.get("candidates"):
            event = "dynamic_frame_conflict"
            raw_name = record["candidates"][0]["raw"]
        elif record.get("ambiguous_match"):
            event = "dynamic_match_ambiguous"
            raw_name = None
        elif not record.get("can_enroll"):
            event = "dynamic_skip_enroll"
            raw_name = None
            yaw_deg = record.get("yaw_deg")
            yaw_text = f"{yaw_deg:.1f}°" if yaw_deg is not None else "N/A"
            print(f"[FaceRec] dynamic skip enroll track {track_id}: "
                  f"yaw={yaw_text}, best={score:.3f}, quality gate failed")
            self._dynamic_pending_enroll.pop(track_id, None)
        else:
            raw_name, pending_hits, pending_reset = self._update_pending_dynamic_enroll(
                track_id,
                feature,
                face,
                record.get("quality") or {},
                frame_id,
            )
            if raw_name is None:
                event = "dynamic_enroll_pending_reset" if pending_reset else "dynamic_enroll_pending"
                _observe(event, None, None)
                self._dump_debug_sample(
                    panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
                    track_id=track_id, frame_id=frame_id, face_id=None,
                    event=event, score=score, bbox=record.get("bbox"),
                    confidence=record.get("confidence"), yaw_deg=record.get("yaw_deg"),
                    is_primary=record.get("is_primary"), raw_face_id=None,
                    second_score=record.get("second_score"),
                    quality=record.get("quality"),
                )
                return
            self._add_dynamic_alias_probe(raw_name, feature)
            final_name = self._final_identity_name(raw_name)
            if self._is_faceid_available_this_frame(final_name, track_id):
                self._dynamic_track_bindings[track_id] = final_name
                face_name_map[track_id] = final_name
                self._reserve_faceid_this_frame(final_name, track_id)
                self._maybe_bind_known_label(final_name, feature, record.get("quality"))
                kind = "primary" if record.get("is_primary") else "supplement"
                yaw_deg = record.get("yaw_deg")
                yaw_text = f"{yaw_deg:.1f}°" if yaw_deg is not None else "N/A"
                alias_text = "" if raw_name == final_name else f" alias->{final_name}"
                print(f"[FaceRec] dynamic enroll new track {track_id} -> "
                      f"{raw_name}{alias_text} (best={score:.3f}, {kind}, "
                      f"yaw={yaw_text}, hits={pending_hits})")
                _observe("dynamic_enroll_confirm", final_name, raw_name)
                self._dump_debug_sample(
                    panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
                    track_id=track_id, frame_id=frame_id, face_id=final_name,
                    event="dynamic_enroll_confirm", score=score, bbox=record.get("bbox"),
                    confidence=record.get("confidence"), yaw_deg=record.get("yaw_deg"),
                    is_primary=record.get("is_primary"), raw_face_id=raw_name,
                    second_score=record.get("second_score"),
                    quality=record.get("quality"),
                )
                return
            event = "dynamic_frame_conflict"

        _observe(event, None, raw_name)
        self._dump_debug_sample(
            panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
            track_id=track_id, frame_id=frame_id, face_id=None,
            event=event, score=score, bbox=record.get("bbox"),
            confidence=record.get("confidence"), yaw_deg=record.get("yaw_deg"),
            is_primary=record.get("is_primary"), raw_face_id=raw_name,
            second_score=record.get("second_score"),
            quality=record.get("quality"),
        )

    def _process_dynamic_frame(
        self,
        panorama_bgr: np.ndarray,
        detections: list,
        face_name_map: Dict[int, str],
        frame_id: int,
    ) -> None:
        self._reset_dynamic_frame_if_needed(frame_id)
        records = []
        for det in detections:
            record = self._build_dynamic_frame_record(
                panorama_bgr, det, frame_id, face_name_map
            )
            if record is not None:
                records.append(record)

        assignments = self._select_dynamic_frame_assignments(records)
        for record in records:
            track_id = int(record["track_id"])
            candidate = assignments.get(track_id)
            if candidate is None:
                self._finalize_unassigned_dynamic_record(
                    panorama_bgr, record, face_name_map, frame_id
                )
            else:
                self._apply_dynamic_frame_assignment(
                    panorama_bgr, record, candidate, face_name_map, frame_id
                )

    def _process_dynamic_detection(
        self,
        panorama_bgr: np.ndarray,
        keypoints,
        track_id: int,
        face_name_map: Dict[int, str],
        frame_id: int,
        bbox=None,
        confidence: Optional[float] = None,
    ) -> None:
        self._reset_dynamic_frame_if_needed(frame_id)

        bound_name = self._dynamic_track_bindings.get(track_id)
        bound_name = self._resolve_dynamic_alias(bound_name)
        current_name = self._resolve_dynamic_alias(bound_name or face_name_map.get(track_id))
        quality = self._dynamic_feature_quality(bbox, keypoints, confidence, panorama_bgr)
        if not quality.get("trigger_ok"):
            if current_name is not None and self._is_faceid_available_this_frame(current_name, track_id):
                face_name_map[track_id] = current_name
                self._dynamic_track_bindings[track_id] = current_name
                self._reserve_faceid_this_frame(current_name, track_id)
            else:
                face_name_map.pop(track_id, None)
                self._dynamic_pending_enroll.pop(track_id, None)
            return

        last = self._last_attempt_frame.get(track_id)
        if (track_id in face_name_map
                and last is not None
                and frame_id - last < self.dynamic_match_interval):
            if current_name is not None:
                if self._is_faceid_available_this_frame(current_name, track_id):
                    face_name_map[track_id] = current_name
                    self._dynamic_track_bindings[track_id] = current_name
                    self._reserve_faceid_this_frame(current_name, track_id)
                else:
                    face_name_map.pop(track_id, None)
            return

        face = self.align_face(panorama_bgr, keypoints)
        if face is None:
            return

        try:
            feature = self.extract_feature(face)
        except Exception as exc:
            print(f"[FaceRec] feature extraction failed for track {track_id}: {exc}")
            return

        self._last_attempt_frame[track_id] = frame_id
        yaw_deg = quality.get("yaw_deg")
        is_primary = bool(quality.get("primary_ok"))
        can_enroll = bool(quality.get("sample_ok"))
        if bound_name is not None:
            bound_score = self._identity_score_for_name(bound_name, feature)
            name, score = self.match(feature)
            bound_name = self._resolve_dynamic_alias(bound_name) or bound_name
            should_switch = (
                not self.dynamic_lock_to_track
                and
                name is not None
                and name != bound_name
                and self._is_faceid_available_this_frame(name, track_id)
                and bound_score < self.dynamic_binding_mismatch_threshold
                and score >= bound_score + self.dynamic_switch_margin
            )
            raw_name = name if should_switch else bound_name
            final_name = self._final_identity_name(raw_name)
            if not self._is_faceid_available_this_frame(final_name, track_id):
                print(f"[FaceRec] dynamic conflict: skip track {track_id}, "
                      f"{final_name} already used this frame")
                face_name_map.pop(track_id, None)
                self._dump_debug_sample(
                    panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
                    track_id=track_id, frame_id=frame_id, face_id=None,
                    event="dynamic_frame_conflict", score=score if should_switch else bound_score,
                    bbox=bbox, confidence=confidence, yaw_deg=yaw_deg,
                    is_primary=is_primary,
                    quality=quality,
                )
                return
            else:
                final_score = score if should_switch else bound_score

            previous = face_name_map.get(track_id)
            face_name_map[track_id] = final_name
            self._dynamic_track_bindings[track_id] = final_name
            self._add_dynamic_alias_probe(raw_name, feature)
            if previous != final_name:
                action = "switch" if should_switch else "lock_keep"
                print(f"[FaceRec] dynamic {action} track {track_id} -> {final_name} "
                      f"(score={final_score:.3f}, bound={bound_score:.3f})")

            update_kind = self._maybe_update_dynamic_identity(
                raw_name,
                feature,
                face,
                quality,
                locked_track_sample=not should_switch,
                frame_id=frame_id,
            )
            if update_kind:
                groups = self._dynamic_id_features.get(raw_name, {})
                n_primary = len(groups.get("primary", []))
                n_supp = len(groups.get("supplement", []))
                yaw_text = f"{yaw_deg:.1f}°" if yaw_deg is not None else "N/A"
                print(f"[FaceRec] dynamic update {raw_name}: "
                      f"{n_primary} primary, {n_supp} supplement "
                      f"(+{update_kind}, yaw={yaw_text})")
            final_name = self._apply_dynamic_alias_to_track(track_id, face_name_map, raw_name)
            self._reserve_faceid_this_frame(final_name, track_id)
            self._maybe_bind_known_label(final_name, feature, quality)
            self._dump_debug_sample(
                panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
                track_id=track_id, frame_id=frame_id, face_id=final_name,
                event="dynamic_switch" if should_switch else "dynamic_lock_keep",
                score=final_score, bbox=bbox, confidence=confidence,
                yaw_deg=yaw_deg, is_primary=is_primary, raw_face_id=raw_name,
                quality=quality,
            )
            return

        name, score = self.match(feature)
        if name is not None:
            raw_name = name
            final_name = self._final_identity_name(raw_name)
            if not self._is_faceid_available_this_frame(final_name, track_id):
                print(f"[FaceRec] dynamic conflict: skip track {track_id}, "
                      f"{final_name} already used this frame")
                face_name_map.pop(track_id, None)
                self._dump_debug_sample(
                    panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
                    track_id=track_id, frame_id=frame_id, face_id=None,
                    event="dynamic_frame_conflict", score=score, bbox=bbox,
                    confidence=confidence, yaw_deg=yaw_deg, is_primary=is_primary,
                    raw_face_id=raw_name,
                    quality=quality,
                )
                return
            previous = face_name_map.get(track_id)
            self._dynamic_track_bindings[track_id] = final_name
            face_name_map[track_id] = final_name
            self._add_dynamic_alias_probe(raw_name, feature)
            if previous != final_name:
                print(f"[FaceRec] dynamic match track {track_id} -> {final_name} ({score:.3f})")
            update_kind = self._maybe_update_dynamic_identity(
                raw_name,
                feature,
                face,
                quality,
                locked_track_sample=False,
                frame_id=frame_id,
            )
            if update_kind:
                groups = self._dynamic_id_features.get(raw_name, {})
                n_primary = len(groups.get("primary", []))
                n_supp = len(groups.get("supplement", []))
                yaw_text = f"{yaw_deg:.1f}°" if yaw_deg is not None else "N/A"
                print(f"[FaceRec] dynamic update {raw_name}: "
                      f"{n_primary} primary, {n_supp} supplement "
                      f"(+{update_kind}, yaw={yaw_text})")
            final_name = self._apply_dynamic_alias_to_track(track_id, face_name_map, raw_name)
            self._reserve_faceid_this_frame(final_name, track_id)
            self._maybe_bind_known_label(final_name, feature, quality)
            self._dump_debug_sample(
                panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
                track_id=track_id, frame_id=frame_id, face_id=final_name,
                event="dynamic_match", score=score, bbox=bbox,
                confidence=confidence, yaw_deg=yaw_deg, is_primary=is_primary,
                raw_face_id=raw_name,
                quality=quality,
            )
            return

        if not can_enroll:
            yaw_text = f"{yaw_deg:.1f}°" if yaw_deg is not None else "N/A"
            print(f"[FaceRec] dynamic skip enroll track {track_id}: "
                  f"yaw={yaw_text}, best={score:.3f}, quality gate failed")
            self._dynamic_pending_enroll.pop(track_id, None)
            self._dump_debug_sample(
                panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
                track_id=track_id, frame_id=frame_id, face_id=None,
                event="dynamic_skip_enroll", score=score, bbox=bbox,
                confidence=confidence, yaw_deg=yaw_deg, is_primary=is_primary,
                quality=quality,
            )
            return

        raw_name, pending_hits, pending_reset = self._update_pending_dynamic_enroll(
            track_id, feature, face, quality, frame_id
        )
        if raw_name is None:
            self._dump_debug_sample(
                panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
                track_id=track_id, frame_id=frame_id, face_id=None,
                event="dynamic_enroll_pending_reset" if pending_reset else "dynamic_enroll_pending",
                score=score, bbox=bbox, confidence=confidence,
                yaw_deg=yaw_deg, is_primary=is_primary,
                quality=quality,
            )
            return
        self._add_dynamic_alias_probe(raw_name, feature)
        final_name = self._apply_dynamic_alias_to_track(track_id, face_name_map, raw_name)
        self._reserve_faceid_this_frame(final_name, track_id)
        self._maybe_bind_known_label(final_name, feature, quality)
        kind = "primary" if is_primary else "supplement"
        yaw_text = f"{yaw_deg:.1f}°" if yaw_deg is not None else "N/A"
        alias_text = "" if raw_name == final_name else f" alias->{final_name}"
        print(f"[FaceRec] dynamic enroll new track {track_id} -> {raw_name}{alias_text} "
              f"(best={score:.3f}, {kind}, yaw={yaw_text}, hits={pending_hits})")
        self._dump_debug_sample(
            panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
            track_id=track_id, frame_id=frame_id, face_id=final_name,
            event="dynamic_enroll_confirm", score=score, bbox=bbox,
            confidence=confidence, yaw_deg=yaw_deg, is_primary=is_primary,
            raw_face_id=raw_name,
            quality=quality,
        )

    def process_detection(
        self,
        panorama_bgr: np.ndarray,
        keypoints,
        track_id: int,
        is_new_track: bool,
        face_name_map: Dict[int, str],
        frame_id: int,
        bbox=None,
        confidence: Optional[float] = None,
    ) -> None:
        if self.dynamic_library:
            self._process_dynamic_detection(
                panorama_bgr, keypoints, track_id, face_name_map, frame_id,
                bbox=bbox, confidence=confidence,
            )
            return

        if track_id in face_name_map:
            return

        last = self._last_attempt_frame.get(track_id)
        if last is not None and frame_id - last < self.cooldown_frames:
            return

        # New tracks get one immediate try; later attempts require a frontal face.
        if not is_new_track and not self.is_frontal(keypoints):
            return

        self._last_attempt_frame[track_id] = frame_id
        face = self.align_face(panorama_bgr, keypoints)
        if face is None:
            return

        try:
            feature = self.extract_feature(face)
        except Exception as exc:
            print(f"[FaceRec] feature extraction failed for track {track_id}: {exc}")
            return

        name, score = self.match(feature)
        self._dump_debug_sample(
            panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
            track_id=track_id, frame_id=frame_id, face_id=name,
            event="static_match" if name is not None else "static_unknown",
            score=score, bbox=bbox, confidence=confidence,
            yaw_deg=self._yaw_deg(keypoints), is_primary=self.is_frontal(keypoints),
            quality=None,
        )
        if name is not None:
            face_name_map[track_id] = name
            print(f"[FaceRec] track {track_id} -> {name} ({score:.3f})")

    def process_frame(
        self,
        panorama_bgr: np.ndarray,
        detections: list,
        new_track_ids: set,
        face_name_map: Dict[int, str],
        frame_id: int,
    ) -> None:
        if self.dynamic_library:
            active_ids = {
                int(det.get("track_id", -1))
                for det in detections
                if int(det.get("track_id", -1)) >= 0
            }
            for tid in active_ids:
                self._dynamic_last_seen_frame[tid] = frame_id
            self._purge_stale_dynamic_state(active_ids, frame_id)

        if self.dynamic_library and self.dynamic_global_assignment:
            self._process_dynamic_frame(
                panorama_bgr, detections, face_name_map, frame_id
            )
            return

        for det in detections:
            tid = int(det.get("track_id", -1))
            self.process_detection(
                panorama_bgr,
                det.get("keypoints", []),
                tid,
                is_new_track=(tid in new_track_ids),
                face_name_map=face_name_map,
                frame_id=frame_id,
                bbox=det.get("bbox"),
                confidence=det.get("confidence"),
            )

    def cleanup_track(self, track_id: int) -> None:
        if self.dynamic_library:
            # Keep TrackID -> FaceID bindings for a TTL. HybridSORT can emit the
            # same TrackID again after a short gap, and board_cpp relies on this
            # binding to carry FaceID without re-enrolling a person.
            return
        self._last_attempt_frame.pop(track_id, None)
        self._track_face_debug.pop(track_id, None)
