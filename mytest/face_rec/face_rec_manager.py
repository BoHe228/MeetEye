"""
Minimal AdaFace manager used by MeetEye.

The wider AdaFace project is intentionally not imported wholesale here. This
module keeps only the pieces MeetEye needs:
  - load an IR-18 AdaFace checkpoint
  - load a directory of .npy identity features
  - align a face from YOLO pose keypoints
  - extract and match a 512D L2-normalized feature
  - cache recognition attempts by TrackID
"""
from __future__ import annotations

import os
import sys
import time
import json
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
_NOSE_SCALE = 0.6


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


class FaceRecManager:
    """Small AdaFace wrapper for track-aware face recognition."""

    def __init__(
        self,
        model_path: str,
        library_dir: str,
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
        debug_dump_dir: Optional[str] = None,
        debug_dump_every: int = 1,
        debug_dump_max: int = 0,
    ):
        self.model_path = model_path
        self.library_dir = library_dir
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
        if self.debug_dump_dir:
            os.makedirs(self.debug_dump_dir, exist_ok=True)
            os.makedirs(os.path.join(self.debug_dump_dir, "aligned_faces"), exist_ok=True)
            os.makedirs(os.path.join(self.debug_dump_dir, "bbox_crops"), exist_ok=True)
            os.makedirs(os.path.join(self.debug_dump_dir, "features"), exist_ok=True)
            print(f"[FaceRecDebug] dump enabled: {self.debug_dump_dir}")

        self.model = self._load_model(model_path)
        if self.dynamic_library:
            os.makedirs(self.dynamic_library_dir, exist_ok=True)
            os.makedirs(os.path.join(self.dynamic_library_dir, "_preview"), exist_ok=True)
            self.lib_names, self.lib_matrix = [], None
            self._dynamic_id_features: Dict[str, Dict[str, list]] = {}
            self._dynamic_track_bindings: Dict[int, str] = {}
            # FaceID is a global identity. TrackID is only a temporary carrier.
            # Same-frame assignments are kept separately to prevent two visible
            # targets from drawing the same faceid in one frame.
            self._dynamic_frame_id: Optional[int] = None
            self._dynamic_frame_assignments: Dict[str, int] = {}
            self._dynamic_aliases: Dict[str, str] = {}
            self._dynamic_alias_votes: Dict[Tuple[str, str], int] = {}
            self._dynamic_alias_probe_features: Dict[str, list] = {}
            self._dynamic_next_id = 1
            print(f"[FaceRec] dynamic library enabled: {self.dynamic_library_dir}")
            print("[FaceRec] dynamic library starts empty; previous sessions are not loaded")
        else:
            self.lib_names, self.lib_matrix = self._load_library(library_dir)
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
        }
        meta_path = os.path.join(self.debug_dump_dir, "metadata.jsonl")
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    def align_face(self, image_bgr: np.ndarray, keypoints) -> Optional[np.ndarray]:
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
        scores = []
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
            return scores

        if self.lib_matrix is None or not self.lib_names:
            return []

        sims = self.lib_matrix @ query
        best_by_name: Dict[str, float] = {}
        for name, score in zip(self.lib_names, sims):
            score = float(score)
            if score > best_by_name.get(name, 0.0):
                best_by_name[name] = score
        scores = sorted(best_by_name.items(), key=lambda item: item[1], reverse=True)
        return scores

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
    ) -> bool:
        feat = self._normalize_feature(feature)
        groups = self._dynamic_id_features.setdefault(
            name, {"primary": [], "supplement": []}
        )
        bucket_name = "primary" if is_primary else "supplement"
        bucket = groups[bucket_name]
        all_feats = groups["primary"] + groups["supplement"]
        if all_feats:
            best_existing = max(float(np.dot(old, feat)) for old in all_feats)
            if best_existing >= 1.0 - self.dynamic_min_sample_diversity:
                return False

        bucket.append(feat)
        if len(bucket) > self.dynamic_max_samples_per_id:
            del bucket[0:len(bucket) - self.dynamic_max_samples_per_id]

        self._save_dynamic_identity(name, face_bgr)
        self._rebuild_dynamic_matrix()
        return True

    def _add_dynamic_alias_probe(self, name: str, feature: np.ndarray) -> None:
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
        self._dynamic_id_features[name] = {"primary": [], "supplement": []}
        self._add_dynamic_sample(name, feature, face_bgr, is_primary=is_primary)
        return name

    def _update_dynamic_identity(
        self,
        name: str,
        feature: np.ndarray,
        face_bgr: np.ndarray,
        *,
        is_primary: bool,
    ) -> bool:
        return self._add_dynamic_sample(name, feature, face_bgr, is_primary=is_primary)

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
        final_name = self._maybe_auto_alias_dynamic_identity(raw_name) or raw_name
        final_name = self._resolve_dynamic_alias(final_name) or final_name
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

        if self._bbox_too_small(bbox, panorama_bgr):
            face_name_map.pop(track_id, None)
            return None

        bound_name = self._resolve_dynamic_alias(
            self._dynamic_track_bindings.get(track_id)
        )
        current_name = self._resolve_dynamic_alias(bound_name or face_name_map.get(track_id))
        last = self._last_attempt_frame.get(track_id)

        record = {
            "track_id": track_id,
            "bbox": bbox,
            "confidence": confidence,
            "keypoints": keypoints,
            "bound_name": bound_name,
            "current_name": current_name,
            "face": None,
            "feature": None,
            "yaw_deg": None,
            "is_primary": None,
            "can_enroll": False,
            "best_score": 0.0,
            "second_score": 0.0,
            "ambiguous_match": False,
            "candidates": [],
        }

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
        yaw_deg = self._yaw_deg(keypoints)
        is_primary = (
            yaw_deg is not None
            and yaw_deg <= self.dynamic_primary_max_yaw_deg
        )
        can_enroll = yaw_deg is not None and yaw_deg <= self.dynamic_enroll_max_yaw_deg

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
            bound_score = self._dynamic_identity_score(bound_name, feature)
            raw_name = bound_name
            event = "dynamic_ambiguous_keep" if ambiguous_match else "dynamic_lock_keep"
            final_name = self._resolve_dynamic_alias(raw_name) or raw_name
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
                final_name = self._resolve_dynamic_alias(name) or name
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
            final_name = self._resolve_dynamic_alias(name) or name
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
        final_name = self._maybe_auto_alias_dynamic_identity(raw_name) or raw_name
        final_name = self._resolve_dynamic_alias(final_name) or final_name
        if not self._is_faceid_available_this_frame(final_name, track_id):
            face_name_map.pop(track_id, None)
            feature = record.get("feature")
            face = record.get("face")
            if feature is not None and face is not None:
                self._dump_debug_sample(
                    panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
                    track_id=track_id, frame_id=frame_id, face_id=None,
                    event="dynamic_frame_conflict", score=candidate.get("score"),
                    bbox=record.get("bbox"), confidence=record.get("confidence"),
                    yaw_deg=record.get("yaw_deg"), is_primary=record.get("is_primary"),
                    raw_face_id=raw_name,
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

        self._add_dynamic_alias_probe(raw_name, feature)
        event = candidate.get("event") or "dynamic_match"
        if event == "carry":
            event = "dynamic_lock_keep"
        score = float(candidate.get("score") or 0.0)
        if previous != final_name:
            action = "switch" if event == "dynamic_switch" else (
                "match" if event == "dynamic_match" else (
                    "ambiguous_keep" if event == "dynamic_ambiguous_keep" else "lock_keep"
                )
            )
            print(f"[FaceRec] dynamic global {action} track {track_id} -> {final_name} "
                  f"(score={score:.3f})")

        if event != "dynamic_ambiguous_keep" and score >= self.dynamic_update_similarity:
            if self._update_dynamic_identity(raw_name, feature, face,
                                             is_primary=bool(record.get("is_primary"))):
                groups = self._dynamic_id_features.get(raw_name, {})
                n_primary = len(groups.get("primary", []))
                n_supp = len(groups.get("supplement", []))
                kind = "primary" if record.get("is_primary") else "supplement"
                yaw_deg = record.get("yaw_deg")
                yaw_text = f"{yaw_deg:.1f}°" if yaw_deg is not None else "N/A"
                print(f"[FaceRec] dynamic update {raw_name}: "
                      f"{n_primary} primary, {n_supp} supplement "
                      f"(+{kind}, yaw={yaw_text})")

        self._dump_debug_sample(
            panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
            track_id=track_id, frame_id=frame_id, face_id=final_name,
            event=event, score=score, bbox=record.get("bbox"),
            confidence=record.get("confidence"), yaw_deg=record.get("yaw_deg"),
            is_primary=record.get("is_primary"), raw_face_id=raw_name,
            second_score=candidate.get("second_score"),
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
                  f"yaw={yaw_text}, best={score:.3f}, "
                  f"need <= {self.dynamic_enroll_max_yaw_deg:.1f}°")
        else:
            raw_name = self._create_dynamic_identity(
                feature, face, is_primary=bool(record.get("is_primary"))
            )
            self._add_dynamic_alias_probe(raw_name, feature)
            final_name = self._maybe_auto_alias_dynamic_identity(raw_name) or raw_name
            final_name = self._resolve_dynamic_alias(final_name) or final_name
            if self._is_faceid_available_this_frame(final_name, track_id):
                self._dynamic_track_bindings[track_id] = final_name
                face_name_map[track_id] = final_name
                self._reserve_faceid_this_frame(final_name, track_id)
                kind = "primary" if record.get("is_primary") else "supplement"
                yaw_deg = record.get("yaw_deg")
                yaw_text = f"{yaw_deg:.1f}°" if yaw_deg is not None else "N/A"
                alias_text = "" if raw_name == final_name else f" alias->{final_name}"
                print(f"[FaceRec] dynamic enroll new track {track_id} -> "
                      f"{raw_name}{alias_text} (best={score:.3f}, {kind}, yaw={yaw_text})")
                self._dump_debug_sample(
                    panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
                    track_id=track_id, frame_id=frame_id, face_id=final_name,
                    event="dynamic_enroll", score=score, bbox=record.get("bbox"),
                    confidence=record.get("confidence"), yaw_deg=record.get("yaw_deg"),
                    is_primary=record.get("is_primary"), raw_face_id=raw_name,
                    second_score=record.get("second_score"),
                )
                return
            event = "dynamic_frame_conflict"

        self._dump_debug_sample(
            panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
            track_id=track_id, frame_id=frame_id, face_id=None,
            event=event, score=score, bbox=record.get("bbox"),
            confidence=record.get("confidence"), yaw_deg=record.get("yaw_deg"),
            is_primary=record.get("is_primary"), raw_face_id=raw_name,
            second_score=record.get("second_score"),
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

        if self._bbox_too_small(bbox, panorama_bgr):
            face_name_map.pop(track_id, None)
            return

        bound_name = self._dynamic_track_bindings.get(track_id)
        bound_name = self._resolve_dynamic_alias(bound_name)
        last = self._last_attempt_frame.get(track_id)
        if (track_id in face_name_map
                and last is not None
                and frame_id - last < self.dynamic_match_interval):
            current_name = self._resolve_dynamic_alias(bound_name or face_name_map.get(track_id))
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
        yaw_proxy = self._yaw_proxy(keypoints)
        yaw_deg = self._yaw_deg(keypoints)
        is_primary = (
            yaw_deg is not None
            and yaw_deg <= self.dynamic_primary_max_yaw_deg
        )
        can_enroll = yaw_deg is not None and yaw_deg <= self.dynamic_enroll_max_yaw_deg
        if bound_name is not None:
            bound_score = self._dynamic_identity_score(bound_name, feature)
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
            final_name = self._resolve_dynamic_alias(raw_name) or raw_name
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

            if final_score >= self.dynamic_update_similarity:
                if self._update_dynamic_identity(raw_name, feature, face, is_primary=is_primary):
                    groups = self._dynamic_id_features.get(raw_name, {})
                    n_primary = len(groups.get("primary", []))
                    n_supp = len(groups.get("supplement", []))
                    kind = "primary" if is_primary else "supplement"
                    yaw_text = f"{yaw_deg:.1f}°" if yaw_deg is not None else "N/A"
                    print(f"[FaceRec] dynamic update {raw_name}: "
                          f"{n_primary} primary, {n_supp} supplement "
                          f"(+{kind}, yaw={yaw_text})")
            final_name = self._apply_dynamic_alias_to_track(track_id, face_name_map, raw_name)
            self._reserve_faceid_this_frame(final_name, track_id)
            self._dump_debug_sample(
                panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
                track_id=track_id, frame_id=frame_id, face_id=final_name,
                event="dynamic_switch" if should_switch else "dynamic_lock_keep",
                score=final_score, bbox=bbox, confidence=confidence,
                yaw_deg=yaw_deg, is_primary=is_primary, raw_face_id=raw_name,
            )
            return

        name, score = self.match(feature)
        if name is not None:
            raw_name = name
            final_name = self._resolve_dynamic_alias(raw_name) or raw_name
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
                )
                return
            previous = face_name_map.get(track_id)
            self._dynamic_track_bindings[track_id] = final_name
            face_name_map[track_id] = final_name
            self._add_dynamic_alias_probe(raw_name, feature)
            if previous != final_name:
                print(f"[FaceRec] dynamic match track {track_id} -> {final_name} ({score:.3f})")
            if score >= self.dynamic_update_similarity:
                if self._update_dynamic_identity(raw_name, feature, face, is_primary=is_primary):
                    groups = self._dynamic_id_features.get(raw_name, {})
                    n_primary = len(groups.get("primary", []))
                    n_supp = len(groups.get("supplement", []))
                    kind = "primary" if is_primary else "supplement"
                    yaw_text = f"{yaw_deg:.1f}°" if yaw_deg is not None else "N/A"
                    print(f"[FaceRec] dynamic update {raw_name}: "
                          f"{n_primary} primary, {n_supp} supplement "
                          f"(+{kind}, yaw={yaw_text})")
            final_name = self._apply_dynamic_alias_to_track(track_id, face_name_map, raw_name)
            self._reserve_faceid_this_frame(final_name, track_id)
            self._dump_debug_sample(
                panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
                track_id=track_id, frame_id=frame_id, face_id=final_name,
                event="dynamic_match", score=score, bbox=bbox,
                confidence=confidence, yaw_deg=yaw_deg, is_primary=is_primary,
                raw_face_id=raw_name,
            )
            return

        if not can_enroll:
            yaw_text = f"{yaw_deg:.1f}°" if yaw_deg is not None else "N/A"
            print(f"[FaceRec] dynamic skip enroll track {track_id}: "
                  f"yaw={yaw_text}, best={score:.3f}, need <= {self.dynamic_enroll_max_yaw_deg:.1f}°")
            self._dump_debug_sample(
                panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
                track_id=track_id, frame_id=frame_id, face_id=None,
                event="dynamic_skip_enroll", score=score, bbox=bbox,
                confidence=confidence, yaw_deg=yaw_deg, is_primary=is_primary,
            )
            return

        raw_name = self._create_dynamic_identity(feature, face, is_primary=is_primary)
        self._add_dynamic_alias_probe(raw_name, feature)
        final_name = self._apply_dynamic_alias_to_track(track_id, face_name_map, raw_name)
        self._reserve_faceid_this_frame(final_name, track_id)
        kind = "primary" if is_primary else "supplement"
        yaw_text = f"{yaw_deg:.1f}°" if yaw_deg is not None else "N/A"
        alias_text = "" if raw_name == final_name else f" alias->{final_name}"
        print(f"[FaceRec] dynamic enroll new track {track_id} -> {raw_name}{alias_text} "
              f"(best={score:.3f}, {kind}, yaw={yaw_text})")
        self._dump_debug_sample(
            panorama_bgr=panorama_bgr, face_bgr=face, feature=feature,
            track_id=track_id, frame_id=frame_id, face_id=final_name,
            event="dynamic_enroll", score=score, bbox=bbox,
            confidence=confidence, yaw_deg=yaw_deg, is_primary=is_primary,
            raw_face_id=raw_name,
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
        self._last_attempt_frame.pop(track_id, None)
        if self.dynamic_library:
            self._dynamic_track_bindings.pop(track_id, None)
