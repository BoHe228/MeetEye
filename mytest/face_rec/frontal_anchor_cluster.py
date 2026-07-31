#!/usr/bin/env python3
"""Cluster wide-angle face samples by frontal anchor features.

This tool is meant for offline verification. It extracts face samples from a
video, computes yaw from nose position relative to the two eyes, chooses
high-quality frontal samples as identity anchors, and assigns every sample to
the closest anchor by AdaFace cosine similarity.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_ADAFACE_DIR = _SCRIPT_DIR / "AdaFace"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_ADAFACE_DIR) not in sys.path:
    sys.path.insert(0, str(_ADAFACE_DIR))

from face_rec_manager import FaceRecManager  # noqa: E402
from net import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wide-angle frontal-anchor clustering with AdaFace features."
    )
    parser.add_argument(
        "--video-path",
        default="Wide-Angle_test/data/广角_小会议室_6.16.mp4",
        help="input video used as the face sample source",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolo_model/yolov8n-face.pt",
        help="YOLO face model with 5-point landmarks",
    )
    parser.add_argument(
        "--adaface-checkpoint",
        default="face_rc/board_cpp/models/adaface_ir50_webface4m.ckpt",
        help="AdaFace checkpoint used for 512D feature extraction",
    )
    parser.add_argument(
        "--adaface-model-name",
        default="ir_50",
        choices=["ir_18", "ir_34", "ir_50", "ir_101", "ir_se_50"],
        help="AdaFace backbone name",
    )
    parser.add_argument(
        "--output-dir",
        default="debug_face_dump/wide_angle_6.16_frontal_anchor_ir50",
        help="output directory for crops, features, metadata and HTML",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--frame-step", type=int, default=5,
                        help="process one frame every N frames")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="maximum saved face samples; 0 means unlimited")
    parser.add_argument("--min-box-size", type=float, default=60.0,
                        help="minimum bbox side for extracted samples")
    parser.add_argument("--min-keypoint-conf", type=float, default=0.6,
                        help="minimum eye/nose keypoint confidence for extracted samples")
    parser.add_argument("--anchor-min-box-size", type=float, default=80.0,
                        help="minimum bbox side for frontal anchor candidates")
    parser.add_argument("--anchor-min-conf", type=float, default=0.5,
                        help="minimum detector confidence for frontal anchor candidates")
    parser.add_argument("--frontal-yaw-deg", type=float, default=10.0,
                        help="yaw threshold for frontal anchor candidates")
    parser.add_argument(
        "--cluster-mode",
        default="frontal-anchor",
        choices=["frontal-anchor", "board-gallery"],
        help="frontal-anchor uses one fixed frontal vector; board-gallery mimics board_cpp primary+samples",
    )
    parser.add_argument("--match-threshold", type=float, default=0.65,
                        help="minimum cosine similarity to assign sample to an anchor/gallery")
    parser.add_argument("--match-margin", type=float, default=0.08,
                        help="top1 must exceed top2 by this margin when multiple anchors/galleries exist")
    parser.add_argument("--update-similarity", type=float, default=0.65,
                        help="board-gallery: existing FaceID similarity required before primary feature update")
    parser.add_argument("--pose-sample-similarity", type=float, default=0.55,
                        help="board-gallery: existing FaceID similarity required before adding a high-quality pose sample")
    parser.add_argument("--update-max-similarity", type=float, default=0.75,
                        help="board-gallery: add pose sample only when similarity is not above this value")
    parser.add_argument("--min-sample-diversity", type=float, default=0.015,
                        help="board-gallery: skip adding near-duplicate samples above 1 - this value")
    parser.add_argument("--update-min-box-size", type=float, default=60.0,
                        help="board-gallery: minimum bbox side for supplement sample updates")
    parser.add_argument("--update-min-keypoint-conf", type=float, default=0.6,
                        help="board-gallery: minimum eye/nose keypoint confidence for supplement samples")
    parser.add_argument("--update-min-score", type=float, default=0.5,
                        help="board-gallery: minimum detector confidence for supplement samples")
    parser.add_argument("--primary-min-box-size", type=float, default=70.0,
                        help="board-gallery: minimum bbox side for primary feature updates")
    parser.add_argument("--primary-min-keypoint-conf", type=float, default=0.7,
                        help="board-gallery: minimum eye/nose keypoint confidence for primary feature updates")
    parser.add_argument("--primary-min-score", type=float, default=0.6,
                        help="board-gallery: minimum detector confidence for primary feature updates")
    parser.add_argument("--primary-ema-alpha", type=float, default=0.1,
                        help="board-gallery: EMA alpha used when updating a primary feature")
    parser.add_argument("--max-samples-per-id", type=int, default=5,
                        help="board-gallery: maximum primary+supplement features kept for each FaceID")
    parser.add_argument("--dynamic-enroll-confirm-frames", type=int, default=3,
                        help="board-gallery: unmatched primary-quality samples required before creating a new FaceID")
    parser.add_argument("--five-point-scale", type=float, default=1.20,
                        help="same 5-point crop scale as board_cpp")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="torch/AdaFace device; YOLO uses the same device when possible")
    parser.add_argument("--max-images-per-group", type=int, default=80,
                        help="maximum samples shown in each HTML group")
    parser.add_argument("--html-name", default="frontal_anchor_clusters.html")
    return parser.parse_args()


def repo_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def unique_output_dir(base: Path) -> Path:
    if not base.exists() or not any(base.iterdir()):
        base.mkdir(parents=True, exist_ok=True)
        return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base.with_name(f"{base.name}_{stamp}")
    idx = 2
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{stamp}_{idx}")
        idx += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
    clean = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        clean[new_key] = value
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        print(f"[AdaFace] missing keys: {len(missing)}")
    if unexpected:
        print(f"[AdaFace] unexpected keys: {len(unexpected)}")


def load_adaface(checkpoint: Path, model_name: str, device: torch.device) -> torch.nn.Module:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"AdaFace checkpoint not found: {checkpoint}")
    model = build_model(model_name)
    load_checkpoint(model, checkpoint)
    model.to(device)
    model.eval()
    return model


def normalize_features(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1.0e-12)
    return features / norms


def extract_features(
    model: torch.nn.Module,
    crops_bgr: Sequence[np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    batch_size = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, len(crops_bgr), batch_size):
            batch = crops_bgr[start:start + batch_size]
            arr = np.stack([
                ((cv2.resize(crop, (112, 112)).astype(np.float32) / 255.0) - 0.5) / 0.5
                for crop in batch
            ])
            arr = np.transpose(arr, (0, 3, 1, 2))
            tensor = torch.from_numpy(arr).to(device)
            pred = model(tensor)
            feat = pred[0] if isinstance(pred, (tuple, list)) else pred
            outputs.append(feat.detach().cpu().numpy().astype(np.float32))
    return normalize_features(np.concatenate(outputs, axis=0))


def make_crop_manager(scale: float) -> FaceRecManager:
    manager = object.__new__(FaceRecManager)
    manager.align_mode = "five-point"
    manager.five_point_scale = float(scale)
    return manager


def safe_bbox_crop(image: np.ndarray, bbox: Sequence[float]) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def valid_five_point_keypoints(keypoints: Sequence[Sequence[float]]) -> bool:
    if keypoints is None or len(keypoints) != 5:
        return False
    for kp in keypoints:
        if len(kp) < 2:
            return False
        x = float(kp[0])
        y = float(kp[1])
        conf = float(kp[2]) if len(kp) >= 3 else 1.0
        if (x == 0.0 and y == 0.0) or conf < 0.1:
            return False
    return True


def min_eye_nose_conf(keypoints: Sequence[Sequence[float]]) -> float:
    if keypoints is None or len(keypoints) < 3:
        return 0.0
    confs = []
    for kp in keypoints[:3]:
        confs.append(float(kp[2]) if len(kp) >= 3 else 1.0)
    return float(min(confs)) if confs else 0.0


def choose_torch_device(name: str) -> torch.device:
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def extract_samples(args: argparse.Namespace, output_dir: Path) -> Tuple[List[dict], List[np.ndarray]]:
    video_path = repo_path(args.video_path)
    yolo_model = repo_path(args.yolo_model)
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    if not yolo_model.is_file():
        raise FileNotFoundError(f"YOLO model not found: {yolo_model}")

    aligned_dir = output_dir / "aligned_faces"
    bbox_dir = output_dir / "bbox_crops"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    bbox_dir.mkdir(parents=True, exist_ok=True)

    crop_manager = make_crop_manager(args.five_point_scale)
    yaw_manager = make_crop_manager(args.five_point_scale)
    model = YOLO(str(yolo_model))
    torch_device = choose_torch_device(args.device)
    yolo_device = 0 if torch_device.type == "cuda" else "cpu"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    rows: List[dict] = []
    crops: List[np.ndarray] = []
    frame_index = 0
    sample_idx = 0
    max_samples = max(0, int(args.max_samples))
    frame_step = max(1, int(args.frame_step))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if (frame_index - 1) % frame_step != 0:
            continue
        pred = model.predict(
            frame,
            imgsz=int(args.imgsz),
            conf=float(args.conf),
            verbose=False,
            device=yolo_device,
        )[0]
        if pred.boxes is None or pred.keypoints is None:
            continue
        boxes = pred.boxes.xyxy.detach().cpu().numpy()
        confs = pred.boxes.conf.detach().cpu().numpy()
        keypoints = pred.keypoints.data.detach().cpu().numpy()
        order = np.argsort(-confs)
        for det_rank, det_idx in enumerate(order.tolist()):
            bbox = boxes[det_idx].astype(np.float32)
            width = float(bbox[2] - bbox[0])
            height = float(bbox[3] - bbox[1])
            if min(width, height) < float(args.min_box_size):
                continue
            kpts = keypoints[det_idx].astype(np.float32).tolist()
            if not valid_five_point_keypoints(kpts):
                continue
            keypoint_min_conf = min_eye_nose_conf(kpts)
            if keypoint_min_conf < float(args.min_keypoint_conf):
                continue
            face = crop_manager.align_face(frame, kpts)
            if face is None:
                continue
            bbox_crop = safe_bbox_crop(frame, bbox)
            yaw_deg = yaw_manager._yaw_deg(kpts)
            sample_idx += 1
            stem = f"{sample_idx:06d}_f{frame_index:06d}_d{det_idx}"
            aligned_rel = Path("aligned_faces") / f"{stem}.jpg"
            bbox_rel = Path("bbox_crops") / f"{stem}.jpg"
            cv2.imwrite(str(output_dir / aligned_rel), face)
            if bbox_crop is not None:
                cv2.imwrite(str(output_dir / bbox_rel), bbox_crop)
            row = {
                "sample_idx": sample_idx,
                "frame_id": frame_index,
                "det_index": int(det_idx),
                "det_rank": int(det_rank),
                "confidence": float(confs[det_idx]),
                "bbox": [float(v) for v in bbox.tolist()],
                "bbox_width": width,
                "bbox_height": height,
                "min_keypoint_conf": keypoint_min_conf,
                "yaw_deg": None if yaw_deg is None else float(yaw_deg),
                "is_frontal_candidate": (
                    yaw_deg is not None
                    and yaw_deg <= float(args.frontal_yaw_deg)
                    and min(width, height) >= float(args.anchor_min_box_size)
                    and float(confs[det_idx]) >= float(args.anchor_min_conf)
                ),
                "aligned_face": str(aligned_rel),
                "bbox_crop": str(bbox_rel) if bbox_crop is not None else None,
            }
            rows.append(row)
            crops.append(face)
            if max_samples and len(rows) >= max_samples:
                cap.release()
                print(f"[Sample] reached max_samples={max_samples}")
                print(f"[Sample] source_fps={source_fps:.3f}, total_frames={total_frames}")
                return rows, crops
    cap.release()
    print(f"[Sample] source_fps={source_fps:.3f}, total_frames={total_frames}")
    return rows, crops


def score_quality(row: dict) -> Tuple[float, float, float, int]:
    yaw = row.get("yaw_deg")
    yaw_abs = float(yaw) if yaw is not None else 999.0
    min_side = min(float(row.get("bbox_width") or 0.0), float(row.get("bbox_height") or 0.0))
    conf = float(row.get("confidence") or 0.0)
    return yaw_abs, -min_side, -conf, int(row.get("frame_id") or 0)


def margin_clear(scores: np.ndarray, best_idx: int, margin: float) -> Tuple[float, float, bool]:
    if scores.size <= 0:
        return 0.0, 0.0, False
    best = float(scores[best_idx])
    if scores.size == 1:
        return best, 0.0, True
    second = float(np.partition(scores, -2)[-2])
    return best, second, best >= second + float(margin)


def build_frontal_anchor_groups(
    rows: List[dict],
    features: np.ndarray,
    match_threshold: float,
    match_margin: float,
) -> Tuple[List[dict], Dict[str, List[int]], List[int]]:
    frontal = [i for i, row in enumerate(rows) if row.get("is_frontal_candidate")]
    frontal.sort(key=lambda idx: score_quality(rows[idx]))

    anchors: List[dict] = []
    for idx in frontal:
        if not anchors:
            anchors.append({"face_id": "face1", "sample_index": idx})
            rows[idx]["is_anchor"] = True
            continue
        anchor_indices = [a["sample_index"] for a in anchors]
        scores = features[anchor_indices] @ features[idx]
        best_idx = int(np.argmax(scores))
        best, second, clear = margin_clear(scores, best_idx, match_margin)
        if best >= float(match_threshold) and clear:
            rows[idx]["anchor_duplicate_of"] = anchors[best_idx]["face_id"]
            rows[idx]["anchor_duplicate_score"] = best
            rows[idx]["anchor_duplicate_second_score"] = second
            continue
        face_id = f"face{len(anchors) + 1}"
        anchors.append({"face_id": face_id, "sample_index": idx})
        rows[idx]["is_anchor"] = True

    groups: Dict[str, List[int]] = defaultdict(list)
    unassigned: List[int] = []
    if not anchors:
        for idx, row in enumerate(rows):
            row["assigned_face_id"] = None
            row["unassigned_reason"] = "no_frontal_anchor"
            unassigned.append(idx)
        return anchors, groups, unassigned

    anchor_indices = [a["sample_index"] for a in anchors]
    anchor_features = features[anchor_indices]
    for idx, row in enumerate(rows):
        scores = anchor_features @ features[idx]
        best_anchor_idx = int(np.argmax(scores))
        best, second, clear = margin_clear(scores, best_anchor_idx, match_margin)
        assigned = best >= float(match_threshold) and clear
        face_id = anchors[best_anchor_idx]["face_id"]
        row["best_anchor_face_id"] = face_id
        row["best_anchor_sample_idx"] = rows[anchors[best_anchor_idx]["sample_index"]]["sample_idx"]
        row["best_anchor_score"] = best
        row["second_anchor_score"] = second
        row["anchor_match_clear"] = bool(clear)
        if assigned:
            row["assigned_face_id"] = face_id
            groups[face_id].append(idx)
        else:
            row["assigned_face_id"] = None
            if best < float(match_threshold):
                row["unassigned_reason"] = "below_threshold"
            else:
                row["unassigned_reason"] = "ambiguous_margin"
            unassigned.append(idx)
    return anchors, groups, unassigned


def normalize_vector(feature: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(feature))
    if norm <= 1.0e-12:
        return feature.astype(np.float32)
    return (feature / norm).astype(np.float32)


def board_feature_quality(row: dict, args: argparse.Namespace) -> dict:
    min_side = min(float(row.get("bbox_width") or 0.0), float(row.get("bbox_height") or 0.0))
    min_kp_conf = float(row.get("min_keypoint_conf") or 0.0)
    det_score = float(row.get("confidence") or 0.0)
    yaw = row.get("yaw_deg")
    yaw_ok = yaw is not None and float(yaw) <= float(args.frontal_yaw_deg)
    sample_ok = (
        min_side >= float(args.update_min_box_size)
        and min_kp_conf >= float(args.update_min_keypoint_conf)
        and det_score >= float(args.update_min_score)
    )
    primary_ok = (
        sample_ok
        and yaw_ok
        and min_side >= float(args.primary_min_box_size)
        and min_kp_conf >= float(args.primary_min_keypoint_conf)
        and det_score >= float(args.primary_min_score)
    )
    side_den = max(1.0, float(args.primary_min_box_size))
    side_score = min(1.0, min_side / side_den)
    quality_score = 0.45 * side_score + 0.35 * min_kp_conf + 0.20 * det_score
    return {
        "sample_ok": bool(sample_ok),
        "primary_ok": bool(primary_ok),
        "quality_score": float(quality_score),
        "min_side": float(min_side),
        "min_keypoint_conf": float(min_kp_conf),
        "det_score": float(det_score),
        "yaw_ok": bool(yaw_ok),
    }


def gallery_score(identity: dict, feature: np.ndarray) -> float:
    best = 0.0
    for old in identity.get("features", []):
        if old.shape == feature.shape:
            best = max(best, float(old @ feature))
    return best


def gallery_scores(identities: Sequence[dict], feature: np.ndarray) -> np.ndarray:
    return np.asarray([gallery_score(identity, feature) for identity in identities], dtype=np.float32)


def update_gallery_primary(identity: dict, feature: np.ndarray, quality_score: float, sample_index: int,
                           args: argparse.Namespace) -> None:
    if not identity["features"] or identity["features"][0].shape != feature.shape:
        identity["features"] = [feature.copy()]
        identity["feature_rows"] = [sample_index]
        identity["primary_quality"] = float(quality_score)
        identity["sample_index"] = sample_index
        return
    if identity.get("primary_quality", 0.0) <= 0.0 or quality_score > float(identity["primary_quality"]) + 0.2:
        identity["features"][0] = feature.copy()
        identity["feature_rows"][0] = sample_index
        identity["primary_quality"] = max(float(identity.get("primary_quality", 0.0)), float(quality_score))
        identity["sample_index"] = sample_index
        return
    alpha = max(0.0, min(1.0, float(args.primary_ema_alpha)))
    identity["features"][0] = normalize_vector((1.0 - alpha) * identity["features"][0] + alpha * feature)
    identity["primary_quality"] = max(float(identity.get("primary_quality", 0.0)), float(quality_score))


def trim_gallery(identity: dict, max_samples: int) -> None:
    max_samples = max(1, int(max_samples))
    if len(identity["features"]) <= max_samples:
        return
    if max_samples == 1:
        identity["features"] = identity["features"][:1]
        identity["feature_rows"] = identity["feature_rows"][:1]
        return
    remove_count = len(identity["features"]) - max_samples
    del identity["features"][1:1 + remove_count]
    del identity["feature_rows"][1:1 + remove_count]


def maybe_update_gallery(
    identities: Sequence[dict],
    identity_index: int,
    feature: np.ndarray,
    quality: dict,
    sample_index: int,
    args: argparse.Namespace,
) -> str:
    identity = identities[identity_index]
    best_existing = gallery_score(identity, feature)
    can_update_primary = (
        bool(quality["primary_ok"])
        and best_existing >= float(args.update_similarity)
    )
    can_add_sample = (
        bool(quality["sample_ok"])
        and best_existing >= float(args.pose_sample_similarity)
        and (
            float(args.update_max_similarity) <= float(args.pose_sample_similarity)
            or best_existing <= float(args.update_max_similarity)
        )
        and best_existing < 1.0 - float(args.min_sample_diversity)
    )
    if not can_update_primary and not can_add_sample:
        return ""
    best_other = 0.0
    for idx, other in enumerate(identities):
        if idx == identity_index:
            continue
        best_other = max(best_other, gallery_score(other, feature))
    if best_existing < best_other + float(args.match_margin):
        return ""

    primary_updated = False
    sample_updated = False
    if can_update_primary:
        update_gallery_primary(identity, feature, quality["quality_score"], sample_index, args)
        primary_updated = True
    if can_add_sample:
        identity["features"].append(feature.copy())
        identity["feature_rows"].append(sample_index)
        trim_gallery(identity, int(args.max_samples_per_id))
        sample_updated = True
    if primary_updated and sample_updated:
        return "primary_sample"
    if primary_updated:
        return "primary"
    if sample_updated:
        return "sample"
    return ""


def build_board_gallery_groups(
    rows: List[dict],
    features: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[List[dict], Dict[str, List[int]], List[int]]:
    identities: List[dict] = []
    groups: Dict[str, List[int]] = defaultdict(list)
    pending_candidates: List[dict] = []
    confirm_frames = max(1, int(args.dynamic_enroll_confirm_frames))
    match_threshold = float(args.match_threshold)

    def mark_unassigned(idx: int, reason: str, action: str = "unassigned") -> None:
        rows[idx]["assigned_face_id"] = None
        rows[idx]["unassigned_reason"] = reason
        rows[idx]["gallery_action"] = action

    def confirm_pending_candidate(candidate: dict, current_idx: int) -> None:
        face_id = f"face{len(identities) + 1}"
        row_indices = list(candidate["rows"])
        if not any(rows[ri].get("gallery_primary_ok") for ri in row_indices):
            return
        anchor_idx = max(
            row_indices,
            key=lambda ri: (
                1 if rows[ri].get("gallery_primary_ok") else 0,
                float(rows[ri].get("gallery_quality_score") or 0.0),
            ),
        )
        ordered_rows = [anchor_idx] + [ri for ri in row_indices if ri != anchor_idx]
        identity = {
            "face_id": face_id,
            "sample_index": anchor_idx,
            "features": [features[anchor_idx].copy()],
            "feature_rows": [anchor_idx],
            "primary_quality": float(rows[anchor_idx].get("gallery_quality_score") or 0.0),
        }
        for ri in ordered_rows[1:]:
            if gallery_score(identity, features[ri]) < 1.0 - float(args.min_sample_diversity):
                identity["features"].append(features[ri].copy())
                identity["feature_rows"].append(ri)
        trim_gallery(identity, int(args.max_samples_per_id))
        identities.append(identity)

        for ri in row_indices:
            rows[ri].pop("unassigned_reason", None)
            rows[ri]["assigned_face_id"] = face_id
            rows[ri]["best_anchor_face_id"] = face_id
            rows[ri]["best_anchor_sample_idx"] = rows[anchor_idx]["sample_idx"]
            rows[ri]["best_anchor_score"] = 1.0 if ri == anchor_idx else float(features[anchor_idx] @ features[ri])
            rows[ri]["second_anchor_score"] = float(rows[ri].get("second_anchor_score") or 0.0)
            rows[ri]["anchor_match_clear"] = True
            rows[ri]["gallery_feature_count"] = len(identity["features"])
            rows[ri]["dynamic_enroll_confirm_hits"] = len(row_indices)
            if ri == anchor_idx:
                rows[ri]["is_anchor"] = True
            if ri == current_idx:
                rows[ri]["gallery_action"] = "new_primary_confirm"
            else:
                rows[ri]["gallery_action"] = "new_primary_confirmed_pending"
            groups[face_id].append(ri)

    for idx, row in enumerate(rows):
        feature = features[idx]
        quality = board_feature_quality(row, args)
        row["gallery_sample_ok"] = quality["sample_ok"]
        row["gallery_primary_ok"] = quality["primary_ok"]
        row["gallery_quality_score"] = quality["quality_score"]
        row["gallery_yaw_ok"] = quality["yaw_ok"]

        if identities:
            scores = gallery_scores(identities, feature)
            best_idx = int(np.argmax(scores))
            best, second, clear = margin_clear(scores, best_idx, float(args.match_margin))
            best_face_id = identities[best_idx]["face_id"]
        else:
            scores = np.empty((0,), dtype=np.float32)
            best_idx = -1
            best = 0.0
            second = 0.0
            clear = False
            best_face_id = ""

        row["best_anchor_face_id"] = best_face_id or None
        row["best_anchor_score"] = float(best)
        row["second_anchor_score"] = float(second)
        row["anchor_match_clear"] = bool(clear)

        if identities and best >= float(args.match_threshold) and clear:
            pending_candidates = [
                candidate for candidate in pending_candidates
                if idx not in candidate.get("rows", [])
            ]
            identity = identities[best_idx]
            face_id = identity["face_id"]
            row["assigned_face_id"] = face_id
            row.pop("unassigned_reason", None)
            row["best_anchor_sample_idx"] = rows[identity["sample_index"]]["sample_idx"]
            update_kind = maybe_update_gallery(identities, best_idx, feature, quality, idx, args)
            row["gallery_action"] = "match_update_" + update_kind if update_kind else "match_no_update"
            row["gallery_feature_count"] = len(identity["features"])
            groups[face_id].append(idx)
            continue

        if quality["sample_ok"]:
            best_pending_idx = -1
            best_pending_score = 0.0
            second_pending_score = 0.0
            for pending_idx, candidate in enumerate(pending_candidates):
                pending_scores = np.asarray(
                    [float(old @ feature) for old in candidate.get("features", [])],
                    dtype=np.float32,
                )
                if pending_scores.size <= 0:
                    continue
                min_pending_score = float(np.min(pending_scores))
                max_pending_score = float(np.max(pending_scores))
                if min_pending_score < match_threshold:
                    continue
                if max_pending_score > best_pending_score:
                    second_pending_score = best_pending_score
                    best_pending_score = max_pending_score
                    best_pending_idx = pending_idx
                elif max_pending_score > second_pending_score:
                    second_pending_score = max_pending_score
            pending_clear = (
                best_pending_idx >= 0
                and best_pending_score >= second_pending_score + float(args.match_margin)
            )
            if confirm_frames <= 1:
                candidate = {"rows": [idx], "features": [feature.copy()]}
                confirm_pending_candidate(candidate, idx)
                continue
            if best_pending_idx >= 0 and pending_clear:
                candidate = pending_candidates[best_pending_idx]
                candidate["rows"].append(idx)
                candidate["features"].append(feature.copy())
                row["pending_best_score"] = best_pending_score
                row["pending_second_score"] = second_pending_score
            else:
                candidate = {"rows": [idx], "features": [feature.copy()]}
                pending_candidates.append(candidate)
                row["pending_best_score"] = 1.0
                row["pending_second_score"] = 0.0
            row["dynamic_enroll_confirm_hits"] = len(candidate["rows"])
            if len(candidate["rows"]) >= confirm_frames:
                if any(rows[ri].get("gallery_primary_ok") for ri in candidate["rows"]):
                    confirm_pending_candidate(candidate, idx)
                    pending_candidates[:] = [
                        item for item in pending_candidates if item is not candidate
                    ]
                else:
                    mark_unassigned(
                        idx,
                        "dynamic_enroll_pending_no_primary",
                        "new_primary_pending_no_primary",
                    )
            else:
                mark_unassigned(idx, "dynamic_enroll_pending", "new_primary_pending")
            continue

        if not quality["sample_ok"]:
            mark_unassigned(idx, "quality_not_enough")
        elif not identities:
            mark_unassigned(idx, "no_primary_gallery")
        elif best < float(args.match_threshold):
            mark_unassigned(idx, "below_threshold")
        else:
            mark_unassigned(idx, "ambiguous_margin")

    anchors = []
    for identity in identities:
        anchors.append({
            "face_id": identity["face_id"],
            "sample_index": identity["sample_index"],
            "feature_count": len(identity["features"]),
            "feature_rows": [int(i) for i in identity["feature_rows"]],
            "primary_quality": float(identity.get("primary_quality") or 0.0),
        })
    unassigned = [idx for idx, row in enumerate(rows) if row.get("assigned_face_id") is None]
    return anchors, groups, unassigned


def write_feature_files(rows: List[dict], features: np.ndarray, output_dir: Path) -> None:
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    for row, feature in zip(rows, features):
        rel = Path("features") / f"{int(row['sample_idx']):06d}_f{int(row['frame_id']):06d}.npy"
        np.save(output_dir / rel, feature.astype(np.float32))
        row["feature"] = str(rel)


def write_metadata(rows: List[dict], output_dir: Path) -> None:
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_group(rows: List[dict], indices: List[int]) -> Tuple[float, float]:
    scores = [
        float(rows[i].get("best_anchor_score") or 0.0)
        for i in indices
        if rows[i].get("best_anchor_score") is not None
    ]
    if not scores:
        return 0.0, 0.0
    return float(np.mean(scores)), float(np.min(scores))


def rel_img(row: dict, key: str = "aligned_face") -> str:
    value = row.get(key)
    return "" if not value else html.escape(str(value))


def sample_card(row: dict, is_anchor: bool = False) -> str:
    badge = "<span class='anchor-badge'>anchor</span>" if is_anchor else ""
    face_id = row.get("assigned_face_id") or "unassigned"
    score = row.get("best_anchor_score")
    second = row.get("second_anchor_score")
    score_text = "-" if score is None else f"{float(score):.3f}"
    second_text = "-" if second is None else f"{float(second):.3f}"
    yaw = row.get("yaw_deg")
    yaw_text = "-" if yaw is None else f"{float(yaw):.1f}"
    conf = float(row.get("confidence") or 0.0)
    box = f"{int(row.get('bbox_width') or 0)}x{int(row.get('bbox_height') or 0)}"
    action = row.get("gallery_action")
    action_line = f"<div class='line'>action={html.escape(str(action))}</div>" if action else ""
    return (
        "<div class='sample'>"
        f"<img src='{rel_img(row)}' alt='sample {row.get('sample_idx')}'>"
        f"<div class='line'>#{row.get('sample_idx')} f{row.get('frame_id')} {badge}</div>"
        f"<div class='line'>face={html.escape(str(face_id))}</div>"
        f"<div class='line'>sim={score_text} second={second_text}</div>"
        f"{action_line}"
        f"<div class='line'>yaw={yaw_text} conf={conf:.2f}</div>"
        f"<div class='line'>box={box}</div>"
        "</div>"
    )


def write_html(
    output_dir: Path,
    html_name: str,
    rows: List[dict],
    anchors: List[dict],
    groups: Dict[str, List[int]],
    unassigned: List[int],
    args: argparse.Namespace,
) -> Path:
    title = "Board Gallery Face Clusters" if args.cluster_mode == "board-gallery" else "Frontal Anchor Face Clusters"
    group_items = []
    for anchor in anchors:
        face_id = anchor["face_id"]
        indices = groups.get(face_id, [])
        anchor_idx = anchor["sample_index"]
        if anchor_idx not in indices:
            indices = [anchor_idx] + indices
        indices = sorted(
            set(indices),
            key=lambda i: (
                0 if rows[i].get("is_anchor") else 1,
                -float(rows[i].get("best_anchor_score") or 0.0),
                int(rows[i].get("frame_id") or 0),
            ),
        )
        group_items.append((face_id, anchor_idx, indices))
    group_items.sort(key=lambda x: len(x[2]), reverse=True)

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;background:#f6f7f9;color:#111}",
        "h1{font-size:24px;margin:0 0 12px}",
        ".meta{font-size:13px;line-height:1.6;margin:0 0 18px;color:#333}",
        ".group{background:#fff;border:1px solid #ddd;margin:0 0 18px;padding:14px}",
        ".head{display:flex;gap:14px;align-items:flex-start;margin-bottom:12px}",
        ".head img{width:112px;height:112px;object-fit:cover;border:1px solid #bbb}",
        ".grid{display:flex;flex-wrap:wrap;gap:10px}",
        ".sample{width:136px;font-size:12px;line-height:1.35;background:#fafafa;border:1px solid #e1e1e1;padding:6px}",
        ".sample img{width:112px;height:112px;object-fit:cover;display:block;margin:0 auto 5px;border:1px solid #ccc}",
        ".line{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}",
        ".anchor-badge{background:#111;color:#fff;border-radius:3px;padding:1px 4px;font-size:11px}",
        "</style>",
        f"<h1>{html.escape(title)}</h1>",
        "<div class='meta'>",
        f"video: {html.escape(args.video_path)}<br>",
        f"adaface: {html.escape(args.adaface_checkpoint)} ({html.escape(args.adaface_model_name)})<br>",
        f"cluster_mode: {html.escape(args.cluster_mode)}<br>",
        f"align: five-point scale={float(args.five_point_scale):.2f}<br>",
        f"frontal_yaw_deg <= {float(args.frontal_yaw_deg):.1f}, "
        f"match_threshold >= {float(args.match_threshold):.2f}, "
        f"match_margin >= {float(args.match_margin):.2f}<br>",
        f"dynamic_enroll_confirm_frames = {int(args.dynamic_enroll_confirm_frames)}<br>",
        f"samples: {len(rows)}, anchors: {len(anchors)}, unassigned: {len(unassigned)}",
        "</div>",
    ]

    max_images = max(1, int(args.max_images_per_group))
    for face_id, anchor_idx, indices in group_items:
        anchor_row = rows[anchor_idx]
        anchor_meta = next((item for item in anchors if item["face_id"] == face_id), {})
        feature_count = int(anchor_meta.get("feature_count") or 1)
        mean_sim, min_sim = summarize_group(rows, indices)
        parts.append("<section class='group'>")
        parts.append("<div class='head'>")
        parts.append(f"<img src='{rel_img(anchor_row)}' alt='{html.escape(face_id)} anchor'>")
        parts.append(
            "<div>"
            f"<h2>{html.escape(face_id)}  n={len(indices)}</h2>"
            f"anchor sample=#{anchor_row.get('sample_idx')} frame={anchor_row.get('frame_id')}<br>"
            f"anchor yaw={float(anchor_row.get('yaw_deg') or 0.0):.1f}, "
            f"conf={float(anchor_row.get('confidence') or 0.0):.2f}, "
            f"box={int(anchor_row.get('bbox_width') or 0)}x{int(anchor_row.get('bbox_height') or 0)}<br>"
            f"gallery_features={feature_count}<br>"
            f"assigned sim mean={mean_sim:.3f}, min={min_sim:.3f}"
            "</div>"
        )
        parts.append("</div><div class='grid'>")
        for idx in indices[:max_images]:
            parts.append(sample_card(rows[idx], is_anchor=(idx == anchor_idx)))
        parts.append("</div></section>")

    if unassigned:
        parts.append("<section class='group'>")
        parts.append(f"<h2>Unassigned / No Gallery Match  n={len(unassigned)}</h2>")
        parts.append("<div class='grid'>")
        ordered = sorted(
            unassigned,
            key=lambda i: (
                rows[i].get("unassigned_reason") or "",
                -float(rows[i].get("best_anchor_score") or 0.0),
            ),
        )
        for idx in ordered[:max_images]:
            parts.append(sample_card(rows[idx]))
        parts.append("</div></section>")

    path = output_dir / html_name
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def write_analysis(
    output_dir: Path,
    rows: List[dict],
    anchors: List[dict],
    groups: Dict[str, List[int]],
    unassigned: List[int],
    args: argparse.Namespace,
) -> Path:
    title = "Board-gallery face clustering" if args.cluster_mode == "board-gallery" else "Frontal-anchor face clustering"
    lines = [
        title,
        "",
        f"video: {args.video_path}",
        f"adaface_checkpoint: {args.adaface_checkpoint}",
        f"adaface_model_name: {args.adaface_model_name}",
        f"cluster_mode: {args.cluster_mode}",
        f"align_mode: five-point",
        f"five_point_scale: {float(args.five_point_scale):.2f}",
        f"samples: {len(rows)}",
        f"anchors: {len(anchors)}",
        f"unassigned: {len(unassigned)}",
        "",
        "thresholds:",
        f"  conf >= {float(args.conf):.3f}",
        f"  min_box_size >= {float(args.min_box_size):.1f}",
        f"  anchor_min_box_size >= {float(args.anchor_min_box_size):.1f}",
        f"  anchor_min_conf >= {float(args.anchor_min_conf):.3f}",
        f"  frontal_yaw_deg <= {float(args.frontal_yaw_deg):.1f}",
        f"  match_threshold >= {float(args.match_threshold):.3f}",
        f"  match_margin >= {float(args.match_margin):.3f}",
    ]
    if args.cluster_mode == "board-gallery":
        lines.extend([
            f"  update_similarity >= {float(args.update_similarity):.3f}",
            f"  pose_sample_similarity >= {float(args.pose_sample_similarity):.3f}",
            f"  update_max_similarity <= {float(args.update_max_similarity):.3f}",
            f"  min_sample_diversity >= {float(args.min_sample_diversity):.3f}",
            f"  update_min_box_size >= {float(args.update_min_box_size):.1f}",
            f"  update_min_keypoint_conf >= {float(args.update_min_keypoint_conf):.3f}",
            f"  update_min_score >= {float(args.update_min_score):.3f}",
            f"  primary_min_box_size >= {float(args.primary_min_box_size):.1f}",
            f"  primary_min_keypoint_conf >= {float(args.primary_min_keypoint_conf):.3f}",
            f"  primary_min_score >= {float(args.primary_min_score):.3f}",
            f"  primary_ema_alpha = {float(args.primary_ema_alpha):.3f}",
            f"  max_samples_per_id = {int(args.max_samples_per_id)}",
            f"  dynamic_enroll_confirm_frames = {int(args.dynamic_enroll_confirm_frames)}",
        ])
    lines.extend(["", "groups:"])
    for anchor in anchors:
        face_id = anchor["face_id"]
        idxs = groups.get(face_id, [])
        anchor_idx = anchor["sample_index"]
        if anchor_idx not in idxs:
            idxs = [anchor_idx] + idxs
        mean_sim, min_sim = summarize_group(rows, idxs)
        row = rows[anchor_idx]
        feature_count = int(anchor.get("feature_count") or 1)
        lines.append(
            f"  {face_id}: n={len(set(idxs))}, anchor_sample={row.get('sample_idx')}, "
            f"frame={row.get('frame_id')}, yaw={float(row.get('yaw_deg') or 0.0):.1f}, "
            f"conf={float(row.get('confidence') or 0.0):.3f}, "
            f"gallery_features={feature_count}, "
            f"sim_mean={mean_sim:.3f}, sim_min={min_sim:.3f}"
        )
    if unassigned:
        reason_counts: Dict[str, int] = defaultdict(int)
        for idx in unassigned:
            reason_counts[str(rows[idx].get("unassigned_reason") or "unknown")] += 1
        reason_text = ", ".join(f"{k}:{v}" for k, v in sorted(reason_counts.items()))
        lines.extend(["", f"unassigned_reasons: {reason_text}"])
    path = output_dir / "cluster_analysis.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    output_dir = unique_output_dir(repo_path(args.output_dir))
    print(f"[Output] {output_dir}")

    samples, crops = extract_samples(args, output_dir)
    if not samples:
        raise RuntimeError("no usable face samples extracted")

    device = choose_torch_device(args.device)
    model = load_adaface(repo_path(args.adaface_checkpoint), args.adaface_model_name, device)
    features = extract_features(model, crops, device, args.batch_size)
    write_feature_files(samples, features, output_dir)

    if args.cluster_mode == "board-gallery":
        anchors, groups, unassigned = build_board_gallery_groups(samples, features, args)
    else:
        anchors, groups, unassigned = build_frontal_anchor_groups(
            samples,
            features,
            match_threshold=float(args.match_threshold),
            match_margin=float(args.match_margin),
        )
    write_metadata(samples, output_dir)
    analysis_path = write_analysis(output_dir, samples, anchors, groups, unassigned, args)
    html_path = write_html(output_dir, args.html_name, samples, anchors, groups, unassigned, args)

    print(f"[Done] samples={len(samples)} anchors={len(anchors)} unassigned={len(unassigned)}")
    print(f"[Done] analysis: {analysis_path}")
    print(f"[Done] html: {html_path}")


if __name__ == "__main__":
    main()
