"""
GPU WebUI entry with face recognition kept separate from TrackID.

This file intentionally leaves main_GPU_webui.py untouched. It reuses the
existing WebUI pipeline, enables the existing FaceRecManager path by default,
and patches only presentation/JSON behavior for this entry:
  - display_id is shown as ID:<display_id> when present
  - face recognition is shown as an additional Face:<name> label
  - JSON keeps track_id separately and adds a separate "face_recognition" object
"""
import importlib.util
import importlib.machinery
import os
import sys


_HERE = os.path.dirname(os.path.abspath(__file__))


def _argv_has_option(option: str) -> bool:
    prefix = option + "="
    return any(arg == option or arg.startswith(prefix) for arg in sys.argv[1:])


def _set_default_arg(args, attr: str, option: str, value) -> None:
    if not _argv_has_option(option):
        setattr(args, attr, value)


def _preload_face_rec_manager() -> None:
    """
    Load face_rec.face_rec_manager before webui.processor is imported.

    Some local checkouts only contain face_rec_manager.pyc. Normal imports do
    not look inside __pycache__ for sourceless modules, so this entry provides a
    compatibility load path without changing the original project files.
    """
    if 'face_rec.face_rec_manager' in sys.modules:
        return

    src_path = os.path.join(_HERE, 'face_rec', 'face_rec_manager.py')
    pyc_path = os.path.join(
        _HERE, 'face_rec', '__pycache__', 'face_rec_manager.cpython-310.pyc'
    )
    use_source = os.path.exists(src_path)
    module_path = src_path if use_source else pyc_path
    if not os.path.exists(module_path):
        print("[FaceRC] 未找到 face_rec_manager.py/.pyc，人脸识别会按原逻辑跳过")
        return

    try:
        if use_source:
            spec = importlib.util.spec_from_file_location(
                'face_rec.face_rec_manager', module_path
            )
        else:
            loader = importlib.machinery.SourcelessFileLoader(
                'face_rec.face_rec_manager', module_path
            )
            spec = importlib.util.spec_from_loader(
                'face_rec.face_rec_manager',
                loader,
                origin=src_path,
            )
        if spec is None or spec.loader is None:
            print(f"[FaceRC] 无法加载人脸识别模块: {module_path}")
            return
        module = importlib.util.module_from_spec(spec)
        module.__file__ = src_path
        sys.modules['face_rec.face_rec_manager'] = module
        spec.loader.exec_module(module)
        print(f"[FaceRC] 已加载人脸识别模块: {module_path}")
    except Exception as exc:
        sys.modules.pop('face_rec.face_rec_manager', None)
        print(f"[FaceRC] 人脸识别模块预加载失败: {type(exc).__name__}: {exc}")


def _patch_parse_args_default_face_rec() -> None:
    """Make this entry enable --use-face-rec by default, while honoring --no-use-face-rec."""
    import config

    original_parse_args = config.parse_args

    def parse_args_face_rec_default():
        args = original_parse_args()
        if '--no-use-face-rec' not in sys.argv:
            args.use_face_rec = True
        args.face_rec_dynamic_library = True
        _set_default_arg(args, 'face_rec_threshold', '--face-rec-threshold', 0.55)
        _set_default_arg(args, 'face_rec_cooldown', '--face-rec-cooldown', 10)
        _set_default_arg(args, 'face_rec_align_mode', '--face-rec-align-mode', 'five-point')
        _set_default_arg(args, 'face_rec_five_point_scale', '--face-rec-five-point-scale', 1.20)
        _set_default_arg(args, 'dynamic_face_max_samples', '--dynamic-face-max-samples', 12)
        _set_default_arg(args, 'dynamic_face_update_similarity', '--dynamic-face-update-similarity', 0.65)
        _set_default_arg(args, 'dynamic_face_pose_sample_similarity', '--dynamic-face-pose-sample-similarity', 0.55)
        _set_default_arg(args, 'dynamic_face_pose_library_similarity', '--dynamic-face-pose-library-similarity', 0.55)
        _set_default_arg(args, 'dynamic_face_pose_library_max_similarity', '--dynamic-face-pose-library-max-similarity', 0.90)
        _set_default_arg(args, 'dynamic_face_pose_sample_interval', '--dynamic-face-pose-sample-interval', 30)
        _set_default_arg(args, 'dynamic_face_pending_seed_samples', '--dynamic-face-pending-seed-samples', 2)
        _set_default_arg(args, 'dynamic_face_locked_sample_similarity', '--dynamic-face-locked-sample-similarity', 0.35)
        _set_default_arg(args, 'dynamic_face_primary_max_yaw', '--dynamic-face-primary-max-yaw', 10.0)
        _set_default_arg(args, 'dynamic_face_primary_ema_alpha', '--dynamic-face-primary-ema-alpha', 0.1)
        _set_default_arg(args, 'dynamic_face_enroll_confirm_frames', '--dynamic-face-enroll-confirm-frames', 3)
        _set_default_arg(args, 'dynamic_face_binding_ttl', '--dynamic-face-binding-ttl', 900)
        _set_default_arg(args, 'dynamic_face_min_height', '--dynamic-face-min-height', 100)
        _set_default_arg(args, 'dynamic_face_min_keypoint_conf', '--dynamic-face-min-keypoint-conf', 0.6)
        _set_default_arg(args, 'dynamic_face_update_min_height', '--dynamic-face-update-min-height', 100)
        _set_default_arg(args, 'dynamic_face_update_min_keypoint_conf', '--dynamic-face-update-min-keypoint-conf', 0.7)
        _set_default_arg(args, 'dynamic_face_update_min_score', '--dynamic-face-update-min-score', 0.5)
        _set_default_arg(args, 'dynamic_face_primary_min_height', '--dynamic-face-primary-min-height', 110)
        _set_default_arg(args, 'dynamic_face_primary_min_keypoint_conf', '--dynamic-face-primary-min-keypoint-conf', 0.8)
        _set_default_arg(args, 'dynamic_face_primary_min_score', '--dynamic-face-primary-min-score', 0.6)
        _set_default_arg(args, 'dynamic_face_match_margin', '--dynamic-face-match-margin', 0.08)
        _set_default_arg(args, 'dynamic_face_alias_threshold', '--dynamic-face-alias-threshold', 0.55)
        if (not _argv_has_option('--face-rec-observer')
                and not _argv_has_option('--no-face-rec-observer')):
            args.face_rec_observer = True
        _set_default_arg(args, 'face_rec_observer_pending_ttl', '--face-rec-observer-pending-ttl', 90)
        return args

    config.parse_args = parse_args_face_rec_default


def _patch_processor_face_rec_manager() -> None:
    import main_GPU_webui as base
    import webui.processor as processor

    BaseProcessor = processor.FisheyePanoramaYOLOPose

    class DynamicFaceRecProcessor(BaseProcessor):
        def initialize(self):
            original_face_rec_cls = processor.FaceRecManager

            class DynamicFaceRecManager(original_face_rec_cls):
                def __init__(self, *args, **kwargs):
                    kwargs['dynamic_library'] = True
                    kwargs['dynamic_library_dir'] = getattr(
                        self_args, 'dynamic_face_library_dir', None
                    )
                    kwargs['dynamic_match_interval'] = getattr(
                        self_args, 'dynamic_face_match_interval', None
                    )
                    kwargs['dynamic_max_samples_per_id'] = getattr(
                        self_args, 'dynamic_face_max_samples', 5
                    )
                    kwargs['dynamic_update_similarity'] = getattr(
                        self_args, 'dynamic_face_update_similarity', None
                    )
                    kwargs['dynamic_pose_sample_similarity'] = getattr(
                        self_args, 'dynamic_face_pose_sample_similarity', None
                    )
                    kwargs['dynamic_pose_library_similarity'] = getattr(
                        self_args, 'dynamic_face_pose_library_similarity', None
                    )
                    kwargs['dynamic_pose_library_max_similarity'] = getattr(
                        self_args, 'dynamic_face_pose_library_max_similarity', 0.90
                    )
                    kwargs['dynamic_pose_sample_interval'] = getattr(
                        self_args, 'dynamic_face_pose_sample_interval', 30
                    )
                    kwargs['dynamic_pending_seed_samples'] = getattr(
                        self_args, 'dynamic_face_pending_seed_samples', 2
                    )
                    kwargs['dynamic_locked_sample_similarity'] = getattr(
                        self_args, 'dynamic_face_locked_sample_similarity', 0.35
                    )
                    kwargs['dynamic_min_sample_diversity'] = getattr(
                        self_args, 'dynamic_face_min_sample_diversity', 0.015
                    )
                    kwargs['dynamic_primary_max_yaw_deg'] = getattr(
                        self_args, 'dynamic_face_primary_max_yaw', 20.0
                    )
                    kwargs['dynamic_primary_ema_alpha'] = getattr(
                        self_args, 'dynamic_face_primary_ema_alpha', 0.1
                    )
                    kwargs['dynamic_enroll_confirm_frames'] = getattr(
                        self_args, 'dynamic_face_enroll_confirm_frames', 3
                    )
                    kwargs['dynamic_binding_ttl_frames'] = getattr(
                        self_args, 'dynamic_face_binding_ttl', 900
                    )
                    kwargs['dynamic_min_keypoint_conf'] = getattr(
                        self_args, 'dynamic_face_min_keypoint_conf', 0.6
                    )
                    kwargs['dynamic_update_min_face_height'] = getattr(
                        self_args, 'dynamic_face_update_min_height', None
                    )
                    kwargs['dynamic_update_min_keypoint_conf'] = getattr(
                        self_args, 'dynamic_face_update_min_keypoint_conf', 0.7
                    )
                    kwargs['dynamic_update_min_score'] = getattr(
                        self_args, 'dynamic_face_update_min_score', 0.5
                    )
                    kwargs['dynamic_primary_min_face_height'] = getattr(
                        self_args, 'dynamic_face_primary_min_height', None
                    )
                    kwargs['dynamic_primary_min_keypoint_conf'] = getattr(
                        self_args, 'dynamic_face_primary_min_keypoint_conf', 0.8
                    )
                    kwargs['dynamic_primary_min_score'] = getattr(
                        self_args, 'dynamic_face_primary_min_score', 0.6
                    )
                    kwargs['dynamic_supplement_fallback_threshold'] = getattr(
                        self_args, 'dynamic_face_supplement_fallback_threshold', None
                    )
                    kwargs['dynamic_match_margin'] = getattr(
                        self_args, 'dynamic_face_match_margin', 0.08
                    )
                    kwargs['dynamic_ambiguous_keep_bound'] = getattr(
                        self_args, 'dynamic_face_ambiguous_keep_bound', True
                    )
                    kwargs['dynamic_ambiguous_keep_min_score'] = getattr(
                        self_args, 'dynamic_face_ambiguous_keep_min_score', None
                    )
                    kwargs['dynamic_global_assignment'] = getattr(
                        self_args, 'dynamic_face_global_assignment', True
                    )
                    kwargs['dynamic_auto_alias'] = getattr(
                        self_args, 'dynamic_face_auto_alias', True
                    )
                    kwargs['dynamic_alias_threshold'] = getattr(
                        self_args, 'dynamic_face_alias_threshold', None
                    )
                    kwargs['dynamic_alias_min_samples'] = getattr(
                        self_args, 'dynamic_face_alias_min_samples', 2
                    )
                    kwargs['dynamic_alias_min_hits'] = getattr(
                        self_args, 'dynamic_face_alias_min_hits', 2
                    )
                    kwargs['dynamic_alias_margin'] = getattr(
                        self_args, 'dynamic_face_alias_margin', 0.03
                    )
                    kwargs['dynamic_alias_probe_samples'] = getattr(
                        self_args, 'dynamic_face_alias_probe_samples', 30
                    )
                    kwargs['dynamic_switch_margin'] = getattr(
                        self_args, 'dynamic_face_switch_margin', 0.15
                    )
                    kwargs['dynamic_min_face_height'] = getattr(
                        self_args, 'dynamic_face_min_height', 64
                    )
                    kwargs['dynamic_enroll_max_yaw_deg'] = getattr(
                        self_args, 'dynamic_face_enroll_max_yaw', 30.0
                    )
                    kwargs['dynamic_binding_mismatch_threshold'] = getattr(
                        self_args, 'dynamic_face_binding_mismatch_threshold', None
                    )
                    kwargs['dynamic_lock_to_track'] = getattr(
                        self_args, 'dynamic_face_lock_to_track', True
                    )
                    kwargs['align_mode'] = getattr(
                        self_args, 'face_rec_align_mode', 'eye'
                    )
                    kwargs['five_point_scale'] = getattr(
                        self_args, 'face_rec_five_point_scale', 1.20
                    )
                    kwargs['debug_dump_dir'] = getattr(
                        self_args, 'face_debug_dump_dir', None
                    )
                    kwargs['debug_dump_every'] = getattr(
                        self_args, 'face_debug_dump_every', 1
                    )
                    kwargs['debug_dump_max'] = getattr(
                        self_args, 'face_debug_dump_max', 0
                    )
                    kwargs['observer_enabled'] = getattr(
                        self_args, 'face_rec_observer', False
                    )
                    kwargs['observer_max_tracks'] = getattr(
                        self_args, 'face_rec_observer_max_tracks', 4
                    )
                    kwargs['observer_jpeg_size'] = getattr(
                        self_args, 'face_rec_observer_jpeg_size', 96
                    )
                    kwargs['observer_pending_ttl_frames'] = getattr(
                        self_args, 'face_rec_observer_pending_ttl', 90
                    )
                    super().__init__(*args, **kwargs)

            self_args = self.args
            processor.FaceRecManager = DynamicFaceRecManager
            try:
                return super().initialize()
            finally:
                processor.FaceRecManager = original_face_rec_cls

    processor.FisheyePanoramaYOLOPose = DynamicFaceRecProcessor
    base.FisheyePanoramaYOLOPose = DynamicFaceRecProcessor
    print("[FaceRC] 人脸库策略: TrackID绑定保留 + 三帧确认动态建库 + 高质量样本更新")


def _draw_detections_keep_track_id(
    image,
    detections,
    tracker=None,
    show_id=True,
    show_conf=True,
    face_name_map=None,
    use_kpt_bbox=False,
    kpt_bbox_conf=0.3,
    kpt_bbox_padding=0.15,
    kpt_bbox_upper_only=True,
    kpt_bbox_padding_v=None,
    draw_kpt=False,
):
    """
    Draw labels for this entry without replacing TrackID with face name.

    Existing utils.visualizer.draw_bounding_boxes treats face_name_map as a
    replacement for ID. Here the two systems stay visually separate.
    """
    import cv2
    from utils.visualizer import compute_stable_bbox_from_keypoints, draw_keypoints

    annotated = image.copy()
    face_name_map = face_name_map or {}

    for det in detections:
        bbox = det['bbox']
        if use_kpt_bbox:
            bbox = compute_stable_bbox_from_keypoints(
                det.get('keypoints', []),
                conf_thresh=kpt_bbox_conf,
                padding=kpt_bbox_padding,
                fallback_bbox=bbox,
                upper_body_only=kpt_bbox_upper_only,
                padding_v=kpt_bbox_padding_v,
            )

        x1, y1, x2, y2 = map(int, bbox)
        confidence = det.get('confidence', 0.0)
        track_id = det.get('track_id', -1)
        is_lost = det.get('_is_lost', False)

        if is_lost:
            color = (255, 191, 0)
        elif confidence > 0.8:
            color = (0, 255, 0)
        elif confidence > 0.6:
            color = (0, 200, 255)
        else:
            color = (0, 165, 255)

        thickness = 1 if is_lost else 2
        if det.get('_sector_rep'):
            color = (0, 0, 255)
            thickness = 3

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        lines = []
        first_line = []
        if show_id and track_id != -1:
            label_id = det.get('display_id') or track_id
            first_line.append(f"ID:{label_id}")
        if show_conf:
            first_line.append(f"{confidence:.2f}")
        if first_line:
            lines.append(" ".join(first_line))

        face_name = face_name_map.get(track_id)
        if face_name:
            lines.append(f"Face:{face_name}")
        if det.get('talking'):
            lines.append("Speaking")

        if not lines:
            continue

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        font_thickness = 2
        sizes = [
            cv2.getTextSize(line, font, scale, font_thickness)[0]
            for line in lines
        ]
        text_w = max(w for w, _h in sizes)
        line_h = max(h for _w, h in sizes) + 6
        box_h = line_h * len(lines) + 4

        label_y2 = max(box_h, y1)
        label_y1 = label_y2 - box_h
        label_x2 = min(annotated.shape[1] - 1, x1 + text_w + 8)
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)

        for idx, line in enumerate(lines):
            ty = label_y1 + 4 + (idx + 1) * line_h - 4
            cv2.putText(
                annotated,
                line,
                (x1 + 4, ty),
                font,
                scale,
                (0, 0, 0),
                font_thickness,
                cv2.LINE_AA,
            )

    if draw_kpt:
        annotated = draw_keypoints(annotated, detections)
    return annotated


def _patch_processor_drawing() -> None:
    import webui.processor as processor

    processor.draw_detections = _draw_detections_keep_track_id
    print("[FaceRC] 显示策略: TrackID 与 Face 识别结果分开显示")


def _patch_inference_json() -> None:
    import json
    import main_GPU_webui as base
    import webui.state as ws
    from utils.sector import aggregate_sectors

    original_build_json = base._build_inference_json

    def build_json_with_face_rec(tracked, angle_info):
        raw = original_build_json(tracked, angle_info)
        try:
            payload = json.loads(raw.decode())
            face_name_map = getattr(ws.processor, '_face_name_map', {}) or {}
            face_rec = getattr(ws.processor, 'face_rec', None)
            if face_rec is not None and hasattr(face_rec, 'get_debug_snapshot'):
                payload['face_rec_observer'] = face_rec.get_debug_snapshot()
            targets = payload.get('targets')
            if isinstance(targets, dict):
                for tid_str, target in targets.items():
                    try:
                        tid = int(target.get('track_id', tid_str))
                    except (TypeError, ValueError):
                        tid = int(target.get('id', -1))
                    name = face_name_map.get(tid)
                    target['face_recognition'] = {
                        'name': name,
                        'face_id': name,
                        'matched': bool(name),
                    }
            sectors = payload.get('sectors')
            if isinstance(sectors, dict):
                _sectors, rep_indices = aggregate_sectors(
                    tracked or [], angle_info, payload.get('num_sectors', len(sectors) or 8)
                )
                for idx in rep_indices:
                    if idx >= len(tracked or []):
                        continue
                    det = tracked[idx]
                    tid = int(det.get('track_id', -1))
                    name = face_name_map.get(tid)
                    persons = (angle_info or {}).get('persons', [])
                    angle = persons[idx] if idx < len(persons) else None
                    if angle is None:
                        continue
                    sector_size = 360.0 / max(1, int(payload.get('num_sectors', len(sectors) or 8)))
                    sector_id = str(int(float(angle['azimuth_deg']) // sector_size)
                                    % max(1, int(payload.get('num_sectors', len(sectors) or 8))))
                    if sector_id in sectors:
                        sectors[sector_id]['track_id'] = tid
                        sectors[sector_id]['face_recognition'] = {
                            'name': name,
                            'face_id': name,
                            'matched': bool(name),
                        }
            return json.dumps(payload, ensure_ascii=False).encode()
        except Exception:
            return raw

    base._build_inference_json = build_json_with_face_rec
    print("[FaceRC] JSON策略: id 保持 TrackID，face_recognition 独立输出")


def main() -> None:
    _preload_face_rec_manager()
    _patch_parse_args_default_face_rec()

    import main_GPU_webui as base

    _patch_processor_drawing()
    _patch_processor_face_rec_manager()
    _patch_inference_json()
    base.main()


if __name__ == "__main__":
    main()
