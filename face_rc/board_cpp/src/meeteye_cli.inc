static void print_usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " --image frame.jpg | --camera-device /dev/video0 [options]\n"
      << "\n"
      << "Options:\n"
      << "  --image PATH           JPEG/MJPEG frame path; may be repeated\n"
      << "  --image-list PATH      text file with one JPEG/MJPEG path per line\n"
      << "  --camera-device PATH   V4L2 MJPEG camera device, e.g. /dev/video0\n"
      << "  --camera-width N       camera capture width, default: 1920\n"
      << "  --camera-height N      camera capture height, default: 1080\n"
      << "  --camera-fps N         camera FPS request, default: 30\n"
      << "  --input-fps VALUE      source FPS for seconds-to-frames windows, default: camera-fps\n"
      << "  --effective-fps VALUE  alias of --input-fps\n"
      << "  --camera-buffers N     mmap capture buffers, default: 4\n"
      << "  --camera-timeout-ms N  camera select timeout, default: 3000\n"
      << "  --max-frames N         frames to read in camera mode, default: unlimited\n"
      << "  --output-jsonl PATH    write Python-compatible targets JSONL\n"
      << "  --debug-jsonl PATH     write full debug detection JSONL\n"
      << "  --no-output-jsonl      do not create default board_output JSONL\n"
      << "  --ws-host HOST         JSON WebSocket bind host, default: 0.0.0.0\n"
      << "  --ws-port PORT         JSON WebSocket port, default: 8001\n"
      << "  --no-ws-json          disable /ws/inference JSON WebSocket server\n"
      << "  --webui               enable C++ WebUI with annotated frame stream\n"
      << "  --webui-host HOST     WebUI bind host, default: 0.0.0.0\n"
	      << "  --webui-port PORT     WebUI HTTP port, default: 8080\n"
	      << "  --webui-jpeg-quality N  annotated JPEG quality, default: 80\n"
	      << "  --webui-frame-scale VALUE  downscale WebUI JPEG frame before sending, default: 1.0\n"
	      << "  --webui-slot-fps VALUE  display slot crop WebSocket FPS limit, default: 10\n"
	      << "  --webui-show-hidden-targets  draw targets without display_id on WebUI, default: on\n"
      << "  --no-webui-show-hidden-targets  hide targets without display_id on WebUI\n"
      << "  --sector-output       emit Python-compatible sector aggregate JSON\n"
      << "  --num-sectors N       sector count for --sector-output, default: 8\n"
      << "  --target-output-limit N  max targets in non-sector JSON, default: 8; 0 means unlimited\n"
      << "  --target-output-compact-ids  write non-sector target keys as 1..N, default\n"
      << "  --no-target-output-compact-ids  keep non-sector target keys as display/public ID\n"
      << "  --display-id-reuse    emit compact reusable display_id while preserving public track_id, default\n"
      << "  --no-display-id-reuse disable reusable display_id output\n"
      << "  --display-id-reuse-spatial     prefer recently lost nearby display_id, default\n"
      << "  --no-display-id-reuse-spatial  reuse the smallest free display_id directly\n"
      << "  --display-id-reuse-max-seconds VALUE  spatial display_id reuse window, default: 30\n"
      << "  --display-id-reuse-fallback-min-seconds VALUE  allow fallback display_id reuse after seconds, default: 5\n"
      << "  --display-id-public-binding-ttl-seconds VALUE  exclusively keep public_id -> display_id binding, default: 2\n"
      << "  --display-id-reuse-max-frames N  legacy frame override for display_id reuse window\n"
      << "  --display-id-reuse-fallback-min-frames N  legacy frame override for display_id fallback reuse\n"
      << "  --display-id-public-binding-ttl N  legacy frame override for display_id exclusive binding\n"
      << "  --display-id-max-count N  max reusable display IDs, default: 8; 0 means unlimited\n"
      << "  --display-id-replace-confirm-seconds VALUE  consecutive seconds before area replacement, default: 1\n"
      << "  --display-id-replace-confirm-frames N  legacy frame override for area replacement confirmation\n"
      << "  --display-id-replace-area-ratio VALUE  replace smallest displayed slot when new area is larger by ratio, default: 1.50\n"
      << "  --display-id-min-box-size VALUE  min bbox short side allowed to occupy display slot, default: 40\n"
      << "  --display-id-reuse-center-thresh VALUE  spatial display_id center gate, default: 2.0\n"
      << "  --display-id-reuse-size-ratio VALUE  spatial display_id size gate, default: 0.4\n"
      << "  --display-id-reuse-ambiguity-margin VALUE  spatial display_id unique-best gate, default: 0.25\n"
      << "  --face-rec            enable AdaFace RKNN face recognition, default: off\n"
      << "  --no-face-rec         disable AdaFace RKNN face recognition\n"
      << "  --face-rec-model PATH AdaFace RKNN model path, default: models/adaface_ir18_112.rknn\n"
      << "  --face-rec-model-preset NAME  AdaFace model preset: ir18_webface4m or ir50_webface4m\n"
      << "  --face-library-dir DIR  static .npy feature library, default: face_library\n"
      << "  --known-face-dir DIR  known face JPEG photo dir, default: face_photos\n"
      << "  --known-face-feature-library  persist/update known face features, default: off\n"
      << "  --no-known-face-feature-library  match known photos only, no feature writeback\n"
      << "  --known-face-feature-dir DIR  known feature dir, default: face_photos/face_photo_features_cpp\n"
      << "  --known-face-feature-update-threshold VALUE  image-feature gate for primary EMA, default: 0.65\n"
      << "  --known-face-feature-pose-threshold VALUE  image-feature gate for pose sample, default: 0.60\n"
      << "  --known-face-feature-update-margin VALUE  top1-top2 known update margin, default: 0.08\n"
      << "  --known-face-feature-primary-ema-alpha VALUE  known primary EMA alpha, default: 0.1\n"
      << "  --known-face-feature-max-samples N  max known pose samples; plus image+primary rows, default: 12\n"
      << "  --face-rec-dynamic-library  enroll unknown faces as runtime face1/face2 ids\n"
      << "  --dynamic-face-library-dir DIR  output dir for C++ dynamic .npy features\n"
      << "  --face-rec-debug-crops  save actual 112x112 AdaFace input crops, default: off\n"
      << "  --face-rec-debug-crop-dir DIR  output dir for debug crops, default: board_output/face_rec_crops\n"
      << "  --face-rec-align-mode MODE  AdaFace crop alignment: eye or five-point, default: five-point\n"
      << "  --face-rec-five-point-scale VALUE  enlarge five-point aligned crop, default: 1.20\n"
      << "  --face-rec-min-box-size VALUE  trigger only when min bbox side >= VALUE, default: 100\n"
      << "  --face-rec-min-keypoint-conf VALUE  eye/nose keypoint confidence gate, default: 0.6\n"
      << "  --face-rec-min-eye-dist VALUE  eye distance gate in pixels, default: 0 (disabled)\n"
      << "  --face-rec-threshold VALUE  cosine match threshold, default: 0.55\n"
      << "  --face-rec-margin VALUE  top1-top2 margin gate, default: 0.08\n"
      << "  --face-rec-relink         allow young dynamic FaceID to relink to older FaceID, default\n"
      << "  --no-face-rec-relink      disable dynamic FaceID relink\n"
      << "  --face-rec-relink-threshold VALUE  relink cosine threshold, default: 0.55\n"
      << "  --face-rec-relink-margin VALUE  relink top1-top2 margin gate, default: 0.08\n"
      << "  --face-rec-relink-confirm-frames N  consecutive AdaFace checks before relink, default: 2\n"
      << "  --face-rec-relink-max-current-samples N  relink only young dynamic FaceID, default: 2\n"
      << "  --face-rec-lock-verify  release existing FaceID after repeated low-similarity checks, default\n"
      << "  --no-face-rec-lock-verify  keep existing public_id FaceID binding without low-score release\n"
      << "  --face-rec-lock-min-similarity VALUE  low-score release threshold, default: 0.25\n"
      << "  --face-rec-lock-fail-frames N  consecutive low-score checks before release, default: 3\n"
      << "  --face-rec-display-tentative  tentatively carry FaceID from display_id memory, default\n"
      << "  --no-face-rec-display-tentative  disable display_id based tentative FaceID carry\n"
      << "  --face-rec-tentative-threshold VALUE  cosine threshold to confirm tentative FaceID, default: 0.50\n"
      << "  --face-rec-tentative-fail-frames N  failed AdaFace checks before releasing tentative FaceID, default: 2\n"
      << "  --face-rec-dynamic-enroll-confirm-frames N  unmatched quality checks before creating a new dynamic FaceID, default: 3\n"
      << "  --face-rec-cooldown-seconds VALUE  unbound AdaFace attempt interval, default: 0.333333\n"
      << "  --face-rec-cooldown N  legacy frame override for unbound AdaFace attempt interval\n"
      << "  --face-rec-newly-bound-cooldown-seconds VALUE  newly bound FaceID check interval, default: 1\n"
      << "  --face-rec-newly-bound-cooldown N  legacy frame override for newly bound check interval\n"
      << "  --face-rec-stable-cooldown-seconds VALUE  stable FaceID check interval, default: 5\n"
      << "  --face-rec-stable-cooldown N  legacy frame override for stable check interval\n"
      << "  --face-rec-reappeared-cooldown-seconds VALUE  reappeared track check interval, default: 0.333333\n"
      << "  --face-rec-reappeared-cooldown N  legacy frame override for reappeared check interval\n"
      << "  --face-rec-reappeared-probe-seconds VALUE  keep reappeared fast-check state, default: 1\n"
      << "  --face-rec-reappeared-probe-frames N  legacy frame override for reappeared fast-check state\n"
      << "  --face-rec-newly-bound-checks N  checks before FaceID becomes stable, default: 3\n"
      << "  --face-rec-max-per-frame N  max parallel AdaFace runs per frame, default: 3, capped at 3\n"
      << "  --face-rec-async      run AdaFace in one background worker, default: off\n"
      << "  --no-face-rec-async   run AdaFace synchronously in the current frame\n"
      << "  --face-rec-async-max-pending N  max queued/in-flight async AdaFace tasks, default: 8\n"
      << "  --face-rec-async-max-age-seconds VALUE  drop async FaceID result older than seconds, default: 1\n"
      << "  --face-rec-async-max-age N  legacy frame override for async FaceID result age\n"
      << "  --face-rec-binding-ttl-seconds VALUE  keep FaceID binding for inactive TrackID, default: 30\n"
      << "  --face-rec-binding-ttl N  legacy frame override for inactive TrackID FaceID binding\n"
      << "  --face-rec-update-similarity VALUE  dynamic FaceID primary update min score, default: 0.65\n"
      << "  --face-rec-pose-sample-similarity VALUE  add high-quality pose samples above this score, default: 0.55\n"
      << "  --face-rec-pose-library-max-similarity VALUE  skip pose samples above this duplicate score, default: 0.90\n"
      << "  --face-rec-update-max-similarity VALUE  legacy alias of --face-rec-pose-library-max-similarity, default: 0.90\n"
      << "  --face-rec-pose-sample-interval N  min frames between pose samples, default: 30\n"
      << "  --face-rec-pending-seed-samples N  max pose samples copied from pending enroll, default: 2\n"
      << "  --face-rec-min-sample-diversity VALUE  skip near-duplicate FaceID samples, default: 0.015\n"
      << "  --face-rec-max-samples-per-id N  max dynamic vectors per FaceID, default: 12\n"
      << "  --face-rec-locked-sample-similarity VALUE  add high-quality locked-track pose samples above this score, default: 0.35\n"
      << "  --face-rec-update-min-box-size VALUE  min bbox side for dynamic enroll/sample update, default: 100\n"
      << "  --face-rec-update-min-keypoint-conf VALUE  min eye/nose confidence for dynamic enroll/sample update, default: 0.7\n"
      << "  --face-rec-update-min-score VALUE  min detection score for dynamic enroll/sample update, default: 0.5\n"
      << "  --face-rec-primary-min-box-size VALUE  min bbox side for primary FaceID update, default: 110\n"
      << "  --face-rec-primary-min-keypoint-conf VALUE  min eye/nose confidence for primary FaceID update, default: 0.8\n"
      << "  --face-rec-primary-min-score VALUE  min detection score for primary FaceID update, default: 0.6\n"
      << "  --face-rec-primary-max-yaw-deg VALUE  max yaw for primary FaceID update, default: 10\n"
      << "  --face-rec-primary-ema-alpha VALUE  high-quality primary FaceID EMA alpha, default: 0.1\n"
      << "  --print-profile-summary  print average per-stage timings to stderr\n"
      << "  --profile-system-load write background CPU/GPU/NPU/memory load JSONL\n"
      << "  --system-load-interval-ms N  hardware load sample interval, default: 200\n"
      << "  --wide-angle-yolo-workers N  wide-angle YOLO NPU workers on core0/core1, default: 2\n"
      << "  --decode-prefetch     pre-decode image-list frames in a worker thread, default: on\n"
      << "  --no-decode-prefetch  disable image-list read/decode worker pipeline\n"
      << "  --no-camera-prefetch  disable camera read/decode worker pipeline\n"
      << "  --no-bound-input      disable RKNN bound-input/OpenCL imported-output path\n"
      << "  --staging-copy-input  OpenCL writes staging buffers, then CPU copies to RKNN input\n"
      << "  --staging-pipeline    overlap next OpenCL staging with current blocking RKNN, default: on\n"
      << "  --no-staging-pipeline disable staging overlap pipeline\n"
      << "  --no-stdout-json       suppress JSON results on stdout\n"
      << "  --stdout-debug-json    print full debug detection JSON to stdout\n"
      << "  --map-dir DIR          exported map directory, default: map_export\n"
      << "  --projection-mode MODE projection geometry: fisheye-panorama or wide-angle, default: fisheye-panorama\n"
      << "  --model PATH           RKNN model path\n"
      << "  --calib-yaml PATH      accepted for Python CLI compatibility; C++ defaults to fit_4\n"
      << "  --conf VALUE           confidence threshold, default: 0.2\n"
      << "  --conf-threshold VALUE alias of --conf\n"
      << "  --iou-threshold VALUE  alias of --decode-iou\n"
      << "  --decode-iou VALUE     decode NMS IoU threshold, default: 0.99\n"
      << "  --merge-iou VALUE      cross-slice merge IoU threshold, default: 0.2\n"
      << "  --nms-iou VALUE        final NMS IoU threshold, default: 0.6\n"
      << "  --overlap VALUE        slice overlap ratio, default: 0.1\n"
      << "  --max-det N            per-slice max detections, default: 100\n"
      << "  --no-tracker           disable native HybridSORT tracking\n"
      << "  --track-buffer-seconds VALUE  tracker lost buffer time, default: 30\n"
      << "  --track-buffer N       legacy tracker lost buffer frame override\n"
      << "  --tracker-min-hits N   consecutive hits before confirming a new track, default: 3\n"
      << "  --tracker-match-thresh VALUE  tracker association IoU threshold, default: 0.12\n"
      << "  --tracker-new-thresh VALUE    new track confidence threshold, default: 0.7\n"
      << "  --new-track-overlap-thresh VALUE  avoid duplicate new tracks, default: 0.4\n"
      << "  --new-track-center-suppress VALUE  suppress tiny new raw tracks near existing tracks, default: 1.0\n"
      << "  --new-track-suppress-max-side VALUE  max side for tiny new-track suppression, default: 80\n"
      << "  --byte-residual-size-ratio VALUE  protect small residual observation updates below this size ratio, default: 0.45\n"
      << "  --byte-residual-max-side VALUE  max side for residual observation protection, default: 40\n"
      << "  --lost-track-reconnect-center-thresh VALUE  strict native lost-track center gate before new track, default: 1.6\n"
      << "  --lost-track-reconnect-multi-target-center-thresh VALUE  stricter lost-track gate near multiple detections, default: 1.2\n"
      << "  --lost-track-reconnect-size-ratio VALUE  strict native lost-track size gate, default: 0.5\n"
      << "  --lost-track-reconnect-ambiguity-margin VALUE  strict native lost-track unique-best gate, default: 0.35\n"
      << "  --lost-velocity-decay VALUE   multiply unmatched track velocity each frame, default: 0.85\n"
      << "  --inherit-center-dist-thresh VALUE  short occlusion ID inheritance gate, default: 2.0\n"
      << "  --inherit-multi-target-center-dist-thresh VALUE  stricter short inheritance gate near multiple targets, default: 1.2\n"
      << "  --inherit-size-ratio-thresh VALUE   short occlusion size gate, default: 0.4\n"
      << "  --inherit-ambiguity-margin VALUE    short occlusion unique-best gate, default: 0.25\n"
      << "  --public-recover-center-dist-thresh VALUE  low-ambiguity public ID recovery gate, default: 2.0\n"
      << "  --public-recover-max-seconds VALUE  low-ambiguity public ID recovery window, default: 30\n"
      << "  --public-recover-max-frames N       legacy frame override for public ID recovery window\n"
      << "  --public-alias-confirm-frames N     consecutive duplicate frames before raw tracks share a public ID, default: 6\n"
      << "  --no-smooth-bbox       disable tracker bbox EMA smoothing\n"
      << "  --smooth-bbox-alpha VALUE  tracker bbox EMA alpha, default: 0.6\n"
      << "  --coast-seconds VALUE  output lost tracks for seconds, default: 0.3\n"
      << "  --coast-frames N       legacy frame override for lost-track output\n"
      << "  --coast-hold           keep coasting bbox at last observation, default\n"
      << "  --no-coast-hold        output coasting tracks with predicted bbox instead of last observation\n"
      << "  --kalman-bbox          use native Kalman state bbox for active tracks\n"
      << "  --no-boundary-recover  disable geometry boundary ID recovery\n"
      << "  --boundary-time-window-seconds VALUE  boundary recovery history window, default: 3\n"
      << "  --boundary-time-window N  legacy frame override for boundary recovery history\n"
      << "  --no-final-boundary-dedup  disable final boundary duplicate suppression\n"
      << "  --final-boundary-dedup-iou VALUE     final min-area IoU duplicate gate, default: 0.70\n"
      << "  --final-boundary-dedup-center VALUE  final spatial duplicate center gate, default: 0.80\n"
      << "  --final-boundary-dedup-size VALUE    final spatial duplicate area-ratio gate, default: 0.25\n"
      << "  --no-filter-invalid-boxes  disable Python-compatible invalid bbox filtering\n"
      << "  --min-box-width VALUE   drop boxes narrower than this before tracker, default: 12\n"
      << "  --min-box-height VALUE  drop boxes shorter than this before tracker, default: 12\n"
      << "  --min-box-aspect-ratio VALUE  drop very thin boxes before tracker, default: 0.35\n"
      << "  --max-width-ratio VALUE  drop invalid huge boxes wider than process_width*ratio, default: 0.6\n"
      << "  --tracker-input-small-dedup  remove adjacent tiny duplicates before tracker, default: on\n"
      << "  --no-tracker-input-small-dedup  keep tiny duplicates in tracker input\n"
      << "  --no-close-small-dedup  disable close-center tiny box duplicate suppression\n"
      << "  --close-small-dedup-center VALUE  legacy normalized tiny duplicate center gate, default: 2.00\n"
      << "  --close-small-dedup-center-px VALUE  tiny duplicate absolute center gate, default: 32\n"
      << "  --close-small-dedup-x-gap-px VALUE  tiny duplicate absolute x-gap gate, default: 8\n"
      << "  --close-small-dedup-y-center-px VALUE  tiny duplicate absolute y-center gate, default: 12\n"
      << "  --close-small-dedup-y-cover VALUE  tiny duplicate vertical overlap gate, default: 0.65\n"
      << "  --close-small-public-alias-center-px VALUE  public alias absolute center gate, default: 20\n"
      << "  --close-small-public-alias-x-gap-px VALUE  public alias absolute x-gap gate, default: 4\n"
      << "  --close-small-public-alias-y-cover VALUE  public alias vertical overlap gate, default: 0.85\n"
      << "  --close-small-dedup-max-area VALUE  tiny duplicate max area, default: 900\n"
      << "  --close-small-dedup-max-side VALUE  tiny duplicate max side length, default: 30\n"
      << "  --no-normal-duplicate-dedup  disable close-center normal box duplicate suppression\n"
      << "  --normal-duplicate-center VALUE  normal duplicate center gate, default: 0.25\n"
      << "  --normal-duplicate-abs-center-px VALUE  normal duplicate absolute center gate, default: 16\n"
      << "  --normal-duplicate-size-ratio VALUE  normal duplicate size gate, default: 0.70\n"
      << "  --normal-duplicate-iou VALUE  normal duplicate IoU gate, default: 0.45\n"
      << "  --normal-duplicate-cover VALUE  normal duplicate x/y cover gate, default: 0.75\n"
      << "  --raw-duplicate-retire  retire duplicate native raw tracks after strict confirmation, default: on\n"
      << "  --no-raw-duplicate-retire  disable duplicate native raw track retirement\n"
      << "  --raw-duplicate-retire-confirm-frames N  duplicate raw retire confirmation frames, default: 5\n"
      << "  --raw-duplicate-retire-center VALUE  strict duplicate center gate, default: 0.45\n"
      << "  --raw-duplicate-retire-iou VALUE  strict duplicate IoU shortcut, default: 0.55\n"
      << "  --raw-duplicate-retire-size-ratio VALUE  strict duplicate size gate, default: 0.65\n"
      << "  --raw-duplicate-retire-y-cover VALUE  strict duplicate vertical cover gate, default: 0.75\n"
      << "  --raw-duplicate-retire-local-dist VALUE  third-target ambiguity radius, default: 1.5\n"
      << "  --raw-duplicate-retire-strong-public-seen N  block retiring two strong different public IDs, default: 10\n"
      << "  --angle-vectorized     accepted; C++ angle path is already vector-style\n"
      << "  --force-build          accepted as no-op; board_cpp is prebuilt\n";
}

static void apply_projection_mode_defaults(Config& cfg) {
  cfg.projection_mode = normalize_projection_mode(cfg.projection_mode);
  if (!is_wide_angle_projection(cfg.projection_mode)) {
    cfg.wrap_boundary_geometry = true;
    return;
  }

  cfg.wrap_boundary_geometry = false;
  cfg.boundary_recover = false;
  cfg.final_boundary_dedup = false;
  cfg.bound_input = false;
  cfg.staging_copy_input = false;
  cfg.staging_pipeline = false;

  auto is_default = [](float value, float default_value) {
    return std::fabs(value - default_value) < 1.0e-6f;
  };
  if (is_default(cfg.normal_duplicate_center_thresh, 0.25f)) {
    cfg.normal_duplicate_center_thresh = 0.50f;
  }
  if (is_default(cfg.normal_duplicate_abs_center_px, 16.0f)) {
    cfg.normal_duplicate_abs_center_px = 48.0f;
  }
  if (is_default(cfg.normal_duplicate_size_ratio_thresh, 0.70f)) {
    cfg.normal_duplicate_size_ratio_thresh = 0.50f;
  }
  if (is_default(cfg.normal_duplicate_iou_thresh, 0.45f)) {
    cfg.normal_duplicate_iou_thresh = 0.25f;
  }
  if (is_default(cfg.normal_duplicate_cover_thresh, 0.75f)) {
    cfg.normal_duplicate_cover_thresh = 0.55f;
  }
}

static Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
      }
      return argv[++i];
    };

    if (arg == "--image") {
      cfg.image_paths.push_back(need_value(arg));
    } else if (arg == "--image-list") {
      cfg.image_list_path = need_value(arg);
    } else if (arg == "--camera-device") {
      cfg.camera_device = need_value(arg);
    } else if (arg == "--camera-width") {
      cfg.camera_width = std::stoi(need_value(arg));
    } else if (arg == "--camera-height") {
      cfg.camera_height = std::stoi(need_value(arg));
    } else if (arg == "--camera-fps") {
      cfg.camera_fps = std::stoi(need_value(arg));
    } else if (arg == "--input-fps" || arg == "--effective-fps") {
      cfg.input_fps = std::stod(need_value(arg));
    } else if (arg == "--camera-buffers") {
      cfg.camera_buffers = std::stoi(need_value(arg));
    } else if (arg == "--camera-timeout-ms") {
      cfg.camera_timeout_ms = std::stoi(need_value(arg));
    } else if (arg == "--max-frames") {
      cfg.max_frames = std::stoi(need_value(arg));
    } else if (arg == "--output-jsonl") {
      cfg.output_jsonl_path = need_value(arg);
    } else if (arg == "--debug-jsonl") {
      cfg.debug_jsonl_path = need_value(arg);
    } else if (arg == "--no-output-jsonl") {
      cfg.no_output_jsonl = true;
    } else if (arg == "--ws-host") {
      cfg.ws_host = need_value(arg);
    } else if (arg == "--ws-port") {
      cfg.ws_port = std::stoi(need_value(arg));
    } else if (arg == "--no-ws-json") {
      cfg.ws_json = false;
    } else if (arg == "--webui") {
      cfg.webui = true;
    } else if (arg == "--webui-host") {
      cfg.webui_host = need_value(arg);
    } else if (arg == "--webui-port" || arg == "--port") {
      cfg.webui_port = std::stoi(need_value(arg));
	    } else if (arg == "--webui-jpeg-quality") {
	      cfg.webui_jpeg_quality = std::stoi(need_value(arg));
	    } else if (arg == "--webui-frame-scale") {
	      cfg.webui_frame_scale = std::stod(need_value(arg));
	    } else if (arg == "--webui-slot-fps") {
	      cfg.webui_slot_fps = std::stod(need_value(arg));
	    } else if (arg == "--webui-show-hidden-targets") {
	      cfg.webui_show_hidden_targets = true;
    } else if (arg == "--no-webui-show-hidden-targets") {
      cfg.webui_show_hidden_targets = false;
    } else if (arg == "--sector-output") {
      cfg.sector_output = true;
    } else if (arg == "--num-sectors") {
      cfg.num_sectors = std::stoi(need_value(arg));
    } else if (arg == "--target-output-limit") {
      cfg.target_output_limit = std::stoi(need_value(arg));
    } else if (arg == "--target-output-compact-ids") {
      cfg.target_output_compact_ids = true;
    } else if (arg == "--no-target-output-compact-ids") {
      cfg.target_output_compact_ids = false;
    } else if (arg == "--display-id-reuse") {
      cfg.display_id_reuse = true;
    } else if (arg == "--no-display-id-reuse") {
      cfg.display_id_reuse = false;
    } else if (arg == "--display-id-reuse-spatial") {
      cfg.display_id_reuse_spatial = true;
    } else if (arg == "--no-display-id-reuse-spatial") {
      cfg.display_id_reuse_spatial = false;
    } else if (arg == "--display-id-reuse-max-seconds") {
      cfg.display_id_reuse_max_seconds = std::stod(need_value(arg));
      cfg.display_id_reuse_max_frames = -1;
    } else if (arg == "--display-id-reuse-max-frames") {
      cfg.display_id_reuse_max_frames = std::stoi(need_value(arg));
    } else if (arg == "--display-id-reuse-fallback-min-seconds") {
      cfg.display_id_reuse_fallback_min_seconds = std::stod(need_value(arg));
      cfg.display_id_reuse_fallback_min_frames = -1;
    } else if (arg == "--display-id-reuse-fallback-min-frames") {
      cfg.display_id_reuse_fallback_min_frames = std::stoi(need_value(arg));
    } else if (arg == "--display-id-public-binding-ttl-seconds") {
      cfg.display_id_public_binding_ttl_seconds = std::stod(need_value(arg));
      cfg.display_id_public_binding_ttl = -1;
    } else if (arg == "--display-id-public-binding-ttl") {
      cfg.display_id_public_binding_ttl = std::stoi(need_value(arg));
    } else if (arg == "--display-id-max-count") {
      cfg.display_id_max_count = std::stoi(need_value(arg));
    } else if (arg == "--display-id-replace-confirm-seconds") {
      cfg.display_id_replace_confirm_seconds = std::stod(need_value(arg));
      cfg.display_id_replace_confirm_frames = -1;
    } else if (arg == "--display-id-replace-confirm-frames") {
      cfg.display_id_replace_confirm_frames = std::stoi(need_value(arg));
    } else if (arg == "--display-id-replace-area-ratio") {
      cfg.display_id_replace_area_ratio = std::stof(need_value(arg));
    } else if (arg == "--display-id-min-box-size") {
      cfg.display_id_min_box_size = std::stof(need_value(arg));
    } else if (arg == "--display-id-reuse-center-thresh") {
      cfg.display_id_reuse_center_thresh = std::stof(need_value(arg));
    } else if (arg == "--display-id-reuse-size-ratio") {
      cfg.display_id_reuse_size_ratio_thresh = std::stof(need_value(arg));
    } else if (arg == "--display-id-reuse-ambiguity-margin") {
      cfg.display_id_reuse_ambiguity_margin = std::stof(need_value(arg));
    } else if (arg == "--face-rec" || arg == "--use-face-rec") {
      cfg.face_rec_enabled = true;
    } else if (arg == "--no-face-rec" || arg == "--no-use-face-rec") {
      cfg.face_rec_enabled = false;
    } else if (arg == "--face-rec-model") {
      cfg.face_rec_model_path = need_value(arg);
    } else if (arg == "--face-rec-model-preset" || arg == "--face-rec-model-name") {
      cfg.face_rec_model_path = face_rec_model_path_for_preset(need_value(arg));
    } else if (arg == "--face-library-dir") {
      cfg.face_rec_library_dir = need_value(arg);
    } else if (arg == "--known-face-dir") {
      cfg.known_face_dir = need_value(arg);
    } else if (arg == "--known-face-feature-library") {
      cfg.known_face_feature_library = true;
    } else if (arg == "--no-known-face-feature-library") {
      cfg.known_face_feature_library = false;
    } else if (arg == "--known-face-feature-dir") {
      cfg.known_face_feature_dir = need_value(arg);
    } else if (arg == "--known-face-feature-update-threshold") {
      cfg.known_face_feature_update_threshold = std::stof(need_value(arg));
    } else if (arg == "--known-face-feature-pose-threshold") {
      cfg.known_face_feature_pose_threshold = std::stof(need_value(arg));
    } else if (arg == "--known-face-feature-update-margin") {
      cfg.known_face_feature_update_margin = std::stof(need_value(arg));
    } else if (arg == "--known-face-feature-primary-ema-alpha") {
      cfg.known_face_feature_primary_ema_alpha = std::stof(need_value(arg));
    } else if (arg == "--known-face-feature-max-samples") {
      cfg.known_face_feature_max_samples = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-dynamic-library" || arg == "--dynamic-face-library") {
      cfg.face_rec_dynamic_library = true;
    } else if (arg == "--no-face-rec-dynamic-library" || arg == "--no-dynamic-face-library") {
      cfg.face_rec_dynamic_library = false;
    } else if (arg == "--dynamic-face-library-dir") {
      cfg.face_rec_dynamic_library_dir = need_value(arg);
    } else if (arg == "--face-rec-debug-crops") {
      cfg.face_rec_debug_crops = true;
    } else if (arg == "--no-face-rec-debug-crops") {
      cfg.face_rec_debug_crops = false;
    } else if (arg == "--face-rec-debug-crop-dir") {
      cfg.face_rec_debug_crop_dir = need_value(arg);
    } else if (arg == "--face-rec-align-mode") {
      cfg.face_rec_align_mode = normalize_face_rec_align_mode(need_value(arg));
    } else if (arg == "--face-rec-five-point-scale") {
      cfg.face_rec_five_point_scale = std::stof(need_value(arg));
    } else if (arg == "--face-rec-min-box-size" || arg == "--dynamic-face-min-height") {
      cfg.face_rec_min_box_size = std::stof(need_value(arg));
    } else if (arg == "--face-rec-min-keypoint-conf") {
      cfg.face_rec_min_keypoint_conf = std::stof(need_value(arg));
    } else if (arg == "--face-rec-min-eye-dist") {
      cfg.face_rec_min_eye_dist = std::stof(need_value(arg));
    } else if (arg == "--face-rec-threshold") {
      cfg.face_rec_threshold = std::stof(need_value(arg));
    } else if (arg == "--face-rec-margin" || arg == "--dynamic-face-match-margin") {
      cfg.face_rec_margin = std::stof(need_value(arg));
    } else if (arg == "--face-rec-relink") {
      cfg.face_rec_relink = true;
    } else if (arg == "--no-face-rec-relink") {
      cfg.face_rec_relink = false;
    } else if (arg == "--face-rec-relink-threshold") {
      cfg.face_rec_relink_threshold = std::stof(need_value(arg));
    } else if (arg == "--face-rec-relink-margin") {
      cfg.face_rec_relink_margin = std::stof(need_value(arg));
    } else if (arg == "--face-rec-relink-confirm-frames") {
      cfg.face_rec_relink_confirm_frames = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-relink-max-current-samples") {
      cfg.face_rec_relink_max_current_samples = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-lock-verify") {
      cfg.face_rec_lock_verify = true;
    } else if (arg == "--no-face-rec-lock-verify") {
      cfg.face_rec_lock_verify = false;
    } else if (arg == "--face-rec-lock-min-similarity") {
      cfg.face_rec_lock_min_similarity = std::stof(need_value(arg));
    } else if (arg == "--face-rec-lock-fail-frames") {
      cfg.face_rec_lock_fail_frames = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-display-tentative") {
      cfg.face_rec_display_tentative = true;
    } else if (arg == "--no-face-rec-display-tentative") {
      cfg.face_rec_display_tentative = false;
    } else if (arg == "--face-rec-tentative-threshold") {
      cfg.face_rec_tentative_threshold = std::stof(need_value(arg));
    } else if (arg == "--face-rec-tentative-fail-frames") {
      cfg.face_rec_tentative_fail_frames = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-dynamic-enroll-confirm-frames") {
      cfg.face_rec_dynamic_enroll_confirm_frames = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-update-similarity") {
      cfg.face_rec_update_similarity = std::stof(need_value(arg));
    } else if (arg == "--face-rec-pose-sample-similarity") {
      cfg.face_rec_pose_sample_similarity = std::stof(need_value(arg));
    } else if (arg == "--face-rec-pose-library-max-similarity") {
      cfg.face_rec_pose_library_max_similarity = std::stof(need_value(arg));
      cfg.face_rec_update_max_similarity = cfg.face_rec_pose_library_max_similarity;
    } else if (arg == "--face-rec-update-max-similarity") {
      cfg.face_rec_update_max_similarity = std::stof(need_value(arg));
      cfg.face_rec_pose_library_max_similarity = cfg.face_rec_update_max_similarity;
    } else if (arg == "--face-rec-pose-sample-interval") {
      cfg.face_rec_pose_sample_interval = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-pending-seed-samples") {
      cfg.face_rec_pending_seed_samples = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-min-sample-diversity") {
      cfg.face_rec_min_sample_diversity = std::stof(need_value(arg));
    } else if (arg == "--face-rec-cooldown-seconds") {
      cfg.face_rec_cooldown_seconds = std::stod(need_value(arg));
      cfg.face_rec_cooldown = -1;
    } else if (arg == "--face-rec-cooldown") {
      cfg.face_rec_cooldown = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-newly-bound-cooldown-seconds") {
      cfg.face_rec_newly_bound_cooldown_seconds = std::stod(need_value(arg));
      cfg.face_rec_newly_bound_cooldown = -1;
    } else if (arg == "--face-rec-newly-bound-cooldown") {
      cfg.face_rec_newly_bound_cooldown = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-stable-cooldown-seconds") {
      cfg.face_rec_stable_cooldown_seconds = std::stod(need_value(arg));
      cfg.face_rec_stable_cooldown = -1;
    } else if (arg == "--face-rec-stable-cooldown") {
      cfg.face_rec_stable_cooldown = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-reappeared-cooldown-seconds") {
      cfg.face_rec_reappeared_cooldown_seconds = std::stod(need_value(arg));
      cfg.face_rec_reappeared_cooldown = -1;
    } else if (arg == "--face-rec-reappeared-cooldown") {
      cfg.face_rec_reappeared_cooldown = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-reappeared-probe-seconds") {
      cfg.face_rec_reappeared_probe_seconds = std::stod(need_value(arg));
      cfg.face_rec_reappeared_probe_frames = -1;
    } else if (arg == "--face-rec-reappeared-probe-frames") {
      cfg.face_rec_reappeared_probe_frames = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-newly-bound-checks") {
      cfg.face_rec_newly_bound_checks = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-max-per-frame") {
      cfg.face_rec_max_per_frame = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-async") {
      cfg.face_rec_async = true;
    } else if (arg == "--no-face-rec-async") {
      cfg.face_rec_async = false;
    } else if (arg == "--face-rec-async-max-pending") {
      cfg.face_rec_async_max_pending = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-async-max-age-seconds") {
      cfg.face_rec_async_max_age_seconds = std::stod(need_value(arg));
      cfg.face_rec_async_max_age = -1;
    } else if (arg == "--face-rec-async-max-age") {
      cfg.face_rec_async_max_age = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-binding-ttl-seconds") {
      cfg.face_rec_binding_ttl_seconds = std::stod(need_value(arg));
      cfg.face_rec_binding_ttl = -1;
    } else if (arg == "--face-rec-binding-ttl") {
      cfg.face_rec_binding_ttl = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-max-samples-per-id") {
      cfg.face_rec_max_samples_per_id = std::stoi(need_value(arg));
    } else if (arg == "--face-rec-locked-sample-similarity") {
      cfg.face_rec_locked_sample_similarity = std::stof(need_value(arg));
    } else if (arg == "--face-rec-update-min-box-size") {
      cfg.face_rec_update_min_box_size = std::stof(need_value(arg));
    } else if (arg == "--face-rec-update-min-keypoint-conf") {
      cfg.face_rec_update_min_keypoint_conf = std::stof(need_value(arg));
    } else if (arg == "--face-rec-update-min-score") {
      cfg.face_rec_update_min_score = std::stof(need_value(arg));
    } else if (arg == "--face-rec-primary-min-box-size") {
      cfg.face_rec_primary_min_box_size = std::stof(need_value(arg));
    } else if (arg == "--face-rec-primary-min-keypoint-conf") {
      cfg.face_rec_primary_min_keypoint_conf = std::stof(need_value(arg));
    } else if (arg == "--face-rec-primary-min-score") {
      cfg.face_rec_primary_min_score = std::stof(need_value(arg));
    } else if (arg == "--face-rec-primary-max-yaw-deg") {
      cfg.face_rec_primary_max_yaw_deg = std::stof(need_value(arg));
    } else if (arg == "--face-rec-primary-ema-alpha") {
      cfg.face_rec_primary_ema_alpha = std::stof(need_value(arg));
    } else if (arg == "--print-profile-summary") {
      cfg.print_profile_summary = true;
    } else if (arg == "--decode-prefetch") {
      cfg.decode_prefetch = true;
    } else if (arg == "--no-decode-prefetch") {
      cfg.decode_prefetch = false;
    } else if (arg == "--no-camera-prefetch") {
      cfg.camera_prefetch = false;
    } else if (arg == "--no-bound-input") {
      cfg.bound_input = false;
    } else if (arg == "--staging-copy-input") {
      cfg.staging_copy_input = true;
    } else if (arg == "--no-staging-copy-input") {
      cfg.staging_copy_input = false;
      cfg.staging_pipeline = false;
    } else if (arg == "--staging-pipeline") {
      cfg.staging_pipeline = true;
      cfg.staging_copy_input = true;
    } else if (arg == "--no-staging-pipeline") {
      cfg.staging_pipeline = false;
    } else if (arg == "--no-stdout-json") {
      cfg.no_stdout_json = true;
    } else if (arg == "--stdout-debug-json") {
      cfg.stdout_debug_json = true;
    } else if (arg == "--map-dir") {
      cfg.map_dir = need_value(arg);
    } else if (arg == "--projection-mode") {
      cfg.projection_mode = need_value(arg);
    } else if (arg == "--model") {
      cfg.model_path = need_value(arg);
    } else if (arg == "--model-path") {
      cfg.model_path = need_value(arg);
    } else if (arg == "--calib-yaml") {
      cfg.calib_yaml = need_value(arg);
    } else if (arg == "--fit-degree") {
      cfg.fit_degree = std::stoi(need_value(arg));
    } else if (arg == "--conf") {
      cfg.conf_threshold = std::stof(need_value(arg));
    } else if (arg == "--conf-threshold") {
      cfg.conf_threshold = std::stof(need_value(arg));
    } else if (arg == "--decode-iou") {
      cfg.decode_iou_threshold = std::stof(need_value(arg));
    } else if (arg == "--iou-threshold") {
      cfg.decode_iou_threshold = std::stof(need_value(arg));
    } else if (arg == "--merge-iou") {
      cfg.merge_iou_threshold = std::stof(need_value(arg));
    } else if (arg == "--nms-iou") {
      cfg.nms_iou_threshold = std::stof(need_value(arg));
    } else if (arg == "--overlap") {
      cfg.overlap_ratio = std::stof(need_value(arg));
    } else if (arg == "--max-det") {
      cfg.max_det = std::stoi(need_value(arg));
    } else if (arg == "--no-tracker") {
      cfg.tracker_enabled = false;
    } else if (arg == "--track-buffer-seconds") {
      cfg.track_buffer_seconds = std::stod(need_value(arg));
      cfg.track_buffer = -1;
    } else if (arg == "--track-buffer") {
      cfg.track_buffer = std::stoi(need_value(arg));
    } else if (arg == "--tracker-min-hits") {
      cfg.tracker_min_hits = std::stoi(need_value(arg));
    } else if (arg == "--tracker-match-thresh") {
      cfg.tracker_match_thresh = std::stof(need_value(arg));
    } else if (arg == "--tracker-new-thresh") {
      cfg.tracker_new_thresh = std::stof(need_value(arg));
    } else if (arg == "--new-track-overlap-thresh") {
      cfg.new_track_overlap_thresh = std::stof(need_value(arg));
    } else if (arg == "--new-track-center-suppress") {
      cfg.new_track_center_suppress_thresh = std::stof(need_value(arg));
    } else if (arg == "--new-track-suppress-max-side") {
      cfg.new_track_suppress_max_side = std::stof(need_value(arg));
    } else if (arg == "--byte-residual-size-ratio") {
      cfg.byte_residual_size_ratio_thresh = std::stof(need_value(arg));
    } else if (arg == "--byte-residual-max-side") {
      cfg.byte_residual_max_side = std::stof(need_value(arg));
    } else if (arg == "--lost-track-reconnect-center-thresh") {
      cfg.lost_track_reconnect_center_thresh = std::stof(need_value(arg));
    } else if (arg == "--lost-track-reconnect-multi-target-center-thresh") {
      cfg.lost_track_reconnect_multi_target_center_thresh = std::stof(need_value(arg));
    } else if (arg == "--lost-track-reconnect-size-ratio") {
      cfg.lost_track_reconnect_size_ratio_thresh = std::stof(need_value(arg));
    } else if (arg == "--lost-track-reconnect-ambiguity-margin") {
      cfg.lost_track_reconnect_ambiguity_margin = std::stof(need_value(arg));
    } else if (arg == "--lost-velocity-decay") {
      cfg.lost_velocity_decay = std::stof(need_value(arg));
    } else if (arg == "--inherit-center-dist-thresh") {
      cfg.inherit_center_dist_thresh = std::stof(need_value(arg));
    } else if (arg == "--inherit-multi-target-center-dist-thresh") {
      cfg.inherit_multi_target_center_dist_thresh = std::stof(need_value(arg));
    } else if (arg == "--inherit-size-ratio-thresh") {
      cfg.inherit_size_ratio_thresh = std::stof(need_value(arg));
    } else if (arg == "--inherit-ambiguity-margin") {
      cfg.inherit_ambiguity_margin = std::stof(need_value(arg));
    } else if (arg == "--public-recover-center-dist-thresh") {
      cfg.public_recover_center_dist_thresh = std::stof(need_value(arg));
    } else if (arg == "--public-recover-max-seconds") {
      cfg.public_recover_max_seconds = std::stod(need_value(arg));
      cfg.public_recover_max_frames = -1;
    } else if (arg == "--public-recover-max-frames") {
      cfg.public_recover_max_frames = std::stoi(need_value(arg));
    } else if (arg == "--public-alias-confirm-frames") {
      cfg.public_alias_confirm_frames = std::stoi(need_value(arg));
    } else if (arg == "--tracker-high-thresh") {
      cfg.tracker_high_thresh = std::stof(need_value(arg));
    } else if (arg == "--tracker-low-thresh") {
      cfg.tracker_low_thresh = std::stof(need_value(arg));
    } else if (arg == "--tracker-byte") {
      cfg.tracker_byte = true;
    } else if (arg == "--no-tracker-byte") {
      cfg.tracker_byte = false;
    } else if (arg == "--smooth-bbox") {
      cfg.smooth_bbox = true;
    } else if (arg == "--no-smooth-bbox") {
      cfg.smooth_bbox = false;
    } else if (arg == "--smooth-bbox-alpha") {
      cfg.smooth_bbox_alpha = std::stof(need_value(arg));
    } else if (arg == "--coast-seconds") {
      cfg.coast_seconds = std::stod(need_value(arg));
      cfg.coast_frames = -1;
    } else if (arg == "--coast-frames") {
      cfg.coast_frames = std::stoi(need_value(arg));
    } else if (arg == "--coast-hold") {
      cfg.coast_hold = true;
    } else if (arg == "--no-coast-hold") {
      cfg.coast_hold = false;
    } else if (arg == "--kalman-bbox") {
      cfg.kalman_bbox = true;
    } else if (arg == "--boundary-recover") {
      cfg.boundary_recover = true;
    } else if (arg == "--no-boundary-recover") {
      cfg.boundary_recover = false;
    } else if (arg == "--boundary-margin") {
      cfg.boundary_margin = std::stof(need_value(arg));
    } else if (arg == "--boundary-time-window-seconds") {
      cfg.boundary_time_window_seconds = std::stod(need_value(arg));
      cfg.boundary_time_window = -1;
    } else if (arg == "--boundary-time-window") {
      cfg.boundary_time_window = std::stoi(need_value(arg));
    } else if (arg == "--final-boundary-dedup") {
      cfg.final_boundary_dedup = true;
    } else if (arg == "--no-final-boundary-dedup") {
      cfg.final_boundary_dedup = false;
    } else if (arg == "--final-boundary-dedup-iou") {
      cfg.final_boundary_dedup_iou_thresh = std::stof(need_value(arg));
    } else if (arg == "--final-boundary-dedup-center") {
      cfg.final_boundary_dedup_center_thresh = std::stof(need_value(arg));
    } else if (arg == "--final-boundary-dedup-size") {
      cfg.final_boundary_dedup_size_ratio_thresh = std::stof(need_value(arg));
    } else if (arg == "--no-filter-invalid-boxes") {
      cfg.filter_invalid_boxes = false;
    } else if (arg == "--min-box-width") {
      cfg.min_box_width = std::stof(need_value(arg));
    } else if (arg == "--min-box-height") {
      cfg.min_box_height = std::stof(need_value(arg));
    } else if (arg == "--min-box-aspect-ratio") {
      cfg.min_box_aspect_ratio = std::stof(need_value(arg));
    } else if (arg == "--max-width-ratio") {
      cfg.max_width_ratio = std::stof(need_value(arg));
    } else if (arg == "--tracker-input-small-dedup") {
      cfg.tracker_input_small_dedup = true;
    } else if (arg == "--no-tracker-input-small-dedup") {
      cfg.tracker_input_small_dedup = false;
    } else if (arg == "--close-small-dedup") {
      cfg.close_small_dedup = true;
    } else if (arg == "--no-close-small-dedup") {
      cfg.close_small_dedup = false;
    } else if (arg == "--close-small-dedup-center") {
      cfg.close_small_dedup_center_thresh = std::stof(need_value(arg));
    } else if (arg == "--close-small-dedup-center-px") {
      cfg.close_small_dedup_center_px = std::stof(need_value(arg));
    } else if (arg == "--close-small-dedup-x-gap-px") {
      cfg.close_small_dedup_x_gap_px = std::stof(need_value(arg));
    } else if (arg == "--close-small-dedup-y-center-px") {
      cfg.close_small_dedup_y_center_px = std::stof(need_value(arg));
    } else if (arg == "--close-small-dedup-y-cover") {
      cfg.close_small_dedup_y_cover_thresh = std::stof(need_value(arg));
    } else if (arg == "--close-small-public-alias-center-px") {
      cfg.close_small_public_alias_center_px = std::stof(need_value(arg));
    } else if (arg == "--close-small-public-alias-x-gap-px") {
      cfg.close_small_public_alias_x_gap_px = std::stof(need_value(arg));
    } else if (arg == "--close-small-public-alias-y-cover") {
      cfg.close_small_public_alias_y_cover_thresh = std::stof(need_value(arg));
    } else if (arg == "--close-small-dedup-max-area") {
      cfg.close_small_dedup_max_area = std::stof(need_value(arg));
    } else if (arg == "--close-small-dedup-max-side") {
      cfg.close_small_dedup_max_side = std::stof(need_value(arg));
    } else if (arg == "--close-small-dedup-iou") {
      (void)std::stof(need_value(arg));  // legacy no-op; tiny duplicates no longer use IoU.
    } else if (arg == "--normal-duplicate-dedup") {
      cfg.normal_duplicate_dedup = true;
    } else if (arg == "--no-normal-duplicate-dedup") {
      cfg.normal_duplicate_dedup = false;
    } else if (arg == "--normal-duplicate-center") {
      cfg.normal_duplicate_center_thresh = std::stof(need_value(arg));
    } else if (arg == "--normal-duplicate-abs-center-px") {
      cfg.normal_duplicate_abs_center_px = std::stof(need_value(arg));
    } else if (arg == "--normal-duplicate-size-ratio") {
      cfg.normal_duplicate_size_ratio_thresh = std::stof(need_value(arg));
    } else if (arg == "--normal-duplicate-iou") {
      cfg.normal_duplicate_iou_thresh = std::stof(need_value(arg));
    } else if (arg == "--normal-duplicate-cover") {
      cfg.normal_duplicate_cover_thresh = std::stof(need_value(arg));
    } else if (arg == "--raw-duplicate-retire") {
      cfg.raw_duplicate_retire = true;
    } else if (arg == "--no-raw-duplicate-retire") {
      cfg.raw_duplicate_retire = false;
    } else if (arg == "--raw-duplicate-retire-confirm-frames") {
      cfg.raw_duplicate_retire_confirm_frames = std::stoi(need_value(arg));
    } else if (arg == "--raw-duplicate-retire-center") {
      cfg.raw_duplicate_retire_center_thresh = std::stof(need_value(arg));
    } else if (arg == "--raw-duplicate-retire-iou") {
      cfg.raw_duplicate_retire_iou_thresh = std::stof(need_value(arg));
    } else if (arg == "--raw-duplicate-retire-size-ratio") {
      cfg.raw_duplicate_retire_size_ratio_thresh = std::stof(need_value(arg));
    } else if (arg == "--raw-duplicate-retire-y-cover") {
      cfg.raw_duplicate_retire_y_cover_thresh = std::stof(need_value(arg));
    } else if (arg == "--raw-duplicate-retire-local-dist") {
      cfg.raw_duplicate_retire_local_dist_thresh = std::stof(need_value(arg));
    } else if (arg == "--raw-duplicate-retire-strong-public-seen") {
      cfg.raw_duplicate_retire_strong_public_seen = std::stoi(need_value(arg));
    } else if (arg == "--profile-interval") {
      cfg.profile_interval = std::stoi(need_value(arg));
    } else if (arg == "--profile-system-load" || arg == "--json-system-load") {
      cfg.profile_system_load = true;
    } else if (arg == "--system-load-interval-ms") {
      cfg.system_load_interval_ms = std::stoi(need_value(arg));
    } else if (arg == "--wide-angle-yolo-workers") {
      cfg.wide_angle_yolo_workers = std::stoi(need_value(arg));
    } else if (arg == "--angle-vectorized") {
      cfg.angle_vectorized = true;
    } else if (arg == "--force-build") {
      cfg.force_build = true;
    } else if (arg == "--no-build") {
      // No-op in board_cpp: native libraries are expected to be prebuilt.
    } else if (arg == "--json-debug-keypoints") {
      cfg.json_debug_keypoints = true;
    } else if (arg == "--imgsz" || arg == "--map-file" ||
               arg == "--queue-size" ||
               arg == "--video-name" || arg == "--video-fps") {
      (void)need_value(arg);
    } else if (arg == "--save-video" || arg == "--save-original-video" ||
               arg == "--tracker-native-assoc" ||
               arg == "--tracker-native-full") {
      // Accepted no-op flags from Python board/main.py.
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (!cfg.image_list_path.empty()) {
    std::vector<std::string> listed = read_image_list(cfg.image_list_path);
    cfg.image_paths.insert(cfg.image_paths.end(), listed.begin(), listed.end());
  }
  if (cfg.image_paths.empty() && cfg.camera_device.empty()) {
    throw std::runtime_error("--image, --image-list, or --camera-device is required");
  }
  if (!cfg.image_paths.empty() && !cfg.camera_device.empty()) {
    throw std::runtime_error("--camera-device cannot be combined with --image/--image-list");
  }
  apply_projection_mode_defaults(cfg);
  if (cfg.staging_pipeline) {
    cfg.staging_copy_input = true;
    cfg.bound_input = true;
  }
  for (const std::string& image_path : cfg.image_paths) {
    if (!file_exists(image_path)) {
      throw std::runtime_error("image not found: " + image_path);
    }
  }
  if (!file_exists(cfg.model_path)) {
    throw std::runtime_error("model not found: " + cfg.model_path);
  }
  if (cfg.face_rec_enabled && !file_exists(cfg.face_rec_model_path)) {
    throw std::runtime_error("face rec model not found: " + cfg.face_rec_model_path);
  }
  cfg.face_rec_align_mode = normalize_face_rec_align_mode(cfg.face_rec_align_mode);
  if (cfg.camera_width <= 0 || cfg.camera_height <= 0) {
    throw std::runtime_error("camera width/height must be positive");
  }
  if (cfg.camera_fps <= 0) {
    throw std::runtime_error("--camera-fps must be positive");
  }
  if (cfg.input_fps < 0.0 || !std::isfinite(cfg.input_fps)) {
    throw std::runtime_error("--input-fps must be non-negative");
  }
  auto require_nonnegative_seconds = [](double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0) {
      throw std::runtime_error(std::string(name) + " must be non-negative seconds");
    }
  };
  require_nonnegative_seconds(cfg.track_buffer_seconds, "--track-buffer-seconds");
  require_nonnegative_seconds(cfg.coast_seconds, "--coast-seconds");
  require_nonnegative_seconds(cfg.public_recover_max_seconds, "--public-recover-max-seconds");
  require_nonnegative_seconds(cfg.display_id_reuse_max_seconds, "--display-id-reuse-max-seconds");
  require_nonnegative_seconds(
      cfg.display_id_reuse_fallback_min_seconds,
      "--display-id-reuse-fallback-min-seconds");
  require_nonnegative_seconds(
      cfg.display_id_public_binding_ttl_seconds,
      "--display-id-public-binding-ttl-seconds");
  require_nonnegative_seconds(
      cfg.display_id_replace_confirm_seconds,
      "--display-id-replace-confirm-seconds");
  require_nonnegative_seconds(cfg.face_rec_cooldown_seconds, "--face-rec-cooldown-seconds");
  require_nonnegative_seconds(
      cfg.face_rec_newly_bound_cooldown_seconds,
      "--face-rec-newly-bound-cooldown-seconds");
  require_nonnegative_seconds(
      cfg.face_rec_stable_cooldown_seconds,
      "--face-rec-stable-cooldown-seconds");
  require_nonnegative_seconds(
      cfg.face_rec_reappeared_cooldown_seconds,
      "--face-rec-reappeared-cooldown-seconds");
  require_nonnegative_seconds(
      cfg.face_rec_reappeared_probe_seconds,
      "--face-rec-reappeared-probe-seconds");
  require_nonnegative_seconds(
      cfg.face_rec_async_max_age_seconds,
      "--face-rec-async-max-age-seconds");
  require_nonnegative_seconds(cfg.face_rec_binding_ttl_seconds, "--face-rec-binding-ttl-seconds");
  require_nonnegative_seconds(cfg.boundary_time_window_seconds, "--boundary-time-window-seconds");
  const double effective_fps = cfg.input_fps > 0.0 ? cfg.input_fps : static_cast<double>(cfg.camera_fps);
  if (cfg.track_buffer < 0) {
    cfg.track_buffer = seconds_to_frames(cfg.track_buffer_seconds, effective_fps);
  }
  if (cfg.coast_frames < 0) {
    cfg.coast_frames = seconds_to_frames(cfg.coast_seconds, effective_fps);
  }
  if (cfg.public_recover_max_frames < 0) {
    cfg.public_recover_max_frames =
        seconds_to_frames(cfg.public_recover_max_seconds, effective_fps);
  }
  if (cfg.display_id_reuse_max_frames < 0) {
    cfg.display_id_reuse_max_frames =
        seconds_to_frames(cfg.display_id_reuse_max_seconds, effective_fps);
  }
  if (cfg.display_id_reuse_fallback_min_frames < 0) {
    cfg.display_id_reuse_fallback_min_frames =
        seconds_to_frames(cfg.display_id_reuse_fallback_min_seconds, effective_fps);
  }
  if (cfg.display_id_public_binding_ttl < 0) {
    cfg.display_id_public_binding_ttl =
        seconds_to_frames(cfg.display_id_public_binding_ttl_seconds, effective_fps);
  }
  if (cfg.display_id_replace_confirm_frames < 0) {
    cfg.display_id_replace_confirm_frames =
        seconds_to_frames(cfg.display_id_replace_confirm_seconds, effective_fps);
  }
  if (cfg.face_rec_cooldown < 0) {
    cfg.face_rec_cooldown =
        seconds_to_frames(cfg.face_rec_cooldown_seconds, effective_fps);
  }
  if (cfg.face_rec_newly_bound_cooldown < 0) {
    cfg.face_rec_newly_bound_cooldown =
        seconds_to_frames(cfg.face_rec_newly_bound_cooldown_seconds, effective_fps);
  }
  if (cfg.face_rec_stable_cooldown < 0) {
    cfg.face_rec_stable_cooldown =
        seconds_to_frames(cfg.face_rec_stable_cooldown_seconds, effective_fps);
  }
  if (cfg.face_rec_reappeared_cooldown < 0) {
    cfg.face_rec_reappeared_cooldown =
        seconds_to_frames(cfg.face_rec_reappeared_cooldown_seconds, effective_fps);
  }
  if (cfg.face_rec_reappeared_probe_frames < 0) {
    cfg.face_rec_reappeared_probe_frames =
        seconds_to_frames(cfg.face_rec_reappeared_probe_seconds, effective_fps);
  }
  if (cfg.face_rec_async_max_age < 0) {
    cfg.face_rec_async_max_age =
        seconds_to_frames(cfg.face_rec_async_max_age_seconds, effective_fps);
  }
  if (cfg.face_rec_binding_ttl < 0) {
    cfg.face_rec_binding_ttl =
        seconds_to_frames(cfg.face_rec_binding_ttl_seconds, effective_fps);
  }
  if (cfg.boundary_time_window < 0) {
    cfg.boundary_time_window =
        seconds_to_frames(cfg.boundary_time_window_seconds, effective_fps);
  }
  if (cfg.ws_port <= 0 || cfg.ws_port > 65535) {
    throw std::runtime_error("--ws-port must be in 1..65535");
  }
  if (cfg.webui_port <= 0 || cfg.webui_port > 65535) {
    throw std::runtime_error("--webui-port must be in 1..65535");
  }
  if (cfg.webui && cfg.ws_json && cfg.webui_port == cfg.ws_port &&
      (cfg.webui_host == cfg.ws_host || cfg.webui_host == "0.0.0.0" || cfg.ws_host == "0.0.0.0")) {
    throw std::runtime_error("--webui-port must differ from --ws-port when both servers are enabled");
  }
  if (cfg.webui_jpeg_quality < 1 || cfg.webui_jpeg_quality > 100) {
    throw std::runtime_error("--webui-jpeg-quality must be in 1..100");
  }
	  if (!(cfg.webui_frame_scale > 0.0) || cfg.webui_frame_scale > 1.0) {
	    throw std::runtime_error("--webui-frame-scale must be in (0, 1]");
	  }
	  if (!(cfg.webui_slot_fps > 0.0) || cfg.webui_slot_fps > 60.0) {
	    throw std::runtime_error("--webui-slot-fps must be in (0, 60]");
	  }
	  if (cfg.max_frames < 0) {
    throw std::runtime_error("--max-frames must be non-negative");
  }
  if (cfg.num_sectors <= 0) {
    throw std::runtime_error("--num-sectors must be positive");
  }
  if (cfg.target_output_limit < 0) {
    throw std::runtime_error("--target-output-limit must be non-negative");
  }
  if (cfg.track_buffer < 0) {
    throw std::runtime_error("--track-buffer must be non-negative");
  }
  if (cfg.tracker_min_hits < 1) {
    throw std::runtime_error("--tracker-min-hits must be positive");
  }
  if (cfg.coast_frames < 0) {
    throw std::runtime_error("--coast-frames must be non-negative");
  }
  if (cfg.boundary_time_window < 0) {
    throw std::runtime_error("--boundary-time-window must be non-negative");
  }
  if (cfg.display_id_reuse_max_frames < 0) {
    throw std::runtime_error("--display-id-reuse-max-frames must be non-negative");
  }
  if (cfg.display_id_reuse_fallback_min_frames < 0) {
    throw std::runtime_error("--display-id-reuse-fallback-min-frames must be non-negative");
  }
  if (cfg.display_id_public_binding_ttl < 0) {
    throw std::runtime_error("--display-id-public-binding-ttl must be non-negative");
  }
  if (cfg.face_rec_cooldown < 1) {
    throw std::runtime_error("--face-rec-cooldown must be positive");
  }
  if (cfg.face_rec_newly_bound_cooldown < 1) {
    throw std::runtime_error("--face-rec-newly-bound-cooldown must be positive");
  }
  if (cfg.face_rec_stable_cooldown < 1) {
    throw std::runtime_error("--face-rec-stable-cooldown must be positive");
  }
  if (cfg.face_rec_reappeared_cooldown < 1) {
    throw std::runtime_error("--face-rec-reappeared-cooldown must be positive");
  }
  if (cfg.face_rec_reappeared_probe_frames < 1) {
    throw std::runtime_error("--face-rec-reappeared-probe-frames must be positive");
  }
  if (cfg.face_rec_max_per_frame < 0) {
    throw std::runtime_error("--face-rec-max-per-frame must be non-negative");
  }
  if (cfg.face_rec_max_per_frame == 0) {
    cfg.face_rec_max_per_frame = 3;
  }
  if (cfg.face_rec_async_max_pending < 1) {
    throw std::runtime_error("--face-rec-async-max-pending must be positive");
  }
  if (cfg.face_rec_async_max_age < 1) {
    throw std::runtime_error("--face-rec-async-max-age must be positive");
  }
  if (cfg.face_rec_binding_ttl < 0) {
    throw std::runtime_error("--face-rec-binding-ttl must be non-negative");
  }
  if (cfg.face_rec_max_samples_per_id < 1) {
    throw std::runtime_error("--face-rec-max-samples-per-id must be positive");
  }
  if (cfg.face_rec_relink_confirm_frames < 1) {
    throw std::runtime_error("--face-rec-relink-confirm-frames must be positive");
  }
  if (cfg.face_rec_relink_max_current_samples < 1) {
    throw std::runtime_error("--face-rec-relink-max-current-samples must be positive");
  }
  if (cfg.face_rec_lock_fail_frames < 1) {
    throw std::runtime_error("--face-rec-lock-fail-frames must be positive");
  }
  if (cfg.face_rec_tentative_fail_frames < 1) {
    throw std::runtime_error("--face-rec-tentative-fail-frames must be positive");
  }
  cfg.system_load_interval_ms = std::max(50, cfg.system_load_interval_ms);
  cfg.wide_angle_yolo_workers = std::max(1, std::min(cfg.wide_angle_yolo_workers, 2));
  if (cfg.fit_degree != 4 && cfg.fit_degree != 5) {
    throw std::runtime_error("--fit-degree must be 4 or 5");
  }
  cfg.smooth_bbox_alpha = std::max(0.0f, std::min(cfg.smooth_bbox_alpha, 0.99f));
  cfg.lost_velocity_decay = std::max(0.0f, std::min(cfg.lost_velocity_decay, 1.0f));
  cfg.inherit_center_dist_thresh =
      std::max(0.0f, std::min(cfg.inherit_center_dist_thresh, 5.0f));
  cfg.inherit_multi_target_center_dist_thresh =
      std::max(0.0f, std::min(cfg.inherit_multi_target_center_dist_thresh, 5.0f));
  cfg.inherit_size_ratio_thresh =
      std::max(0.0f, std::min(cfg.inherit_size_ratio_thresh, 1.0f));
  cfg.inherit_ambiguity_margin =
      std::max(0.0f, std::min(cfg.inherit_ambiguity_margin, 5.0f));
  cfg.public_recover_center_dist_thresh =
      std::max(0.0f, std::min(cfg.public_recover_center_dist_thresh, 5.0f));
  cfg.public_recover_max_frames = std::max(0, std::min(cfg.public_recover_max_frames, 10000));
  cfg.public_alias_confirm_frames = std::max(1, std::min(cfg.public_alias_confirm_frames, 30));
  cfg.boundary_time_window = std::max(0, std::min(cfg.boundary_time_window, 10000));
  cfg.boundary_margin = std::max(0.01f, std::min(cfg.boundary_margin, 0.4f));
  cfg.max_width_ratio = std::max(0.05f, std::min(cfg.max_width_ratio, 1.0f));
  cfg.min_box_width = std::max(0.0f, std::min(cfg.min_box_width, 200.0f));
  cfg.min_box_height = std::max(0.0f, std::min(cfg.min_box_height, 200.0f));
  cfg.min_box_aspect_ratio = std::max(0.0f, std::min(cfg.min_box_aspect_ratio, 1.0f));
  cfg.new_track_center_suppress_thresh =
      std::max(0.0f, std::min(cfg.new_track_center_suppress_thresh, 5.0f));
  cfg.new_track_suppress_max_side =
      std::max(0.0f, std::min(cfg.new_track_suppress_max_side, 300.0f));
  cfg.byte_residual_size_ratio_thresh =
      std::max(0.0f, std::min(cfg.byte_residual_size_ratio_thresh, 1.0f));
  cfg.byte_residual_max_side =
      std::max(0.0f, std::min(cfg.byte_residual_max_side, 300.0f));
  cfg.lost_track_reconnect_center_thresh =
      std::max(0.0f, std::min(cfg.lost_track_reconnect_center_thresh, 5.0f));
  cfg.lost_track_reconnect_multi_target_center_thresh =
      std::max(0.0f, std::min(cfg.lost_track_reconnect_multi_target_center_thresh, 5.0f));
  cfg.lost_track_reconnect_size_ratio_thresh =
      std::max(0.0f, std::min(cfg.lost_track_reconnect_size_ratio_thresh, 1.0f));
  cfg.lost_track_reconnect_ambiguity_margin =
      std::max(0.0f, std::min(cfg.lost_track_reconnect_ambiguity_margin, 5.0f));
  cfg.close_small_dedup_center_thresh =
      std::max(0.0f, std::min(cfg.close_small_dedup_center_thresh, 5.0f));
  cfg.close_small_dedup_center_px =
      std::max(0.0f, std::min(cfg.close_small_dedup_center_px, 300.0f));
  cfg.close_small_dedup_x_gap_px =
      std::max(0.0f, std::min(cfg.close_small_dedup_x_gap_px, 300.0f));
  cfg.close_small_dedup_y_center_px =
      std::max(0.0f, std::min(cfg.close_small_dedup_y_center_px, 300.0f));
  cfg.close_small_dedup_y_cover_thresh =
      std::max(0.0f, std::min(cfg.close_small_dedup_y_cover_thresh, 1.0f));
  cfg.close_small_public_alias_center_px =
      std::max(0.0f, std::min(cfg.close_small_public_alias_center_px, 300.0f));
  cfg.close_small_public_alias_x_gap_px =
      std::max(0.0f, std::min(cfg.close_small_public_alias_x_gap_px, 300.0f));
  cfg.close_small_public_alias_y_cover_thresh =
      std::max(0.0f, std::min(cfg.close_small_public_alias_y_cover_thresh, 1.0f));
  cfg.close_small_dedup_max_area =
      std::max(0.0f, std::min(cfg.close_small_dedup_max_area, 10000.0f));
  cfg.close_small_dedup_max_side =
      std::max(0.0f, std::min(cfg.close_small_dedup_max_side, 300.0f));
  cfg.normal_duplicate_center_thresh =
      std::max(0.0f, std::min(cfg.normal_duplicate_center_thresh, 5.0f));
  cfg.normal_duplicate_abs_center_px =
      std::max(0.0f, std::min(cfg.normal_duplicate_abs_center_px, 300.0f));
  cfg.normal_duplicate_size_ratio_thresh =
      std::max(0.0f, std::min(cfg.normal_duplicate_size_ratio_thresh, 1.0f));
  cfg.normal_duplicate_iou_thresh =
      std::max(0.0f, std::min(cfg.normal_duplicate_iou_thresh, 1.0f));
  cfg.normal_duplicate_cover_thresh =
      std::max(0.0f, std::min(cfg.normal_duplicate_cover_thresh, 1.0f));
  cfg.raw_duplicate_retire_confirm_frames =
      std::max(1, std::min(cfg.raw_duplicate_retire_confirm_frames, 30));
  cfg.raw_duplicate_retire_strong_public_seen =
      std::max(0, std::min(cfg.raw_duplicate_retire_strong_public_seen, 10000));
  cfg.raw_duplicate_retire_center_thresh =
      std::max(0.0f, std::min(cfg.raw_duplicate_retire_center_thresh, 5.0f));
  cfg.raw_duplicate_retire_iou_thresh =
      std::max(0.0f, std::min(cfg.raw_duplicate_retire_iou_thresh, 1.0f));
  cfg.raw_duplicate_retire_size_ratio_thresh =
      std::max(0.0f, std::min(cfg.raw_duplicate_retire_size_ratio_thresh, 1.0f));
  cfg.raw_duplicate_retire_y_cover_thresh =
      std::max(0.0f, std::min(cfg.raw_duplicate_retire_y_cover_thresh, 1.0f));
  cfg.raw_duplicate_retire_local_dist_thresh =
      std::max(0.0f, std::min(cfg.raw_duplicate_retire_local_dist_thresh, 10.0f));
  cfg.display_id_reuse_max_frames =
      std::max(0, std::min(cfg.display_id_reuse_max_frames, 10000));
  cfg.display_id_reuse_fallback_min_frames =
      std::max(0, std::min(cfg.display_id_reuse_fallback_min_frames, 10000));
  cfg.display_id_public_binding_ttl =
      std::max(0, std::min(cfg.display_id_public_binding_ttl, 10000));
  cfg.display_id_max_count =
      std::max(0, std::min(cfg.display_id_max_count, 1000));
  cfg.display_id_replace_confirm_frames =
      std::max(1, std::min(cfg.display_id_replace_confirm_frames, 1000));
  cfg.display_id_replace_area_ratio =
      std::max(1.0f, std::min(cfg.display_id_replace_area_ratio, 10.0f));
  cfg.display_id_min_box_size =
      std::max(0.0f, std::min(cfg.display_id_min_box_size, 1000.0f));
  cfg.display_id_reuse_center_thresh =
      std::max(0.0f, std::min(cfg.display_id_reuse_center_thresh, 10.0f));
  cfg.display_id_reuse_size_ratio_thresh =
      std::max(0.0f, std::min(cfg.display_id_reuse_size_ratio_thresh, 1.0f));
  cfg.display_id_reuse_ambiguity_margin =
      std::max(0.0f, std::min(cfg.display_id_reuse_ambiguity_margin, 5.0f));
  cfg.face_rec_min_box_size =
      std::max(0.0f, std::min(cfg.face_rec_min_box_size, 1000.0f));
  cfg.face_rec_min_keypoint_conf =
      std::max(0.0f, std::min(cfg.face_rec_min_keypoint_conf, 1.0f));
  cfg.face_rec_min_eye_dist =
      std::max(0.0f, std::min(cfg.face_rec_min_eye_dist, 500.0f));
  cfg.face_rec_threshold =
      std::max(-1.0f, std::min(cfg.face_rec_threshold, 1.0f));
  cfg.face_rec_margin =
      std::max(0.0f, std::min(cfg.face_rec_margin, 1.0f));
  cfg.face_rec_relink_threshold =
      std::max(-1.0f, std::min(cfg.face_rec_relink_threshold, 1.0f));
  cfg.face_rec_relink_margin =
      std::max(0.0f, std::min(cfg.face_rec_relink_margin, 1.0f));
  cfg.face_rec_lock_min_similarity =
      std::max(-1.0f, std::min(cfg.face_rec_lock_min_similarity, 1.0f));
  cfg.face_rec_tentative_threshold =
      std::max(-1.0f, std::min(cfg.face_rec_tentative_threshold, 1.0f));
  cfg.face_rec_locked_sample_similarity =
      std::max(-1.0f, std::min(cfg.face_rec_locked_sample_similarity, 1.0f));
  cfg.face_rec_pose_sample_similarity =
      std::max(-1.0f, std::min(cfg.face_rec_pose_sample_similarity, 1.0f));
  cfg.face_rec_pose_library_max_similarity =
      std::max(-1.0f, std::min(cfg.face_rec_pose_library_max_similarity, 1.0f));
  cfg.face_rec_update_similarity =
      std::max(-1.0f, std::min(cfg.face_rec_update_similarity, 1.0f));
  cfg.face_rec_update_max_similarity =
      std::max(-1.0f, std::min(cfg.face_rec_update_max_similarity, 1.0f));
  cfg.face_rec_pose_library_max_similarity =
      std::min(cfg.face_rec_pose_library_max_similarity, cfg.face_rec_update_max_similarity);
  cfg.face_rec_min_sample_diversity =
      std::max(0.0f, std::min(cfg.face_rec_min_sample_diversity, 1.0f));
  cfg.face_rec_pose_sample_interval =
      std::max(0, std::min(cfg.face_rec_pose_sample_interval, 10000));
  cfg.face_rec_pending_seed_samples =
      std::max(0, std::min(cfg.face_rec_pending_seed_samples, 100));
  cfg.face_rec_update_min_box_size =
      std::max(0.0f, std::min(cfg.face_rec_update_min_box_size, 1000.0f));
  cfg.face_rec_update_min_keypoint_conf =
      std::max(0.0f, std::min(cfg.face_rec_update_min_keypoint_conf, 1.0f));
  cfg.face_rec_update_min_score =
      std::max(0.0f, std::min(cfg.face_rec_update_min_score, 1.0f));
  cfg.face_rec_primary_min_box_size =
      std::max(cfg.face_rec_update_min_box_size,
               std::min(cfg.face_rec_primary_min_box_size, 1000.0f));
  cfg.face_rec_primary_min_keypoint_conf =
      std::max(cfg.face_rec_update_min_keypoint_conf,
               std::min(cfg.face_rec_primary_min_keypoint_conf, 1.0f));
  cfg.face_rec_primary_min_score =
      std::max(cfg.face_rec_update_min_score,
               std::min(cfg.face_rec_primary_min_score, 1.0f));
  cfg.face_rec_primary_max_yaw_deg =
      std::max(0.0f, std::min(cfg.face_rec_primary_max_yaw_deg, 90.0f));
  cfg.face_rec_primary_ema_alpha =
      std::max(0.0f, std::min(cfg.face_rec_primary_ema_alpha, 1.0f));
  cfg.face_rec_five_point_scale =
      std::max(0.70f, std::min(cfg.face_rec_five_point_scale, 1.60f));
  cfg.known_face_feature_update_threshold =
      std::max(-1.0f, std::min(cfg.known_face_feature_update_threshold, 1.0f));
  cfg.known_face_feature_pose_threshold =
      std::max(-1.0f, std::min(cfg.known_face_feature_pose_threshold, 1.0f));
  cfg.known_face_feature_update_margin =
      std::max(0.0f, std::min(cfg.known_face_feature_update_margin, 1.0f));
  cfg.known_face_feature_primary_ema_alpha =
      std::max(0.0f, std::min(cfg.known_face_feature_primary_ema_alpha, 1.0f));
  cfg.known_face_feature_max_samples =
      std::max(0, std::min(cfg.known_face_feature_max_samples, 100));
  cfg.face_rec_max_per_frame = std::min(cfg.face_rec_max_per_frame, 3);
  cfg.face_rec_newly_bound_cooldown =
      std::max(1, std::min(cfg.face_rec_newly_bound_cooldown, 10000));
  cfg.face_rec_stable_cooldown =
      std::max(1, std::min(cfg.face_rec_stable_cooldown, 10000));
  cfg.face_rec_reappeared_cooldown =
      std::max(1, std::min(cfg.face_rec_reappeared_cooldown, 10000));
  cfg.face_rec_reappeared_probe_frames =
      std::max(1, std::min(cfg.face_rec_reappeared_probe_frames, 10000));
  cfg.face_rec_async_max_pending =
      std::max(1, std::min(cfg.face_rec_async_max_pending, 100));
  cfg.face_rec_async_max_age =
      std::max(1, std::min(cfg.face_rec_async_max_age, 10000));
  cfg.face_rec_binding_ttl = std::min(cfg.face_rec_binding_ttl, 10000);
  cfg.face_rec_max_samples_per_id = std::min(cfg.face_rec_max_samples_per_id, 100);
  cfg.face_rec_relink_confirm_frames =
      std::min(cfg.face_rec_relink_confirm_frames, 100);
  cfg.face_rec_relink_max_current_samples =
      std::min(cfg.face_rec_relink_max_current_samples, 100);
  cfg.face_rec_lock_fail_frames =
      std::min(cfg.face_rec_lock_fail_frames, 100);
  cfg.face_rec_tentative_fail_frames =
      std::min(cfg.face_rec_tentative_fail_frames, 100);
  cfg.face_rec_dynamic_enroll_confirm_frames =
      std::max(1, std::min(cfg.face_rec_dynamic_enroll_confirm_frames, 100));
  cfg.face_rec_newly_bound_checks =
      std::max(1, std::min(cfg.face_rec_newly_bound_checks, 100));
  cfg.final_boundary_dedup_iou_thresh =
      std::max(0.0f, std::min(cfg.final_boundary_dedup_iou_thresh, 1.0f));
  cfg.final_boundary_dedup_center_thresh =
      std::max(0.05f, std::min(cfg.final_boundary_dedup_center_thresh, 5.0f));
  cfg.final_boundary_dedup_size_ratio_thresh =
      std::max(0.0f, std::min(cfg.final_boundary_dedup_size_ratio_thresh, 1.0f));
  return cfg;
}
