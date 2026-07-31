enum ProfileSlot {
  kProfileDecode = 0,
  kProfileFileRead,
  kProfileJpegDecode,
  kProfileDecodeWait,
  kProfileCameraRead,
  kProfileOpenclEnsure,
  kProfileRknnInputAlloc,
  kProfileOpenclRunOuter,
  kProfileRknnTotalOuter,
  kProfileBoundPrepare,
  kProfileBoundImport,
  kProfileStagingCopy,
  kProfileTrackerOuter,
  kProfileFaceRec,
  kProfileAngleTargets,
  kProfileBuildPayload,
  kProfileStdout,
  kProfileJsonl,
  kProfileWebSocket,
  kProfileDebugJsonl,
  kProfileWriteOutputs,
  kProfileFrameTotal,
  kProfileCount,
};

struct Config {
  std::vector<std::string> image_paths;
  std::string image_list_path;
  std::string output_jsonl_path;
  std::string debug_jsonl_path;
  std::string calib_yaml = "fisheye_calib.yaml";
  std::string camera_device;
  std::string map_dir = "map_export";
  std::string projection_mode = "fisheye_panorama";
  std::string model_path =
      "models/yolov8n-face-640-b1-int8-hybrid-split-kptconf-rk3588.rknn";
  std::string ws_host = "0.0.0.0";
  std::string ws_path = "/ws/inference";
  std::string webui_host = "0.0.0.0";
  std::string face_rec_model_path = "models/adaface_ir18_112.rknn";
  std::string face_rec_library_dir = "face_library";
  std::string face_rec_dynamic_library_dir = "face_library_dynamic_cpp";
  std::string face_rec_debug_crop_dir = "board_output/face_rec_crops";
  std::string face_rec_align_mode = "eye";
  bool no_stdout_json = false;
  bool stdout_debug_json = false;
  bool no_output_jsonl = false;
  bool ws_json = true;
  bool webui = false;
  bool json_debug_keypoints = false;
  bool sector_output = false;
  bool display_id_reuse = true;
  bool target_output_compact_ids = true;
  bool face_rec_enabled = false;
  bool face_rec_dynamic_library = false;
  bool face_rec_debug_crops = false;
  bool face_rec_async = false;
  bool face_rec_relink = true;
  bool face_rec_lock_verify = true;
  bool face_rec_display_tentative = true;
  bool display_id_reuse_spatial = true;
  bool print_profile_summary = false;
  bool profile_system_load = false;
  bool decode_prefetch = true;
  bool camera_prefetch = true;
  bool bound_input = true;
  bool staging_copy_input = true;
  bool staging_pipeline = true;
  bool angle_vectorized = false;
  bool force_build = false;
  bool wrap_boundary_geometry = true;
  int camera_width = 1920;
  int camera_height = 1080;
  int camera_fps = 30;
  double input_fps = 0.0;
  int camera_buffers = 4;
  int camera_timeout_ms = 3000;
  int max_frames = 0;
  int ws_port = 8001;
  int webui_port = 8080;
  int webui_jpeg_quality = 80;
  double webui_frame_scale = 1.0;
  int num_sectors = 8;
  int target_output_limit = 8;
  int display_id_reuse_max_frames = -1;
  int display_id_reuse_fallback_min_frames = -1;
  int display_id_public_binding_ttl = -1;
  int display_id_max_count = 8;
  int display_id_replace_confirm_frames = 3;
  float display_id_replace_area_ratio = 1.10f;
  int face_rec_cooldown = -1;
  int face_rec_newly_bound_cooldown = -1;
  int face_rec_stable_cooldown = -1;
  int face_rec_reappeared_cooldown = -1;
  int face_rec_reappeared_probe_frames = -1;
  int face_rec_max_per_frame = 3;
  int face_rec_async_max_pending = 8;
  int face_rec_async_max_age = -1;
  int face_rec_binding_ttl = -1;
  int face_rec_max_samples_per_id = 12;
  int face_rec_relink_confirm_frames = 2;
  int face_rec_relink_max_current_samples = 2;
  int face_rec_lock_fail_frames = 3;
  int face_rec_tentative_fail_frames = 2;
  int face_rec_dynamic_enroll_confirm_frames = 3;
  int face_rec_newly_bound_checks = 3;
  int profile_interval = 30;
  int system_load_interval_ms = 200;
  int wide_angle_yolo_workers = 2;
  int fit_degree = 4;
  float conf_threshold = 0.2f;
  float decode_iou_threshold = 0.99f;
  float merge_iou_threshold = 0.2f;
  float nms_iou_threshold = 0.6f;
  float overlap_ratio = 0.1f;
  int max_det = kDefaultMaxDet;
  int max_nms = kDefaultMaxNms;
  int max_output_dets = kDefaultMaxOutputDets;
  bool tracker_enabled = true;
  bool tracker_byte = true;
  bool smooth_bbox = true;
  bool kalman_bbox = false;
  bool coast_hold = true;
  bool boundary_recover = true;
  bool filter_invalid_boxes = true;
  int track_buffer = -1;
  int coast_frames = -1;
  int tracker_frame_rate = 30;
  int tracker_delta_t = 3;
  int tracker_min_hits = 3;
  float tracker_high_thresh = 0.5f;
  float tracker_low_thresh = 0.1f;
  float tracker_match_thresh = 0.12f;
  float tracker_new_thresh = 0.7f;
  float new_track_overlap_thresh = 0.4f;
  float new_track_center_suppress_thresh = 1.0f;
  float new_track_suppress_max_side = 80.0f;
  float byte_residual_size_ratio_thresh = 0.45f;
  float byte_residual_max_side = 40.0f;
  float lost_track_reconnect_center_thresh = 1.6f;
  float lost_track_reconnect_multi_target_center_thresh = 1.2f;
  float lost_track_reconnect_size_ratio_thresh = 0.5f;
  float lost_track_reconnect_ambiguity_margin = 0.35f;
  float tracker_inertia = 0.1f;
  float lost_velocity_decay = 0.85f;
  float smooth_bbox_alpha = 0.6f;
  float max_width_ratio = 0.6f;
  float min_box_width = 12.0f;
  float min_box_height = 12.0f;
  float min_box_aspect_ratio = 0.35f;
  bool tracker_input_small_dedup = true;
  bool close_small_dedup = true;
  bool raw_duplicate_retire = true;
  float close_small_dedup_center_thresh = 2.00f;
  float close_small_dedup_center_px = 32.0f;
  float close_small_dedup_x_gap_px = 8.0f;
  float close_small_dedup_y_center_px = 12.0f;
  float close_small_dedup_y_cover_thresh = 0.65f;
  float close_small_public_alias_center_px = 20.0f;
  float close_small_public_alias_x_gap_px = 4.0f;
  float close_small_public_alias_y_cover_thresh = 0.85f;
  float close_small_dedup_max_area = 900.0f;
  float close_small_dedup_max_side = 30.0f;
  bool normal_duplicate_dedup = true;
  float normal_duplicate_center_thresh = 0.25f;
  float normal_duplicate_abs_center_px = 16.0f;
  float normal_duplicate_size_ratio_thresh = 0.70f;
  float normal_duplicate_iou_thresh = 0.45f;
  float normal_duplicate_cover_thresh = 0.75f;
  int raw_duplicate_retire_confirm_frames = 5;
  int raw_duplicate_retire_strong_public_seen = 10;
  float raw_duplicate_retire_center_thresh = 0.45f;
  float raw_duplicate_retire_iou_thresh = 0.55f;
  float raw_duplicate_retire_size_ratio_thresh = 0.65f;
  float raw_duplicate_retire_y_cover_thresh = 0.75f;
  float raw_duplicate_retire_local_dist_thresh = 1.5f;
  float display_id_reuse_center_thresh = 2.0f;
  float display_id_reuse_size_ratio_thresh = 0.4f;
  float display_id_reuse_ambiguity_margin = 0.25f;
  float face_rec_min_box_size = 100.0f;
  float face_rec_min_keypoint_conf = 0.6f;
  float face_rec_min_eye_dist = 0.0f;
  float face_rec_threshold = 0.55f;
  float face_rec_margin = 0.08f;
  float face_rec_relink_threshold = 0.55f;
  float face_rec_relink_margin = 0.08f;
  float face_rec_lock_min_similarity = 0.25f;
  float face_rec_tentative_threshold = 0.50f;
  float face_rec_locked_sample_similarity = 0.35f;
  float face_rec_pose_sample_similarity = 0.55f;
  float face_rec_pose_library_max_similarity = 0.90f;
  float face_rec_update_similarity = 0.65f;
  float face_rec_update_max_similarity = 0.90f;
  float face_rec_min_sample_diversity = 0.015f;
  int face_rec_pose_sample_interval = 30;
  int face_rec_pending_seed_samples = 2;
  float face_rec_update_min_box_size = 100.0f;
  float face_rec_update_min_keypoint_conf = 0.7f;
  float face_rec_update_min_score = 0.5f;
  float face_rec_primary_min_box_size = 110.0f;
  float face_rec_primary_min_keypoint_conf = 0.8f;
  float face_rec_primary_min_score = 0.6f;
  float face_rec_primary_max_yaw_deg = 10.0f;
  float face_rec_primary_ema_alpha = 0.1f;
  float face_rec_five_point_scale = 1.20f;
  float inherit_center_dist_thresh = 2.0f;
  float inherit_multi_target_center_dist_thresh = 1.2f;
  float inherit_size_ratio_thresh = 0.4f;
  float inherit_ambiguity_margin = 0.25f;
  float public_recover_center_dist_thresh = 2.0f;
  int public_recover_max_frames = -1;
  int public_alias_confirm_frames = 6;
  float boundary_margin = 0.04f;
  float boundary_size_ratio_thresh = 0.45f;
  float boundary_center_dist_thresh = 1.8f;
  int boundary_time_window = -1;
  bool final_boundary_dedup = true;
  float final_boundary_dedup_iou_thresh = 0.70f;
  float final_boundary_dedup_center_thresh = 0.80f;
  float final_boundary_dedup_size_ratio_thresh = 0.25f;
  double track_buffer_seconds = 30.0;
  double coast_seconds = 0.3;
  double public_recover_max_seconds = 30.0;
  double display_id_reuse_max_seconds = 30.0;
  double display_id_reuse_fallback_min_seconds = 5.0;
  double display_id_public_binding_ttl_seconds = 2.0;
  double face_rec_cooldown_seconds = 0.333333;
  double face_rec_newly_bound_cooldown_seconds = 1.0;
  double face_rec_stable_cooldown_seconds = 5.0;
  double face_rec_reappeared_cooldown_seconds = 0.333333;
  double face_rec_reappeared_probe_seconds = 1.0;
  double face_rec_async_max_age_seconds = 1.0;
  double face_rec_binding_ttl_seconds = 30.0;
  double boundary_time_window_seconds = 3.0;
};

struct SliceMeta {
  int start_x = 0;
  int slice_width = 0;
  int slice_height = 0;
  int original_width = 0;
  int original_height = 0;
  int wrap_around = 0;
  float gain = 1.0f;
  int left = 0;
  int top = 0;
  int new_width = 0;
  int new_height = 0;
};

struct MapMeta {
  std::string projection_mode = "fisheye_panorama";
  int num_slices = 0;
  int roi_h = 0;
  int roi_w = 0;
  int imgsz = 0;
  int process_width = 0;
  int process_height = 0;
  int img_width = 0;
  int img_height = 0;
  int center_x = 0;
  int center_y = 0;
  int radius = 0;
  int base_map_width = 0;
  int base_map_height = 0;
  double wide_fx = 0.0;
  double wide_fy = 0.0;
  double wide_cx = 0.0;
  double wide_cy = 0.0;
  std::vector<SliceMeta> slices;
};

struct Image {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> bgr;
};

struct TrackerSnapshotDebug {
  int raw_id = -1;
  int public_id = 0;
  int time_since_update = 0;
  std::array<float, 5> last_observation{{-1.0f, -1.0f, -1.0f, -1.0f, -1.0f}};
  std::array<float, 4> state{{0.0f, 0.0f, 0.0f, 0.0f}};
  bool finite_last_observation = false;
  bool lost_confirmed = false;
  std::string reject_reason;
};

struct ShortInheritCandidateDebug {
  int old_raw_id = -1;
  int old_public_id = 0;
  std::array<float, 4> bbox{{0.0f, 0.0f, 0.0f, 0.0f}};
  float dist = 0.0f;
  float size_score = 0.0f;
  bool used_state = false;
  std::string reject_reason;
};

struct ShortInheritDebug {
  int raw_id = -1;
  std::array<float, 4> bbox{{0.0f, 0.0f, 0.0f, 0.0f}};
  int local_target_count = 0;
  bool local_multi_target = false;
  float effective_center_dist_thresh = 0.0f;
  int chosen_old_raw_id = -1;
  int chosen_public_id = 0;
  std::string result;
  std::vector<ShortInheritCandidateDebug> candidates;
};

struct PublicRecoverCandidateDebug {
  int public_id = 0;
  int missing_frames = 0;
  std::array<float, 4> last_bbox{{0.0f, 0.0f, 0.0f, 0.0f}};
  float dist = 0.0f;
  float size_score = 0.0f;
  bool strict_duplicate = false;
  bool strict_ok = false;
  bool relaxed_ok = false;
  std::string reject_reason;
};

struct PublicRecoverDebug {
  int raw_id = -1;
  std::array<float, 4> bbox{{0.0f, 0.0f, 0.0f, 0.0f}};
  int local_target_count = 0;
  bool local_single_target = false;
  int recovery_window = 0;
  int chosen_public_id = 0;
  std::string mode;
  std::string result;
  std::vector<PublicRecoverCandidateDebug> candidates;
};

struct DisplayBindingDebug {
  int public_id = 0;
  int display_id = 0;
  int last_frame = -1;
  int missing_frames = -1;
  bool active_public = false;
  bool ttl_reserved = false;
};

struct DisplayReleaseDebug {
  int public_id = 0;
  int display_id = 0;
  int frame_id = -1;
  int last_frame = -1;
  int missing_frames = -1;
  std::string reason;
};

struct DisplayReuseCandidateDebug {
  int display_id = 0;
  int previous_public_id = 0;
  int missing_frames = -1;
  float dist = 0.0f;
  float size_score = 0.0f;
  bool spatial = false;
  std::string reject_reason;
};

struct DisplayAssignDebug {
  int original_index = -1;
  int public_id = 0;
  int raw_track_id = -1;
  int assigned_display_id = 0;
  int previous_display_id = 0;
  int reused_from_public_id = 0;
  int replaced_public_id = 0;
  int replaced_display_id = 0;
  int replaced_original_index = -1;
  int public_consecutive_seen = 0;
  int replaced_public_consecutive_seen = 0;
  int replace_pending_hits = 0;
  int replace_confirm_frames = 0;
  int display_counter = 0;
  int display_max_count = 0;
  float bbox_area = 0.0f;
  float replaced_bbox_area = 0.0f;
  bool had_existing_binding = false;
  bool reuse_spatial = false;
  bool wait_for_free_display = false;
  bool hidden = false;
  std::string action;
  std::string reason;
  std::vector<DisplayReuseCandidateDebug> reuse_candidates;
};

struct FrameResult {
  int frame_index = 0;
  std::string image_path;
  int frame_width = 0;
  int frame_height = 0;
  double remap_timings[4] = {0.0, 0.0, 0.0, 0.0};
  double rknn_timings[24] = {0.0};
  double tracker_timings[16] = {0.0};
  double profile_ms[kProfileCount] = {0.0};
  int merge_stats[8] = {0};
  int tracker_stats[16] = {0};
  std::vector<float> detections;
  std::vector<int> raw_track_ids;
  std::vector<int> track_ids;
  std::vector<int> display_ids;
  std::vector<int> display_reuse_from_public_ids;
  std::vector<uint8_t> display_reuse_spatial;
  int raw_detection_count = 0;
  int detection_count = 0;
  int filtered_invalid = 0;
  int inherited_ids = 0;
  int boundary_recovered = 0;
  int coasting_added = 0;
  int final_dedup_removed = 0;
  int close_small_dedup_removed = 0;
  int normal_duplicate_dedup_removed = 0;
  int raw_duplicate_retired = 0;
  bool tracker_enabled = false;
  std::vector<TrackerSnapshotDebug> tracker_snapshots_debug;
  std::vector<ShortInheritDebug> short_inherit_debug;
  std::vector<PublicRecoverDebug> public_recover_debug;
  std::vector<DisplayBindingDebug> display_bindings_debug;
  std::vector<DisplayReleaseDebug> display_release_debug;
  std::vector<int> display_free_ids_debug;
  std::vector<DisplayAssignDebug> display_assign_debug;
};

struct PreparedStagingFrame {
  bool valid = false;
  Image image;
  std::string image_path;
  int frame_index = 0;
  double frame_start_s = 0.0;
  double file_read_ms = 0.0;
  double camera_read_ms = 0.0;
  double jpeg_decode_ms = 0.0;
  double decode_wait_ms = 0.0;
  float original_width = 0.0f;
  std::vector<uint8_t> rknn_inputs;
  FrameResult result;
};

struct FrameRateStats {
  double frame_ms = 0.0;
  double instant_fps = 0.0;
  double average_fps = 0.0;
  double elapsed_s = 0.0;
  int frames = 0;
};
