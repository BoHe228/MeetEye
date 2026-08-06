class OpenCLHandle {
 public:
  ~OpenCLHandle() {
    if (handle_) {
      ds_opencl_fused_destroy(handle_);
    }
  }

  void** out() { return &handle_; }
  void* get() const { return handle_; }
  void reset() {
    if (handle_) {
      ds_opencl_fused_destroy(handle_);
      handle_ = nullptr;
    }
  }

 private:
  void* handle_ = nullptr;
};

class RknnHandle {
 public:
  ~RknnHandle() {
    if (handle_) {
      face_rknn_parallel_destroy(handle_);
    }
  }

  void** out() { return &handle_; }
  void* get() const { return handle_; }

 private:
  void* handle_ = nullptr;
};

class HybridSortHandle {
 public:
  ~HybridSortHandle() {
    if (handle_) {
      hybrid_sort_native_destroy(handle_);
    }
  }

  void** out() { return &handle_; }
  void* get() const { return handle_; }

 private:
  void* handle_ = nullptr;
};

struct IouPair {
  int track_index = -1;
  int det_index = -1;
  float iou = 0.0f;
};

struct NativeTrackSnapshot {
  int raw_id = -1;
  int time_since_update = 0;
  std::array<float, 5> last_observation{{-1.0f, -1.0f, -1.0f, -1.0f, -1.0f}};
  std::array<float, 4> state{{0.0f, 0.0f, 0.0f, 0.0f}};
};

enum class BoundarySide {
  None = 0,
  Left,
  Right,
  Top,
  Bottom,
};

struct BoundaryLostTarget {
  int raw_id = -1;
  int frame_index = 0;
  BoundarySide side = BoundarySide::None;
  std::array<float, 4> bbox{{0.0f, 0.0f, 0.0f, 0.0f}};
  std::array<float, 2> velocity{{0.0f, 0.0f}};
  bool has_velocity = false;
};

static std::array<float, 4> bbox4_from_ptr(const float* p) {
  return std::array<float, 4>{{p[0], p[1], p[2], p[3]}};
}

static float bbox_width(const std::array<float, 4>& b) {
  return std::max(0.0f, b[2] - b[0]);
}

static float bbox_height(const std::array<float, 4>& b) {
  return std::max(0.0f, b[3] - b[1]);
}

static float bbox_area4(const std::array<float, 4>& b) {
  return bbox_width(b) * bbox_height(b);
}

static float bbox_min_side(const std::array<float, 4>& b) {
  return std::min(bbox_width(b), bbox_height(b));
}

static std::array<float, 2> bbox_center(const std::array<float, 4>& b) {
  return std::array<float, 2>{{(b[0] + b[2]) * 0.5f, (b[1] + b[3]) * 0.5f}};
}

static float center_distance_norm(const std::array<float, 4>& a, const std::array<float, 4>& b) {
  const auto ca = bbox_center(a);
  const auto cb = bbox_center(b);
  const float avg_h = std::max((bbox_height(a) + bbox_height(b)) * 0.5f, 1.0f);
  const float dx = ca[0] - cb[0];
  const float dy = ca[1] - cb[1];
  return std::sqrt(dx * dx + dy * dy) / avg_h;
}

static float size_ratio_score(const std::array<float, 4>& a, const std::array<float, 4>& b) {
  const float aw = std::max(bbox_width(a), 1.0f);
  const float ah = std::max(bbox_height(a), 1.0f);
  const float bw = std::max(bbox_width(b), 1.0f);
  const float bh = std::max(bbox_height(b), 1.0f);
  return std::min(aw, bw) / std::max(aw, bw) * std::min(ah, bh) / std::max(ah, bh);
}

static bool finite_bbox(const std::array<float, 4>& b) {
  return std::isfinite(b[0]) && std::isfinite(b[1]) && std::isfinite(b[2]) &&
         std::isfinite(b[3]) && bbox_width(b) > 0.0f && bbox_height(b) > 0.0f;
}

class NativeHybridSortTracker {
 public:
  NativeHybridSortTracker(const Config& cfg, const MapMeta& meta)
      : cfg_(cfg), meta_(meta) {
    char err[2048] = {0};
    max_age_ = std::max(0, cfg_.track_buffer);
    const int ret = hybrid_sort_native_create(
        cfg_.tracker_high_thresh,
        cfg_.tracker_low_thresh,
        max_age_,
        std::max(1, cfg_.tracker_min_hits),
        cfg_.tracker_match_thresh,
        std::max(1, cfg_.tracker_delta_t),
        cfg_.tracker_inertia,
        cfg_.tracker_byte ? 1 : 0,
        1,
        1.0f,
        1,
        1.0f,
        cfg_.tracker_new_thresh,
        cfg_.new_track_overlap_thresh,
        cfg_.lost_velocity_decay,
        cfg_.new_track_center_suppress_thresh,
        cfg_.new_track_suppress_max_side,
        cfg_.byte_residual_size_ratio_thresh,
        cfg_.byte_residual_max_side,
        cfg_.lost_track_reconnect_center_thresh,
        cfg_.lost_track_reconnect_multi_target_center_thresh,
        cfg_.lost_track_reconnect_size_ratio_thresh,
        cfg_.lost_track_reconnect_ambiguity_margin,
        handle_.out(),
        err,
        sizeof(err));
    if (ret != 0 || handle_.get() == nullptr) {
      throw std::runtime_error(std::string("hybrid_sort_native_create failed: ") + err);
    }
  }

  void update(FrameResult& result) {
    frame_id_ += 1;
    result.tracker_enabled = true;
    result.raw_detection_count = result.detection_count;
    filter_invalid_detections(result);
    apply_tracker_input_small_dedup(result);
    const int tracker_input_count = result.detection_count;

    std::vector<float> raw = result.detections;
    raw.resize(static_cast<size_t>(tracker_input_count) * kDetectionFields);

    std::vector<float> tracker_input(
        std::max<size_t>(1, static_cast<size_t>(tracker_input_count) * 5),
        0.0f);
    for (int i = 0; i < tracker_input_count; ++i) {
      const float* det = raw.data() + static_cast<size_t>(i) * kDetectionFields;
      float* row = tracker_input.data() + static_cast<size_t>(i) * 5;
      row[0] = det[0];
      row[1] = det[1];
      row[2] = det[2];
      row[3] = det[3];
      row[4] = det[4];
    }

    const int max_track_rows = std::max(1024, tracker_input_count + 128);
    std::vector<float> tracker_rows(static_cast<size_t>(max_track_rows) * 5, 0.0f);
    int tracker_count = 0;
    char err[2048] = {0};
    const float tracker_h =
        meta_.process_height > 0 ? static_cast<float>(meta_.process_height)
                                 : static_cast<float>(result.frame_height);
    const float tracker_w =
        meta_.process_width > 0 ? static_cast<float>(meta_.process_width)
                                : static_cast<float>(result.frame_width);
    const float native_wrap_h = tracker_h;
    const float native_wrap_w = cfg_.wrap_boundary_geometry ? tracker_w : 0.0f;
    const int ret = hybrid_sort_native_update(
        handle_.get(),
        tracker_input.data(),
        tracker_input_count,
        5,
        tracker_h,
        native_wrap_w,
        native_wrap_h,
        native_wrap_w,
        tracker_rows.data(),
        max_track_rows,
        &tracker_count,
        result.tracker_stats,
        16,
        result.tracker_timings,
        16,
        err,
        sizeof(err));
    if (ret != 0) {
      throw std::runtime_error(std::string("hybrid_sort_native_update failed: ") + err);
    }

    tracker_count = std::max(0, std::min(tracker_count, max_track_rows));
    std::vector<NativeTrackSnapshot> snapshots = get_track_snapshots();
    std::map<int, NativeTrackSnapshot> snapshot_by_raw;
    for (const NativeTrackSnapshot& snapshot : snapshots) {
      if (snapshot.raw_id > 0) {
        snapshot_by_raw[snapshot.raw_id] = snapshot;
      }
    }
    record_tracker_snapshot_debug(result, snapshots);

    const std::vector<int> track_to_det = match_tracks_to_detections(tracker_rows, tracker_count, raw);
    std::vector<float> tracked(static_cast<size_t>(tracker_count) * kDetectionFields, 0.0f);
    std::vector<int> raw_ids(static_cast<size_t>(tracker_count), -1);
    result.track_ids.assign(static_cast<size_t>(tracker_count), -1);
    std::vector<int> current_active_raw_ids;
    current_active_raw_ids.reserve(static_cast<size_t>(tracker_count));

    for (int ti = 0; ti < tracker_count; ++ti) {
      const float* track = tracker_rows.data() + static_cast<size_t>(ti) * 5;
      const int raw_id = std::max(1, static_cast<int>(std::lround(track[4])));
      raw_ids[static_cast<size_t>(ti)] = raw_id;
      current_active_raw_ids.push_back(raw_id);
      float* out = tracked.data() + static_cast<size_t>(ti) * kDetectionFields;

      const int det_index = track_to_det[static_cast<size_t>(ti)];
      if (det_index >= 0) {
        const float* src = raw.data() + static_cast<size_t>(det_index) * kDetectionFields;
        std::copy(src, src + kDetectionFields, out);
        meta_cache_[raw_id] = std::array<float, kDetectionFields>{};
        std::copy(src, src + kDetectionFields, meta_cache_[raw_id].begin());
      } else {
        const auto cached = meta_cache_.find(raw_id);
        if (cached != meta_cache_.end()) {
          std::copy(cached->second.begin(), cached->second.end(), out);
        } else {
          out[4] = 0.5f;
        }
      }

      std::array<float, 4> bbox{{track[0], track[1], track[2], track[3]}};
      const auto snapshot_it = snapshot_by_raw.find(raw_id);
      if (cfg_.kalman_bbox && snapshot_it != snapshot_by_raw.end() &&
          finite_bbox(snapshot_it->second.state)) {
        bbox = snapshot_it->second.state;
      }
      if (cfg_.smooth_bbox && !cfg_.kalman_bbox) {
        bbox = smooth_bbox(raw_id, bbox);
      }
      out[0] = bbox[0];
      out[1] = bbox[1];
      out[2] = bbox[2];
      out[3] = bbox[3];

      result.track_ids[static_cast<size_t>(ti)] = -1;
    }

    result.detections.swap(tracked);
    result.detection_count = tracker_count;

    retired_raw_ids_this_frame_.clear();
    const std::map<int, std::array<float, 4>> current_bboxes =
        current_bbox_by_raw_ids(result, raw_ids);
    cleanup_pending_public_aliases(current_bboxes);
    cleanup_pending_raw_duplicate_retires(current_bboxes);
    register_lost_targets(current_active_raw_ids);
    apply_short_occlusion_inheritance(result, raw_ids, snapshot_by_raw);
    apply_boundary_recovery(result, raw_ids);
    if (!cfg_.coast_hold) {
      add_coasting_tracks(result, raw_ids, snapshot_by_raw);
    }
    apply_final_boundary_dedup(result, raw_ids);
    apply_close_small_dedup(result, raw_ids);
    apply_recent_public_occlusion_recovery(result, raw_ids);
    resolve_public_id_conflicts(result, raw_ids);
    result.raw_track_ids = raw_ids;
    apply_public_ids(result, raw_ids);
    update_public_seen_counts(result);
    update_public_last_bboxes(result);
    const int count_before_public_coast = result.detection_count;
    add_public_coasting_tracks(result);
    apply_boundary_output_dedup(result, count_before_public_coast);
    apply_close_small_output_dedup(result, count_before_public_coast);
    apply_display_ids(result);
    remove_retired_raw_ids(current_active_raw_ids);
    update_active_track_state(current_active_raw_ids, snapshot_by_raw);
  }

 private:
  struct PendingPublicAlias {
    int public_raw = -1;
    int alias_raw = -1;
    int public_id = -1;
    int hits = 0;
    int last_frame = -1;
  };

  struct PendingRawDuplicateRetire {
    int keep_raw = -1;
    int retire_raw = -1;
    int hits = 0;
    int last_frame = -1;
  };

  struct PendingDisplayAreaReplacement {
    int new_public_id = 0;
    int old_public_id = 0;
    int hits = 0;
    int last_frame = -1;
  };

  struct DisplayReuseChoice {
    int display_id = 0;
    int previous_public_id = 0;
    bool spatial = false;
    bool wait_for_free_display = false;
    std::vector<DisplayReuseCandidateDebug> candidates;
  };

  struct DisplayReplacementChoice {
    int display_id = 0;
    int replaced_public_id = 0;
    int replaced_index = -1;
    int replaced_hits = 0;
    int pending_hits = 0;
    float replaced_area = 0.0f;
    bool replace = false;
    std::string reject_reason;
  };

  void filter_invalid_detections(FrameResult& result) const {
    if (!cfg_.filter_invalid_boxes || result.detection_count <= 0) {
      return;
    }
    const int process_w = meta_.process_width > 0 ? meta_.process_width : result.frame_width;
    const int process_h = meta_.process_height > 0 ? meta_.process_height : result.frame_height;
    const float boundary_threshold = process_w * 0.05f;
    const float max_allowed_width = process_w * cfg_.max_width_ratio;
    int write = 0;
    for (int i = 0; i < result.detection_count; ++i) {
      float* det = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
      float x1 = det[0];
      float y1 = det[1];
      float x2 = det[2];
      float y2 = det[3];
      if (x1 >= process_w || x2 <= 0.0f || y1 >= process_h || y2 <= 0.0f || x2 < x1) {
        continue;
      }
      x1 = std::max(0.0f, std::min(x1, static_cast<float>(process_w)));
      x2 = std::max(0.0f, std::min(x2, static_cast<float>(process_w)));
      y1 = std::max(0.0f, std::min(y1, static_cast<float>(process_h)));
      y2 = std::max(0.0f, std::min(y2, static_cast<float>(process_h)));
      const float box_w = x2 - x1;
      const float box_h = y2 - y1;
      if (box_w <= 0.0f || box_h <= 0.0f) {
        continue;
      }
      if (box_w < cfg_.min_box_width || box_h < cfg_.min_box_height) {
        continue;
      }
      const float aspect_ratio = std::min(box_w, box_h) / (std::max(box_w, box_h) + 1.0e-6f);
      if (aspect_ratio < cfg_.min_box_aspect_ratio) {
        continue;
      }
      if (box_w > max_allowed_width || (x1 < boundary_threshold && x2 > process_w - boundary_threshold)) {
        continue;
      }
      det[0] = x1;
      det[1] = y1;
      det[2] = x2;
      det[3] = y2;
      if (write != i) {
        float* dst = result.detections.data() + static_cast<size_t>(write) * kDetectionFields;
        std::copy(det, det + kDetectionFields, dst);
      }
      write += 1;
    }
    result.filtered_invalid = result.detection_count - write;
    result.detection_count = write;
  }

  void apply_tracker_input_small_dedup(FrameResult& result) {
    if (!cfg_.tracker_input_small_dedup ||
        (!cfg_.close_small_dedup && !cfg_.normal_duplicate_dedup) ||
        result.detection_count <= 1) {
      return;
    }

    const int n = result.detection_count;
    std::vector<uint8_t> keep(static_cast<size_t>(n), 1);
    int removed = 0;
    for (int i = 0; i < n; ++i) {
      if (!keep[static_cast<size_t>(i)]) {
        continue;
      }
      const std::array<float, 4> a =
          bbox4_from_ptr(result.detections.data() + static_cast<size_t>(i) * kDetectionFields);
      for (int j = i + 1; j < n; ++j) {
        if (!keep[static_cast<size_t>(j)]) {
          continue;
        }
        const std::array<float, 4> b =
            bbox4_from_ptr(result.detections.data() + static_cast<size_t>(j) * kDetectionFields);
        const bool close_small_duplicate = is_close_small_duplicate(a, b);
        const bool normal_duplicate =
            !close_small_duplicate && is_normal_close_duplicate(a, b);
        if (!close_small_duplicate && !normal_duplicate) {
          continue;
        }
        const int drop = choose_tracker_input_small_duplicate_drop(result, i, j);
        keep[static_cast<size_t>(drop)] = 0;
        if (close_small_duplicate) {
          result.close_small_dedup_removed += 1;
        } else {
          result.normal_duplicate_dedup_removed += 1;
        }
        removed += 1;
        if (drop == i) {
          break;
        }
      }
    }
    if (removed <= 0) {
      return;
    }
    compact_result_with_track_ids(result, keep);
  }

  int choose_tracker_input_small_duplicate_drop(const FrameResult& result,
                                                int i,
                                                int j) const {
    const float* det_i = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
    const float* det_j = result.detections.data() + static_cast<size_t>(j) * kDetectionFields;
    if (std::fabs(det_i[4] - det_j[4]) > 1.0e-6f) {
      return det_i[4] >= det_j[4] ? j : i;
    }

    const std::array<float, 4> box_i = bbox4_from_ptr(det_i);
    const std::array<float, 4> box_j = bbox4_from_ptr(det_j);
    return bbox_area4(box_i) >= bbox_area4(box_j) ? j : i;
  }

  std::vector<int> match_tracks_to_detections(const std::vector<float>& tracker_rows,
                                              int tracker_count,
                                              const std::vector<float>& raw) const {
    const int det_count = static_cast<int>(raw.size() / kDetectionFields);
    std::vector<IouPair> pairs;
    pairs.reserve(static_cast<size_t>(tracker_count) * std::max(1, det_count));
    for (int ti = 0; ti < tracker_count; ++ti) {
      const float* track = tracker_rows.data() + static_cast<size_t>(ti) * 5;
      for (int di = 0; di < det_count; ++di) {
        const float* det = raw.data() + static_cast<size_t>(di) * kDetectionFields;
        const float iou = box_iou(track, det);
        if (iou > 0.3f) {
          IouPair pair;
          pair.track_index = ti;
          pair.det_index = di;
          pair.iou = iou;
          pairs.push_back(pair);
        }
      }
    }
    std::sort(pairs.begin(), pairs.end(), [](const IouPair& a, const IouPair& b) {
      return a.iou > b.iou;
    });

    std::vector<int> track_to_det(static_cast<size_t>(tracker_count), -1);
    std::vector<uint8_t> used_track(static_cast<size_t>(tracker_count), 0);
    std::vector<uint8_t> used_det(static_cast<size_t>(det_count), 0);
    for (const IouPair& pair : pairs) {
      if (used_track[static_cast<size_t>(pair.track_index)] ||
          used_det[static_cast<size_t>(pair.det_index)]) {
        continue;
      }
      track_to_det[static_cast<size_t>(pair.track_index)] = pair.det_index;
      used_track[static_cast<size_t>(pair.track_index)] = 1;
      used_det[static_cast<size_t>(pair.det_index)] = 1;
    }
    return track_to_det;
  }

  std::vector<NativeTrackSnapshot> get_track_snapshots() const {
    const int max_rows = 1024;
    std::vector<float> rows(static_cast<size_t>(max_rows) * 11, 0.0f);
    int count = 0;
    char err[2048] = {0};
    const int ret = hybrid_sort_native_get_tracks(
        handle_.get(),
        rows.data(),
        max_rows,
        &count,
        err,
        sizeof(err));
    if (ret != 0) {
      throw std::runtime_error(std::string("hybrid_sort_native_get_tracks failed: ") + err);
    }
    count = std::max(0, std::min(count, max_rows));
    std::vector<NativeTrackSnapshot> snapshots;
    snapshots.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
      const float* row = rows.data() + static_cast<size_t>(i) * 11;
      NativeTrackSnapshot snapshot;
      snapshot.raw_id = static_cast<int>(std::lround(row[0]));
      snapshot.time_since_update = static_cast<int>(std::lround(row[1]));
      for (int c = 0; c < 5; ++c) {
        snapshot.last_observation[static_cast<size_t>(c)] = row[2 + c];
      }
      for (int c = 0; c < 4; ++c) {
        snapshot.state[static_cast<size_t>(c)] = row[7 + c];
      }
      if (snapshot.raw_id >= 0) {
        snapshot.raw_id += 1;
      }
      snapshots.push_back(snapshot);
    }
    return snapshots;
  }

  void record_tracker_snapshot_debug(FrameResult& result,
                                     const std::vector<NativeTrackSnapshot>& snapshots) const {
    result.tracker_snapshots_debug.clear();
    result.tracker_snapshots_debug.reserve(snapshots.size());
    for (const NativeTrackSnapshot& snapshot : snapshots) {
      TrackerSnapshotDebug debug;
      debug.raw_id = snapshot.raw_id;
      debug.time_since_update = snapshot.time_since_update;
      debug.last_observation = snapshot.last_observation;
      debug.state = snapshot.state;
      debug.finite_last_observation = finite_bbox(bbox_from_last_observation(snapshot));
      const auto public_it = public_id_map_.find(snapshot.raw_id);
      debug.public_id = public_it == public_id_map_.end() ? 0 : public_it->second;
      if (snapshot.time_since_update <= 0) {
        debug.reject_reason = "active";
      } else if (snapshot.time_since_update > max_age_) {
        debug.reject_reason = "too_old";
      } else if (!debug.finite_last_observation) {
        debug.reject_reason = "invalid_last_observation";
      } else if (debug.public_id <= 0) {
        debug.reject_reason = "no_public_id";
      } else {
        debug.lost_confirmed = true;
        debug.reject_reason = "candidate";
      }
      result.tracker_snapshots_debug.push_back(debug);
    }
  }

  void apply_short_occlusion_inheritance(FrameResult& result,
                                         const std::vector<int>& raw_ids,
                                         const std::map<int, NativeTrackSnapshot>& snapshot_by_raw) {
    std::map<int, NativeTrackSnapshot> lost_confirmed;
    for (const auto& item : snapshot_by_raw) {
      const int raw_id = item.first;
      const NativeTrackSnapshot& snapshot = item.second;
      if (snapshot.time_since_update <= 0) {
        continue;
      }
      if (snapshot.time_since_update > max_age_) {
        continue;
      }
      if (!finite_bbox(bbox_from_last_observation(snapshot))) {
        continue;
      }
      lost_confirmed[raw_id] = snapshot;
    }
    if (lost_confirmed.empty()) {
      return;
    }

    for (size_t i = 0; i < raw_ids.size(); ++i) {
      const int raw_id = raw_ids[i];
      if (raw_id <= 0 || public_id_map_.find(raw_id) != public_id_map_.end()) {
        continue;
      }
      if (i >= static_cast<size_t>(result.detection_count)) {
        continue;
      }
      const float* det = result.detections.data() + i * kDetectionFields;
      const std::array<float, 4> new_bbox = bbox4_from_ptr(det);
      ShortInheritDebug debug;
      debug.raw_id = raw_id;
      debug.bbox = new_bbox;
      debug.result = "no_candidate";
      int best_old_raw = -1;
      const int local_target_count =
          count_local_current_targets(result, new_bbox, i, cfg_.inherit_center_dist_thresh);
      debug.local_target_count = local_target_count;
      debug.local_multi_target = local_target_count > 1;
      const float inherit_dist_thresh =
          debug.local_multi_target
              ? std::min(cfg_.inherit_center_dist_thresh,
                         cfg_.inherit_multi_target_center_dist_thresh)
              : cfg_.inherit_center_dist_thresh;
      debug.effective_center_dist_thresh = inherit_dist_thresh;
      float best_dist = std::numeric_limits<float>::infinity();
      float second_best_dist = std::numeric_limits<float>::infinity();
      for (const auto& item : lost_confirmed) {
        const int old_raw = item.first;
        ShortInheritCandidateDebug cand;
        cand.old_raw_id = old_raw;
        const auto public_it = public_id_map_.find(old_raw);
        cand.old_public_id = public_it == public_id_map_.end() ? 0 : public_it->second;
        if (old_raw == raw_id) {
          cand.reject_reason = "same_raw";
          debug.candidates.push_back(cand);
          continue;
        }
        if (public_it == public_id_map_.end()) {
          cand.reject_reason = "no_public_id";
          debug.candidates.push_back(cand);
          continue;
        }
        const NativeTrackSnapshot& old_track = item.second;
        const std::array<float, 4> obs_bbox = bbox_from_last_observation(old_track);
        cand.bbox = obs_bbox;
        float d = inheritance_center_distance_norm(new_bbox, obs_bbox);
        float size_score = size_ratio_score(new_bbox, obs_bbox);
        if (finite_bbox(old_track.state)) {
          const float state_dist = inheritance_center_distance_norm(new_bbox, old_track.state);
          if (state_dist < d) {
            d = state_dist;
            size_score = size_ratio_score(new_bbox, old_track.state);
            cand.bbox = old_track.state;
            cand.used_state = true;
          }
        }
        cand.dist = d;
        cand.size_score = size_score;
        if (size_score < cfg_.inherit_size_ratio_thresh) {
          cand.reject_reason = "size_gate";
          debug.candidates.push_back(cand);
          continue;
        }
        if (d >= inherit_dist_thresh) {
          cand.reject_reason =
              debug.local_multi_target && d < cfg_.inherit_center_dist_thresh
                  ? "multi_target_distance_gate"
                  : "distance_gate";
          debug.candidates.push_back(cand);
          continue;
        }
        if (d < best_dist) {
          if (std::isfinite(best_dist)) {
            second_best_dist = best_dist;
          }
          best_dist = d;
          best_old_raw = old_raw;
          cand.reject_reason = "best";
        } else if (d < second_best_dist) {
          second_best_dist = d;
          cand.reject_reason = "second_best";
        } else {
          cand.reject_reason = "distance_gate";
        }
        debug.candidates.push_back(cand);
      }
      if (best_old_raw > 0) {
        const float nearest_other_dist =
            nearest_other_current_target_distance(result, new_bbox, i, cfg_.inherit_center_dist_thresh);
        if (std::isfinite(nearest_other_dist) &&
            nearest_other_dist < cfg_.inherit_center_dist_thresh &&
            (nearest_other_dist - best_dist) < cfg_.inherit_ambiguity_margin) {
          debug.chosen_old_raw_id = best_old_raw;
          debug.chosen_public_id = public_id_map_[best_old_raw];
          debug.result = "current_target_ambiguity";
          result.short_inherit_debug.push_back(debug);
          continue;
        }
        if (std::isfinite(second_best_dist) &&
            (second_best_dist - best_dist) < cfg_.inherit_ambiguity_margin) {
          debug.chosen_old_raw_id = best_old_raw;
          debug.chosen_public_id = public_id_map_[best_old_raw];
          debug.result = "ambiguity";
          result.short_inherit_debug.push_back(debug);
          continue;
        }
        const int old_public = public_id_map_[best_old_raw];
        public_id_map_[raw_id] = old_public;
        remove_boundary_lost(best_old_raw);
        result.inherited_ids += 1;
        debug.chosen_old_raw_id = best_old_raw;
        debug.chosen_public_id = old_public;
        debug.result = "assigned";
      }
      result.short_inherit_debug.push_back(debug);
    }
  }

  void apply_boundary_recovery(FrameResult& result, const std::vector<int>& raw_ids) {
    if (!cfg_.boundary_recover || !cfg_.wrap_boundary_geometry) {
      return;
    }
    cleanup_boundary_lost();
    std::vector<int> claimed_old;
    std::set<int> claimed_public;
    std::set<int> active_public;
    for (int raw_id : raw_ids) {
      const auto map_it = public_id_map_.find(raw_id);
      if (map_it != public_id_map_.end() && map_it->second > 0) {
        active_public.insert(map_it->second);
      }
    }
    for (size_t i = 0; i < raw_ids.size(); ++i) {
      const int raw_id = raw_ids[i];
      if (raw_id <= 0 || public_id_map_.find(raw_id) != public_id_map_.end()) {
        continue;
      }
      if (i >= static_cast<size_t>(result.detection_count)) {
        continue;
      }
      const float* det = result.detections.data() + i * kDetectionFields;
      const std::array<float, 4> bbox = bbox4_from_ptr(det);
      const BoundarySide appear_side = boundary_side_for_bbox(bbox);
      if (appear_side == BoundarySide::None) {
        continue;
      }
      int best_old_raw = -1;
      float best_score = 1.0e9f;
      for (const BoundaryLostTarget& lost : boundary_lost_) {
        if (std::find(claimed_old.begin(), claimed_old.end(), lost.raw_id) != claimed_old.end()) {
          continue;
        }
        if (public_id_map_.find(lost.raw_id) == public_id_map_.end()) {
          continue;
        }
        const bool same_side = lost.side == appear_side;
        const bool opposite_side = opposite_boundary(lost.side) == appear_side;
        if (!same_side && !opposite_side) {
          continue;
        }
        const float size_score = size_ratio_score(lost.bbox, bbox);
        if (size_score < cfg_.boundary_size_ratio_thresh) {
          continue;
        }
        float dist = boundary_distance_score(lost, bbox, appear_side, opposite_side);
        if (dist > cfg_.boundary_center_dist_thresh) {
          continue;
        }
        if (same_side && lost.has_velocity) {
          const auto c_last = bbox_center(lost.bbox);
          const auto c_new = bbox_center(bbox);
          const int dt = std::max(1, frame_id_ - lost.frame_index);
          const float pred_x = c_last[0] + lost.velocity[0] * dt;
          const float pred_y = c_last[1] + lost.velocity[1] * dt;
          const float dx = c_new[0] - pred_x;
          const float dy = c_new[1] - pred_y;
          const float max_dim = std::max(bbox_width(lost.bbox), bbox_height(lost.bbox));
          if (std::sqrt(dx * dx + dy * dy) > max_dim * 2.0f) {
            continue;
          }
        }
        if (dist < best_score) {
          best_score = dist;
          best_old_raw = lost.raw_id;
        }
      }
      if (best_old_raw > 0) {
        const int public_id = public_id_map_[best_old_raw];
        public_id_map_[raw_id] = public_id;
        claimed_public.insert(public_id);
        active_public.insert(public_id);
        claimed_old.push_back(best_old_raw);
        remove_boundary_lost(best_old_raw);
        result.boundary_recovered += 1;
      }
    }
    apply_recent_public_boundary_recovery(result, raw_ids, active_public, claimed_public);
  }

  void apply_public_ids(FrameResult& result, const std::vector<int>& raw_ids) {
    result.track_ids.assign(static_cast<size_t>(result.detection_count), -1);
    for (int i = 0; i < result.detection_count; ++i) {
      const int raw_id = i < static_cast<int>(raw_ids.size()) ? raw_ids[static_cast<size_t>(i)] : -1;
      result.track_ids[static_cast<size_t>(i)] = raw_id > 0 ? public_id_for(raw_id) : -1;
    }
  }

  void resolve_public_id_conflicts(FrameResult& result, std::vector<int>& raw_ids) {
    if (result.detection_count <= 1) {
      return;
    }

    const int n = result.detection_count;
    std::map<int, std::vector<int>> indices_by_public;
    for (int i = 0; i < n; ++i) {
      const int raw_id = i < static_cast<int>(raw_ids.size()) ? raw_ids[static_cast<size_t>(i)] : -1;
      const auto map_it = public_id_map_.find(raw_id);
      if (map_it == public_id_map_.end() || map_it->second <= 0) {
        continue;
      }
      indices_by_public[map_it->second].push_back(i);
    }

    std::vector<uint8_t> keep(static_cast<size_t>(n), 1);
    bool changed = false;
    for (const auto& item : indices_by_public) {
      const int public_id = item.first;
      const std::vector<int>& indices = item.second;
      if (indices.size() <= 1) {
        continue;
      }

      const int owner = choose_public_conflict_owner(result, public_id, indices);
      const std::array<float, 4> owner_bbox =
          bbox4_from_ptr(result.detections.data() + static_cast<size_t>(owner) * kDetectionFields);
      for (int idx : indices) {
        if (idx == owner) {
          continue;
        }
        const std::array<float, 4> bbox =
            bbox4_from_ptr(result.detections.data() + static_cast<size_t>(idx) * kDetectionFields);
        if (should_split_same_public_conflict(owner_bbox, bbox)) {
          const int raw_id =
              idx < static_cast<int>(raw_ids.size()) ? raw_ids[static_cast<size_t>(idx)] : -1;
          if (raw_id > 0) {
            assign_new_public_id(raw_id);
            changed = true;
            continue;
          }
        }
        keep[static_cast<size_t>(idx)] = 0;
        result.final_dedup_removed += 1;
        changed = true;
      }
    }

    if (!changed) {
      return;
    }
    compact_result_with_raw_ids(result, raw_ids, keep);
  }

  void add_coasting_tracks(FrameResult& result,
                           std::vector<int>& active_raw_ids,
                           const std::map<int, NativeTrackSnapshot>& snapshot_by_raw) {
    if (!cfg_.kalman_bbox && cfg_.coast_frames <= 0) {
      return;
    }
    const int coast_limit = cfg_.coast_frames > 0 ? cfg_.coast_frames : max_age_;
    std::vector<int> active = active_raw_ids;
    for (const auto& item : snapshot_by_raw) {
      const int raw_id = item.first;
      const NativeTrackSnapshot& snapshot = item.second;
      if (raw_id <= 0 || snapshot.time_since_update <= 0 || snapshot.time_since_update > coast_limit) {
        continue;
      }
      if (std::find(active.begin(), active.end(), raw_id) != active.end()) {
        continue;
      }
      if (result.detection_count >= cfg_.max_output_dets) {
        break;
      }
      const size_t required =
          (static_cast<size_t>(result.detection_count) + 1) * kDetectionFields;
      if (result.detections.size() < required) {
        result.detections.resize(required, 0.0f);
      }
      float* out = result.detections.data() + static_cast<size_t>(result.detection_count) * kDetectionFields;
      const auto cached = meta_cache_.find(raw_id);
      if (cached != meta_cache_.end()) {
        std::copy(cached->second.begin(), cached->second.end(), out);
      } else {
        std::fill(out, out + kDetectionFields, 0.0f);
        out[4] = 0.5f;
      }
      std::array<float, 4> bbox = bbox_from_last_observation(snapshot);
      if (!cfg_.coast_hold && finite_bbox(snapshot.state)) {
        bbox = snapshot.state;
      }
      if (!finite_bbox(bbox)) {
        continue;
      }
      out[0] = bbox[0];
      out[1] = bbox[1];
      out[2] = bbox[2];
      out[3] = bbox[3];
      active_raw_ids.push_back(raw_id);
      result.detection_count += 1;
      result.coasting_added += 1;
    }
  }

  void apply_final_boundary_dedup(FrameResult& result, std::vector<int>& raw_ids) {
    if (!cfg_.final_boundary_dedup || !cfg_.wrap_boundary_geometry ||
        result.detection_count <= 1) {
      return;
    }
    const int n = result.detection_count;
    std::vector<uint8_t> keep(static_cast<size_t>(n), 1);
    for (int i = 0; i < n; ++i) {
      if (!keep[static_cast<size_t>(i)]) {
        continue;
      }
      const std::array<float, 4> a =
          bbox4_from_ptr(result.detections.data() + static_cast<size_t>(i) * kDetectionFields);
      const BoundarySide side_a = boundary_duplicate_side_for_bbox(a);
      if (side_a == BoundarySide::None) {
        continue;
      }
      for (int j = i + 1; j < n; ++j) {
        if (!keep[static_cast<size_t>(j)]) {
          continue;
        }
        const std::array<float, 4> b =
            bbox4_from_ptr(result.detections.data() + static_cast<size_t>(j) * kDetectionFields);
        const BoundarySide side_b = boundary_duplicate_side_for_bbox(b);
        const int raw_i =
            i < static_cast<int>(raw_ids.size()) ? raw_ids[static_cast<size_t>(i)] : -1;
        const int raw_j =
            j < static_cast<int>(raw_ids.size()) ? raw_ids[static_cast<size_t>(j)] : -1;
        const bool same_public_boundary_residual =
            same_mapped_public_for_raws(raw_i, raw_j) &&
            is_boundary_residual_duplicate(a, side_a, b, side_b);
        if (!is_final_boundary_duplicate(a, side_a, b, side_b) &&
            !same_public_boundary_residual) {
          continue;
        }
        const int drop = choose_final_boundary_duplicate_drop(result, raw_ids, i, j);
        const int keep_idx = drop == i ? j : i;
        confirm_pending_duplicate_public_alias(raw_ids, keep_idx, drop, a, b);
        confirm_pending_raw_duplicate_retire(result, raw_ids, keep_idx, drop, a, b);
        keep[static_cast<size_t>(drop)] = 0;
        result.final_dedup_removed += 1;
        if (drop == i) {
          break;
        }
      }
    }
    if (result.final_dedup_removed <= 0) {
      return;
    }
    compact_result_with_raw_ids(result, raw_ids, keep);
  }

  void apply_close_small_dedup(FrameResult& result, std::vector<int>& raw_ids) {
    if ((!cfg_.close_small_dedup && !cfg_.normal_duplicate_dedup) ||
        result.detection_count <= 1) {
      return;
    }
    const int n = result.detection_count;
    std::vector<uint8_t> keep(static_cast<size_t>(n), 1);
    int removed = 0;
    for (int i = 0; i < n; ++i) {
      if (!keep[static_cast<size_t>(i)]) {
        continue;
      }
      const std::array<float, 4> a =
          bbox4_from_ptr(result.detections.data() + static_cast<size_t>(i) * kDetectionFields);
      for (int j = i + 1; j < n; ++j) {
        if (!keep[static_cast<size_t>(j)]) {
          continue;
        }
        const std::array<float, 4> b =
            bbox4_from_ptr(result.detections.data() + static_cast<size_t>(j) * kDetectionFields);
        const bool close_small_duplicate = is_close_small_duplicate(a, b);
        const bool normal_duplicate =
            !close_small_duplicate && is_normal_close_duplicate(a, b);
        if (!close_small_duplicate && !normal_duplicate) {
          continue;
        }
        const int drop = choose_close_small_duplicate_drop(result, raw_ids, i, j);
        const int keep_idx = drop == i ? j : i;
        if (should_alias_close_small_public_id(raw_ids, keep_idx, drop, a, b)) {
          confirm_pending_duplicate_public_alias(raw_ids, keep_idx, drop, a, b);
        }
        confirm_pending_raw_duplicate_retire(result, raw_ids, keep_idx, drop, a, b);
        keep[static_cast<size_t>(drop)] = 0;
        if (close_small_duplicate) {
          result.close_small_dedup_removed += 1;
        } else {
          result.normal_duplicate_dedup_removed += 1;
        }
        removed += 1;
        if (drop == i) {
          break;
        }
      }
    }
    if (removed <= 0) {
      return;
    }
    compact_result_with_raw_ids(result, raw_ids, keep);
  }

  void apply_close_small_output_dedup(FrameResult& result, int active_count) {
    if ((!cfg_.close_small_dedup && !cfg_.normal_duplicate_dedup) ||
        result.detection_count <= 1 ||
        result.track_ids.size() < static_cast<size_t>(result.detection_count)) {
      return;
    }
    const int n = result.detection_count;
    active_count = std::max(0, std::min(active_count, n));
    std::vector<uint8_t> keep(static_cast<size_t>(n), 1);
    int removed = 0;
    for (int i = 0; i < n; ++i) {
      if (!keep[static_cast<size_t>(i)]) {
        continue;
      }
      const std::array<float, 4> a =
          bbox4_from_ptr(result.detections.data() + static_cast<size_t>(i) * kDetectionFields);
      for (int j = i + 1; j < n; ++j) {
        if (!keep[static_cast<size_t>(j)]) {
          continue;
        }
        const std::array<float, 4> b =
            bbox4_from_ptr(result.detections.data() + static_cast<size_t>(j) * kDetectionFields);
        const bool close_small_duplicate = is_close_small_duplicate(a, b);
        const bool normal_duplicate =
            !close_small_duplicate && is_normal_close_duplicate(a, b);
        if (!close_small_duplicate && !normal_duplicate) {
          continue;
        }
        const int drop = choose_close_small_output_duplicate_drop(result, i, j, active_count);
        keep[static_cast<size_t>(drop)] = 0;
        if (close_small_duplicate) {
          result.close_small_dedup_removed += 1;
        } else {
          result.normal_duplicate_dedup_removed += 1;
        }
        removed += 1;
        if (drop == i) {
          break;
        }
      }
    }
    if (removed <= 0) {
      return;
    }
    compact_result_with_track_ids(result, keep);
  }

  void apply_boundary_output_dedup(FrameResult& result, int active_count) {
    if (!cfg_.final_boundary_dedup || !cfg_.wrap_boundary_geometry ||
        result.detection_count <= 1 ||
        result.track_ids.size() < static_cast<size_t>(result.detection_count)) {
      return;
    }

    const int n = result.detection_count;
    active_count = std::max(0, std::min(active_count, n));
    std::vector<uint8_t> keep(static_cast<size_t>(n), 1);
    int removed = 0;
    for (int i = 0; i < n; ++i) {
      if (!keep[static_cast<size_t>(i)]) {
        continue;
      }
      const std::array<float, 4> a =
          bbox4_from_ptr(result.detections.data() + static_cast<size_t>(i) * kDetectionFields);
      const BoundarySide side_a = boundary_side_for_bbox(a);
      if (side_a == BoundarySide::None) {
        continue;
      }
      for (int j = i + 1; j < n; ++j) {
        if (!keep[static_cast<size_t>(j)]) {
          continue;
        }
        if (i < active_count && j < active_count) {
          continue;
        }
        const std::array<float, 4> b =
            bbox4_from_ptr(result.detections.data() + static_cast<size_t>(j) * kDetectionFields);
        const BoundarySide side_b = boundary_side_for_bbox(b);
        if (!is_wrap_boundary_duplicate(a, side_a, b, side_b)) {
          continue;
        }
        const int drop = choose_boundary_output_duplicate_drop(result, i, j, active_count);
        keep[static_cast<size_t>(drop)] = 0;
        result.final_dedup_removed += 1;
        removed += 1;
        if (drop == i) {
          break;
        }
      }
    }
    if (removed <= 0) {
      return;
    }
    compact_result_with_track_ids(result, keep);
  }

  void compact_result_with_raw_ids(FrameResult& result,
                                   std::vector<int>& raw_ids,
                                   const std::vector<uint8_t>& keep) {
    const int n = result.detection_count;
    std::vector<float> compact;
    std::vector<int> compact_raw_ids;
    compact.reserve(static_cast<size_t>(n) * kDetectionFields);
    compact_raw_ids.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      if (!keep[static_cast<size_t>(i)]) {
        continue;
      }
      const float* src = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
      compact.insert(compact.end(), src, src + kDetectionFields);
      compact_raw_ids.push_back(i < static_cast<int>(raw_ids.size())
                                    ? raw_ids[static_cast<size_t>(i)]
                                    : -1);
    }
    result.detections.swap(compact);
    raw_ids.swap(compact_raw_ids);
    result.detection_count = static_cast<int>(raw_ids.size());
    result.raw_track_ids.clear();
    result.track_ids.clear();
    result.display_ids.clear();
    result.display_reuse_from_public_ids.clear();
    result.display_reuse_spatial.clear();
  }

  void compact_result_with_track_ids(FrameResult& result,
                                     const std::vector<uint8_t>& keep) {
    const int n = result.detection_count;
    std::vector<float> compact;
    std::vector<int> compact_raw_track_ids;
    std::vector<int> compact_track_ids;
    compact.reserve(static_cast<size_t>(n) * kDetectionFields);
    compact_raw_track_ids.reserve(static_cast<size_t>(n));
    compact_track_ids.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      if (!keep[static_cast<size_t>(i)]) {
        continue;
      }
      const float* src = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
      compact.insert(compact.end(), src, src + kDetectionFields);
      compact_raw_track_ids.push_back(i < static_cast<int>(result.raw_track_ids.size())
                                          ? result.raw_track_ids[static_cast<size_t>(i)]
                                          : -1);
      compact_track_ids.push_back(i < static_cast<int>(result.track_ids.size())
                                      ? result.track_ids[static_cast<size_t>(i)]
                                      : -1);
    }
    result.detections.swap(compact);
    result.raw_track_ids.swap(compact_raw_track_ids);
    result.track_ids.swap(compact_track_ids);
    result.detection_count = static_cast<int>(result.track_ids.size());
    result.display_ids.clear();
    result.display_reuse_from_public_ids.clear();
    result.display_reuse_spatial.clear();
  }

  bool is_final_boundary_duplicate(const std::array<float, 4>& a,
                                   BoundarySide side_a,
                                   const std::array<float, 4>& b,
                                   BoundarySide side_b) const {
    if (side_a == BoundarySide::None || side_b == BoundarySide::None ||
        !finite_bbox(a) || !finite_bbox(b)) {
      return false;
    }
    const bool same_side = side_a == side_b;
    const bool wrap_side = opposite_boundary(side_a) == side_b;
    if (!same_side && !wrap_side) {
      return false;
    }
    float x_cover = 0.0f;
    float y_cover = 0.0f;
    float min_iou = 0.0f;
    const float inter = boundary_intersection_stats(a, side_a, b, side_b, &x_cover, &y_cover, &min_iou);
    const float dist = boundary_center_distance_norm(a, side_a, b, side_b);
    const float area_ratio =
        std::min(bbox_area4(a), bbox_area4(b)) / (std::max(bbox_area4(a), bbox_area4(b)) + 1.0e-6f);
    if (wrap_side &&
        dist < cfg_.final_boundary_dedup_center_thresh &&
        y_cover > 0.45f &&
        area_ratio > cfg_.final_boundary_dedup_size_ratio_thresh) {
      return true;
    }
    if (same_side) {
      const float x_gap = std::max(0.0f, std::max(a[0], b[0]) - std::min(a[2], b[2]));
      const float avg_h = std::max((bbox_height(a) + bbox_height(b)) * 0.5f, 1.0f);
      if (dist < 1.25f &&
          x_gap / avg_h < 0.35f &&
          y_cover > 0.55f &&
          area_ratio > cfg_.final_boundary_dedup_size_ratio_thresh) {
        return true;
      }
    }
    if (inter <= 0.0f) {
      return false;
    }
    if (min_iou > cfg_.final_boundary_dedup_iou_thresh) {
      return true;
    }
    return dist < cfg_.final_boundary_dedup_center_thresh &&
           x_cover > 0.35f &&
           y_cover > 0.35f &&
           area_ratio > cfg_.final_boundary_dedup_size_ratio_thresh;
  }

  bool is_boundary_residual_duplicate(const std::array<float, 4>& a,
                                      BoundarySide side_a,
                                      const std::array<float, 4>& b,
                                      BoundarySide side_b) const {
    if (side_a == BoundarySide::None || side_b == BoundarySide::None ||
        opposite_boundary(side_a) != side_b || !finite_bbox(a) || !finite_bbox(b)) {
      return false;
    }
    const float area_a = bbox_area4(a);
    const float area_b = bbox_area4(b);
    const float small_area = std::min(area_a, area_b);
    const float large_area = std::max(area_a, area_b);
    if (small_area <= 0.0f || large_area <= 0.0f) {
      return false;
    }

    // This branch is only for seam residuals: one side is a clipped fragment
    // of a larger target on the opposite side. Similar-size boxes still use
    // the normal boundary duplicate gates above.
    const float area_ratio = small_area / (large_area + 1.0e-6f);
    if (area_ratio >= cfg_.final_boundary_dedup_size_ratio_thresh) {
      return false;
    }

    const float y_cover = bbox_y_cover(a, b);
    if (y_cover < 0.55f) {
      return false;
    }
    const float dist = boundary_center_distance_norm(a, side_a, b, side_b);
    if (dist > 2.20f) {
      return false;
    }
    const float avg_h = std::max((bbox_height(a) + bbox_height(b)) * 0.5f, 1.0f);
    const float x_gap = boundary_x_gap(a, side_a, b, side_b);
    return x_gap / avg_h <= 0.80f;
  }

  float boundary_intersection_stats(const std::array<float, 4>& a,
                                    BoundarySide side_a,
                                    const std::array<float, 4>& b,
                                    BoundarySide side_b,
                                    float* out_x_cover,
                                    float* out_y_cover,
                                    float* out_min_iou) const {
    const int w = meta_.process_width > 0 ? meta_.process_width : meta_.img_width;
    std::array<float, 4> shifted_b = b;
    if (w > 0 && opposite_boundary(side_a) == side_b) {
      const auto ca = bbox_center(a);
      const auto cb = bbox_center(b);
      const float fw = static_cast<float>(w);
      if (ca[0] - cb[0] > fw * 0.5f) {
        shifted_b[0] += fw;
        shifted_b[2] += fw;
      } else if (ca[0] - cb[0] < -fw * 0.5f) {
        shifted_b[0] -= fw;
        shifted_b[2] -= fw;
      }
    }
    const float ix1 = std::max(a[0], shifted_b[0]);
    const float iy1 = std::max(a[1], shifted_b[1]);
    const float ix2 = std::min(a[2], shifted_b[2]);
    const float iy2 = std::min(a[3], shifted_b[3]);
    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;
    const float min_area = std::min(bbox_area4(a), bbox_area4(b));
    const float min_w = std::min(bbox_width(a), bbox_width(b));
    const float min_h = std::min(bbox_height(a), bbox_height(b));
    if (out_x_cover != nullptr) {
      *out_x_cover = min_w > 0.0f ? iw / (min_w + 1.0e-6f) : 0.0f;
    }
    if (out_y_cover != nullptr) {
      *out_y_cover = min_h > 0.0f ? ih / (min_h + 1.0e-6f) : 0.0f;
    }
    if (out_min_iou != nullptr) {
      *out_min_iou = min_area > 0.0f ? inter / (min_area + 1.0e-6f) : 0.0f;
    }
    return inter;
  }

  float boundary_center_distance_norm(const std::array<float, 4>& a,
                                      BoundarySide side_a,
                                      const std::array<float, 4>& b,
                                      BoundarySide side_b) const {
    const int w = meta_.process_width > 0 ? meta_.process_width : meta_.img_width;
    const auto ca = bbox_center(a);
    const auto cb = bbox_center(b);
    float dx = ca[0] - cb[0];
    if (w > 0 && opposite_boundary(side_a) == side_b) {
      const float fw = static_cast<float>(w);
      if (dx > fw * 0.5f) {
        dx -= fw;
      } else if (dx < -fw * 0.5f) {
        dx += fw;
      }
    }
    const float dy = ca[1] - cb[1];
    const float avg_h = std::max((bbox_height(a) + bbox_height(b)) * 0.5f, 1.0f);
    return std::sqrt(dx * dx + dy * dy) / avg_h;
  }

  float boundary_x_gap(const std::array<float, 4>& a,
                       BoundarySide side_a,
                       const std::array<float, 4>& b,
                       BoundarySide side_b) const {
    const int w = meta_.process_width > 0 ? meta_.process_width : meta_.img_width;
    std::array<float, 4> shifted_b = b;
    if (w > 0 && opposite_boundary(side_a) == side_b) {
      const float fw = static_cast<float>(w);
      if (side_a == BoundarySide::Right && side_b == BoundarySide::Left) {
        shifted_b[0] += fw;
        shifted_b[2] += fw;
      } else if (side_a == BoundarySide::Left && side_b == BoundarySide::Right) {
        shifted_b[0] -= fw;
        shifted_b[2] -= fw;
      }
    }
    return std::max(0.0f, std::max(a[0], shifted_b[0]) - std::min(a[2], shifted_b[2]));
  }

  int choose_final_boundary_duplicate_drop(const FrameResult& result,
                                           const std::vector<int>& raw_ids,
                                           int i,
                                           int j) const {
    const int raw_i = i < static_cast<int>(raw_ids.size()) ? raw_ids[static_cast<size_t>(i)] : -1;
    const int raw_j = j < static_cast<int>(raw_ids.size()) ? raw_ids[static_cast<size_t>(j)] : -1;
    const int seen_i = public_seen_count_for_raw(raw_i);
    const int seen_j = public_seen_count_for_raw(raw_j);
    if (seen_i != seen_j) {
      return seen_i > seen_j ? j : i;
    }

    const float* det_i = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
    const float* det_j = result.detections.data() + static_cast<size_t>(j) * kDetectionFields;
    if (std::fabs(det_i[4] - det_j[4]) > 1.0e-6f) {
      return det_i[4] >= det_j[4] ? j : i;
    }

    const std::array<float, 4> box_i = bbox4_from_ptr(det_i);
    const std::array<float, 4> box_j = bbox4_from_ptr(det_j);
    return bbox_area4(box_i) >= bbox_area4(box_j) ? j : i;
  }

  bool is_wrap_boundary_duplicate(const std::array<float, 4>& a,
                                  BoundarySide side_a,
                                  const std::array<float, 4>& b,
                                  BoundarySide side_b) const {
    if (side_a == BoundarySide::None || side_b == BoundarySide::None ||
        opposite_boundary(side_a) != side_b) {
      return false;
    }
    return is_final_boundary_duplicate(a, side_a, b, side_b);
  }

  int choose_boundary_output_duplicate_drop(const FrameResult& result,
                                            int i,
                                            int j,
                                            int active_count) const {
    const bool active_i = i < active_count;
    const bool active_j = j < active_count;
    if (active_i != active_j) {
      return active_i ? j : i;
    }

    const int public_i =
        i < static_cast<int>(result.track_ids.size()) ? result.track_ids[static_cast<size_t>(i)] : -1;
    const int public_j =
        j < static_cast<int>(result.track_ids.size()) ? result.track_ids[static_cast<size_t>(j)] : -1;
    if (!active_i && !active_j) {
      const auto frame_i = public_last_frame_.find(public_i);
      const auto frame_j = public_last_frame_.find(public_j);
      const int last_i = frame_i == public_last_frame_.end() ? -1 : frame_i->second;
      const int last_j = frame_j == public_last_frame_.end() ? -1 : frame_j->second;
      if (last_i != last_j) {
        return last_i > last_j ? j : i;
      }
    }

    const int seen_i = public_i > 0 ? public_seen_count(public_i) : 0;
    const int seen_j = public_j > 0 ? public_seen_count(public_j) : 0;
    if (seen_i != seen_j) {
      return seen_i > seen_j ? j : i;
    }

    const float* det_i = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
    const float* det_j = result.detections.data() + static_cast<size_t>(j) * kDetectionFields;
    if (std::fabs(det_i[4] - det_j[4]) > 1.0e-6f) {
      return det_i[4] >= det_j[4] ? j : i;
    }

    const std::array<float, 4> box_i = bbox4_from_ptr(det_i);
    const std::array<float, 4> box_j = bbox4_from_ptr(det_j);
    return bbox_area4(box_i) >= bbox_area4(box_j) ? j : i;
  }

  bool is_close_small_duplicate(const std::array<float, 4>& a,
                                const std::array<float, 4>& b) const {
    if (!cfg_.close_small_dedup || !finite_bbox(a) || !finite_bbox(b)) {
      return false;
    }
    const float area_a = bbox_area4(a);
    const float area_b = bbox_area4(b);
    const float max_side_a = std::max(bbox_width(a), bbox_height(a));
    const float max_side_b = std::max(bbox_width(b), bbox_height(b));
    if (area_a <= 0.0f || area_b <= 0.0f ||
        area_a > cfg_.close_small_dedup_max_area ||
        area_b > cfg_.close_small_dedup_max_area ||
        max_side_a > cfg_.close_small_dedup_max_side ||
        max_side_b > cfg_.close_small_dedup_max_side) {
      return false;
    }

    const float dist_norm = inheritance_center_distance_norm(a, b);
    if (dist_norm >= cfg_.close_small_dedup_center_thresh) {
      return false;
    }
    const float dist_px = center_distance_px(a, b);
    if (dist_px > cfg_.close_small_dedup_center_px) {
      return false;
    }
    const auto ca = bbox_center(a);
    const auto cb = bbox_center(b);
    const float y_center_delta = std::fabs(ca[1] - cb[1]);
    if (y_center_delta > cfg_.close_small_dedup_y_center_px) {
      return false;
    }
    const float y_cover = bbox_y_cover(a, b);
    const float x_gap = bbox_x_gap(a, b);
    return y_cover >= cfg_.close_small_dedup_y_cover_thresh &&
           x_gap <= cfg_.close_small_dedup_x_gap_px;
  }

  bool is_normal_close_duplicate(const std::array<float, 4>& a,
                                 const std::array<float, 4>& b) const {
    if (!cfg_.normal_duplicate_dedup || !finite_bbox(a) || !finite_bbox(b)) {
      return false;
    }
    const float size_score = size_ratio_score(a, b);
    const float partial_size_thresh =
        std::max(0.45f, cfg_.normal_duplicate_size_ratio_thresh * 0.75f);
    if (size_score < std::min(cfg_.normal_duplicate_size_ratio_thresh,
                              partial_size_thresh)) {
      return false;
    }
    const float dist_norm = inheritance_center_distance_norm(a, b);
    const float dist_px = center_distance_px(a, b);
    const bool center_ok =
        dist_px <= cfg_.normal_duplicate_abs_center_px ||
        (dist_norm <= cfg_.normal_duplicate_center_thresh &&
         dist_px <= cfg_.normal_duplicate_abs_center_px * 1.5f);
    const float iou = box_iou(a.data(), b.data());
    const float x_cover = bbox_x_cover(a, b);
    const float y_cover = bbox_y_cover(a, b);
    if (size_score >= cfg_.normal_duplicate_size_ratio_thresh && center_ok &&
        (iou >= cfg_.normal_duplicate_iou_thresh ||
        (x_cover >= cfg_.normal_duplicate_cover_thresh &&
         y_cover >= cfg_.normal_duplicate_cover_thresh))) {
      return true;
    }

    const float partial_center_thresh =
        std::max(0.65f, cfg_.normal_duplicate_center_thresh);
    const float partial_center_px =
        std::max(48.0f, cfg_.normal_duplicate_abs_center_px);
    const float partial_x_cover_thresh =
        std::min(0.45f, std::max(0.35f, cfg_.normal_duplicate_cover_thresh * 0.75f));
    const float partial_y_cover_thresh =
        std::max(0.65f, cfg_.normal_duplicate_cover_thresh);
    return size_score >= partial_size_thresh &&
           dist_norm <= partial_center_thresh &&
           dist_px <= partial_center_px &&
           x_cover >= partial_x_cover_thresh &&
           y_cover >= partial_y_cover_thresh;
  }

  float center_distance_px(const std::array<float, 4>& a,
                           const std::array<float, 4>& b) const {
    const auto ca = bbox_center(a);
    const auto cb = bbox_center(b);
    float dx = ca[0] - cb[0];
    if (cfg_.wrap_boundary_geometry && finite_bbox(a) && finite_bbox(b)) {
      const BoundarySide side_a = boundary_side_for_bbox(a);
      const BoundarySide side_b = boundary_side_for_bbox(b);
      const int w = meta_.process_width > 0 ? meta_.process_width : meta_.img_width;
      if (w > 0 && side_a != BoundarySide::None &&
          side_b != BoundarySide::None && opposite_boundary(side_a) == side_b) {
        const float fw = static_cast<float>(w);
        if (dx > fw * 0.5f) {
          dx -= fw;
        } else if (dx < -fw * 0.5f) {
          dx += fw;
        }
      }
    }
    const float dy = ca[1] - cb[1];
    return std::sqrt(dx * dx + dy * dy);
  }

  int choose_close_small_duplicate_drop(const FrameResult& result,
                                        const std::vector<int>& raw_ids,
                                        int i,
                                        int j) const {
    const int raw_i = i < static_cast<int>(raw_ids.size()) ? raw_ids[static_cast<size_t>(i)] : -1;
    const int raw_j = j < static_cast<int>(raw_ids.size()) ? raw_ids[static_cast<size_t>(j)] : -1;
    const int seen_i = public_seen_count_for_raw(raw_i);
    const int seen_j = public_seen_count_for_raw(raw_j);
    if (seen_i != seen_j) {
      return seen_i > seen_j ? j : i;
    }

    const float* det_i = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
    const float* det_j = result.detections.data() + static_cast<size_t>(j) * kDetectionFields;
    if (std::fabs(det_i[4] - det_j[4]) > 1.0e-6f) {
      return det_i[4] >= det_j[4] ? j : i;
    }

    const std::array<float, 4> box_i = bbox4_from_ptr(det_i);
    const std::array<float, 4> box_j = bbox4_from_ptr(det_j);
    return bbox_area4(box_i) >= bbox_area4(box_j) ? j : i;
  }

  int choose_close_small_output_duplicate_drop(const FrameResult& result,
                                               int i,
                                               int j,
                                               int active_count) const {
    const bool active_i = i < active_count;
    const bool active_j = j < active_count;
    if (active_i != active_j) {
      return active_i ? j : i;
    }

    const int public_i =
        i < static_cast<int>(result.track_ids.size()) ? result.track_ids[static_cast<size_t>(i)] : -1;
    const int public_j =
        j < static_cast<int>(result.track_ids.size()) ? result.track_ids[static_cast<size_t>(j)] : -1;
    const int seen_i = public_i > 0 ? public_seen_count(public_i) : 0;
    const int seen_j = public_j > 0 ? public_seen_count(public_j) : 0;
    if (seen_i != seen_j) {
      return seen_i > seen_j ? j : i;
    }

    const float* det_i = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
    const float* det_j = result.detections.data() + static_cast<size_t>(j) * kDetectionFields;
    if (std::fabs(det_i[4] - det_j[4]) > 1.0e-6f) {
      return det_i[4] >= det_j[4] ? j : i;
    }

    const std::array<float, 4> box_i = bbox4_from_ptr(det_i);
    const std::array<float, 4> box_j = bbox4_from_ptr(det_j);
    return bbox_area4(box_i) >= bbox_area4(box_j) ? j : i;
  }

  std::map<int, std::array<float, 4>> current_bbox_by_raw_ids(
      const FrameResult& result,
      const std::vector<int>& raw_ids) const {
    std::map<int, std::array<float, 4>> bbox_by_raw;
    const int n = std::min(result.detection_count, static_cast<int>(raw_ids.size()));
    for (int i = 0; i < n; ++i) {
      const int raw_id = raw_ids[static_cast<size_t>(i)];
      if (raw_id <= 0) {
        continue;
      }
      const std::array<float, 4> bbox =
          bbox4_from_ptr(result.detections.data() + static_cast<size_t>(i) * kDetectionFields);
      if (finite_bbox(bbox)) {
        bbox_by_raw[raw_id] = bbox;
      }
    }
    return bbox_by_raw;
  }

  void cleanup_pending_public_aliases(
      const std::map<int, std::array<float, 4>>& bbox_by_raw) {
    for (auto it = pending_public_aliases_.begin(); it != pending_public_aliases_.end();) {
      const PendingPublicAlias& pending = it->second;
      if (frame_id_ - pending.last_frame > 1) {
        it = pending_public_aliases_.erase(it);
        continue;
      }
      const auto public_bbox = bbox_by_raw.find(pending.public_raw);
      const auto alias_bbox = bbox_by_raw.find(pending.alias_raw);
      if (public_bbox != bbox_by_raw.end() && alias_bbox != bbox_by_raw.end() &&
          should_split_same_public_conflict(public_bbox->second, alias_bbox->second)) {
        it = pending_public_aliases_.erase(it);
        continue;
      }
      ++it;
    }
  }

  static std::pair<int, int> raw_pair_key(int raw_a, int raw_b) {
    if (raw_a > raw_b) {
      std::swap(raw_a, raw_b);
    }
    return std::make_pair(raw_a, raw_b);
  }

  bool strict_raw_duplicate_geometry(const std::array<float, 4>& a,
                                     const std::array<float, 4>& b) const {
    if (!finite_bbox(a) || !finite_bbox(b)) {
      return false;
    }
    const float iou = box_iou(a.data(), b.data());
    const float dist = inheritance_center_distance_norm(a, b);
    const float size_score = size_ratio_score(a, b);
    const float y_cover = bbox_y_cover(a, b);
    if (size_score < cfg_.raw_duplicate_retire_size_ratio_thresh ||
        y_cover < cfg_.raw_duplicate_retire_y_cover_thresh) {
      return is_normal_close_duplicate(a, b);
    }
    return iou >= cfg_.raw_duplicate_retire_iou_thresh ||
           dist <= cfg_.raw_duplicate_retire_center_thresh ||
           is_normal_close_duplicate(a, b);
  }

  void cleanup_pending_raw_duplicate_retires(
      const std::map<int, std::array<float, 4>>& bbox_by_raw) {
    for (auto it = pending_raw_duplicate_retires_.begin();
         it != pending_raw_duplicate_retires_.end();) {
      const PendingRawDuplicateRetire& pending = it->second;
      if (frame_id_ - pending.last_frame > 1) {
        it = pending_raw_duplicate_retires_.erase(it);
        continue;
      }
      const auto keep_bbox = bbox_by_raw.find(pending.keep_raw);
      const auto retire_bbox = bbox_by_raw.find(pending.retire_raw);
      if (keep_bbox == bbox_by_raw.end() || retire_bbox == bbox_by_raw.end() ||
          !strict_raw_duplicate_geometry(keep_bbox->second, retire_bbox->second)) {
        it = pending_raw_duplicate_retires_.erase(it);
        continue;
      }
      ++it;
    }
  }

  void erase_pending_public_alias_pair(int raw_a, int raw_b) {
    if (raw_a <= 0 || raw_b <= 0 || raw_a == raw_b) {
      return;
    }
    pending_public_aliases_.erase(std::make_pair(raw_a, raw_b));
    pending_public_aliases_.erase(std::make_pair(raw_b, raw_a));
  }

  void erase_pending_raw_duplicate_retire_pair(int raw_a, int raw_b) {
    if (raw_a <= 0 || raw_b <= 0 || raw_a == raw_b) {
      return;
    }
    pending_raw_duplicate_retires_.erase(raw_pair_key(raw_a, raw_b));
  }

  void erase_pending_public_aliases_for_raw(int raw_id) {
    if (raw_id <= 0) {
      return;
    }
    for (auto it = pending_public_aliases_.begin(); it != pending_public_aliases_.end();) {
      if (it->second.public_raw == raw_id || it->second.alias_raw == raw_id) {
        it = pending_public_aliases_.erase(it);
      } else {
        ++it;
      }
    }
  }

  void erase_pending_raw_duplicate_retires_for_raw(int raw_id) {
    if (raw_id <= 0) {
      return;
    }
    for (auto it = pending_raw_duplicate_retires_.begin();
         it != pending_raw_duplicate_retires_.end();) {
      if (it->second.keep_raw == raw_id || it->second.retire_raw == raw_id) {
        it = pending_raw_duplicate_retires_.erase(it);
      } else {
        ++it;
      }
    }
  }

  void confirm_pending_duplicate_public_alias(const std::vector<int>& raw_ids,
                                              int keep_idx,
                                              int drop_idx,
                                              const std::array<float, 4>& a,
                                              const std::array<float, 4>& b) {
    if (keep_idx < 0 || drop_idx < 0 ||
        keep_idx >= static_cast<int>(raw_ids.size()) ||
        drop_idx >= static_cast<int>(raw_ids.size()) ||
        !finite_bbox(a) || !finite_bbox(b)) {
      return;
    }
    const int keep_raw = raw_ids[static_cast<size_t>(keep_idx)];
    const int drop_raw = raw_ids[static_cast<size_t>(drop_idx)];
    if (keep_raw <= 0 || drop_raw <= 0 || keep_raw == drop_raw) {
      return;
    }

    const int keep_public = mapped_public_id_for_raw(keep_raw);
    const int drop_public = mapped_public_id_for_raw(drop_raw);
    if (keep_public > 0 && keep_public == drop_public) {
      erase_pending_public_alias_pair(keep_raw, drop_raw);
      return;
    }
    if (should_split_same_public_conflict(a, b)) {
      erase_pending_public_alias_pair(keep_raw, drop_raw);
      return;
    }

    int public_raw = -1;
    int alias_raw = -1;
    int public_id = -1;
    if (keep_public > 0) {
      public_raw = keep_raw;
      alias_raw = drop_raw;
      public_id = keep_public;
    } else if (drop_public > 0) {
      public_raw = drop_raw;
      alias_raw = keep_raw;
      public_id = drop_public;
    } else {
      erase_pending_public_alias_pair(keep_raw, drop_raw);
      return;
    }

    const auto key = std::make_pair(public_raw, alias_raw);
    PendingPublicAlias& pending = pending_public_aliases_[key];
    if (pending.last_frame == frame_id_ - 1 &&
        pending.public_raw == public_raw &&
        pending.alias_raw == alias_raw &&
        pending.public_id == public_id) {
      pending.hits += 1;
    } else {
      pending.hits = 1;
    }
    pending.public_raw = public_raw;
    pending.alias_raw = alias_raw;
    pending.public_id = public_id;
    pending.last_frame = frame_id_;

    if (pending.hits >= cfg_.public_alias_confirm_frames) {
      public_id_map_[alias_raw] = public_id;
      pending_public_aliases_.erase(key);
    }
  }

  bool raw_duplicate_retire_has_third_target(const FrameResult& result,
                                             int keep_idx,
                                             int drop_idx,
                                             const std::array<float, 4>& a,
                                             const std::array<float, 4>& b) const {
    if (cfg_.raw_duplicate_retire_local_dist_thresh <= 0.0f) {
      return false;
    }
    for (int i = 0; i < result.detection_count; ++i) {
      if (i == keep_idx || i == drop_idx) {
        continue;
      }
      const std::array<float, 4> other =
          bbox4_from_ptr(result.detections.data() + static_cast<size_t>(i) * kDetectionFields);
      if (!finite_bbox(other)) {
        continue;
      }
      if (inheritance_center_distance_norm(a, other) <
              cfg_.raw_duplicate_retire_local_dist_thresh ||
          inheritance_center_distance_norm(b, other) <
              cfg_.raw_duplicate_retire_local_dist_thresh) {
        return true;
      }
    }
    return false;
  }

  bool raw_duplicate_retire_public_conflict_blocks(int keep_raw, int retire_raw) const {
    const int keep_public = mapped_public_id_for_raw(keep_raw);
    const int retire_public = mapped_public_id_for_raw(retire_raw);
    if (keep_public <= 0 || retire_public <= 0 || keep_public == retire_public) {
      return false;
    }
    if (cfg_.raw_duplicate_retire_strong_public_seen <= 0) {
      return true;
    }
    return public_seen_count(keep_public) >= cfg_.raw_duplicate_retire_strong_public_seen &&
           public_seen_count(retire_public) >= cfg_.raw_duplicate_retire_strong_public_seen;
  }

  bool should_confirm_raw_duplicate_retire(const FrameResult& result,
                                           const std::vector<int>& raw_ids,
                                           int keep_idx,
                                           int drop_idx,
                                           const std::array<float, 4>& a,
                                           const std::array<float, 4>& b) const {
    if (!cfg_.raw_duplicate_retire ||
        keep_idx < 0 || drop_idx < 0 ||
        keep_idx >= static_cast<int>(raw_ids.size()) ||
        drop_idx >= static_cast<int>(raw_ids.size()) ||
        !strict_raw_duplicate_geometry(a, b)) {
      return false;
    }
    const int keep_raw = raw_ids[static_cast<size_t>(keep_idx)];
    const int retire_raw = raw_ids[static_cast<size_t>(drop_idx)];
    if (keep_raw <= 0 || retire_raw <= 0 || keep_raw == retire_raw) {
      return false;
    }
    if (retired_raw_ids_this_frame_.find(keep_raw) != retired_raw_ids_this_frame_.end() ||
        retired_raw_ids_this_frame_.find(retire_raw) != retired_raw_ids_this_frame_.end()) {
      return false;
    }
    if (raw_duplicate_retire_has_third_target(result, keep_idx, drop_idx, a, b)) {
      return false;
    }
    if (raw_duplicate_retire_public_conflict_blocks(keep_raw, retire_raw)) {
      return false;
    }
    return true;
  }

  bool public_has_other_raw_mapping(int public_id, int raw_id) const {
    if (public_id <= 0) {
      return false;
    }
    for (const auto& item : public_id_map_) {
      if (item.first != raw_id && item.second == public_id) {
        return true;
      }
    }
    return false;
  }

  void cleanup_retired_raw_track_state(int raw_id) {
    if (raw_id <= 0) {
      return;
    }
    int retired_public = 0;
    const auto map_it = public_id_map_.find(raw_id);
    if (map_it != public_id_map_.end()) {
      retired_public = map_it->second;
      public_id_map_.erase(map_it);
    }
    meta_cache_.erase(raw_id);
    bbox_cache_.erase(raw_id);
    prev_bbox_.erase(raw_id);
    prev_prev_bbox_.erase(raw_id);
    remove_boundary_lost(raw_id);
    erase_pending_public_aliases_for_raw(raw_id);
    erase_pending_raw_duplicate_retires_for_raw(raw_id);
    if (retired_public <= 0 || public_has_other_raw_mapping(retired_public, raw_id)) {
      return;
    }
    public_last_bbox_.erase(retired_public);
    public_last_frame_.erase(retired_public);
    public_last_detection_.erase(retired_public);
    public_seen_count_.erase(retired_public);
    public_consecutive_seen_count_.erase(retired_public);
    auto display_it = display_id_by_public_.find(retired_public);
    if (display_it != display_id_by_public_.end()) {
      if (display_it->second > 0) {
        free_display_ids_.insert(display_it->second);
        DisplayReleaseDebug release_debug;
        release_debug.public_id = retired_public;
        release_debug.display_id = display_it->second;
        release_debug.frame_id = frame_id_;
        const auto last_it = display_public_last_frame_.find(retired_public);
        if (last_it != display_public_last_frame_.end()) {
          release_debug.last_frame = last_it->second;
          release_debug.missing_frames = frame_id_ - last_it->second;
        }
        release_debug.reason = "raw_track_state_cleanup";
        pending_display_release_debug_.push_back(release_debug);
      }
      display_public_last_frame_.erase(retired_public);
      display_id_by_public_.erase(display_it);
    }
  }

  bool retire_native_raw_track(int raw_id) {
    char err[2048] = {0};
    const int ret = hybrid_sort_native_retire_track(handle_.get(), raw_id, err, sizeof(err));
    if (ret < 0) {
      throw std::runtime_error(std::string("hybrid_sort_native_retire_track failed: ") + err);
    }
    if (ret <= 0) {
      return false;
    }
    retired_raw_ids_this_frame_.insert(raw_id);
    cleanup_retired_raw_track_state(raw_id);
    return true;
  }

  void confirm_pending_raw_duplicate_retire(FrameResult& result,
                                            const std::vector<int>& raw_ids,
                                            int keep_idx,
                                            int drop_idx,
                                            const std::array<float, 4>& a,
                                            const std::array<float, 4>& b) {
    if (!should_confirm_raw_duplicate_retire(result, raw_ids, keep_idx, drop_idx, a, b)) {
      if (keep_idx >= 0 && keep_idx < static_cast<int>(raw_ids.size()) &&
          drop_idx >= 0 && drop_idx < static_cast<int>(raw_ids.size())) {
        erase_pending_raw_duplicate_retire_pair(raw_ids[static_cast<size_t>(keep_idx)],
                                               raw_ids[static_cast<size_t>(drop_idx)]);
      }
      return;
    }
    const int keep_raw = raw_ids[static_cast<size_t>(keep_idx)];
    const int retire_raw = raw_ids[static_cast<size_t>(drop_idx)];
    const auto key = raw_pair_key(keep_raw, retire_raw);
    PendingRawDuplicateRetire& pending = pending_raw_duplicate_retires_[key];
    if (pending.last_frame == frame_id_ - 1 &&
        pending.keep_raw == keep_raw &&
        pending.retire_raw == retire_raw) {
      pending.hits += 1;
    } else {
      pending.hits = 1;
    }
    pending.keep_raw = keep_raw;
    pending.retire_raw = retire_raw;
    pending.last_frame = frame_id_;

    if (pending.hits >= cfg_.raw_duplicate_retire_confirm_frames) {
      if (retire_native_raw_track(retire_raw)) {
        result.raw_duplicate_retired += 1;
      }
      pending_raw_duplicate_retires_.erase(key);
    }
  }

  bool should_alias_close_small_public_id(const std::vector<int>& raw_ids,
                                          int keep_idx,
                                          int drop_idx,
                                          const std::array<float, 4>& a,
                                          const std::array<float, 4>& b) const {
    if (keep_idx < 0 || drop_idx < 0 ||
        keep_idx >= static_cast<int>(raw_ids.size()) ||
        drop_idx >= static_cast<int>(raw_ids.size()) ||
        !finite_bbox(a) || !finite_bbox(b)) {
      return false;
    }
    const int keep_raw = raw_ids[static_cast<size_t>(keep_idx)];
    const int drop_raw = raw_ids[static_cast<size_t>(drop_idx)];
    if (keep_raw <= 0 || drop_raw <= 0 || keep_raw == drop_raw) {
      return false;
    }

    const int keep_public = mapped_public_id_for_raw(keep_raw);
    const int drop_public = mapped_public_id_for_raw(drop_raw);
    if (keep_public > 0 && keep_public == drop_public) {
      return true;
    }

    const float dist_px = center_distance_px(a, b);
    if (dist_px > cfg_.close_small_public_alias_center_px) {
      return false;
    }
    const float x_gap = bbox_x_gap(a, b);
    if (x_gap > cfg_.close_small_public_alias_x_gap_px) {
      return false;
    }
    const float y_cover = bbox_y_cover(a, b);
    return y_cover >= cfg_.close_small_public_alias_y_cover_thresh;
  }

  int mapped_public_id_for_raw(int raw_id) const {
    const auto it = public_id_map_.find(raw_id);
    return it == public_id_map_.end() ? -1 : it->second;
  }

  bool should_split_same_public_conflict(const std::array<float, 4>& a,
                                         const std::array<float, 4>& b) const {
    if (!finite_bbox(a) || !finite_bbox(b) || same_public_duplicate(a, b)) {
      return false;
    }
    const float dist_norm = inheritance_center_distance_norm(a, b);
    const float dist_px = center_distance_px(a, b);
    const float x_gap = bbox_x_gap(a, b);
    const float y_cover = bbox_y_cover(a, b);

    if (dist_norm >= 2.10f && dist_px >= 70.0f && x_gap >= 30.0f) {
      return true;
    }
    if (dist_px >= 90.0f && x_gap >= 40.0f && y_cover < 0.85f) {
      return true;
    }
    return false;
  }

  int choose_public_conflict_owner(const FrameResult& result,
                                   int public_id,
                                   const std::vector<int>& indices) const {
    const auto last_it = public_last_bbox_.find(public_id);
    if (last_it != public_last_bbox_.end() && finite_bbox(last_it->second)) {
      int best = indices.empty() ? -1 : indices[0];
      float best_dist = std::numeric_limits<float>::infinity();
      for (int idx : indices) {
        if (idx < 0 || idx >= result.detection_count) {
          continue;
        }
        const std::array<float, 4> bbox =
            bbox4_from_ptr(result.detections.data() + static_cast<size_t>(idx) * kDetectionFields);
        if (!finite_bbox(bbox)) {
          continue;
        }
        const float dist = inheritance_center_distance_norm(bbox, last_it->second);
        if (dist < best_dist) {
          best_dist = dist;
          best = idx;
        }
      }
      if (best >= 0) {
        return best;
      }
    }

    int best = indices.empty() ? -1 : indices[0];
    float best_score = -1.0f;
    for (int idx : indices) {
      if (idx < 0 || idx >= result.detection_count) {
        continue;
      }
      const float* det = result.detections.data() + static_cast<size_t>(idx) * kDetectionFields;
      if (det[4] > best_score) {
        best_score = det[4];
        best = idx;
      }
    }
    return best;
  }

  bool same_public_duplicate(const std::array<float, 4>& a,
                             const std::array<float, 4>& b) const {
    if (!finite_bbox(a) || !finite_bbox(b)) {
      return false;
    }
    const float iou = box_iou(a.data(), b.data());
    const float dist = inheritance_center_distance_norm(a, b);
    const float size_score = size_ratio_score(a, b);
    const float x_cover = bbox_x_cover(a, b);
    const float y_cover = bbox_y_cover(a, b);
    const BoundarySide side_a = boundary_side_for_bbox(a);
    const BoundarySide side_b = boundary_side_for_bbox(b);
    if (side_a != BoundarySide::None &&
        side_b != BoundarySide::None &&
        opposite_boundary(side_a) == side_b) {
      return is_final_boundary_duplicate(a, side_a, b, side_b);
    }
    const BoundarySide dup_side_a = boundary_duplicate_side_for_bbox(a);
    const BoundarySide dup_side_b = boundary_duplicate_side_for_bbox(b);
    if (dup_side_a != BoundarySide::None &&
        dup_side_b != BoundarySide::None &&
        opposite_boundary(dup_side_a) == dup_side_b &&
        is_boundary_residual_duplicate(a, dup_side_a, b, dup_side_b)) {
      return true;
    }
    const bool same_boundary_side = side_a != BoundarySide::None && side_a == side_b;
    const float x_gap = bbox_x_gap(a, b);
    const float avg_h = std::max((bbox_height(a) + bbox_height(b)) * 0.5f, 1.0f);
    if (same_boundary_side &&
        dist < 1.0f &&
        size_score > 0.45f &&
        y_cover > 0.55f &&
        (x_cover > 0.25f || x_gap / avg_h < 0.15f)) {
      return true;
    }
    return iou > 0.35f ||
           (dist < 0.65f &&
            size_score > 0.35f &&
            x_cover > 0.30f &&
            y_cover > 0.35f);
  }

  int count_local_current_targets(const FrameResult& result,
                                  const std::array<float, 4>& bbox,
                                  size_t self_index,
                                  float dist_thresh) const {
    if (!finite_bbox(bbox) || dist_thresh <= 0.0f) {
      return 0;
    }
    int count = 0;
    for (int i = 0; i < result.detection_count; ++i) {
      const float* det = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
      const std::array<float, 4> other = bbox4_from_ptr(det);
      if (!finite_bbox(other)) {
        continue;
      }
      if (static_cast<size_t>(i) == self_index ||
          inheritance_center_distance_norm(bbox, other) <= dist_thresh) {
        count += 1;
      }
    }
    return count;
  }

  float inheritance_center_distance_norm(const std::array<float, 4>& a,
                                         const std::array<float, 4>& b) const {
    if (!finite_bbox(a) || !finite_bbox(b)) {
      return center_distance_norm(a, b);
    }
    if (!cfg_.wrap_boundary_geometry) {
      return center_distance_norm(a, b);
    }
    const BoundarySide side_a = boundary_side_for_bbox(a);
    const BoundarySide side_b = boundary_side_for_bbox(b);
    if (side_a != BoundarySide::None &&
        side_b != BoundarySide::None &&
        opposite_boundary(side_a) == side_b) {
      return boundary_center_distance_norm(a, side_a, b, side_b);
    }
    return center_distance_norm(a, b);
  }

  float nearest_other_current_target_distance(const FrameResult& result,
                                              const std::array<float, 4>& bbox,
                                              size_t self_index,
                                              float dist_limit) const {
    if (!finite_bbox(bbox) || dist_limit <= 0.0f) {
      return std::numeric_limits<float>::infinity();
    }
    float best = std::numeric_limits<float>::infinity();
    for (int i = 0; i < result.detection_count; ++i) {
      if (static_cast<size_t>(i) == self_index) {
        continue;
      }
      const float* det = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
      const std::array<float, 4> other = bbox4_from_ptr(det);
      if (!finite_bbox(other)) {
        continue;
      }
      const float dist = inheritance_center_distance_norm(bbox, other);
      if (dist < best && dist < dist_limit) {
        best = dist;
      }
    }
    return best;
  }

  float bbox_x_cover(const std::array<float, 4>& a,
                     const std::array<float, 4>& b) const {
    std::array<float, 4> shifted_b = b;
    if (cfg_.wrap_boundary_geometry && finite_bbox(a) && finite_bbox(b)) {
      const BoundarySide side_a = boundary_side_for_bbox(a);
      const BoundarySide side_b = boundary_side_for_bbox(b);
      const int w = meta_.process_width > 0 ? meta_.process_width : meta_.img_width;
      if (w > 0 && side_a != BoundarySide::None &&
          side_b != BoundarySide::None && opposite_boundary(side_a) == side_b) {
        const auto ca = bbox_center(a);
        const auto cb = bbox_center(b);
        const float fw = static_cast<float>(w);
        if (ca[0] - cb[0] > fw * 0.5f) {
          shifted_b[0] += fw;
          shifted_b[2] += fw;
        } else if (ca[0] - cb[0] < -fw * 0.5f) {
          shifted_b[0] -= fw;
          shifted_b[2] -= fw;
        }
      }
    }
    const float iw =
        std::max(0.0f, std::min(a[2], shifted_b[2]) - std::max(a[0], shifted_b[0]));
    const float min_w = std::min(bbox_width(a), bbox_width(b));
    return min_w > 0.0f ? iw / (min_w + 1.0e-6f) : 0.0f;
  }

  float bbox_x_gap(const std::array<float, 4>& a,
                   const std::array<float, 4>& b) const {
    std::array<float, 4> shifted_b = b;
    if (cfg_.wrap_boundary_geometry && finite_bbox(a) && finite_bbox(b)) {
      const BoundarySide side_a = boundary_side_for_bbox(a);
      const BoundarySide side_b = boundary_side_for_bbox(b);
      const int w = meta_.process_width > 0 ? meta_.process_width : meta_.img_width;
      if (w > 0 && side_a != BoundarySide::None &&
          side_b != BoundarySide::None && opposite_boundary(side_a) == side_b) {
        const auto ca = bbox_center(a);
        const auto cb = bbox_center(b);
        const float fw = static_cast<float>(w);
        if (ca[0] - cb[0] > fw * 0.5f) {
          shifted_b[0] += fw;
          shifted_b[2] += fw;
        } else if (ca[0] - cb[0] < -fw * 0.5f) {
          shifted_b[0] -= fw;
          shifted_b[2] -= fw;
        }
      }
    }
    return std::max(0.0f, std::max(a[0], shifted_b[0]) - std::min(a[2], shifted_b[2]));
  }

  float bbox_y_cover(const std::array<float, 4>& a,
                     const std::array<float, 4>& b) const {
    const float ih = std::max(0.0f, std::min(a[3], b[3]) - std::max(a[1], b[1]));
    const float min_h = std::min(bbox_height(a), bbox_height(b));
    return min_h > 0.0f ? ih / (min_h + 1.0e-6f) : 0.0f;
  }

  int public_seen_count(int public_id) const {
    const auto it = public_seen_count_.find(public_id);
    return it == public_seen_count_.end() ? 0 : it->second;
  }

  int public_consecutive_seen_count(int public_id) const {
    const auto it = public_consecutive_seen_count_.find(public_id);
    return it == public_consecutive_seen_count_.end() ? 0 : it->second;
  }

  int public_seen_count_for_raw(int raw_id) const {
    const auto it = public_id_map_.find(raw_id);
    return it == public_id_map_.end() ? 0 : public_seen_count(it->second);
  }

  int assign_new_public_id(int raw_id) {
    public_id_counter_ += 1;
    public_id_map_[raw_id] = public_id_counter_;
    return public_id_counter_;
  }

  void apply_display_ids(FrameResult& result) {
    result.display_ids.clear();
    result.display_reuse_from_public_ids.clear();
    result.display_reuse_spatial.clear();
    result.display_bindings_debug.clear();
    result.display_release_debug = pending_display_release_debug_;
    pending_display_release_debug_.clear();
    result.display_free_ids_debug.clear();
    result.display_assign_debug.clear();
    if (!cfg_.display_id_reuse ||
        result.track_ids.size() < static_cast<size_t>(result.detection_count)) {
      return;
    }

    std::set<int> active_public_ids;
    std::map<int, int> active_public_index;
    std::map<int, std::array<float, 4>> active_public_bbox;
    std::map<int, int> visible_public_raw_id;
    std::set<int> display_eligible_public_ids;
    std::map<int, int> display_eligible_public_index;
    std::map<int, std::array<float, 4>> display_eligible_public_bbox;
    for (int i = 0; i < result.detection_count; ++i) {
      const int public_id = result.track_ids[static_cast<size_t>(i)];
      if (public_id > 0) {
        const std::array<float, 4> bbox = bbox4_from_ptr(
            result.detections.data() + static_cast<size_t>(i) * kDetectionFields);
        active_public_ids.insert(public_id);
        active_public_index[public_id] = i;
        active_public_bbox[public_id] = bbox;
        if (display_slot_bbox_eligible(bbox)) {
          display_eligible_public_ids.insert(public_id);
          display_eligible_public_index[public_id] = i;
          display_eligible_public_bbox[public_id] = bbox;
        }
        const int raw_id =
            i < static_cast<int>(result.raw_track_ids.size())
                ? result.raw_track_ids[static_cast<size_t>(i)]
                : -1;
        auto raw_it = visible_public_raw_id.find(public_id);
        if (raw_it == visible_public_raw_id.end() || raw_it->second <= 0) {
          visible_public_raw_id[public_id] = raw_id;
        }
      }
    }

    for (auto it = display_id_by_public_.begin(); it != display_id_by_public_.end();) {
      if (display_eligible_public_ids.find(it->first) != display_eligible_public_ids.end()) {
        ++it;
        continue;
      }
      const auto last_it = display_public_last_frame_.find(it->first);
      const int missing_frames =
          last_it == display_public_last_frame_.end()
              ? cfg_.display_id_public_binding_ttl + 1
              : frame_id_ - last_it->second;
      if (cfg_.display_id_public_binding_ttl > 0 &&
          missing_frames <= cfg_.display_id_public_binding_ttl) {
        ++it;
        continue;
      }
      const int display_id = it->second;
      if (display_id > 0) {
        free_display_ids_.insert(display_id);
        display_last_public_[display_id] = it->first;
        DisplayReleaseDebug release_debug;
        release_debug.public_id = it->first;
        release_debug.display_id = display_id;
        release_debug.frame_id = frame_id_;
        release_debug.last_frame =
            last_it == display_public_last_frame_.end() ? -1 : last_it->second;
        release_debug.missing_frames = missing_frames;
        release_debug.reason = "display_binding_ttl_expired";
        result.display_release_debug.push_back(release_debug);
      }
      display_public_last_frame_.erase(it->first);
      it = display_id_by_public_.erase(it);
    }

    record_display_state_debug(result, active_public_ids);

    result.display_ids.assign(static_cast<size_t>(result.detection_count), -1);
    result.display_reuse_from_public_ids.assign(static_cast<size_t>(result.detection_count), 0);
    result.display_reuse_spatial.assign(static_cast<size_t>(result.detection_count), 0);
    const bool bootstrap_display_assignment =
        display_id_counter_ <= 0 && display_id_by_public_.empty();
    std::vector<int> process_order;
    process_order.reserve(static_cast<size_t>(result.detection_count));
    for (int i = 0; i < result.detection_count; ++i) {
      process_order.push_back(i);
    }
    std::sort(process_order.begin(), process_order.end(), [&](int a, int b) {
      const int public_a =
          a < static_cast<int>(result.track_ids.size())
              ? result.track_ids[static_cast<size_t>(a)]
              : -1;
      const int public_b =
          b < static_cast<int>(result.track_ids.size())
              ? result.track_ids[static_cast<size_t>(b)]
              : -1;
      const auto bbox_a_it = active_public_bbox.find(public_a);
      const auto bbox_b_it = active_public_bbox.find(public_b);
      const float area_a =
          bbox_a_it == active_public_bbox.end() ? 0.0f : bbox_area4(bbox_a_it->second);
      const float area_b =
          bbox_b_it == active_public_bbox.end() ? 0.0f : bbox_area4(bbox_b_it->second);
      if (bootstrap_display_assignment && area_a != area_b) {
        return area_a > area_b;
      }
      const bool bound_a = display_id_by_public_.find(public_a) != display_id_by_public_.end();
      const bool bound_b = display_id_by_public_.find(public_b) != display_id_by_public_.end();
      if (bound_a != bound_b) {
        return bound_a;
      }
      const auto raw_a_it = visible_public_raw_id.find(public_a);
      const auto raw_b_it = visible_public_raw_id.find(public_b);
      const bool live_a = raw_a_it != visible_public_raw_id.end() && raw_a_it->second > 0;
      const bool live_b = raw_b_it != visible_public_raw_id.end() && raw_b_it->second > 0;
      if (live_a != live_b) {
        return live_a;
      }
      const int hits_a = public_a > 0 ? public_consecutive_seen_count(public_a) : 0;
      const int hits_b = public_b > 0 ? public_consecutive_seen_count(public_b) : 0;
      if (hits_a != hits_b) {
        return hits_a > hits_b;
      }
      if (area_a != area_b) {
        return area_a > area_b;
      }
      return public_a < public_b;
    });
    std::set<int> hidden_public_ids;
    for (int i : process_order) {
      const int public_id = result.track_ids[static_cast<size_t>(i)];
      if (public_id <= 0 || hidden_public_ids.find(public_id) != hidden_public_ids.end()) {
        continue;
      }
      const int raw_track_id =
          i < static_cast<int>(result.raw_track_ids.size())
              ? result.raw_track_ids[static_cast<size_t>(i)]
              : -1;
      const float* debug_det =
          result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
      const std::array<float, 4> debug_bbox = bbox4_from_ptr(debug_det);
      DisplayAssignDebug assign_debug;
      assign_debug.original_index = i;
      assign_debug.public_id = public_id;
      assign_debug.raw_track_id = raw_track_id;
      assign_debug.public_consecutive_seen = public_consecutive_seen_count(public_id);
      assign_debug.display_counter = display_id_counter_;
      assign_debug.display_max_count = cfg_.display_id_max_count;
      assign_debug.bbox_area = finite_bbox(debug_bbox) ? bbox_area4(debug_bbox) : 0.0f;
      assign_debug.bbox_min_side = finite_bbox(debug_bbox) ? bbox_min_side(debug_bbox) : 0.0f;

      auto display_it = display_id_by_public_.find(public_id);
      assign_debug.had_existing_binding = display_it != display_id_by_public_.end();
      assign_debug.previous_display_id =
          assign_debug.had_existing_binding ? display_it->second : 0;
      DisplayReuseChoice reuse_choice;
      if (!display_slot_bbox_eligible(debug_bbox)) {
        assign_debug.action = "hidden";
        assign_debug.hidden = true;
        assign_debug.reason = "display_box_too_small";
        result.display_assign_debug.push_back(assign_debug);
        hidden_public_ids.insert(public_id);
        continue;
      }
      if (display_it == display_id_by_public_.end()) {
        if (raw_track_id <= 0) {
          assign_debug.action = "hidden";
          assign_debug.hidden = true;
          assign_debug.reason = "display_coast_without_binding";
          result.display_assign_debug.push_back(assign_debug);
          hidden_public_ids.insert(public_id);
          continue;
        }
        const float* det =
            result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
        const std::array<float, 4> bbox = bbox4_from_ptr(det);
        reuse_choice = choose_reusable_display_id(bbox, active_public_ids, public_id);
        assign_debug.reuse_candidates = reuse_choice.candidates;
        assign_debug.wait_for_free_display = reuse_choice.wait_for_free_display;
        int display_id = reuse_choice.display_id;
        if (display_id > 0) {
          assign_debug.action = reuse_choice.spatial ? "spatial_reuse" : "fallback_reuse";
          assign_debug.reused_from_public_id = reuse_choice.previous_public_id;
          assign_debug.reuse_spatial = reuse_choice.spatial;
          free_display_ids_.erase(display_id);
          detach_display_id_from_other_public(display_id, public_id);
        } else if (cfg_.display_id_max_count <= 0 ||
                   display_id_counter_ < cfg_.display_id_max_count) {
          display_id = ++display_id_counter_;
          assign_debug.action =
              bootstrap_display_assignment ? "bootstrap_new_display" : "new_display";
        } else if (bootstrap_display_assignment) {
          assign_debug.action = "hidden";
          assign_debug.hidden = true;
          assign_debug.reason = "bootstrap_over_limit_area_rank";
          result.display_assign_debug.push_back(assign_debug);
          hidden_public_ids.insert(public_id);
          continue;
        } else if (reuse_choice.wait_for_free_display) {
          assign_debug.action = "hidden";
          assign_debug.hidden = true;
          assign_debug.reason = "display_free_id_cooling_down";
          result.display_assign_debug.push_back(assign_debug);
          hidden_public_ids.insert(public_id);
          continue;
        } else {
          const DisplayReplacementChoice replacement =
              choose_display_area_replacement(
                  public_id,
                  raw_track_id,
                  bbox,
                  active_public_ids,
                  active_public_index,
                  active_public_bbox,
                  display_eligible_public_ids,
                  display_eligible_public_index,
                  display_eligible_public_bbox,
                  visible_public_raw_id);
          assign_debug.replace_pending_hits = replacement.pending_hits;
          assign_debug.replace_confirm_frames = cfg_.display_id_replace_confirm_frames;
          if (replacement.replace && replacement.display_id > 0) {
            display_id = replacement.display_id;
            assign_debug.action = "replace_by_area";
            assign_debug.replaced_public_id = replacement.replaced_public_id;
            assign_debug.replaced_display_id = replacement.display_id;
            assign_debug.replaced_original_index = replacement.replaced_index;
            assign_debug.replaced_public_consecutive_seen = replacement.replaced_hits;
            assign_debug.replaced_bbox_area = replacement.replaced_area;
            if (replacement.replaced_index >= 0 &&
                replacement.replaced_index < static_cast<int>(result.display_ids.size())) {
              result.display_ids[static_cast<size_t>(replacement.replaced_index)] = -1;
              result.display_reuse_from_public_ids[static_cast<size_t>(replacement.replaced_index)] = 0;
              result.display_reuse_spatial[static_cast<size_t>(replacement.replaced_index)] = 0;
              mark_display_assign_replaced_by_area(
                  result,
                  replacement.replaced_public_id,
                  replacement.display_id,
                  replacement.replaced_index,
                  public_id);
            }
            if (replacement.replaced_public_id > 0) {
              hidden_public_ids.insert(replacement.replaced_public_id);
              display_id_by_public_.erase(replacement.replaced_public_id);
              display_public_last_frame_.erase(replacement.replaced_public_id);
            }
          } else {
            assign_debug.action = "hidden";
            assign_debug.hidden = true;
            assign_debug.reason =
                replacement.reject_reason.empty()
                    ? "display_slots_full_wait_for_release"
                    : replacement.reject_reason;
            result.display_assign_debug.push_back(assign_debug);
            hidden_public_ids.insert(public_id);
            continue;
          }
        }
        display_id_by_public_[public_id] = display_id;
        display_it = display_id_by_public_.find(public_id);
      } else {
        assign_debug.action = "keep_binding";
      }
      const int display_id = display_it->second;
      if (display_id > 0) {
        free_display_ids_.erase(display_id);
      }
      result.display_ids[static_cast<size_t>(i)] = display_id;
      result.display_reuse_from_public_ids[static_cast<size_t>(i)] =
          reuse_choice.spatial ? reuse_choice.previous_public_id : 0;
      result.display_reuse_spatial[static_cast<size_t>(i)] =
          reuse_choice.spatial ? 1 : 0;
      assign_debug.assigned_display_id = display_id;
      assign_debug.reused_from_public_id =
          reuse_choice.spatial ? reuse_choice.previous_public_id : assign_debug.reused_from_public_id;
      assign_debug.reuse_spatial = reuse_choice.spatial || assign_debug.reuse_spatial;
      if (assign_debug.action.empty()) {
        assign_debug.action = "assigned";
      }
      result.display_assign_debug.push_back(assign_debug);
      const float* det =
          result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
      const std::array<float, 4> bbox = bbox4_from_ptr(det);
      if (display_id > 0 && finite_bbox(bbox)) {
        display_last_bbox_[display_id] = bbox;
        display_last_frame_[display_id] = frame_id_;
        display_public_last_frame_[public_id] = frame_id_;
        display_last_public_[display_id] = public_id;
      }
    }
    compact_result_to_displayed_targets(result);
  }

  void record_display_state_debug(FrameResult& result,
                                  const std::set<int>& active_public_ids) const {
    result.display_bindings_debug.clear();
    result.display_bindings_debug.reserve(display_id_by_public_.size());
    for (const auto& item : display_id_by_public_) {
      DisplayBindingDebug debug;
      debug.public_id = item.first;
      debug.display_id = item.second;
      const auto last_it = display_public_last_frame_.find(item.first);
      if (last_it != display_public_last_frame_.end()) {
        debug.last_frame = last_it->second;
        debug.missing_frames = frame_id_ - last_it->second;
      }
      debug.active_public = active_public_ids.find(item.first) != active_public_ids.end();
      debug.ttl_reserved =
          cfg_.display_id_public_binding_ttl > 0 &&
          debug.missing_frames >= 0 &&
          debug.missing_frames <= cfg_.display_id_public_binding_ttl;
      result.display_bindings_debug.push_back(debug);
    }
    result.display_free_ids_debug.assign(free_display_ids_.begin(), free_display_ids_.end());
  }

  void mark_display_assign_replaced_by_area(FrameResult& result,
                                            int replaced_public_id,
                                            int display_id,
                                            int replaced_index,
                                            int by_public_id) const {
    for (DisplayAssignDebug& debug : result.display_assign_debug) {
      if (debug.public_id != replaced_public_id) {
        continue;
      }
      debug.hidden = true;
      debug.action = "replaced_by_larger_area_public";
      debug.reason = "display_slot_taken_by_larger_bbox";
      debug.replaced_public_id = by_public_id;
      debug.replaced_display_id = display_id;
      debug.replaced_original_index = replaced_index;
      return;
    }
  }

  bool display_slot_bbox_eligible(const std::array<float, 4>& bbox) const {
    if (!finite_bbox(bbox)) {
      return false;
    }
    if (cfg_.display_id_min_box_size <= 0.0f) {
      return true;
    }
    return bbox_min_side(bbox) >= cfg_.display_id_min_box_size;
  }

  DisplayReplacementChoice choose_display_area_replacement(
      int new_public_id,
      int new_raw_track_id,
      const std::array<float, 4>& new_bbox,
      const std::set<int>& active_public_ids,
      const std::map<int, int>& active_public_index,
      const std::map<int, std::array<float, 4>>& active_public_bbox,
      const std::set<int>& display_eligible_public_ids,
      const std::map<int, int>& display_eligible_public_index,
      const std::map<int, std::array<float, 4>>& display_eligible_public_bbox,
      const std::map<int, int>& visible_public_raw_id) {
    DisplayReplacementChoice choice;
    if (cfg_.display_id_max_count <= 0 || new_public_id <= 0 ||
        !finite_bbox(new_bbox)) {
      choice.reject_reason = "replace_invalid_candidate";
      return choice;
    }
    if (new_raw_track_id <= 0) {
      choice.reject_reason = "replace_candidate_not_live_raw";
      return choice;
    }
    const float new_area = bbox_area4(new_bbox);
    int worst_public_id = 0;
    int worst_display_id = 0;
    int worst_index = -1;
    int worst_hits = 0;
    bool worst_skip_area_ratio = false;
    std::string worst_pending_reason = "display_area_replace_pending";
    float worst_area = std::numeric_limits<float>::infinity();
    int stale_missing_frames = -1;
    const int stale_protect_frames =
        cfg_.coast_frames > 0 ? cfg_.coast_frames : cfg_.display_id_public_binding_ttl;

    // Active publics whose current bbox is too small for a display slot should
    // not keep a slot from an eligible target. They stay tracked/public-mapped,
    // but become first-class replacement candidates after confirmation.
    for (const auto& item : display_id_by_public_) {
      const int old_public_id = item.first;
      const int display_id = item.second;
      if (old_public_id == new_public_id || display_id <= 0 ||
          active_public_ids.find(old_public_id) == active_public_ids.end() ||
          display_eligible_public_ids.find(old_public_id) != display_eligible_public_ids.end()) {
        continue;
      }
      const auto raw_it = visible_public_raw_id.find(old_public_id);
      if (raw_it == visible_public_raw_id.end() || raw_it->second <= 0) {
        continue;
      }
      const auto bbox_it = active_public_bbox.find(old_public_id);
      if (bbox_it == active_public_bbox.end() || !finite_bbox(bbox_it->second)) {
        continue;
      }
      const float old_area = bbox_area4(bbox_it->second);
      const auto index_it = active_public_index.find(old_public_id);
      const bool is_worse =
          worst_public_id <= 0 ||
          old_area < worst_area ||
          (std::fabs(old_area - worst_area) <= 1.0e-3f &&
           display_id < worst_display_id);
      if (is_worse) {
        worst_public_id = old_public_id;
        worst_display_id = display_id;
        worst_index =
            index_it == active_public_index.end() ? -1 : index_it->second;
        worst_hits = public_consecutive_seen_count(old_public_id);
        worst_area = old_area;
        worst_skip_area_ratio = true;
        worst_pending_reason = "display_small_box_replace_pending";
      }
    }

    // Prefer stale non-live display bindings after the raw track has been
    // missing beyond the coast protection window. Short occlusions still keep
    // their display slot; stale coast slots stop blocking new live tracks.
    for (const auto& item : display_id_by_public_) {
      const int old_public_id = item.first;
      const int display_id = item.second;
      if (old_public_id == new_public_id || display_id <= 0) {
        continue;
      }
      const auto raw_it = visible_public_raw_id.find(old_public_id);
      const bool old_live =
          active_public_ids.find(old_public_id) != active_public_ids.end() &&
          raw_it != visible_public_raw_id.end() && raw_it->second > 0;
      if (old_live) {
        continue;
      }
      const auto last_active_it = public_last_frame_.find(old_public_id);
      const int raw_missing_frames =
          last_active_it == public_last_frame_.end()
              ? stale_protect_frames + 1
              : frame_id_ - last_active_it->second;
      if (raw_missing_frames <= stale_protect_frames) {
        continue;
      }

      std::array<float, 4> old_bbox{{0.0f, 0.0f, 0.0f, 0.0f}};
      const auto active_bbox_it = active_public_bbox.find(old_public_id);
      if (active_bbox_it != active_public_bbox.end() &&
          finite_bbox(active_bbox_it->second)) {
        old_bbox = active_bbox_it->second;
      } else {
        const auto display_bbox_it = display_last_bbox_.find(display_id);
        if (display_bbox_it != display_last_bbox_.end() &&
            finite_bbox(display_bbox_it->second)) {
          old_bbox = display_bbox_it->second;
        } else {
          const auto public_bbox_it = public_last_bbox_.find(old_public_id);
          if (public_bbox_it != public_last_bbox_.end() &&
              finite_bbox(public_bbox_it->second)) {
            old_bbox = public_bbox_it->second;
          }
        }
      }
      const float old_area = finite_bbox(old_bbox) ? bbox_area4(old_bbox) : 0.0f;
      const auto index_it = active_public_index.find(old_public_id);
      const bool is_staler =
          worst_public_id <= 0 || raw_missing_frames > stale_missing_frames ||
          (raw_missing_frames == stale_missing_frames &&
           (old_area < worst_area ||
            (std::fabs(old_area - worst_area) <= 1.0e-3f &&
             display_id < worst_display_id)));
      if (is_staler) {
        worst_public_id = old_public_id;
        worst_display_id = display_id;
        worst_index =
            index_it == active_public_index.end() ? -1 : index_it->second;
        worst_hits = public_consecutive_seen_count(old_public_id);
        worst_area = old_area;
        worst_skip_area_ratio = true;
        worst_pending_reason = "display_stale_coast_replace_pending";
        stale_missing_frames = raw_missing_frames;
      }
    }

    if (worst_public_id <= 0) {
      for (const auto& item : display_id_by_public_) {
        const int old_public_id = item.first;
        const int display_id = item.second;
        if (old_public_id == new_public_id || display_id <= 0 ||
            display_eligible_public_ids.find(old_public_id) == display_eligible_public_ids.end()) {
          continue;
        }
        const auto index_it = display_eligible_public_index.find(old_public_id);
        const auto bbox_it = display_eligible_public_bbox.find(old_public_id);
        if (index_it == display_eligible_public_index.end() ||
            bbox_it == display_eligible_public_bbox.end() ||
            !finite_bbox(bbox_it->second)) {
          continue;
        }
        const auto raw_it = visible_public_raw_id.find(old_public_id);
        const bool old_live =
            raw_it != visible_public_raw_id.end() && raw_it->second > 0;
        if (!old_live) {
          continue;
        }
        const float old_area = bbox_area4(bbox_it->second);
        const int old_hits = public_consecutive_seen_count(old_public_id);
        const bool is_worse =
            worst_public_id <= 0 ||
            (std::fabs(old_area - worst_area) > 1.0e-3f
                 ? old_area < worst_area
                 : old_hits < worst_hits);
        if (is_worse) {
          worst_public_id = old_public_id;
          worst_display_id = display_id;
          worst_index = index_it->second;
          worst_hits = old_hits;
          worst_area = old_area;
          worst_skip_area_ratio = false;
          worst_pending_reason = "display_area_replace_pending";
        }
      }
    }
    if (worst_public_id <= 0 || worst_display_id <= 0 ||
        !std::isfinite(worst_area)) {
      choice.reject_reason = "replace_no_active_display_candidate";
      return choice;
    }
    if (!worst_skip_area_ratio &&
        new_area < worst_area * cfg_.display_id_replace_area_ratio) {
      choice.replaced_public_id = worst_public_id;
      choice.replaced_index = worst_index;
      choice.replaced_hits = worst_hits;
      choice.replaced_area = worst_area;
      choice.reject_reason = "replace_area_not_large_enough";
      return choice;
    }
    const auto key = std::make_pair(new_public_id, worst_public_id);
    PendingDisplayAreaReplacement& pending = pending_display_area_replacements_[key];
    if (pending.last_frame == frame_id_ - 1 &&
        pending.new_public_id == new_public_id &&
        pending.old_public_id == worst_public_id) {
      pending.hits += 1;
    } else {
      pending.hits = 1;
    }
    pending.new_public_id = new_public_id;
    pending.old_public_id = worst_public_id;
    pending.last_frame = frame_id_;
    choice.pending_hits = pending.hits;
    if (pending.hits < cfg_.display_id_replace_confirm_frames) {
      choice.replaced_public_id = worst_public_id;
      choice.replaced_index = worst_index;
      choice.replaced_hits = worst_hits;
      choice.replaced_area = worst_area;
      choice.reject_reason =
          worst_pending_reason.empty() ? "display_area_replace_pending"
                                       : worst_pending_reason;
      return choice;
    }
    choice.display_id = worst_display_id;
    choice.replaced_public_id = worst_public_id;
    choice.replaced_index = worst_index;
    choice.replaced_hits = worst_hits;
    choice.replaced_area = worst_area;
    choice.replace = true;
    pending_display_area_replacements_.erase(key);
    return choice;
  }

  DisplayReuseChoice choose_reusable_display_id(
      const std::array<float, 4>& bbox,
      const std::set<int>& active_public_ids,
      int public_id) const {
    if (!cfg_.display_id_reuse_spatial) {
      return choose_unconditional_reusable_display_id(public_id);
    }
    if (cfg_.display_id_reuse_max_frames <= 0 || !finite_bbox(bbox)) {
      return choose_unconditional_reusable_display_id(public_id);
    }

    DisplayReuseChoice best_choice;
    float best_score = std::numeric_limits<float>::infinity();
    auto consider_display = [&](int display_id) {
      DisplayReuseCandidateDebug candidate;
      candidate.display_id = display_id;
      const auto owner_it = display_last_public_.find(display_id);
      candidate.previous_public_id =
          owner_it == display_last_public_.end() ? 0 : owner_it->second;
      if (display_id_reserved_by_other_public(display_id, public_id)) {
        candidate.reject_reason = "reserved_by_public_ttl";
        best_choice.candidates.push_back(candidate);
        return;
      }
      const auto bbox_it = display_last_bbox_.find(display_id);
      const auto frame_it = display_last_frame_.find(display_id);
      if (bbox_it == display_last_bbox_.end() ||
          frame_it == display_last_frame_.end() ||
          !finite_bbox(bbox_it->second)) {
        candidate.reject_reason = "missing_last_bbox_or_frame";
        best_choice.candidates.push_back(candidate);
        return;
      }
      const int missing_frames = frame_id_ - frame_it->second;
      candidate.missing_frames = missing_frames;
      if (missing_frames <= 0 || missing_frames > cfg_.display_id_reuse_max_frames) {
        candidate.reject_reason = "outside_reuse_window";
        best_choice.candidates.push_back(candidate);
        return;
      }
      if (should_delay_existing_public_display_reuse(public_id, missing_frames)) {
        candidate.reject_reason = "existing_public_wait_for_display_cooldown";
        best_choice.wait_for_free_display = true;
        best_choice.candidates.push_back(candidate);
        return;
      }
      const float size_score = size_ratio_score(bbox, bbox_it->second);
      candidate.size_score = size_score;
      if (size_score < cfg_.display_id_reuse_size_ratio_thresh) {
        candidate.reject_reason = "size_ratio_too_low";
        best_choice.candidates.push_back(candidate);
        return;
      }
      const float dist = inheritance_center_distance_norm(bbox, bbox_it->second);
      candidate.dist = dist;
      if (dist >= cfg_.display_id_reuse_center_thresh) {
        candidate.reject_reason = "center_distance_too_far";
        best_choice.candidates.push_back(candidate);
        return;
      }
      if (display_reuse_ambiguous_with_active_publics(
              bbox, display_id, dist, active_public_ids, public_id)) {
        candidate.reject_reason = "ambiguous_with_active_public";
        best_choice.candidates.push_back(candidate);
        return;
      }
      candidate.spatial = true;
      candidate.reject_reason = "ok";
      best_choice.candidates.push_back(candidate);
      const float time_score =
          static_cast<float>(missing_frames) /
          static_cast<float>(std::max(1, cfg_.display_id_reuse_max_frames));
      const float score = dist + time_score * 0.10f;
      if (score < best_score) {
        best_score = score;
        best_choice.display_id = display_id;
        best_choice.previous_public_id = candidate.previous_public_id;
        best_choice.spatial = true;
      }
    };

    for (int display_id : free_display_ids_) {
      consider_display(display_id);
    }

    if (best_choice.display_id > 0) {
      return best_choice;
    }
    DisplayReuseChoice fallback_choice = choose_unconditional_reusable_display_id(public_id);
    fallback_choice.candidates.insert(
        fallback_choice.candidates.begin(),
        best_choice.candidates.begin(),
        best_choice.candidates.end());
    fallback_choice.wait_for_free_display =
        fallback_choice.wait_for_free_display || best_choice.wait_for_free_display;
    return fallback_choice;
  }

  bool display_id_reserved_by_other_public(int display_id, int public_id) const {
    if (display_id <= 0 || cfg_.display_id_public_binding_ttl <= 0) {
      return false;
    }
    for (const auto& item : display_id_by_public_) {
      const int old_public_id = item.first;
      if (old_public_id == public_id || item.second != display_id) {
        continue;
      }
      const auto last_it = display_public_last_frame_.find(old_public_id);
      if (last_it == display_public_last_frame_.end()) {
        return true;
      }
      const int missing_frames = frame_id_ - last_it->second;
      if (missing_frames <= cfg_.display_id_public_binding_ttl) {
        return true;
      }
    }
    return false;
  }

  bool display_reuse_ambiguous_with_active_publics(
      const std::array<float, 4>& bbox,
      int candidate_display_id,
      float candidate_dist,
      const std::set<int>& active_public_ids,
      int current_public_id) const {
    if (!finite_bbox(bbox) || candidate_display_id <= 0 ||
        !std::isfinite(candidate_dist)) {
      return false;
    }
    for (const auto& item : display_id_by_public_) {
      const int active_public_id = item.first;
      const int active_display_id = item.second;
      if (active_public_id == current_public_id ||
          active_display_id == candidate_display_id ||
          active_public_ids.find(active_public_id) == active_public_ids.end()) {
        continue;
      }
      const auto bbox_it = display_last_bbox_.find(active_display_id);
      if (bbox_it == display_last_bbox_.end() || !finite_bbox(bbox_it->second)) {
        continue;
      }
      const float active_dist = inheritance_center_distance_norm(bbox, bbox_it->second);
      if (active_dist < cfg_.display_id_reuse_center_thresh &&
          (active_dist - candidate_dist) < cfg_.display_id_reuse_ambiguity_margin) {
        return true;
      }
    }
    return false;
  }

  void detach_display_id_from_other_public(int display_id, int keep_public_id) {
    if (display_id <= 0) {
      return;
    }
    for (auto it = display_id_by_public_.begin(); it != display_id_by_public_.end();) {
      if (it->first != keep_public_id && it->second == display_id) {
        display_public_last_frame_.erase(it->first);
        it = display_id_by_public_.erase(it);
      } else {
        ++it;
      }
    }
  }

  DisplayReuseChoice choose_unconditional_reusable_display_id(int public_id = 0) const {
    DisplayReuseChoice choice;
    if (cfg_.display_id_reuse_fallback_min_frames <= 0 || free_display_ids_.empty()) {
      return choice;
    }

    int best_display_id = 0;
    int best_missing_frames = -1;
    for (int display_id : free_display_ids_) {
      DisplayReuseCandidateDebug candidate;
      candidate.display_id = display_id;
      const auto owner_it = display_last_public_.find(display_id);
      candidate.previous_public_id =
          owner_it == display_last_public_.end() ? 0 : owner_it->second;
      if (display_id_reserved_by_other_public(display_id, public_id)) {
        candidate.reject_reason = "fallback_reserved_by_public_ttl";
        choice.candidates.push_back(candidate);
        continue;
      }
      const auto frame_it = display_last_frame_.find(display_id);
      if (frame_it == display_last_frame_.end()) {
        candidate.reject_reason = "fallback_missing_last_frame";
        choice.candidates.push_back(candidate);
        continue;
      }
      const int missing_frames = frame_id_ - frame_it->second;
      candidate.missing_frames = missing_frames;
      if (missing_frames < cfg_.display_id_reuse_fallback_min_frames) {
        candidate.reject_reason = "fallback_too_recent";
        choice.wait_for_free_display = true;
        choice.candidates.push_back(candidate);
        continue;
      }
      candidate.reject_reason = "fallback_ok";
      choice.candidates.push_back(candidate);
      if (missing_frames > best_missing_frames ||
          (missing_frames == best_missing_frames &&
           (best_display_id <= 0 || display_id < best_display_id))) {
        best_missing_frames = missing_frames;
        best_display_id = display_id;
      }
    }
    choice.display_id = best_display_id;
    if (best_display_id > 0) {
      const auto owner_it = display_last_public_.find(best_display_id);
      choice.previous_public_id =
          owner_it == display_last_public_.end() ? 0 : owner_it->second;
    }
    choice.spatial = false;
    return choice;
  }

  bool should_delay_existing_public_display_reuse(int public_id, int missing_frames) const {
    if (public_id <= 0 || cfg_.display_id_max_count <= 0 ||
        display_id_counter_ < cfg_.display_id_max_count) {
      return false;
    }
    if (cfg_.display_id_reuse_fallback_min_frames <= 0 ||
        missing_frames >= cfg_.display_id_reuse_fallback_min_frames) {
      return false;
    }
    return public_seen_count(public_id) > 1;
  }

  bool same_mapped_public_for_raws(int raw_a, int raw_b) const {
    if (raw_a <= 0 || raw_b <= 0 || raw_a == raw_b) {
      return false;
    }
    const int public_a = mapped_public_id_for_raw(raw_a);
    const int public_b = mapped_public_id_for_raw(raw_b);
    return public_a > 0 && public_a == public_b;
  }

  void compact_result_to_displayed_targets(FrameResult& result) const {
    result.hidden_detections.clear();
    result.hidden_raw_track_ids.clear();
    result.hidden_track_ids.clear();
    result.hidden_detection_count = 0;
    if (!cfg_.display_id_reuse ||
        result.display_ids.size() < static_cast<size_t>(result.detection_count)) {
      return;
    }
    bool has_hidden = false;
    for (int i = 0; i < result.detection_count; ++i) {
      if (result.display_ids[static_cast<size_t>(i)] <= 0) {
        has_hidden = true;
        break;
      }
    }
    if (!has_hidden) {
      return;
    }

    std::vector<float> kept_detections;
    std::vector<int> kept_raw_ids;
    std::vector<int> kept_track_ids;
    std::vector<int> kept_display_ids;
    std::vector<int> kept_display_from;
    std::vector<uint8_t> kept_display_spatial;
    kept_detections.reserve(result.detections.size());
    for (int i = 0; i < result.detection_count; ++i) {
      if (result.display_ids[static_cast<size_t>(i)] <= 0) {
        const float* det =
            result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
        result.hidden_detections.insert(
            result.hidden_detections.end(), det, det + kDetectionFields);
        result.hidden_raw_track_ids.push_back(
            i < static_cast<int>(result.raw_track_ids.size())
                ? result.raw_track_ids[static_cast<size_t>(i)]
                : -1);
        result.hidden_track_ids.push_back(
            i < static_cast<int>(result.track_ids.size())
                ? result.track_ids[static_cast<size_t>(i)]
                : -1);
        result.hidden_detection_count += 1;
        continue;
      }
      const float* det =
          result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
      kept_detections.insert(kept_detections.end(), det, det + kDetectionFields);
      if (i < static_cast<int>(result.raw_track_ids.size())) {
        kept_raw_ids.push_back(result.raw_track_ids[static_cast<size_t>(i)]);
      }
      if (i < static_cast<int>(result.track_ids.size())) {
        kept_track_ids.push_back(result.track_ids[static_cast<size_t>(i)]);
      }
      kept_display_ids.push_back(result.display_ids[static_cast<size_t>(i)]);
      kept_display_from.push_back(
          i < static_cast<int>(result.display_reuse_from_public_ids.size())
              ? result.display_reuse_from_public_ids[static_cast<size_t>(i)]
              : 0);
      kept_display_spatial.push_back(
          i < static_cast<int>(result.display_reuse_spatial.size())
              ? result.display_reuse_spatial[static_cast<size_t>(i)]
              : 0);
    }
    result.detections.swap(kept_detections);
    result.raw_track_ids.swap(kept_raw_ids);
    result.track_ids.swap(kept_track_ids);
    result.display_ids.swap(kept_display_ids);
    result.display_reuse_from_public_ids.swap(kept_display_from);
    result.display_reuse_spatial.swap(kept_display_spatial);
    result.detection_count = static_cast<int>(result.display_ids.size());
  }

  void update_public_seen_counts(const FrameResult& result) {
    std::set<int> seen_this_frame;
    for (int i = 0; i < result.detection_count; ++i) {
      if (i >= static_cast<int>(result.track_ids.size())) {
        continue;
      }
      const int public_id = result.track_ids[static_cast<size_t>(i)];
      if (public_id > 0 && seen_this_frame.insert(public_id).second) {
        const auto last_it = public_last_frame_.find(public_id);
        const bool consecutive =
            last_it != public_last_frame_.end() &&
            last_it->second == frame_id_ - 1;
        public_consecutive_seen_count_[public_id] =
            consecutive ? public_consecutive_seen_count(public_id) + 1 : 1;
        public_seen_count_[public_id] += 1;
      }
    }
  }

  void update_public_last_bboxes(const FrameResult& result) {
    for (int i = 0; i < result.detection_count; ++i) {
      if (i >= static_cast<int>(result.track_ids.size())) {
        continue;
      }
      const int public_id = result.track_ids[static_cast<size_t>(i)];
      if (public_id <= 0) {
        continue;
      }
      const std::array<float, 4> bbox =
          bbox4_from_ptr(result.detections.data() + static_cast<size_t>(i) * kDetectionFields);
      if (finite_bbox(bbox)) {
        public_last_bbox_[public_id] = bbox;
        public_last_frame_[public_id] = frame_id_;
        std::array<float, kDetectionFields> cached{};
        const float* det = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
        std::copy(det, det + kDetectionFields, cached.begin());
        public_last_detection_[public_id] = cached;
      }
    }
  }

  void add_public_coasting_tracks(FrameResult& result) {
    if (!cfg_.coast_hold || cfg_.coast_frames <= 0 || public_last_detection_.empty()) {
      return;
    }

    const int real_count = result.detection_count;
    std::set<int> active_public;
    std::vector<std::array<float, 4>> active_bboxes;
    active_bboxes.reserve(static_cast<size_t>(real_count));
    for (int i = 0; i < real_count; ++i) {
      if (i >= static_cast<int>(result.track_ids.size())) {
        continue;
      }
      const int public_id = result.track_ids[static_cast<size_t>(i)];
      if (public_id > 0) {
        active_public.insert(public_id);
      }
      const std::array<float, 4> bbox =
          bbox4_from_ptr(result.detections.data() + static_cast<size_t>(i) * kDetectionFields);
      if (finite_bbox(bbox)) {
        active_bboxes.push_back(bbox);
      }
    }

    for (const auto& item : public_last_detection_) {
      const int public_id = item.first;
      if (active_public.find(public_id) != active_public.end()) {
        continue;
      }
      const auto frame_it = public_last_frame_.find(public_id);
      if (frame_it == public_last_frame_.end()) {
        continue;
      }
      const int missing_frames = frame_id_ - frame_it->second;
      if (missing_frames <= 0 || missing_frames > cfg_.coast_frames) {
        continue;
      }
      const auto bbox_it = public_last_bbox_.find(public_id);
      if (bbox_it == public_last_bbox_.end() || !finite_bbox(bbox_it->second)) {
        continue;
      }
      bool covered_by_current_output = false;
      for (const std::array<float, 4>& active_bbox : active_bboxes) {
        if (same_public_duplicate(bbox_it->second, active_bbox)) {
          covered_by_current_output = true;
          break;
        }
      }
      if (covered_by_current_output) {
        continue;
      }
      if (result.detection_count >= cfg_.max_output_dets) {
        break;
      }

      const size_t required =
          (static_cast<size_t>(result.detection_count) + 1) * kDetectionFields;
      if (result.detections.size() < required) {
        result.detections.resize(required, 0.0f);
      }
      float* out = result.detections.data() +
                   static_cast<size_t>(result.detection_count) * kDetectionFields;
      std::copy(item.second.begin(), item.second.end(), out);
      result.raw_track_ids.push_back(-1);
      result.track_ids.push_back(public_id);
      active_public.insert(public_id);
      active_bboxes.push_back(bbox_it->second);
      result.detection_count += 1;
      result.coasting_added += 1;
    }
  }

  void apply_recent_public_occlusion_recovery(FrameResult& result,
                                              const std::vector<int>& raw_ids) {
    if (public_last_bbox_.empty() || result.detection_count <= 0) {
      return;
    }

    std::set<int> active_public;
    for (int raw_id : raw_ids) {
      const auto map_it = public_id_map_.find(raw_id);
      if (map_it != public_id_map_.end() && map_it->second > 0) {
        active_public.insert(map_it->second);
      }
    }

    std::set<int> claimed_public;
    for (size_t i = 0; i < raw_ids.size(); ++i) {
      const int raw_id = raw_ids[i];
      if (raw_id <= 0 || public_id_map_.find(raw_id) != public_id_map_.end()) {
        continue;
      }
      if (i >= static_cast<size_t>(result.detection_count)) {
        continue;
      }
      const float* det = result.detections.data() + i * kDetectionFields;
      const std::array<float, 4> bbox = bbox4_from_ptr(det);
      if (!finite_bbox(bbox)) {
        continue;
      }

      const int recovery_window = std::max(cfg_.coast_frames, max_age_);
      if (recovery_window <= 0) {
        return;
      }

      const float strict_dist_thresh = std::min(cfg_.inherit_center_dist_thresh, 0.75f);
      const float relaxed_dist_thresh = cfg_.public_recover_center_dist_thresh;
      const int local_target_count =
          relaxed_dist_thresh > 0.0f
              ? count_local_current_targets(result, bbox, i, relaxed_dist_thresh)
              : 0;
      const bool local_single_target =
          relaxed_dist_thresh > 0.0f && local_target_count == 1;

      PublicRecoverDebug debug;
      debug.raw_id = raw_id;
      debug.bbox = bbox;
      debug.local_target_count = local_target_count;
      debug.local_single_target = local_single_target;
      debug.recovery_window = recovery_window;
      debug.result = "no_candidate";

      int strict_best_public = -1;
      float strict_best_dist = strict_dist_thresh;
      float strict_second_best_dist = std::numeric_limits<float>::infinity();
      int relaxed_best_public = -1;
      float relaxed_best_dist = std::numeric_limits<float>::infinity();
      float relaxed_second_best_dist = std::numeric_limits<float>::infinity();
      float second_best_dist = std::numeric_limits<float>::infinity();
      for (const auto& item : public_last_bbox_) {
        const int public_id = item.first;
        PublicRecoverCandidateDebug cand;
        cand.public_id = public_id;
        if (active_public.find(public_id) != active_public.end() ||
            claimed_public.find(public_id) != claimed_public.end()) {
          cand.reject_reason = "active_or_claimed";
          debug.candidates.push_back(cand);
          continue;
        }
        const auto frame_it = public_last_frame_.find(public_id);
        if (frame_it == public_last_frame_.end()) {
          cand.reject_reason = "no_last_frame";
          debug.candidates.push_back(cand);
          continue;
        }
        const int missing_frames = frame_id_ - frame_it->second;
        cand.missing_frames = missing_frames;
        if (missing_frames <= 0 || missing_frames > recovery_window) {
          cand.reject_reason = "outside_recovery_window";
          debug.candidates.push_back(cand);
          continue;
        }
        const std::array<float, 4>& last_bbox = item.second;
        cand.last_bbox = last_bbox;
        if (!finite_bbox(last_bbox)) {
          cand.reject_reason = "invalid_last_bbox";
          debug.candidates.push_back(cand);
          continue;
        }
        const float size_score = size_ratio_score(last_bbox, bbox);
        cand.size_score = size_score;
        if (size_score < cfg_.inherit_size_ratio_thresh) {
          cand.reject_reason = "size_gate";
          debug.candidates.push_back(cand);
          continue;
        }
        const float dist = inheritance_center_distance_norm(last_bbox, bbox);
        cand.dist = dist;
        cand.strict_duplicate = same_public_duplicate(last_bbox, bbox);
        const float nearest_other_dist =
            nearest_other_current_target_distance(result, bbox, i, relaxed_dist_thresh);
        if (std::isfinite(nearest_other_dist) &&
            nearest_other_dist < relaxed_dist_thresh &&
            (nearest_other_dist - dist) < cfg_.inherit_ambiguity_margin) {
          cand.reject_reason = "current_target_ambiguity";
          debug.candidates.push_back(cand);
          continue;
        }
        const bool strict_ok = cand.strict_duplicate && dist < strict_dist_thresh;
        cand.strict_ok = strict_ok;
        if (strict_ok) {
          if (dist < strict_best_dist) {
            strict_second_best_dist = strict_best_dist;
            strict_best_dist = dist;
            strict_best_public = public_id;
          } else if (dist < strict_second_best_dist) {
            strict_second_best_dist = dist;
          }
        }

        const bool relaxed_window_ok =
            cfg_.public_recover_max_frames > 0 && missing_frames <= cfg_.public_recover_max_frames;
        cand.relaxed_ok = local_single_target && relaxed_window_ok && dist < relaxed_dist_thresh;
        if (cand.relaxed_ok) {
          if (dist < relaxed_best_dist) {
            relaxed_second_best_dist = relaxed_best_dist;
            relaxed_best_dist = dist;
            relaxed_best_public = public_id;
          } else if (dist < relaxed_second_best_dist) {
            relaxed_second_best_dist = dist;
          }
        }
        if (cand.strict_ok || cand.relaxed_ok) {
          cand.reject_reason = "candidate";
        } else if (!local_single_target) {
          cand.reject_reason = "not_local_single";
        } else if (!relaxed_window_ok) {
          cand.reject_reason = "outside_public_recover_window";
        } else {
          cand.reject_reason = "distance_or_duplicate_gate";
        }
        debug.candidates.push_back(cand);
      }

      int best_public = strict_best_public;
      float best_dist = strict_best_dist;
      second_best_dist = strict_second_best_dist;
      debug.mode = "strict";
      if (best_public <= 0 && local_single_target) {
        best_public = relaxed_best_public;
        best_dist = relaxed_best_dist;
        second_best_dist = relaxed_second_best_dist;
        debug.mode = "relaxed";
      }

      if (best_public <= 0) {
        result.public_recover_debug.push_back(debug);
        continue;
      }
      if (std::isfinite(second_best_dist) &&
          (second_best_dist - best_dist) < cfg_.inherit_ambiguity_margin) {
        debug.chosen_public_id = best_public;
        debug.result = "ambiguity";
        result.public_recover_debug.push_back(debug);
        continue;
      }
      public_id_map_[raw_id] = best_public;
      active_public.insert(best_public);
      claimed_public.insert(best_public);
      result.inherited_ids += 1;
      debug.chosen_public_id = best_public;
      debug.result = "assigned";
      result.public_recover_debug.push_back(debug);
    }
  }

  void apply_recent_public_boundary_recovery(FrameResult& result,
                                             const std::vector<int>& raw_ids,
                                             std::set<int>& active_public,
                                             std::set<int>& claimed_public) {
    if (public_last_bbox_.empty() || result.detection_count <= 0) {
      return;
    }
    const int recent_window = std::max(1, std::min(cfg_.boundary_time_window, 3));
    for (size_t i = 0; i < raw_ids.size(); ++i) {
      const int raw_id = raw_ids[i];
      if (raw_id <= 0 || public_id_map_.find(raw_id) != public_id_map_.end()) {
        continue;
      }
      if (i >= static_cast<size_t>(result.detection_count)) {
        continue;
      }
      const float* det = result.detections.data() + i * kDetectionFields;
      const std::array<float, 4> bbox = bbox4_from_ptr(det);
      const BoundarySide appear_side = boundary_side_for_bbox(bbox);
      if (appear_side == BoundarySide::None) {
        continue;
      }

      int best_public = -1;
      float best_score = cfg_.boundary_center_dist_thresh;
      float second_best_score = std::numeric_limits<float>::infinity();
      for (const auto& item : public_last_bbox_) {
        const int public_id = item.first;
        if (active_public.find(public_id) != active_public.end() ||
            claimed_public.find(public_id) != claimed_public.end()) {
          continue;
        }
        const auto frame_it = public_last_frame_.find(public_id);
        if (frame_it == public_last_frame_.end() ||
            frame_id_ - frame_it->second > recent_window) {
          continue;
        }
        const std::array<float, 4>& last_bbox = item.second;
        const BoundarySide lost_side = boundary_side_for_bbox(last_bbox);
        if (lost_side == BoundarySide::None ||
            opposite_boundary(lost_side) != appear_side) {
          continue;
        }
        const float size_score = size_ratio_score(last_bbox, bbox);
        if (size_score < cfg_.boundary_size_ratio_thresh) {
          continue;
        }
        BoundaryLostTarget pseudo_lost;
        pseudo_lost.raw_id = -1;
        pseudo_lost.frame_index = frame_it->second;
        pseudo_lost.side = lost_side;
        pseudo_lost.bbox = last_bbox;
        const float dist = boundary_distance_score(pseudo_lost, bbox, appear_side, true);
        if (dist > cfg_.boundary_center_dist_thresh) {
          continue;
        }
        if (dist < best_score) {
          second_best_score = best_score;
          best_score = dist;
          best_public = public_id;
        } else if (dist < second_best_score) {
          second_best_score = dist;
        }
      }
      if (best_public <= 0) {
        continue;
      }
      if (std::isfinite(second_best_score) &&
          (second_best_score - best_score) < cfg_.inherit_ambiguity_margin) {
        continue;
      }
      public_id_map_[raw_id] = best_public;
      active_public.insert(best_public);
      claimed_public.insert(best_public);
      result.boundary_recovered += 1;
    }
  }

  void register_lost_targets(const std::vector<int>& current_active_raw_ids) {
    std::vector<int> lost_ids;
    for (int prev_raw : prev_active_raw_ids_) {
      if (std::find(current_active_raw_ids.begin(), current_active_raw_ids.end(), prev_raw) ==
          current_active_raw_ids.end()) {
        lost_ids.push_back(prev_raw);
      }
    }
    for (int lost_raw : lost_ids) {
      bbox_cache_.erase(lost_raw);
      if (public_id_map_.find(lost_raw) == public_id_map_.end()) {
        continue;
      }
      const auto prev_it = prev_bbox_.find(lost_raw);
      if (prev_it == prev_bbox_.end()) {
        continue;
      }
      BoundarySide side = boundary_side_for_bbox(prev_it->second);
      if (side == BoundarySide::None) {
        continue;
      }
      BoundaryLostTarget lost;
      lost.raw_id = lost_raw;
      lost.frame_index = frame_id_;
      lost.side = side;
      lost.bbox = prev_it->second;
      const auto prev_prev_it = prev_prev_bbox_.find(lost_raw);
      if (prev_prev_it != prev_prev_bbox_.end()) {
        const auto c0 = bbox_center(prev_prev_it->second);
        const auto c1 = bbox_center(prev_it->second);
        lost.velocity = std::array<float, 2>{{c1[0] - c0[0], c1[1] - c0[1]}};
        lost.has_velocity = true;
      }
      boundary_lost_.push_back(lost);
    }
    cleanup_boundary_lost();
  }

  void remove_retired_raw_ids(std::vector<int>& raw_ids) const {
    if (retired_raw_ids_this_frame_.empty()) {
      return;
    }
    raw_ids.erase(std::remove_if(raw_ids.begin(), raw_ids.end(), [&](int raw_id) {
                    return retired_raw_ids_this_frame_.find(raw_id) !=
                           retired_raw_ids_this_frame_.end();
                  }),
                  raw_ids.end());
  }

  void update_active_track_state(const std::vector<int>& current_active_raw_ids,
                                 const std::map<int, NativeTrackSnapshot>& snapshot_by_raw) {
    prev_prev_bbox_ = prev_bbox_;
    prev_bbox_.clear();
    for (int raw_id : current_active_raw_ids) {
      const auto snap_it = snapshot_by_raw.find(raw_id);
      if (snap_it != snapshot_by_raw.end()) {
        std::array<float, 4> bbox = bbox_from_last_observation(snap_it->second);
        if (!finite_bbox(bbox) && finite_bbox(snap_it->second.state)) {
          bbox = snap_it->second.state;
        }
        if (finite_bbox(bbox)) {
          prev_bbox_[raw_id] = bbox;
        }
      }
    }
    prev_active_raw_ids_ = current_active_raw_ids;
  }

  std::array<float, 4> smooth_bbox(int raw_id, const std::array<float, 4>& bbox) {
    float cx = (bbox[0] + bbox[2]) * 0.5f;
    float cy = (bbox[1] + bbox[3]) * 0.5f;
    float w = bbox[2] - bbox[0];
    float h = bbox[3] - bbox[1];
    const auto it = bbox_cache_.find(raw_id);
    if (it != bbox_cache_.end()) {
      const float alpha = cfg_.smooth_bbox_alpha;
      cx = alpha * it->second[0] + (1.0f - alpha) * cx;
      cy = alpha * it->second[1] + (1.0f - alpha) * cy;
      w = alpha * it->second[2] + (1.0f - alpha) * w;
      h = alpha * it->second[3] + (1.0f - alpha) * h;
    }
    bbox_cache_[raw_id] = std::array<float, 4>{{cx, cy, w, h}};
    return std::array<float, 4>{{cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f}};
  }

  std::array<float, 4> bbox_from_last_observation(const NativeTrackSnapshot& snapshot) const {
    return std::array<float, 4>{{snapshot.last_observation[0],
                                 snapshot.last_observation[1],
                                 snapshot.last_observation[2],
                                 snapshot.last_observation[3]}};
  }

  BoundarySide boundary_side_for_bbox(const std::array<float, 4>& bbox) const {
    const int w = meta_.process_width > 0 ? meta_.process_width : meta_.img_width;
    const int h = meta_.process_height > 0 ? meta_.process_height : meta_.img_height;
    if (w <= 0 || h <= 0 || !finite_bbox(bbox)) {
      return BoundarySide::None;
    }
    const float margin_w = std::max(1.0f, w * cfg_.boundary_margin);
    float best_overlap = 0.0f;
    BoundarySide best_side = BoundarySide::None;
    auto check = [&](BoundarySide side, float bx1, float by1, float bx2, float by2) {
      const float ix1 = std::max(bbox[0], bx1);
      const float iy1 = std::max(bbox[1], by1);
      const float ix2 = std::min(bbox[2], bx2);
      const float iy2 = std::min(bbox[3], by2);
      const float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
      const float ratio = inter / (bbox_area4(bbox) + 1.0e-6f);
      if (ratio > std::max(0.3f, best_overlap)) {
        best_overlap = ratio;
        best_side = side;
      }
    };
    check(BoundarySide::Left, 0.0f, 0.0f, margin_w, static_cast<float>(h));
    check(BoundarySide::Right, static_cast<float>(w) - margin_w, 0.0f, static_cast<float>(w), static_cast<float>(h));
    return best_side;
  }

  BoundarySide boundary_duplicate_side_for_bbox(const std::array<float, 4>& bbox) const {
    const BoundarySide strict_side = boundary_side_for_bbox(bbox);
    if (strict_side != BoundarySide::None) {
      return strict_side;
    }
    const int w = meta_.process_width > 0 ? meta_.process_width : meta_.img_width;
    const int h = meta_.process_height > 0 ? meta_.process_height : meta_.img_height;
    if (w <= 0 || h <= 0 || !finite_bbox(bbox)) {
      return BoundarySide::None;
    }
    const float margin_w = std::max(1.0f, w * cfg_.boundary_margin);
    const float left_gap = std::max(0.0f, bbox[0]);
    const float right_gap = std::max(0.0f, static_cast<float>(w) - bbox[2]);
    const bool near_left = left_gap <= margin_w;
    const bool near_right = right_gap <= margin_w;
    if (near_left && near_right) {
      return left_gap <= right_gap ? BoundarySide::Left : BoundarySide::Right;
    }
    if (near_left) {
      return BoundarySide::Left;
    }
    if (near_right) {
      return BoundarySide::Right;
    }
    return BoundarySide::None;
  }

  BoundarySide opposite_boundary(BoundarySide side) const {
    if (side == BoundarySide::Left) {
      return BoundarySide::Right;
    }
    if (side == BoundarySide::Right) {
      return BoundarySide::Left;
    }
    if (side == BoundarySide::Top) {
      return BoundarySide::Bottom;
    }
    if (side == BoundarySide::Bottom) {
      return BoundarySide::Top;
    }
    return BoundarySide::None;
  }

  float boundary_distance_score(const BoundaryLostTarget& lost,
                                const std::array<float, 4>& bbox,
                                BoundarySide appear_side,
                                bool opposite_side) const {
    const int w = meta_.process_width > 0 ? meta_.process_width : meta_.img_width;
    const auto c_lost = bbox_center(lost.bbox);
    const auto c_new = bbox_center(bbox);
    float dx = c_new[0] - c_lost[0];
    if (opposite_side && w > 0) {
      if (lost.side == BoundarySide::Left && appear_side == BoundarySide::Right) {
        dx = c_new[0] - (c_lost[0] + static_cast<float>(w));
      } else if (lost.side == BoundarySide::Right && appear_side == BoundarySide::Left) {
        dx = (c_new[0] + static_cast<float>(w)) - c_lost[0];
      }
    }
    const float dy = c_new[1] - c_lost[1];
    const float avg_h = std::max((bbox_height(lost.bbox) + bbox_height(bbox)) * 0.5f, 1.0f);
    return std::sqrt(dx * dx + dy * dy) / avg_h;
  }

  void cleanup_boundary_lost() {
    std::vector<BoundaryLostTarget> kept;
    kept.reserve(boundary_lost_.size());
    for (const BoundaryLostTarget& lost : boundary_lost_) {
      if (frame_id_ - lost.frame_index <= cfg_.boundary_time_window) {
        kept.push_back(lost);
      }
    }
    boundary_lost_.swap(kept);
  }

  void remove_boundary_lost(int raw_id) {
    std::vector<BoundaryLostTarget> kept;
    kept.reserve(boundary_lost_.size());
    for (const BoundaryLostTarget& lost : boundary_lost_) {
      if (lost.raw_id != raw_id) {
        kept.push_back(lost);
      }
    }
    boundary_lost_.swap(kept);
  }

  int public_id_for(int raw_id) {
    const auto it = public_id_map_.find(raw_id);
    if (it != public_id_map_.end()) {
      return it->second;
    }
    public_id_counter_ += 1;
    public_id_map_[raw_id] = public_id_counter_;
    return public_id_counter_;
  }

  Config cfg_;
  const MapMeta& meta_;
  HybridSortHandle handle_;
  std::map<int, int> public_id_map_;
  std::map<std::pair<int, int>, PendingPublicAlias> pending_public_aliases_;
  std::map<std::pair<int, int>, PendingRawDuplicateRetire> pending_raw_duplicate_retires_;
  std::map<int, int> public_seen_count_;
  std::map<int, int> public_consecutive_seen_count_;
  std::map<int, std::array<float, 4>> public_last_bbox_;
  std::map<int, int> public_last_frame_;
  std::map<int, std::array<float, kDetectionFields>> public_last_detection_;
  std::map<int, int> display_id_by_public_;
  std::set<int> free_display_ids_;
  std::map<int, std::array<float, 4>> display_last_bbox_;
  std::map<int, int> display_last_frame_;
  std::map<int, int> display_last_public_;
  std::map<int, int> display_public_last_frame_;
  std::vector<DisplayReleaseDebug> pending_display_release_debug_;
  std::map<std::pair<int, int>, PendingDisplayAreaReplacement> pending_display_area_replacements_;
  int public_id_counter_ = 0;
  int display_id_counter_ = 0;
  int frame_id_ = 0;
  int max_age_ = 0;
  std::set<int> retired_raw_ids_this_frame_;
  std::map<int, std::array<float, kDetectionFields>> meta_cache_;
  std::map<int, std::array<float, 4>> bbox_cache_;
  std::vector<int> prev_active_raw_ids_;
  std::map<int, std::array<float, 4>> prev_bbox_;
  std::map<int, std::array<float, 4>> prev_prev_bbox_;
  std::vector<BoundaryLostTarget> boundary_lost_;
};
