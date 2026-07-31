static void draw_sector_overlay(Image& image,
                                const Config& cfg,
                                const std::vector<TargetInfo>& targets) {
  if (!cfg.sector_output || cfg.num_sectors <= 0 || image.width <= 0 || image.height <= 0) {
    return;
  }
  const int sectors = std::max(1, cfg.num_sectors);
  const double sector_size = 360.0 / static_cast<double>(sectors);
  std::vector<bool> active(static_cast<size_t>(sectors), false);
  for (const TargetInfo& target : targets) {
    if (!target.has_azimuth || !std::isfinite(target.azimuth)) {
      continue;
    }
    double azimuth = std::fmod(target.azimuth, 360.0);
    if (azimuth < 0.0) {
      azimuth += 360.0;
    }
    const int sector =
        ((static_cast<int>(std::floor(azimuth / sector_size)) % sectors) + sectors) % sectors;
    active[static_cast<size_t>(sector)] = true;
  }

  for (int s = 0; s < sectors; ++s) {
    const int x0 = static_cast<int>(std::lround(static_cast<double>(image.width) * s / sectors));
    const int x1 = static_cast<int>(std::lround(static_cast<double>(image.width) * (s + 1) / sectors)) - 1;
    const bool has_target = active[static_cast<size_t>(s)];
    std::ostringstream label;
    label << "S" << s;
    draw_text(image, std::max(2, x0 + 4), 10, label.str(),
              has_target ? 0 : 160,
              has_target ? 0 : 160,
              has_target ? 255 : 160,
              2);
  }
  for (int s = 1; s < sectors; ++s) {
    const int x = static_cast<int>(std::lround(static_cast<double>(image.width) * s / sectors));
    draw_vertical_dashed_line(image, x, 0, image.height - 1, 12, 8, 110, 110, 110);
  }
}

static std::vector<size_t> select_inference_target_indices(
    const std::vector<TargetInfo>& targets,
    const FrameResult& result,
    int target_output_limit);

static void draw_annotation(Image& image,
                            const Config& cfg,
                            const FrameResult& result,
                            const std::vector<TargetInfo>& targets,
                            const FrameRateStats& fps) {
  draw_text(image, 12, 12, "FPS:" + std::to_string(static_cast<int>(fps.average_fps + 0.5)),
            32, 255, 255, 3);
  draw_sector_overlay(image, cfg, targets);
  const std::vector<size_t> draw_indices =
      select_inference_target_indices(targets, result, cfg.target_output_limit);
  for (size_t draw_pos = 0; draw_pos < draw_indices.size(); ++draw_pos) {
    const int i = static_cast<int>(draw_indices[draw_pos]);
    if (i < 0 || i >= result.detection_count) {
      continue;
    }
    const float* det = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
    const int public_id =
        (i < static_cast<int>(result.track_ids.size()) && result.track_ids[static_cast<size_t>(i)] > 0)
            ? result.track_ids[static_cast<size_t>(i)]
            : i + 1;
    const int display_id =
        (cfg.display_id_reuse &&
         i < static_cast<int>(result.display_ids.size()) &&
         result.display_ids[static_cast<size_t>(i)] > 0)
            ? result.display_ids[static_cast<size_t>(i)]
            : public_id;
    const int x1 = static_cast<int>(std::lround(det[0]));
    const int y1 = static_cast<int>(std::lround(det[1]));
    const int x2 = static_cast<int>(std::lround(det[2]));
    const int y2 = static_cast<int>(std::lround(det[3]));
    bool sector_target = false;
    if (cfg.sector_output && i < static_cast<int>(targets.size())) {
      const TargetInfo& target = targets[static_cast<size_t>(i)];
      sector_target = target.has_azimuth && target.has_elevation &&
                      std::isfinite(target.azimuth) && std::isfinite(target.elevation);
    }
    const uint8_t b = sector_target ? 0 : static_cast<uint8_t>((display_id * 53) % 180 + 60);
    const uint8_t g = sector_target ? 0 : static_cast<uint8_t>((display_id * 97) % 180 + 60);
    const uint8_t r = sector_target ? 255 : static_cast<uint8_t>((display_id * 131) % 180 + 60);
    draw_rect(image, x1, y1, x2, y2, b, g, r, 3);
    const float* left_mouth = det + 5 + 3 * 3;
    const float* right_mouth = det + 5 + 4 * 3;
    if (valid_keypoint(left_mouth)) {
      const int kx = static_cast<int>(std::lround(left_mouth[0]));
      const int ky = static_cast<int>(std::lround(left_mouth[1]));
      fill_rect(image, kx - 3, ky - 3, kx + 3, ky + 3, 0, 255, 255);
    }
    if (valid_keypoint(right_mouth)) {
      const int kx = static_cast<int>(std::lround(right_mouth[0]));
      const int ky = static_cast<int>(std::lround(right_mouth[1]));
      fill_rect(image, kx - 3, ky - 3, kx + 3, ky + 3, 0, 255, 255);
    }
    if (valid_keypoint(left_mouth) && valid_keypoint(right_mouth)) {
      const int cx = static_cast<int>(std::lround((left_mouth[0] + right_mouth[0]) * 0.5f));
      const int cy = static_cast<int>(std::lround((left_mouth[1] + right_mouth[1]) * 0.5f));
      fill_rect(image, cx - 4, cy - 4, cx + 4, cy + 4, 0, 255, 0);
    }
    std::ostringstream label;
    label << "ID:" << display_id;
    const TargetInfo* target =
        i < static_cast<int>(targets.size()) ? &targets[static_cast<size_t>(i)] : nullptr;
    if (target != nullptr && target->has_face_recognition && !target->face_id.empty()) {
      label << " F:" << target->face_id;
    }
    if (target != nullptr && target->has_azimuth && std::isfinite(target->azimuth)) {
      label << " A:" << static_cast<int>(std::lround(target->azimuth));
    }
    if (target != nullptr && target->has_elevation && std::isfinite(target->elevation)) {
      label << " E:" << static_cast<int>(std::lround(target->elevation));
    }
    const std::string label_text = label.str();
    const int label_scale = 2;
    const int label_pad_x = 4;
    const int label_pad_y = 3;
    const int label_w = std::max(28, text_pixel_width(label_text, label_scale) + label_pad_x * 2);
    const int label_h = 7 * label_scale + label_pad_y * 2;
    const int max_label_x = std::max(0, image.width - label_w - 1);
    const int label_x = std::max(0, std::min(x1, max_label_x));
    const int label_y = std::max(0, y1 - label_h - 2);
    fill_rect_alpha(image,
                    label_x,
                    label_y,
                    label_x + label_w - 1,
                    label_y + label_h - 1,
                    24,
                    28,
                    34,
                    0.72f);
    draw_text(image, label_x + label_pad_x, label_y + label_pad_y, label_text, b, g, r, label_scale);
  }
}

class AngleAndDistanceRuntime {
 public:
  AngleAndDistanceRuntime(const Config& cfg,
                          const MapMeta& meta,
                          const std::vector<float>* base_map_x,
                          const std::vector<float>* base_map_y)
      : cfg_(cfg), meta_(meta), base_map_x_(base_map_x), base_map_y_(base_map_y) {
    if (cfg_.fit_degree == 5) {
      coeffs_ = std::vector<double>{
          -1.1658012518547278e-11,
          1.1878284840943126e-08,
          -4.2396349229203516e-06,
          0.0005668408282723097,
          -0.19620578806477496,
          90.56394245932009,
      };
    } else {
      coeffs_ = std::vector<double>{
          -2.405707485718247e-09,
          1.9929171857454457e-06,
          -0.0005816463691548895,
          -0.1157626905427259,
          89.28693676949699,
      };
    }
    if (is_wide_angle_projection(meta_.projection_mode)) {
      const int w = meta_.process_width > 0 ? meta_.process_width : meta_.img_width;
      const int h = meta_.process_height > 0 ? meta_.process_height : meta_.img_height;
      const double center_x = w > 1 ? (static_cast<double>(w) - 1.0) * 0.5 : 0.0;
      const double center_y = h > 1 ? (static_cast<double>(h) - 1.0) * 0.5 : 0.0;
      wide_center_azimuth_ = wide_raw_azimuth(center_x);
      wide_center_elevation_ = wide_raw_elevation(center_y);
    }
  }

  std::vector<TargetInfo> build_targets(const FrameResult& result) {
    std::vector<TargetInfo> targets;
    targets.reserve(static_cast<size_t>(result.detection_count));
    for (int i = 0; i < result.detection_count; ++i) {
      const float* det = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
      const int track_id =
          (i < static_cast<int>(result.track_ids.size()) && result.track_ids[static_cast<size_t>(i)] > 0)
              ? result.track_ids[static_cast<size_t>(i)]
              : i + 1;
      const int raw_track_id =
          (i < static_cast<int>(result.raw_track_ids.size()) &&
           result.raw_track_ids[static_cast<size_t>(i)] > 0)
              ? result.raw_track_ids[static_cast<size_t>(i)]
              : -1;
      const int display_id =
          (cfg_.display_id_reuse &&
           i < static_cast<int>(result.display_ids.size()) &&
           result.display_ids[static_cast<size_t>(i)] > 0)
              ? result.display_ids[static_cast<size_t>(i)]
              : -1;
      TargetInfo info;
      info.public_id = track_id;
      info.raw_track_id = raw_track_id;
      info.display_id = display_id;
      info.has_display_id = cfg_.display_id_reuse && display_id > 0;
      info.id = info.has_display_id ? display_id : track_id;
      fill_angle(det, info);
      fill_distance(track_id, det, info);
      targets.push_back(info);
    }
    cleanup_distance_states(targets);
    return targets;
  }

 private:
  void fill_angle(const float* det, TargetInfo& info) const {
    float px = 0.0f;
    float py = 0.0f;
    pick_angle_point(det, &px, &py);
    if (is_wide_angle_projection(meta_.projection_mode)) {
      fill_wide_angle(px, py, info);
      return;
    }
    const int pano_w = meta_.process_width > 0 ? meta_.process_width : meta_.img_width;
    const int pano_h = meta_.process_height > 0 ? meta_.process_height : meta_.img_height;
    if (pano_w <= 0 || pano_h <= 1 || !finite_point(px, py)) {
      return;
    }
    double azimuth = std::fmod(360.0 * (static_cast<double>(px) / pano_w), 360.0);
    if (azimuth < 0.0) {
      azimuth += 360.0;
    }

    const double radius = meta_.radius > 0 ? static_cast<double>(meta_.radius) : 500.0;
    const double cx = meta_.center_x > 0 ? static_cast<double>(meta_.center_x) : 922.0;
    const double cy = meta_.center_y > 0 ? static_cast<double>(meta_.center_y) : 564.0;
    double x_fisheye = 0.0;
    double y_fisheye = 0.0;
    if (has_base_map()) {
      const int map_w = meta_.base_map_width;
      const int map_h = meta_.base_map_height;
      const int xi = std::max(0, std::min(static_cast<int>(px), map_w - 1));
      const int yi = std::max(0, std::min(static_cast<int>(py), map_h - 1));
      const size_t offset = static_cast<size_t>(yi) * static_cast<size_t>(map_w) +
                            static_cast<size_t>(xi);
      x_fisheye = static_cast<double>((*base_map_x_)[offset]);
      y_fisheye = static_cast<double>((*base_map_y_)[offset]);
    } else {
      const double angle = -2.0 * kPi * static_cast<double>(px) / static_cast<double>(pano_w);
      const double y_original = py;
      const double r_ratio = std::max(0.0, std::min(y_original / static_cast<double>(pano_h - 1), 1.0));
      x_fisheye = cx + r_ratio * radius * std::cos(angle);
      y_fisheye = cy + r_ratio * radius * std::sin(angle);
    }
    const double rr = std::sqrt((x_fisheye - cx) * (x_fisheye - cx) +
                                (y_fisheye - cy) * (y_fisheye - cy));

    info.has_azimuth = true;
    info.has_elevation = true;
    info.azimuth = round_to(azimuth, 1000.0);
    info.elevation = round_to(polyval(rr), 1000.0);
  }

  void fill_wide_angle(float px, float py, TargetInfo& info) const {
    if (!finite_point(px, py) || meta_.wide_fx <= 0.0 || meta_.wide_fy <= 0.0) {
      return;
    }
    const double azimuth = wide_raw_azimuth(static_cast<double>(px)) - wide_center_azimuth_;
    const double elevation = wide_raw_elevation(static_cast<double>(py)) - wide_center_elevation_;
    info.has_azimuth = true;
    info.has_elevation = true;
    info.azimuth = round_to(azimuth, 1000.0);
    info.elevation = round_to(elevation, 1000.0);
  }

  double wide_raw_azimuth(double x) const {
    if (meta_.wide_fx <= 0.0) {
      return 0.0;
    }
    return std::atan2((x - meta_.wide_cx) / meta_.wide_fx, 1.0) * 180.0 / kPi;
  }

  double wide_raw_elevation(double y) const {
    if (meta_.wide_fy <= 0.0) {
      return 0.0;
    }
    return std::atan2((y - meta_.wide_cy) / meta_.wide_fy, 1.0) * 180.0 / kPi;
  }

  void pick_angle_point(const float* det, float* out_x, float* out_y) const {
    const float* left_mouth = det + 5 + 3 * 3;
    const float* right_mouth = det + 5 + 4 * 3;
    const float* face_nose = det + 5 + 2 * 3;
    const float* fallback0 = det + 5;
    if (valid_keypoint(left_mouth) && valid_keypoint(right_mouth)) {
      *out_x = (left_mouth[0] + right_mouth[0]) * 0.5f;
      *out_y = (left_mouth[1] + right_mouth[1]) * 0.5f;
      return;
    }
    if (valid_keypoint(left_mouth)) {
      *out_x = left_mouth[0];
      *out_y = left_mouth[1];
      return;
    }
    if (valid_keypoint(right_mouth)) {
      *out_x = right_mouth[0];
      *out_y = right_mouth[1];
      return;
    }
    if (valid_keypoint(face_nose)) {
      *out_x = face_nose[0];
      *out_y = face_nose[1];
      return;
    }
    if (valid_keypoint(fallback0)) {
      *out_x = fallback0[0];
      *out_y = fallback0[1];
      return;
    }
    *out_x = (det[0] + det[2]) * 0.5f;
    *out_y = det[1] + 0.12f * (det[3] - det[1]);
  }

  bool has_base_map() const {
    if (base_map_x_ == nullptr || base_map_y_ == nullptr) {
      return false;
    }
    if (meta_.base_map_width <= 0 || meta_.base_map_height <= 0) {
      return false;
    }
    const size_t expected =
        static_cast<size_t>(meta_.base_map_width) * static_cast<size_t>(meta_.base_map_height);
    return base_map_x_->size() == expected && base_map_y_->size() == expected;
  }

  void fill_distance(int track_id, const float* det, TargetInfo& info) {
    const float* nose = det + 5 + 0 * 3;
    const float* left_eye = det + 5 + 1 * 3;
    const float* right_eye = det + 5 + 2 * 3;
    if (!valid_keypoint(left_eye) || !valid_keypoint(right_eye)) {
      const auto it = distance_states_.find(track_id);
      if (it != distance_states_.end() && it->second.has_last_distance) {
        info.has_distance = true;
        info.distance = round_to(it->second.last_distance, 1000.0);
      }
      return;
    }

    const double dx = static_cast<double>(right_eye[0] - left_eye[0]);
    const double dy = static_cast<double>(right_eye[1] - left_eye[1]);
    const double dapp = std::sqrt(dx * dx + dy * dy);
    info.has_eye_pixel_dist = true;
    info.eye_pixel_dist = round_to(dapp, 100.0);

    if (!valid_keypoint(nose)) {
      return;
    }
    const double dapp_x = std::abs(dx);
    if (dapp_x < 1.0e-6) {
      use_last_distance(track_id, info);
      return;
    }
    const double eye_mid_x = (static_cast<double>(left_eye[0]) + right_eye[0]) * 0.5;
    const double nose_lateral = static_cast<double>(nose[0]) - eye_mid_x;
    const double nose_norm = nose_lateral / (dapp_x * 0.5);
    const double tan_yaw = std::abs(nose_norm) / 0.6;
    const double yaw_deg = std::atan(tan_yaw) * 180.0 / kPi;
    if (std::isfinite(yaw_deg)) {
      info.has_face_yaw = true;
      info.face_yaw = round_to(yaw_deg, 1000.0);
    }
    if (yaw_deg > 45.0) {
      use_last_distance(track_id, info);
      return;
    }
    const double cos_yaw = 1.0 / std::sqrt(1.0 + tan_yaw * tan_yaw);
    const double dreal = dapp / cos_yaw;
    const double denom = 0.024030 * dreal + 0.044812;
    if (denom <= 0.0) {
      use_last_distance(track_id, info);
      return;
    }
    const double distance = 1.0 / denom;
    distance_states_[track_id].has_last_distance = true;
    distance_states_[track_id].last_distance = distance;
    info.has_distance = true;
    info.distance = round_to(distance, 1000.0);
  }

  void use_last_distance(int track_id, TargetInfo& info) const {
    const auto it = distance_states_.find(track_id);
    if (it != distance_states_.end() && it->second.has_last_distance) {
      info.has_distance = true;
      info.distance = round_to(it->second.last_distance, 1000.0);
    }
  }

  void cleanup_distance_states(const std::vector<TargetInfo>& targets) {
    std::vector<int> active_ids;
    active_ids.reserve(targets.size());
    for (const TargetInfo& target : targets) {
      active_ids.push_back(target.public_id > 0 ? target.public_id : target.id);
    }
    std::map<int, AngleDistanceState> kept;
    for (const auto& item : distance_states_) {
      if (std::find(active_ids.begin(), active_ids.end(), item.first) != active_ids.end()) {
        kept[item.first] = item.second;
      }
    }
    distance_states_.swap(kept);
  }

  double polyval(double x) const {
    double y = 0.0;
    for (double c : coeffs_) {
      y = y * x + c;
    }
    return y;
  }

  const Config& cfg_;
  const MapMeta& meta_;
  const std::vector<float>* base_map_x_ = nullptr;
  const std::vector<float>* base_map_y_ = nullptr;
  std::vector<double> coeffs_;
  std::map<int, AngleDistanceState> distance_states_;
  double wide_center_azimuth_ = 0.0;
  double wide_center_elevation_ = 0.0;
};

static float output_target_score(const FrameResult& result, size_t index) {
  if (index >= static_cast<size_t>(result.detection_count)) {
    return 0.0f;
  }
  const float* det = result.detections.data() + index * kDetectionFields;
  return std::isfinite(det[4]) ? det[4] : 0.0f;
}

static float output_target_area(const FrameResult& result, size_t index) {
  if (index >= static_cast<size_t>(result.detection_count)) {
    return 0.0f;
  }
  const float* det = result.detections.data() + index * kDetectionFields;
  const float w = det[2] - det[0];
  const float h = det[3] - det[1];
  if (!std::isfinite(w) || !std::isfinite(h) || w <= 0.0f || h <= 0.0f) {
    return 0.0f;
  }
  return w * h;
}

static std::vector<size_t> select_inference_target_indices(
    const std::vector<TargetInfo>& targets,
    const FrameResult& result,
    int target_output_limit) {
  std::vector<size_t> indices;
  indices.reserve(targets.size());
  for (size_t i = 0; i < targets.size(); ++i) {
    indices.push_back(i);
  }
  if (target_output_limit <= 0 ||
      indices.size() <= static_cast<size_t>(target_output_limit)) {
    return indices;
  }

  std::stable_sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
    const float score_a = output_target_score(result, a);
    const float score_b = output_target_score(result, b);
    if (std::fabs(score_a - score_b) > 1.0e-6f) {
      return score_a > score_b;
    }
    const float area_a = output_target_area(result, a);
    const float area_b = output_target_area(result, b);
    if (std::fabs(area_a - area_b) > 1.0e-3f) {
      return area_a > area_b;
    }
    const int id_a = targets[a].has_display_id ? targets[a].display_id : targets[a].public_id;
    const int id_b = targets[b].has_display_id ? targets[b].display_id : targets[b].public_id;
    if (id_a != id_b) {
      return id_a < id_b;
    }
    return a < b;
  });
  indices.resize(static_cast<size_t>(target_output_limit));
  return indices;
}

static std::string build_inference_json(const Config& cfg,
                                        const std::vector<TargetInfo>& targets,
                                        int frame_id,
                                        bool debug_keypoints,
                                        bool include_face_recognition,
                                        const FrameResult& result,
	                                        const FrameRateStats& fps) {
	  (void)debug_keypoints;
	  std::ostringstream os;
	  os << "{\"timestamp\":" << std::fixed << std::setprecision(3) << wall_time_seconds()
	     << ",\"frame_id\":" << frame_id
	     << ",\"fps\":{\"instant\":" << fps.instant_fps
	     << ",\"average\":" << fps.average_fps
	     << ",\"frame_ms\":" << fps.frame_ms
	     << ",\"elapsed_s\":" << fps.elapsed_s
	     << ",\"frames\":" << fps.frames << "}"
	     << ",\"timings_ms\":{\"frame_total\":" << fps.frame_ms << "}"
	     << ",\"display_id_max_count\":" << cfg.display_id_max_count
	     << ",\"target_output_limit\":" << cfg.target_output_limit
	     << ",\"targets\":{";
  const std::vector<size_t> output_indices =
      select_inference_target_indices(targets, result, cfg.target_output_limit);
  for (size_t i = 0; i < output_indices.size(); ++i) {
    if (i > 0) {
      os << ",";
    }
	    const TargetInfo& target = targets[output_indices[i]];
	    const size_t result_index = output_indices[i];
	    const int output_key =
	        cfg.target_output_compact_ids ? static_cast<int>(i) + 1 : target.id;
	    const int target_id = target.id;
	    os << "\"" << output_key << "\":{\"id\":" << target_id;
	    if (target.has_display_id) {
	      os << ",\"display_id\":" << target.display_id
	         << ",\"public_id\":" << target.public_id;
	    } else if (target.public_id > 0 && target.public_id != target_id) {
	      os << ",\"public_id\":" << target.public_id;
	    }
	    if (target.raw_track_id > 0) {
	      os << ",\"raw_track_id\":" << target.raw_track_id;
	    }
	    if (result_index < static_cast<size_t>(result.detection_count) &&
	        result.detections.size() >= (result_index + 1) * kDetectionFields) {
	      const float* det = result.detections.data() + result_index * kDetectionFields;
	      os << ",\"bbox\":["
	         << std::fixed << std::setprecision(3)
	         << det[0] << "," << det[1] << "," << det[2] << "," << det[3] << "]"
	         << ",\"score\":" << det[4];
	    }
	    os << ",";
	    write_target_face_id_json(os, &target);
	    os << ",\"azimuth\":";
	    print_json_nullable(os, target.has_azimuth, target.azimuth, 3);
	    os << ",\"elevation\":";
	    print_json_nullable(os, target.has_elevation, target.elevation, 3);
	    os << ",\"face_yaw\":";
	    print_json_nullable(os, target.has_face_yaw, target.face_yaw, 3);
	    os << ",\"eye_pixel_dist\":";
    print_json_nullable(os, target.has_eye_pixel_dist, target.eye_pixel_dist, 2);
    os << ",\"distance\":";
    print_json_nullable(os, target.has_distance, target.distance, 3);
    if (include_face_recognition) {
      os << ",";
      write_face_recognition_json(os, target);
    }
    os << "}";
  }
  os << "}}";
  return os.str();
}

static std::string build_sector_json(const std::vector<TargetInfo>& targets,
                                     int frame_id,
                                     int num_sectors,
                                     bool include_face_recognition,
                                     const FrameResult& result,
                                     const FrameRateStats& fps) {
  const int sectors_count = std::max(1, num_sectors);
  const double sector_size = 360.0 / static_cast<double>(sectors_count);

  struct SectorBest {
    bool valid = false;
    double area = 0.0;
    double azimuth = 0.0;
    double elevation = 0.0;
    TargetInfo target;
  };
  std::vector<SectorBest> best(static_cast<size_t>(sectors_count));

  const int count = std::min(
      static_cast<int>(targets.size()),
      std::min(result.detection_count,
               static_cast<int>(result.detections.size() / kDetectionFields)));
  for (int i = 0; i < count; ++i) {
    const TargetInfo& target = targets[static_cast<size_t>(i)];
    if (!target.has_azimuth || !target.has_elevation ||
        !std::isfinite(target.azimuth) || !std::isfinite(target.elevation)) {
      continue;
    }
    const float* det = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
    const double w = std::max(0.0f, det[2] - det[0]);
    const double h = std::max(0.0f, det[3] - det[1]);
    const double area = w * h;
    double azimuth = std::fmod(target.azimuth, 360.0);
    if (azimuth < 0.0) {
      azimuth += 360.0;
    }
    const int sector =
        ((static_cast<int>(std::floor(azimuth / sector_size)) % sectors_count) + sectors_count) %
        sectors_count;
    SectorBest& slot = best[static_cast<size_t>(sector)];
    if (!slot.valid || area > slot.area) {
      slot.valid = true;
      slot.area = area;
      slot.azimuth = azimuth;
      slot.elevation = target.elevation;
      slot.target = target;
    }
  }

  std::ostringstream os;
  os << "{\"timestamp\":" << std::fixed << std::setprecision(3) << wall_time_seconds()
     << ",\"frame_id\":" << frame_id
     << ",\"fps\":{\"instant\":" << fps.instant_fps
     << ",\"average\":" << fps.average_fps
     << ",\"frame_ms\":" << fps.frame_ms
     << ",\"elapsed_s\":" << fps.elapsed_s
     << ",\"frames\":" << fps.frames << "}"
     << ",\"timings_ms\":{\"frame_total\":" << fps.frame_ms << "}"
     << ",\"num_sectors\":" << sectors_count
     << ",\"sectors\":{";
  for (int s = 0; s < sectors_count; ++s) {
    if (s > 0) {
      os << ",";
    }
    const SectorBest& slot = best[static_cast<size_t>(s)];
    os << "\"" << s << "\":{\"has_target\":" << (slot.valid ? "true" : "false")
       << ",\"azimuth\":";
    print_json_rounded_or_null(os, slot.valid, slot.azimuth);
    os << ",\"elevation\":";
    print_json_rounded_or_null(os, slot.valid, slot.elevation);
    if (include_face_recognition) {
      os << ",";
      if (slot.valid) {
        write_face_recognition_json(os, slot.target);
      } else {
        TargetInfo empty_target;
        write_face_recognition_json(os, empty_target);
      }
    }
    os << "}";
  }
  os << "}}";
  return os.str();
}

static std::string build_output_json(const Config& cfg,
                                     const std::vector<TargetInfo>& targets,
                                     int frame_id,
                                     const FrameResult& result,
                                     const FrameRateStats& fps) {
  if (cfg.sector_output) {
    return build_sector_json(
        targets, frame_id, cfg.num_sectors, cfg.face_rec_enabled, result, fps);
  }
  return build_inference_json(
      cfg, targets, frame_id, cfg.json_debug_keypoints, cfg.face_rec_enabled, result, fps);
}

static void print_inference_jsonl(std::ostream& os,
                                  const Config& cfg,
                                  const std::vector<TargetInfo>& targets,
                                  int frame_id,
                                  bool debug_keypoints,
                                  const FrameResult& result) {
  FrameRateStats empty_fps;
  os << build_inference_json(
            cfg, targets, frame_id, debug_keypoints, false, result, empty_fps)
     << "\n";
}

static void print_result_json(std::ostream& os,
                              const MapMeta& meta,
                              int channels,
                              int anchors,
                              const FrameResult& result,
                              const FrameRateStats& fps) {
  os << "{\n";
  os << "  \"frame_index\": " << result.frame_index << ",\n";
  os << "  \"image\": \"" << json_escape(result.image_path) << "\",\n";
  os << "  \"frame\": {\"width\": " << result.frame_width << ", \"height\": " << result.frame_height << "},\n";
  os << "  \"map\": {\"projection_mode\": \"" << json_escape(meta.projection_mode)
            << "\", \"num_slices\": " << meta.num_slices << ", \"roi_w\": " << meta.roi_w
            << ", \"roi_h\": " << meta.roi_h << ", \"imgsz\": " << meta.imgsz << "},\n";
  os << "  \"rknn_shape\": {\"channels\": " << channels << ", \"anchors\": " << anchors << "},\n";
  os << "  \"timings_ms\": {\n";
  os << "    \"opencl_upload\": " << result.remap_timings[0] << ",\n";
  os << "    \"opencl_kernel\": " << result.remap_timings[1] << ",\n";
  os << "    \"opencl_read\": " << result.remap_timings[2] << ",\n";
  os << "    \"opencl_total\": " << result.remap_timings[3] << ",\n";
  os << "    \"rknn_wall\": " << result.rknn_timings[0] << ",\n";
  os << "    \"rknn_run_max\": " << result.rknn_timings[1] << ",\n";
  os << "    \"rknn_output_max\": " << result.rknn_timings[2] << ",\n";
  os << "    \"rknn_decode_max\": " << result.rknn_timings[3] << ",\n";
  os << "    \"native_merge\": " << result.rknn_timings[4] << ",\n";
  os << "    \"tracker_total\": " << result.tracker_timings[0] << ",\n";
  os << "    \"frame_total\": " << fps.frame_ms << ",\n";
  os << "    \"instant_fps\": " << fps.instant_fps << ",\n";
  os << "    \"average_fps\": " << fps.average_fps << "\n";
  os << "  },\n";
  os << "  \"fps\": {\"instant\": " << fps.instant_fps
            << ", \"average\": " << fps.average_fps
            << ", \"frame_ms\": " << fps.frame_ms
            << ", \"elapsed_s\": " << fps.elapsed_s
            << ", \"frames\": " << fps.frames << "},\n";
  os << "  \"profile_ms\": {"
            << "\"decode\": " << result.profile_ms[kProfileDecode]
            << ", \"file_read\": " << result.profile_ms[kProfileFileRead]
            << ", \"jpeg_decode\": " << result.profile_ms[kProfileJpegDecode]
            << ", \"decode_wait\": " << result.profile_ms[kProfileDecodeWait]
            << ", \"camera_read\": " << result.profile_ms[kProfileCameraRead]
            << ", \"opencl_ensure\": " << result.profile_ms[kProfileOpenclEnsure]
            << ", \"rknn_input_alloc\": " << result.profile_ms[kProfileRknnInputAlloc]
            << ", \"opencl_run_outer\": " << result.profile_ms[kProfileOpenclRunOuter]
            << ", \"opencl_upload\": " << result.remap_timings[0]
            << ", \"opencl_kernel\": " << result.remap_timings[1]
            << ", \"opencl_read\": " << result.remap_timings[2]
            << ", \"opencl_total_inner\": " << result.remap_timings[3]
            << ", \"rknn_total_outer\": " << result.profile_ms[kProfileRknnTotalOuter]
            << ", \"bound_prepare\": " << result.profile_ms[kProfileBoundPrepare]
            << ", \"bound_import\": " << result.profile_ms[kProfileBoundImport]
            << ", \"staging_copy\": " << result.profile_ms[kProfileStagingCopy]
            << ", \"rknn_wall_inner\": " << result.rknn_timings[0]
            << ", \"rknn_run_max\": " << result.rknn_timings[1]
            << ", \"rknn_output_max\": " << result.rknn_timings[2]
            << ", \"rknn_decode_max\": " << result.rknn_timings[3]
            << ", \"native_merge\": " << result.rknn_timings[4]
            << ", \"tracker_outer\": " << result.profile_ms[kProfileTrackerOuter]
            << ", \"tracker_inner\": " << result.tracker_timings[0]
            << ", \"face_rec\": " << result.profile_ms[kProfileFaceRec]
            << ", \"angle_targets\": " << result.profile_ms[kProfileAngleTargets]
            << ", \"build_payload\": " << result.profile_ms[kProfileBuildPayload]
            << ", \"stdout\": " << result.profile_ms[kProfileStdout]
            << ", \"jsonl\": " << result.profile_ms[kProfileJsonl]
            << ", \"websocket\": " << result.profile_ms[kProfileWebSocket]
            << ", \"debug_jsonl\": " << result.profile_ms[kProfileDebugJsonl]
            << ", \"write_outputs\": " << result.profile_ms[kProfileWriteOutputs]
            << ", \"frame_total\": " << result.profile_ms[kProfileFrameTotal]
            << "},\n";
  os << "  \"merge_stats\": {\"decoded\": " << result.merge_stats[0] << ", \"nms_keep\": "
            << result.merge_stats[1] << ", \"final\": " << result.merge_stats[2] << "},\n";
  os << "  \"tracker\": {\"enabled\": " << (result.tracker_enabled ? "true" : "false")
            << ", \"raw_detections\": " << result.raw_detection_count
            << ", \"filtered_invalid\": " << result.filtered_invalid
            << ", \"tracks\": " << result.detection_count
            << ", \"created\": " << result.tracker_stats[8]
            << ", \"matches_first\": " << result.tracker_stats[5]
            << ", \"matches_byte\": " << result.tracker_stats[6]
            << ", \"matches_ocr\": " << result.tracker_stats[7]
            << ", \"new_track_center_suppressed\": " << result.tracker_stats[13]
            << ", \"byte_residual_protected\": " << result.tracker_stats[14]
            << ", \"lost_track_reconnected\": " << result.tracker_stats[15]
            << ", \"inherited\": " << result.inherited_ids
            << ", \"boundary_recovered\": " << result.boundary_recovered
            << ", \"coasting_added\": " << result.coasting_added
            << ", \"final_dedup_removed\": " << result.final_dedup_removed
            << ", \"close_small_dedup_removed\": " << result.close_small_dedup_removed
            << ", \"normal_duplicate_dedup_removed\": "
            << result.normal_duplicate_dedup_removed
            << ", \"raw_duplicate_retired\": " << result.raw_duplicate_retired << "},\n";
  os << "  \"tracker_debug\": ";
  print_tracker_debug_json(os, result);
  os << ",\n";
  os << "  \"detection_count\": " << result.detection_count << ",\n";
  os << "  \"detections\": [\n";
  for (int i = 0; i < result.detection_count; ++i) {
    const float* det = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
    os << "    {\"bbox\": ";
    print_json_array(os, det, 4);
    if (i < static_cast<int>(result.track_ids.size()) && result.track_ids[static_cast<size_t>(i)] > 0) {
      os << ", \"track_id\": " << result.track_ids[static_cast<size_t>(i)];
    }
    if (i < static_cast<int>(result.raw_track_ids.size()) &&
        result.raw_track_ids[static_cast<size_t>(i)] > 0) {
      os << ", \"raw_track_id\": " << result.raw_track_ids[static_cast<size_t>(i)];
    }
    if (i < static_cast<int>(result.display_ids.size()) &&
        result.display_ids[static_cast<size_t>(i)] > 0) {
      os << ", \"display_id\": " << result.display_ids[static_cast<size_t>(i)];
    }
    if (i < static_cast<int>(result.display_reuse_from_public_ids.size()) &&
        result.display_reuse_from_public_ids[static_cast<size_t>(i)] > 0) {
      os << ", \"display_reuse_from_public\": "
         << result.display_reuse_from_public_ids[static_cast<size_t>(i)]
         << ", \"display_reuse_spatial\": "
         << (i < static_cast<int>(result.display_reuse_spatial.size()) &&
             result.display_reuse_spatial[static_cast<size_t>(i)] != 0
                 ? "true"
                 : "false");
    }
    os << ", \"score\": " << std::fixed << std::setprecision(6) << det[4]
              << ", \"keypoints\": [";
    for (int k = 0; k < 5; ++k) {
      if (k > 0) {
        os << ",";
      }
      print_json_array(os, det + 5 + k * 3, 3);
    }
    os << "]}";
    if (i + 1 < result.detection_count) {
      os << ",";
    }
    os << "\n";
  }
  os << "  ]\n";
  os << "}\n";
}

static void print_result_jsonl(std::ostream& os,
                               const MapMeta& meta,
                               int channels,
                               int anchors,
                               const FrameResult& result,
                               const FrameRateStats& fps,
                               const std::vector<TargetInfo>* targets = nullptr) {
  os << "{\"frame_index\":" << result.frame_index
     << ",\"image\":\"" << json_escape(result.image_path) << "\""
     << ",\"frame\":{\"width\":" << result.frame_width << ",\"height\":" << result.frame_height << "}"
     << ",\"map\":{\"projection_mode\":\"" << json_escape(meta.projection_mode)
     << "\",\"num_slices\":" << meta.num_slices << ",\"roi_w\":" << meta.roi_w
     << ",\"roi_h\":" << meta.roi_h << ",\"imgsz\":" << meta.imgsz << "}"
     << ",\"rknn_shape\":{\"channels\":" << channels << ",\"anchors\":" << anchors << "}"
     << ",\"timings_ms\":{\"opencl_upload\":" << result.remap_timings[0]
     << ",\"opencl_kernel\":" << result.remap_timings[1]
     << ",\"opencl_read\":" << result.remap_timings[2]
     << ",\"opencl_total\":" << result.remap_timings[3]
     << ",\"rknn_wall\":" << result.rknn_timings[0]
     << ",\"rknn_run_max\":" << result.rknn_timings[1]
     << ",\"rknn_output_max\":" << result.rknn_timings[2]
     << ",\"rknn_decode_max\":" << result.rknn_timings[3]
     << ",\"native_merge\":" << result.rknn_timings[4]
     << ",\"tracker_total\":" << result.tracker_timings[0]
     << ",\"frame_total\":" << fps.frame_ms
     << ",\"instant_fps\":" << fps.instant_fps
     << ",\"average_fps\":" << fps.average_fps << "}"
     << ",\"fps\":{\"instant\":" << fps.instant_fps
     << ",\"average\":" << fps.average_fps
     << ",\"frame_ms\":" << fps.frame_ms
     << ",\"elapsed_s\":" << fps.elapsed_s
     << ",\"frames\":" << fps.frames << "}"
     << ",\"profile_ms\":{\"decode\":" << result.profile_ms[kProfileDecode]
     << ",\"file_read\":" << result.profile_ms[kProfileFileRead]
     << ",\"jpeg_decode\":" << result.profile_ms[kProfileJpegDecode]
     << ",\"decode_wait\":" << result.profile_ms[kProfileDecodeWait]
     << ",\"camera_read\":" << result.profile_ms[kProfileCameraRead]
     << ",\"opencl_ensure\":" << result.profile_ms[kProfileOpenclEnsure]
     << ",\"rknn_input_alloc\":" << result.profile_ms[kProfileRknnInputAlloc]
     << ",\"opencl_run_outer\":" << result.profile_ms[kProfileOpenclRunOuter]
     << ",\"opencl_upload\":" << result.remap_timings[0]
     << ",\"opencl_kernel\":" << result.remap_timings[1]
     << ",\"opencl_read\":" << result.remap_timings[2]
     << ",\"opencl_total_inner\":" << result.remap_timings[3]
     << ",\"rknn_total_outer\":" << result.profile_ms[kProfileRknnTotalOuter]
     << ",\"bound_prepare\":" << result.profile_ms[kProfileBoundPrepare]
     << ",\"bound_import\":" << result.profile_ms[kProfileBoundImport]
     << ",\"staging_copy\":" << result.profile_ms[kProfileStagingCopy]
     << ",\"rknn_wall_inner\":" << result.rknn_timings[0]
     << ",\"rknn_run_max\":" << result.rknn_timings[1]
     << ",\"rknn_output_max\":" << result.rknn_timings[2]
     << ",\"rknn_decode_max\":" << result.rknn_timings[3]
     << ",\"native_merge\":" << result.rknn_timings[4]
     << ",\"tracker_outer\":" << result.profile_ms[kProfileTrackerOuter]
     << ",\"tracker_inner\":" << result.tracker_timings[0]
     << ",\"face_rec\":" << result.profile_ms[kProfileFaceRec]
     << ",\"angle_targets\":" << result.profile_ms[kProfileAngleTargets]
     << ",\"build_payload\":" << result.profile_ms[kProfileBuildPayload]
     << ",\"stdout\":" << result.profile_ms[kProfileStdout]
     << ",\"jsonl\":" << result.profile_ms[kProfileJsonl]
     << ",\"websocket\":" << result.profile_ms[kProfileWebSocket]
     << ",\"debug_jsonl\":" << result.profile_ms[kProfileDebugJsonl]
     << ",\"write_outputs\":" << result.profile_ms[kProfileWriteOutputs]
     << ",\"frame_total\":" << result.profile_ms[kProfileFrameTotal] << "}"
     << ",\"merge_stats\":{\"decoded\":" << result.merge_stats[0]
     << ",\"nms_keep\":" << result.merge_stats[1]
     << ",\"final\":" << result.merge_stats[2] << "}"
     << ",\"tracker\":{\"enabled\":" << (result.tracker_enabled ? "true" : "false")
     << ",\"raw_detections\":" << result.raw_detection_count
     << ",\"filtered_invalid\":" << result.filtered_invalid
     << ",\"tracks\":" << result.detection_count
     << ",\"created\":" << result.tracker_stats[8]
     << ",\"matches_first\":" << result.tracker_stats[5]
     << ",\"matches_byte\":" << result.tracker_stats[6]
     << ",\"matches_ocr\":" << result.tracker_stats[7]
     << ",\"new_track_center_suppressed\":" << result.tracker_stats[13]
     << ",\"byte_residual_protected\":" << result.tracker_stats[14]
     << ",\"lost_track_reconnected\":" << result.tracker_stats[15]
     << ",\"inherited\":" << result.inherited_ids
     << ",\"boundary_recovered\":" << result.boundary_recovered
     << ",\"coasting_added\":" << result.coasting_added
     << ",\"final_dedup_removed\":" << result.final_dedup_removed
     << ",\"close_small_dedup_removed\":" << result.close_small_dedup_removed
     << ",\"normal_duplicate_dedup_removed\":" << result.normal_duplicate_dedup_removed
     << ",\"raw_duplicate_retired\":" << result.raw_duplicate_retired << "}"
     << ",\"tracker_debug\":";
  print_tracker_debug_json(os, result);
  os
     << ",\"detection_count\":" << result.detection_count
     << ",\"detections\":[";
  for (int i = 0; i < result.detection_count; ++i) {
    if (i > 0) {
      os << ",";
    }
    const float* det = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
    os << "{\"bbox\":";
    print_json_array(os, det, 4);
    if (i < static_cast<int>(result.track_ids.size()) && result.track_ids[static_cast<size_t>(i)] > 0) {
      os << ",\"track_id\":" << result.track_ids[static_cast<size_t>(i)];
    }
    if (i < static_cast<int>(result.raw_track_ids.size()) &&
        result.raw_track_ids[static_cast<size_t>(i)] > 0) {
      os << ",\"raw_track_id\":" << result.raw_track_ids[static_cast<size_t>(i)];
    }
    if (i < static_cast<int>(result.display_ids.size()) &&
        result.display_ids[static_cast<size_t>(i)] > 0) {
      os << ",\"display_id\":" << result.display_ids[static_cast<size_t>(i)];
    }
    if (i < static_cast<int>(result.display_reuse_from_public_ids.size()) &&
        result.display_reuse_from_public_ids[static_cast<size_t>(i)] > 0) {
      os << ",\"display_reuse_from_public\":"
         << result.display_reuse_from_public_ids[static_cast<size_t>(i)]
         << ",\"display_reuse_spatial\":"
         << (i < static_cast<int>(result.display_reuse_spatial.size()) &&
             result.display_reuse_spatial[static_cast<size_t>(i)] != 0
                 ? "true"
                 : "false");
    }
    os << ",";
    write_target_face_id_json(
        os, (targets != nullptr && i < static_cast<int>(targets->size()))
                ? &(*targets)[static_cast<size_t>(i)]
                : nullptr);
    os << ",\"score\":" << std::fixed << std::setprecision(6) << det[4] << ",\"keypoints\":[";
    for (int k = 0; k < 5; ++k) {
      if (k > 0) {
        os << ",";
      }
      print_json_array(os, det + 5 + k * 3, 3);
    }
    os << "]";
    if (targets != nullptr && i < static_cast<int>(targets->size())) {
      os << ",";
      write_face_recognition_json(os, (*targets)[static_cast<size_t>(i)]);
    }
    os << "}";
  }
  os << "]}\n";
}
