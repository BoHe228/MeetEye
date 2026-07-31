static float box_iou(const float* a, const float* b) {
  const float x1 = std::max(a[0], b[0]);
  const float y1 = std::max(a[1], b[1]);
  const float x2 = std::min(a[2], b[2]);
  const float y2 = std::min(a[3], b[3]);
  const float inter_w = std::max(0.0f, x2 - x1);
  const float inter_h = std::max(0.0f, y2 - y1);
  const float inter = inter_w * inter_h;
  const float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
  const float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
  const float denom = area_a + area_b - inter;
  return denom > 0.0f ? inter / denom : 0.0f;
}

static void print_json_array(std::ostream& os, const float* values, int count) {
  os << "[";
  for (int i = 0; i < count; ++i) {
    if (i > 0) {
      os << ",";
    }
    os << std::fixed << std::setprecision(3) << values[i];
  }
  os << "]";
}

template <size_t N>
static void print_json_array(std::ostream& os, const std::array<float, N>& values) {
  print_json_array(os, values.data(), static_cast<int>(N));
}

static void print_tracker_debug_json(std::ostream& os, const FrameResult& result) {
  os << "{\"snapshots\":[";
  for (size_t i = 0; i < result.tracker_snapshots_debug.size(); ++i) {
    if (i > 0) {
      os << ",";
    }
    const TrackerSnapshotDebug& s = result.tracker_snapshots_debug[i];
    os << "{\"raw_id\":" << s.raw_id
       << ",\"public_id\":" << s.public_id
       << ",\"time_since_update\":" << s.time_since_update
       << ",\"last_observation\":";
    print_json_array(os, s.last_observation);
    os << ",\"state\":";
    print_json_array(os, s.state);
    os << ",\"finite_last_observation\":" << (s.finite_last_observation ? "true" : "false")
       << ",\"lost_confirmed\":" << (s.lost_confirmed ? "true" : "false")
       << ",\"reject_reason\":\"" << json_escape(s.reject_reason) << "\"}";
  }

  os << "],\"short_inherit\":[";
  for (size_t i = 0; i < result.short_inherit_debug.size(); ++i) {
    if (i > 0) {
      os << ",";
    }
    const ShortInheritDebug& d = result.short_inherit_debug[i];
    os << "{\"raw_id\":" << d.raw_id
       << ",\"bbox\":";
    print_json_array(os, d.bbox);
    os << ",\"local_target_count\":" << d.local_target_count
       << ",\"local_multi_target\":" << (d.local_multi_target ? "true" : "false")
       << ",\"effective_center_dist_thresh\":" << d.effective_center_dist_thresh
       << ",\"chosen_old_raw_id\":" << d.chosen_old_raw_id
       << ",\"chosen_public_id\":" << d.chosen_public_id
       << ",\"result\":\"" << json_escape(d.result) << "\""
       << ",\"candidates\":[";
    for (size_t j = 0; j < d.candidates.size(); ++j) {
      if (j > 0) {
        os << ",";
      }
      const ShortInheritCandidateDebug& c = d.candidates[j];
      os << "{\"old_raw_id\":" << c.old_raw_id
         << ",\"old_public_id\":" << c.old_public_id
         << ",\"bbox\":";
      print_json_array(os, c.bbox);
      os << ",\"dist\":" << c.dist
         << ",\"size_score\":" << c.size_score
         << ",\"used_state\":" << (c.used_state ? "true" : "false")
         << ",\"reject_reason\":\"" << json_escape(c.reject_reason) << "\"}";
    }
    os << "]}";
  }

  os << "],\"public_recover\":[";
  for (size_t i = 0; i < result.public_recover_debug.size(); ++i) {
    if (i > 0) {
      os << ",";
    }
    const PublicRecoverDebug& d = result.public_recover_debug[i];
    os << "{\"raw_id\":" << d.raw_id
       << ",\"bbox\":";
    print_json_array(os, d.bbox);
    os << ",\"local_target_count\":" << d.local_target_count
       << ",\"local_single_target\":" << (d.local_single_target ? "true" : "false")
       << ",\"recovery_window\":" << d.recovery_window
       << ",\"chosen_public_id\":" << d.chosen_public_id
       << ",\"mode\":\"" << json_escape(d.mode) << "\""
       << ",\"result\":\"" << json_escape(d.result) << "\""
       << ",\"candidates\":[";
    for (size_t j = 0; j < d.candidates.size(); ++j) {
      if (j > 0) {
        os << ",";
      }
      const PublicRecoverCandidateDebug& c = d.candidates[j];
      os << "{\"public_id\":" << c.public_id
         << ",\"missing_frames\":" << c.missing_frames
         << ",\"last_bbox\":";
      print_json_array(os, c.last_bbox);
      os << ",\"dist\":" << c.dist
         << ",\"size_score\":" << c.size_score
         << ",\"strict_duplicate\":" << (c.strict_duplicate ? "true" : "false")
         << ",\"strict_ok\":" << (c.strict_ok ? "true" : "false")
         << ",\"relaxed_ok\":" << (c.relaxed_ok ? "true" : "false")
         << ",\"reject_reason\":\"" << json_escape(c.reject_reason) << "\"}";
    }
    os << "]}";
  }
  os << "],\"display_release\":[";
  for (size_t i = 0; i < result.display_release_debug.size(); ++i) {
    if (i > 0) {
      os << ",";
    }
    const DisplayReleaseDebug& d = result.display_release_debug[i];
    os << "{\"public_id\":" << d.public_id
       << ",\"display_id\":" << d.display_id
       << ",\"frame_id\":" << d.frame_id
       << ",\"last_frame\":" << d.last_frame
       << ",\"missing_frames\":" << d.missing_frames
       << ",\"reason\":\"" << json_escape(d.reason) << "\"}";
  }
  os << "],\"display_bindings\":[";
  for (size_t i = 0; i < result.display_bindings_debug.size(); ++i) {
    if (i > 0) {
      os << ",";
    }
    const DisplayBindingDebug& d = result.display_bindings_debug[i];
    os << "{\"public_id\":" << d.public_id
       << ",\"display_id\":" << d.display_id
       << ",\"last_frame\":" << d.last_frame
       << ",\"missing_frames\":" << d.missing_frames
       << ",\"active_public\":" << (d.active_public ? "true" : "false")
       << ",\"ttl_reserved\":" << (d.ttl_reserved ? "true" : "false")
       << "}";
  }
  os << "],\"display_free_ids\":[";
  for (size_t i = 0; i < result.display_free_ids_debug.size(); ++i) {
    if (i > 0) {
      os << ",";
    }
    os << result.display_free_ids_debug[i];
  }
  os << "],\"display_assign\":[";
  for (size_t i = 0; i < result.display_assign_debug.size(); ++i) {
    if (i > 0) {
      os << ",";
    }
    const DisplayAssignDebug& d = result.display_assign_debug[i];
    os << "{\"original_index\":" << d.original_index
       << ",\"public_id\":" << d.public_id
       << ",\"raw_track_id\":" << d.raw_track_id
       << ",\"assigned_display_id\":" << d.assigned_display_id
       << ",\"previous_display_id\":" << d.previous_display_id
       << ",\"reused_from_public_id\":" << d.reused_from_public_id
       << ",\"replaced_public_id\":" << d.replaced_public_id
       << ",\"replaced_display_id\":" << d.replaced_display_id
       << ",\"replaced_original_index\":" << d.replaced_original_index
       << ",\"public_consecutive_seen\":" << d.public_consecutive_seen
       << ",\"replaced_public_consecutive_seen\":" << d.replaced_public_consecutive_seen
       << ",\"replace_pending_hits\":" << d.replace_pending_hits
       << ",\"replace_confirm_frames\":" << d.replace_confirm_frames
       << ",\"display_counter\":" << d.display_counter
       << ",\"display_max_count\":" << d.display_max_count
       << ",\"bbox_area\":" << d.bbox_area
       << ",\"replaced_bbox_area\":" << d.replaced_bbox_area
       << ",\"had_existing_binding\":" << (d.had_existing_binding ? "true" : "false")
       << ",\"reuse_spatial\":" << (d.reuse_spatial ? "true" : "false")
       << ",\"wait_for_free_display\":" << (d.wait_for_free_display ? "true" : "false")
       << ",\"hidden\":" << (d.hidden ? "true" : "false")
       << ",\"action\":\"" << json_escape(d.action) << "\""
       << ",\"reason\":\"" << json_escape(d.reason) << "\""
       << ",\"reuse_candidates\":[";
    for (size_t j = 0; j < d.reuse_candidates.size(); ++j) {
      if (j > 0) {
        os << ",";
      }
      const DisplayReuseCandidateDebug& c = d.reuse_candidates[j];
      os << "{\"display_id\":" << c.display_id
         << ",\"previous_public_id\":" << c.previous_public_id
         << ",\"missing_frames\":" << c.missing_frames
         << ",\"dist\":" << c.dist
         << ",\"size_score\":" << c.size_score
         << ",\"spatial\":" << (c.spatial ? "true" : "false")
         << ",\"reject_reason\":\"" << json_escape(c.reject_reason) << "\"}";
    }
    os << "]}";
  }
  os << "]}";
}

static double wall_time_seconds() {
  using clock = std::chrono::system_clock;
  const auto now = clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now).count() / 1000.0;
}

static double steady_seconds() {
  using clock = std::chrono::steady_clock;
  const auto now = clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::microseconds>(now).count() / 1000000.0;
}

static std::string default_output_jsonl_path() {
  mkdir("board_output", 0775);
  std::time_t now = std::time(nullptr);
  std::tm tm{};
  localtime_r(&now, &tm);
  char buf[128] = {0};
  std::strftime(buf, sizeof(buf), "board_output/board_%Y%m%d_%H%M%S.jsonl", &tm);
  return std::string(buf);
}

static std::string system_profile_path_for(const Config& cfg) {
  std::string base = !cfg.debug_jsonl_path.empty() ? cfg.debug_jsonl_path : cfg.output_jsonl_path;
  if (base.empty()) {
    mkdir("board_output", 0775);
    base = "board_output/board_cpp_system.jsonl";
  }
  return stem_without_ext(base) + "_system_profile.jsonl";
}

static std::string system_summary_path_for(const std::string& system_profile_path) {
  return stem_without_ext(system_profile_path) + "_summary.json";
}

struct MetricStats {
  int count = 0;
  double sum = 0.0;
  double min = 0.0;
  double max = 0.0;

  void add(double value) {
    if (!std::isfinite(value)) {
      return;
    }
    if (count == 0) {
      min = value;
      max = value;
    } else {
      min = std::min(min, value);
      max = std::max(max, value);
    }
    sum += value;
    count += 1;
  }
};

static bool extract_json_number(const std::string& line, const std::string& key, double* value) {
  if (value == nullptr) {
    return false;
  }
  const std::string marker = "\"" + key + "\":";
  size_t pos = line.find(marker);
  if (pos == std::string::npos) {
    return false;
  }
  pos += marker.size();
  while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) {
    ++pos;
  }
  if (line.compare(pos, 4, "null") == 0) {
    return false;
  }
  char* end = nullptr;
  const double parsed = std::strtod(line.c_str() + pos, &end);
  if (end == line.c_str() + pos || !std::isfinite(parsed)) {
    return false;
  }
  *value = parsed;
  return true;
}

static void write_metric_stats_json(std::ostream& os, const MetricStats& stats) {
  if (stats.count <= 0) {
    os << "{\"avg\":null,\"max\":null,\"min\":null}";
    return;
  }
  os << "{\"avg\":" << std::fixed << std::setprecision(2) << (stats.sum / stats.count)
     << ",\"max\":" << stats.max
     << ",\"min\":" << stats.min << "}";
}

static void write_system_load_summary(const std::string& system_profile_path,
                                      const std::string& output_jsonl_path,
                                      double elapsed_s,
                                      int frames) {
  if (system_profile_path.empty() || !file_exists(system_profile_path)) {
    return;
  }
  std::ifstream in(system_profile_path.c_str());
  if (!in) {
    return;
  }
  std::map<std::string, MetricStats> stats;
  const std::vector<std::string> keys = {
      "cpu_percent", "memory_percent", "memory_available_mb",
      "gpu_percent", "gpu_freq_hz", "npu_percent",
      "npu_core0_percent", "npu_core1_percent", "npu_core2_percent",
      "npu_freq_hz", "thermal_max_c"};
  std::string line;
  int samples = 0;
  std::string first_sample;
  std::string last_sample;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty()) {
      continue;
    }
    if (first_sample.empty()) {
      first_sample = line;
    }
    last_sample = line;
    samples += 1;
    for (const std::string& key : keys) {
      double value = 0.0;
      if (extract_json_number(line, key, &value)) {
        stats[key].add(value);
      }
    }
  }
  if (samples <= 0) {
    return;
  }
  const std::string summary_path = system_summary_path_for(system_profile_path);
  std::ofstream out(summary_path.c_str(), std::ios::out | std::ios::trunc);
  if (!out) {
    std::cerr << "[system-load] cannot write summary: " << summary_path << "\n";
    return;
  }
  out << "{\n";
  out << "  \"output_jsonl\": \"" << json_escape(output_jsonl_path) << "\",\n";
  out << "  \"system_profile_jsonl\": \"" << json_escape(system_profile_path) << "\",\n";
  out << "  \"frames\": " << frames << ",\n";
  out << "  \"elapsed_s\": " << std::fixed << std::setprecision(3) << elapsed_s << ",\n";
  out << "  \"avg_fps\": " << (elapsed_s > 0.0 ? static_cast<double>(frames) / elapsed_s : 0.0) << ",\n";
  out << "  \"samples\": " << samples << ",\n";
  for (size_t i = 0; i < keys.size(); ++i) {
    out << "  \"" << keys[i] << "\": ";
    write_metric_stats_json(out, stats[keys[i]]);
    out << ",\n";
  }
  out << "  \"first_sample\": " << (first_sample.empty() ? "{}" : first_sample) << ",\n";
  out << "  \"last_sample\": " << (last_sample.empty() ? "{}" : last_sample) << "\n";
  out << "}\n";
  std::cerr << "[system-load] summary JSON: " << summary_path << "\n";
  auto print_stat = [&](const std::string& key, const std::string& unit) {
    const MetricStats& s = stats[key];
    if (s.count <= 0) {
      return std::string(key + "=N/A");
    }
    std::ostringstream oss;
    oss << key << "=avg " << std::fixed << std::setprecision(2) << (s.sum / s.count)
        << unit << " max " << s.max << unit;
    return oss.str();
  };
  std::cerr << "[system-load] summary "
            << print_stat("cpu_percent", "%") << " | "
            << print_stat("gpu_percent", "%") << " | "
            << print_stat("npu_core0_percent", "%") << " | "
            << print_stat("npu_core1_percent", "%") << " | "
            << print_stat("npu_core2_percent", "%") << " | "
            << print_stat("memory_percent", "%") << " | "
            << print_stat("thermal_max_c", "C") << "\n";
}

static bool finite_point(float x, float y) {
  return std::isfinite(x) && std::isfinite(y) && !(x == 0.0f && y == 0.0f);
}

static bool valid_keypoint(const float* kp) {
  return finite_point(kp[0], kp[1]) && kp[2] > 0.0f;
}

static void put_pixel(Image& image, int x, int y, uint8_t b, uint8_t g, uint8_t r) {
  if (x < 0 || y < 0 || x >= image.width || y >= image.height) {
    return;
  }
  const size_t idx = (static_cast<size_t>(y) * image.width + static_cast<size_t>(x)) * 3;
  image.bgr[idx + 0] = b;
  image.bgr[idx + 1] = g;
  image.bgr[idx + 2] = r;
}

static void draw_line(Image& image,
                      int x0,
                      int y0,
                      int x1,
                      int y1,
                      uint8_t b,
                      uint8_t g,
                      uint8_t r) {
  const int dx = std::abs(x1 - x0);
  const int sx = x0 < x1 ? 1 : -1;
  const int dy = -std::abs(y1 - y0);
  const int sy = y0 < y1 ? 1 : -1;
  int err = dx + dy;
  while (true) {
    put_pixel(image, x0, y0, b, g, r);
    if (x0 == x1 && y0 == y1) {
      break;
    }
    const int e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x0 += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y0 += sy;
    }
  }
}

static void draw_rect(Image& image,
                      int x1,
                      int y1,
                      int x2,
                      int y2,
                      uint8_t b,
                      uint8_t g,
                      uint8_t r,
                      int thickness) {
  x1 = std::max(0, std::min(x1, image.width - 1));
  x2 = std::max(0, std::min(x2, image.width - 1));
  y1 = std::max(0, std::min(y1, image.height - 1));
  y2 = std::max(0, std::min(y2, image.height - 1));
  if (x2 < x1) {
    std::swap(x1, x2);
  }
  if (y2 < y1) {
    std::swap(y1, y2);
  }
  for (int t = 0; t < std::max(1, thickness); ++t) {
    draw_line(image, x1, y1 + t, x2, y1 + t, b, g, r);
    draw_line(image, x1, y2 - t, x2, y2 - t, b, g, r);
    draw_line(image, x1 + t, y1, x1 + t, y2, b, g, r);
    draw_line(image, x2 - t, y1, x2 - t, y2, b, g, r);
  }
}

static void fill_rect(Image& image,
                      int x1,
                      int y1,
                      int x2,
                      int y2,
                      uint8_t b,
                      uint8_t g,
                      uint8_t r) {
  x1 = std::max(0, std::min(x1, image.width - 1));
  x2 = std::max(0, std::min(x2, image.width - 1));
  y1 = std::max(0, std::min(y1, image.height - 1));
  y2 = std::max(0, std::min(y2, image.height - 1));
  if (x2 < x1) {
    std::swap(x1, x2);
  }
  if (y2 < y1) {
    std::swap(y1, y2);
  }
  for (int y = y1; y <= y2; ++y) {
    for (int x = x1; x <= x2; ++x) {
      put_pixel(image, x, y, b, g, r);
    }
  }
}

static void fill_rect_alpha(Image& image,
                            int x1,
                            int y1,
                            int x2,
                            int y2,
                            uint8_t b,
                            uint8_t g,
                            uint8_t r,
                            float alpha) {
  if (image.width <= 0 || image.height <= 0 || image.bgr.empty()) {
    return;
  }
  alpha = std::max(0.0f, std::min(1.0f, alpha));
  x1 = std::max(0, std::min(x1, image.width - 1));
  x2 = std::max(0, std::min(x2, image.width - 1));
  y1 = std::max(0, std::min(y1, image.height - 1));
  y2 = std::max(0, std::min(y2, image.height - 1));
  if (x2 < x1) {
    std::swap(x1, x2);
  }
  if (y2 < y1) {
    std::swap(y1, y2);
  }
  const float inv_alpha = 1.0f - alpha;
  for (int y = y1; y <= y2; ++y) {
    for (int x = x1; x <= x2; ++x) {
      const size_t idx = (static_cast<size_t>(y) * image.width + x) * 3U;
      image.bgr[idx + 0] = static_cast<uint8_t>(image.bgr[idx + 0] * inv_alpha + b * alpha);
      image.bgr[idx + 1] = static_cast<uint8_t>(image.bgr[idx + 1] * inv_alpha + g * alpha);
      image.bgr[idx + 2] = static_cast<uint8_t>(image.bgr[idx + 2] * inv_alpha + r * alpha);
    }
  }
}

static const char* glyph5x7(char c) {
  switch (c) {
    case '0': return "111101101101101101111";
    case '1': return "010110010010010010111";
    case '2': return "111001001111100100111";
    case '3': return "111001001111001001111";
    case '4': return "101101101111001001001";
    case '5': return "111100100111001001111";
    case '6': return "111100100111101101111";
    case '7': return "111001001010010010010";
    case '8': return "111101101111101101111";
    case '9': return "111101101111001001111";
    case 'A': return "010101101111101101101";
    case 'B': return "110101101110101101110";
    case 'C': return "111100100100100100111";
    case 'D': return "110101101101101101110";
    case 'E': return "111100100111100100111";
    case 'F': return "111100100111100100100";
    case 'I': return "111010010010010010111";
    case 'L': return "100100100100100100111";
    case 'M': return "101111111101101101101";
    case 'P': return "110101101110100100100";
    case 'S': return "111100100111001001111";
    case 'T': return "111010010010010010010";
    case 'Z': return "111001001010100100111";
    case ':': return "000010010000010010000";
    case '.': return "000000000000000010010";
    case '-': return "000000000111000000000";
    case ' ': return "000000000000000000000";
    default: return "111001010010010000010";
  }
}

static void draw_text(Image& image,
                      int x,
                      int y,
                      const std::string& text,
                      uint8_t b,
                      uint8_t g,
                      uint8_t r,
                      int scale) {
  scale = std::max(1, scale);
  int cursor = x;
  for (char raw : text) {
    const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(raw)));
    const char* bits = glyph5x7(c);
    for (int row = 0; row < 7; ++row) {
      for (int col = 0; col < 3; ++col) {
        if (bits[row * 3 + col] != '1') {
          continue;
        }
        fill_rect(
            image,
            cursor + col * scale,
            y + row * scale,
            cursor + (col + 1) * scale - 1,
            y + (row + 1) * scale - 1,
            b,
            g,
            r);
      }
    }
    cursor += 4 * scale;
  }
}

static int text_pixel_width(const std::string& text, int scale) {
  scale = std::max(1, scale);
  if (text.empty()) {
    return 0;
  }
  return static_cast<int>(text.size()) * 4 * scale - scale;
}

static void draw_vertical_dashed_line(Image& image,
                                      int x,
                                      int y1,
                                      int y2,
                                      int dash,
                                      int gap,
                                      uint8_t b,
                                      uint8_t g,
                                      uint8_t r) {
  if (image.width <= 0 || image.height <= 0) {
    return;
  }
  x = std::max(0, std::min(x, image.width - 1));
  y1 = std::max(0, std::min(y1, image.height - 1));
  y2 = std::max(0, std::min(y2, image.height - 1));
  if (y2 < y1) {
    std::swap(y1, y2);
  }
  dash = std::max(1, dash);
  gap = std::max(0, gap);
  for (int y = y1; y <= y2; y += dash + gap) {
    draw_line(image, x, y, x, std::min(y + dash - 1, y2), b, g, r);
  }
}

static Image remap_panorama_cpu(const Image& source,
                                const MapMeta& meta,
                                const std::vector<float>* base_map_x,
                                const std::vector<float>* base_map_y) {
  if (base_map_x == nullptr || base_map_y == nullptr ||
      meta.base_map_width <= 0 || meta.base_map_height <= 0 ||
      base_map_x->size() != static_cast<size_t>(meta.base_map_width) * meta.base_map_height ||
      base_map_y->size() != static_cast<size_t>(meta.base_map_width) * meta.base_map_height) {
    return source;
  }

  Image panorama;
  panorama.width = meta.base_map_width;
  panorama.height = meta.base_map_height;
  panorama.bgr.assign(static_cast<size_t>(panorama.width) * panorama.height * 3, 0);
  for (int y = 0; y < panorama.height; ++y) {
    for (int x = 0; x < panorama.width; ++x) {
      const size_t out_idx = (static_cast<size_t>(y) * panorama.width + x) * 3;
      const size_t map_idx = static_cast<size_t>(y) * panorama.width + x;
      const float sx_f = (*base_map_x)[map_idx];
      const float sy_f = (*base_map_y)[map_idx];
      if (!std::isfinite(sx_f) || !std::isfinite(sy_f)) {
        continue;
      }
      const int sx = static_cast<int>(std::lround(sx_f));
      const int sy = static_cast<int>(std::lround(sy_f));
      if (sx < 0 || sy < 0 || sx >= source.width || sy >= source.height) {
        continue;
      }
      const size_t src_idx = (static_cast<size_t>(sy) * source.width + sx) * 3;
      panorama.bgr[out_idx + 0] = source.bgr[src_idx + 0];
      panorama.bgr[out_idx + 1] = source.bgr[src_idx + 1];
      panorama.bgr[out_idx + 2] = source.bgr[src_idx + 2];
    }
  }
  return panorama;
}

static double round_to(double value, double scale) {
  return std::round(value * scale) / scale;
}

static void print_json_nullable(std::ostream& os, bool valid, double value, int precision) {
  if (!valid || !std::isfinite(value)) {
    os << "null";
    return;
  }
  os << std::fixed << std::setprecision(precision) << value;
}

static void print_json_rounded_or_null(std::ostream& os, bool valid, double value) {
  if (!valid || !std::isfinite(value)) {
    os << "null";
    return;
  }
  os << std::fixed << std::setprecision(3) << round_to(value, 1000.0);
}

struct FaceRecognitionBinding {
  bool has_face = false;
  std::string face_id;
  float score = 0.0f;
  float second_score = 0.0f;
  bool dynamic = false;
  bool quality_passed = false;
  bool updated = false;
  bool tentative = false;
  int tentative_fail_count = 0;
  int lock_fail_count = 0;
  int face_check_count = 0;
  int face_bound_frame = 0;
  std::string event;
  bool has_debug = false;
  bool debug_rknn_ran = false;
  bool debug_strong_identity_match = false;
  bool debug_best_reserved_by_other = false;
  bool debug_used_by_other_this_frame = false;
  bool debug_blocked_by_owner = false;
  bool debug_has_nearest_feature = false;
  bool debug_pose_sample_quality_ok = false;
  bool debug_locked_pose_quality_ok = false;
  bool debug_has_pose_library_similarity = false;
  int debug_reserved_owner = 0;
  int debug_nearest_public_id = 0;
  int debug_pose_sample_missing_frames = -1;
  std::string debug_reason;
  std::string debug_best_name;
  std::string debug_update_reason;
  float debug_best_score = 0.0f;
  float debug_second_score = 0.0f;
  float debug_nearest_feature_score = 0.0f;
  float debug_identity_score = 0.0f;
  float debug_sample_similarity = 0.0f;
  float debug_pose_library_similarity = 0.0f;
  float debug_pose_library_max_similarity = 0.0f;
};

struct FaceIdentity {
  std::string name;
  std::vector<std::vector<float>> features;
  bool dynamic = false;
  float primary_quality = 0.0f;
  int last_pose_sample_frame = 0;
};

struct FaceFeatureQuality {
  bool sample_ok = false;
  bool primary_ok = false;
  float quality_score = 0.0f;
  float min_side = 0.0f;
  float min_keypoint_conf = 0.0f;
  float det_score = 0.0f;
  float yaw_deg = 0.0f;
  bool yaw_valid = false;
};

struct AngleDistanceState {
  bool has_last_distance = false;
  double last_distance = 0.0;
};

struct TargetInfo {
  int id = -1;
  int public_id = -1;
  int raw_track_id = -1;
  int display_id = -1;
  bool has_display_id = false;
  bool has_azimuth = false;
  bool has_elevation = false;
  bool has_face_yaw = false;
  bool has_eye_pixel_dist = false;
  bool has_distance = false;
  bool has_face_recognition = false;
  bool has_face_recognition_debug = false;
  bool face_recognition_dynamic = false;
  bool face_recognition_quality = false;
  bool face_recognition_updated = false;
  bool face_recognition_tentative = false;
  bool face_recognition_debug_rknn_ran = false;
  bool face_recognition_debug_strong_identity_match = false;
  bool face_recognition_debug_best_reserved_by_other = false;
  bool face_recognition_debug_used_by_other_this_frame = false;
  bool face_recognition_debug_blocked_by_owner = false;
  bool face_recognition_debug_has_nearest_feature = false;
  bool face_recognition_debug_pose_sample_quality_ok = false;
  bool face_recognition_debug_locked_pose_quality_ok = false;
  bool face_recognition_debug_has_pose_library_similarity = false;
  int face_recognition_debug_reserved_owner = 0;
  int face_recognition_debug_nearest_public_id = 0;
  int face_recognition_debug_pose_sample_missing_frames = -1;
  double azimuth = 0.0;
  double elevation = 0.0;
  double face_yaw = 0.0;
  double eye_pixel_dist = 0.0;
  double distance = 0.0;
  std::string face_id;
  std::string face_recognition_event;
  std::string face_recognition_debug_reason;
  std::string face_recognition_debug_best_name;
  std::string face_recognition_debug_update_reason;
  double face_recognition_score = 0.0;
  double face_recognition_second_score = 0.0;
  double face_recognition_debug_best_score = 0.0;
  double face_recognition_debug_second_score = 0.0;
  double face_recognition_debug_nearest_feature_score = 0.0;
  double face_recognition_debug_identity_score = 0.0;
  double face_recognition_debug_sample_similarity = 0.0;
  double face_recognition_debug_pose_library_similarity = 0.0;
  double face_recognition_debug_pose_library_max_similarity = 0.0;
};

static void write_face_recognition_debug_json(std::ostream& os, const TargetInfo& target) {
  os << "\"debug\":{"
     << "\"reason\":\"" << json_escape(target.face_recognition_debug_reason) << "\""
     << ",\"rknn_ran\":" << (target.face_recognition_debug_rknn_ran ? "true" : "false")
     << ",\"best_name\":";
  if (target.face_recognition_debug_best_name.empty()) {
    os << "null";
  } else {
    os << "\"" << json_escape(target.face_recognition_debug_best_name) << "\"";
  }
  os << ",\"best_score\":" << std::fixed << std::setprecision(3)
     << target.face_recognition_debug_best_score
     << ",\"second_score\":" << target.face_recognition_debug_second_score
     << ",\"strong_identity_match\":"
     << (target.face_recognition_debug_strong_identity_match ? "true" : "false")
     << ",\"best_reserved_by_other\":"
     << (target.face_recognition_debug_best_reserved_by_other ? "true" : "false")
     << ",\"used_by_other_this_frame\":"
     << (target.face_recognition_debug_used_by_other_this_frame ? "true" : "false")
     << ",\"blocked_by_owner\":"
     << (target.face_recognition_debug_blocked_by_owner ? "true" : "false")
     << ",\"reserved_owner\":";
  if (target.face_recognition_debug_reserved_owner > 0) {
    os << target.face_recognition_debug_reserved_owner;
  } else {
    os << "null";
  }
  os << ",\"nearest_frame_public_id\":";
  if (target.face_recognition_debug_has_nearest_feature &&
      target.face_recognition_debug_nearest_public_id > 0) {
    os << target.face_recognition_debug_nearest_public_id;
  } else {
    os << "null";
  }
  os << ",\"nearest_frame_feature_score\":";
  if (target.face_recognition_debug_has_nearest_feature) {
    os << target.face_recognition_debug_nearest_feature_score;
  } else {
    os << "null";
  }
  os << ",\"update_reason\":";
  if (target.face_recognition_debug_update_reason.empty()) {
    os << "null";
  } else {
    os << "\"" << json_escape(target.face_recognition_debug_update_reason) << "\"";
  }
  os << ",\"identity_score\":" << target.face_recognition_debug_identity_score
     << ",\"sample_similarity\":" << target.face_recognition_debug_sample_similarity
     << ",\"pose_library_similarity\":";
  if (target.face_recognition_debug_has_pose_library_similarity) {
    os << target.face_recognition_debug_pose_library_similarity;
  } else {
    os << "null";
  }
  os << ",\"pose_library_max_similarity\":"
     << target.face_recognition_debug_pose_library_max_similarity
     << ",\"pose_sample_missing_frames\":";
  if (target.face_recognition_debug_pose_sample_missing_frames >= 0) {
    os << target.face_recognition_debug_pose_sample_missing_frames;
  } else {
    os << "null";
  }
  os << ",\"pose_sample_quality_ok\":"
     << (target.face_recognition_debug_pose_sample_quality_ok ? "true" : "false")
     << ",\"locked_pose_quality_ok\":"
     << (target.face_recognition_debug_locked_pose_quality_ok ? "true" : "false");
  os << "}";
}

static void write_face_recognition_json(std::ostream& os, const TargetInfo& target) {
  os << "\"face_recognition\":{";
  if (!target.has_face_recognition || target.face_id.empty()) {
    os << "\"matched\":false,\"face_id\":null,\"name\":null";
    if (target.has_face_recognition_debug) {
      os << ",";
      write_face_recognition_debug_json(os, target);
    }
    os << "}";
    return;
  }
  os << "\"matched\":true"
     << ",\"face_id\":\"" << json_escape(target.face_id) << "\""
     << ",\"name\":\"" << json_escape(target.face_id) << "\""
     << ",\"score\":" << std::fixed << std::setprecision(3) << target.face_recognition_score
     << ",\"second_score\":" << target.face_recognition_second_score
     << ",\"dynamic\":" << (target.face_recognition_dynamic ? "true" : "false")
     << ",\"quality_passed\":" << (target.face_recognition_quality ? "true" : "false")
     << ",\"updated\":" << (target.face_recognition_updated ? "true" : "false")
     << ",\"tentative\":" << (target.face_recognition_tentative ? "true" : "false")
     << ",\"event\":\"" << json_escape(target.face_recognition_event) << "\"";
  if (target.has_face_recognition_debug) {
    os << ",";
    write_face_recognition_debug_json(os, target);
  }
  os << "}";
}

static void write_target_face_id_json(std::ostream& os, const TargetInfo* target) {
  os << "\"face_id\":";
  if (target != nullptr && target->has_face_recognition && !target->face_id.empty()) {
    os << "\"" << json_escape(target->face_id) << "\"";
  } else {
    os << "null";
  }
}

static std::string basename_without_ext(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  const size_t begin = slash == std::string::npos ? 0 : slash + 1;
  const size_t dot = path.find_last_of('.');
  const size_t end = (dot == std::string::npos || dot < begin) ? path.size() : dot;
  return path.substr(begin, end - begin);
}

static bool parse_npy_float32(const std::string& path,
                              std::vector<std::vector<float>>* rows,
                              std::string* error) {
  if (rows == nullptr) {
    return false;
  }
  rows->clear();
  const std::vector<uint8_t> data = read_bytes(path);
  if (data.size() < 16 || data[0] != 0x93 || data[1] != 'N' || data[2] != 'U' ||
      data[3] != 'M' || data[4] != 'P' || data[5] != 'Y') {
    if (error != nullptr) {
      *error = "not a .npy file";
    }
    return false;
  }
  const uint8_t major = data[6];
  size_t header_len = 0;
  size_t header_start = 0;
  if (major == 1) {
    if (data.size() < 10) {
      return false;
    }
    header_len = static_cast<size_t>(data[8]) | (static_cast<size_t>(data[9]) << 8);
    header_start = 10;
  } else if (major == 2 || major == 3) {
    if (data.size() < 12) {
      return false;
    }
    header_len = static_cast<size_t>(data[8]) |
                 (static_cast<size_t>(data[9]) << 8) |
                 (static_cast<size_t>(data[10]) << 16) |
                 (static_cast<size_t>(data[11]) << 24);
    header_start = 12;
  } else {
    if (error != nullptr) {
      *error = "unsupported .npy version";
    }
    return false;
  }
  if (header_start + header_len > data.size()) {
    return false;
  }
  const std::string header(
      reinterpret_cast<const char*>(data.data() + header_start), header_len);
  if (header.find("'descr': '<f4'") == std::string::npos &&
      header.find("\"descr\": \"<f4\"") == std::string::npos &&
      header.find("'descr': '|f4'") == std::string::npos &&
      header.find("\"descr\": \"|f4\"") == std::string::npos) {
    if (error != nullptr) {
      *error = "only float32 .npy features are supported";
    }
    return false;
  }
  if (header.find("True") != std::string::npos &&
      header.find("fortran_order") != std::string::npos) {
    if (error != nullptr) {
      *error = "fortran_order .npy is not supported";
    }
    return false;
  }
  const size_t lp = header.find('(');
  const size_t rp = header.find(')', lp == std::string::npos ? 0 : lp);
  if (lp == std::string::npos || rp == std::string::npos || rp <= lp + 1) {
    return false;
  }
  std::vector<int> shape;
  std::stringstream ss(header.substr(lp + 1, rp - lp - 1));
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = trim(item);
    if (item.empty()) {
      continue;
    }
    shape.push_back(std::stoi(item));
  }
  if (shape.empty() || shape.size() > 2) {
    if (error != nullptr) {
      *error = "expected 1D or 2D feature array";
    }
    return false;
  }
  const int dim = shape.size() == 1 ? shape[0] : shape[1];
  const int n_rows = shape.size() == 1 ? 1 : shape[0];
  if (dim <= 0 || n_rows <= 0) {
    return false;
  }
  const size_t offset = header_start + header_len;
  const size_t expected = static_cast<size_t>(n_rows) * dim * sizeof(float);
  if (offset + expected > data.size()) {
    return false;
  }
  const float* ptr = reinterpret_cast<const float*>(data.data() + offset);
  rows->reserve(static_cast<size_t>(n_rows));
  for (int r = 0; r < n_rows; ++r) {
    std::vector<float> feat(ptr + static_cast<size_t>(r) * dim,
                            ptr + static_cast<size_t>(r + 1) * dim);
    double sum = 0.0;
    for (float v : feat) {
      sum += static_cast<double>(v) * v;
    }
    const double norm = std::sqrt(sum);
    if (norm > 1.0e-12) {
      const float inv = static_cast<float>(1.0 / norm);
      for (float& v : feat) {
        v *= inv;
      }
    }
    rows->push_back(std::move(feat));
  }
  return true;
}

static void write_npy_float32_2d(const std::string& path,
                                 const std::vector<std::vector<float>>& rows,
                                 int dim) {
  if (rows.empty() || dim <= 0) {
    return;
  }
  std::ostringstream header_ss;
  header_ss << "{'descr': '<f4', 'fortran_order': False, 'shape': ("
            << rows.size() << ", " << dim << "), }";
  std::string header = header_ss.str();
  const size_t preamble = 10;
  size_t pad = 16 - ((preamble + header.size() + 1) % 16);
  if (pad == 16) {
    pad = 0;
  }
  header.append(pad, ' ');
  header.push_back('\n');
  if (header.size() > 65535) {
    throw std::runtime_error("npy header too large: " + path);
  }
  std::ofstream out(path.c_str(), std::ios::binary | std::ios::out | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("cannot write npy: " + path);
  }
  const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  out.write(reinterpret_cast<const char*>(magic), sizeof(magic));
  const uint16_t header_len = static_cast<uint16_t>(header.size());
  const uint8_t h0 = static_cast<uint8_t>(header_len & 0xff);
  const uint8_t h1 = static_cast<uint8_t>((header_len >> 8) & 0xff);
  out.put(static_cast<char>(h0));
  out.put(static_cast<char>(h1));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (const auto& row : rows) {
    if (static_cast<int>(row.size()) != dim) {
      continue;
    }
    out.write(reinterpret_cast<const char*>(row.data()),
              static_cast<std::streamsize>(row.size() * sizeof(float)));
  }
}

static bool ensure_dir_exists(const std::string& dir) {
  if (dir.empty()) {
    return false;
  }
  struct stat st {};
  if (stat(dir.c_str(), &st) == 0) {
    return S_ISDIR(st.st_mode);
  }
  size_t pos = 0;
  while (pos < dir.size()) {
    pos = dir.find('/', pos + 1);
    const std::string part =
        pos == std::string::npos ? dir : dir.substr(0, pos);
    if (part.empty()) {
      if (pos == std::string::npos) {
        break;
      }
      continue;
    }
    if (mkdir(part.c_str(), 0775) != 0 && errno != EEXIST) {
      return false;
    }
    if (pos == std::string::npos) {
      break;
    }
  }
  if (stat(dir.c_str(), &st) == 0) {
    return S_ISDIR(st.st_mode);
  }
  if (mkdir(dir.c_str(), 0775) == 0) {
    return true;
  }
  return errno == EEXIST;
}
