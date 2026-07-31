static const char* profile_slot_name(int slot) {
  switch (slot) {
    case kProfileDecode:
      return "decode";
    case kProfileFileRead:
      return "file_read";
    case kProfileJpegDecode:
      return "jpeg_decode";
    case kProfileDecodeWait:
      return "decode_wait";
    case kProfileCameraRead:
      return "camera_read";
    case kProfileOpenclEnsure:
      return "opencl_ensure";
    case kProfileRknnInputAlloc:
      return "rknn_input_alloc";
    case kProfileOpenclRunOuter:
      return "opencl_run_outer";
    case kProfileRknnTotalOuter:
      return "rknn_total_outer";
    case kProfileBoundPrepare:
      return "bound_prepare";
    case kProfileBoundImport:
      return "bound_import";
    case kProfileStagingCopy:
      return "staging_copy";
    case kProfileTrackerOuter:
      return "tracker_outer";
    case kProfileFaceRec:
      return "face_rec";
    case kProfileAngleTargets:
      return "angle_targets";
    case kProfileBuildPayload:
      return "build_payload";
    case kProfileStdout:
      return "stdout";
    case kProfileJsonl:
      return "jsonl";
    case kProfileWebSocket:
      return "websocket";
    case kProfileDebugJsonl:
      return "debug_jsonl";
    case kProfileWriteOutputs:
      return "write_outputs";
    case kProfileFrameTotal:
      return "frame_total";
    default:
      return "unknown";
  }
}

class ProfileSummary {
 public:
  void add(const FrameResult& result, const FrameRateStats& fps) {
    frames_ += 1;
    latest_average_fps_ = fps.average_fps;
    latest_instant_fps_ = fps.instant_fps;
    for (int i = 0; i < kProfileCount; ++i) {
      sums_[i] += result.profile_ms[i];
      maxes_[i] = std::max(maxes_[i], result.profile_ms[i]);
    }
  }

  int frames() const { return frames_; }

  void print(std::ostream& os, const char* title) const {
    if (frames_ <= 0) {
      return;
    }
    std::vector<std::pair<double, int>> order;
    order.reserve(kProfileCount);
    for (int i = 0; i < kProfileCount; ++i) {
      order.push_back(std::make_pair(sums_[i] / static_cast<double>(frames_), i));
    }
    std::sort(order.begin(), order.end(), [](const std::pair<double, int>& a,
                                             const std::pair<double, int>& b) {
      return a.first > b.first;
    });

    os << "[profile] " << title << " frames=" << frames_
       << " avg_fps=" << std::fixed << std::setprecision(3) << latest_average_fps_
       << " last_fps=" << latest_instant_fps_ << "\n";
    for (const auto& item : order) {
      const double avg = item.first;
      const int slot = item.second;
      if (avg <= 0.0005 && maxes_[slot] <= 0.0005) {
        continue;
      }
      os << "[profile]   " << std::setw(22) << std::left << profile_slot_name(slot)
         << " avg=" << std::setw(9) << std::right << std::setprecision(3) << avg
         << " ms"
         << " max=" << std::setw(9) << maxes_[slot] << " ms\n";
    }
  }

 private:
  int frames_ = 0;
  double latest_average_fps_ = 0.0;
  double latest_instant_fps_ = 0.0;
  double sums_[kProfileCount] = {0.0};
  double maxes_[kProfileCount] = {0.0};
};

static std::string trim(const std::string& value) {
  size_t begin = 0;
  while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  size_t end = value.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return value.substr(begin, end - begin);
}

static std::string join_path(const std::string& base, const std::string& name) {
  if (base.empty()) {
    return name;
  }
  if (base[base.size() - 1] == '/') {
    return base + name;
  }
  return base + "/" + name;
}

static std::string face_rec_model_path_for_preset(const std::string& preset) {
  if (preset == "ir18" || preset == "ir18_webface4m") {
    return "models/adaface_ir18_112.rknn";
  }
  if (preset == "ir50" || preset == "ir50_webface4m" ||
      preset == "r50" || preset == "r50_webface4m") {
    return "models/adaface_ir50_webface4m_112.rknn";
  }
  throw std::invalid_argument(
      "--face-rec-model-preset must be one of: ir18_webface4m, ir50_webface4m");
}

static std::string normalize_face_rec_align_mode(std::string mode) {
  std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  std::replace(mode.begin(), mode.end(), '-', '_');
  if (mode == "eye" || mode == "eye_crop" || mode == "legacy") {
    return "eye";
  }
  if (mode == "five_point" || mode == "5point" || mode == "arcface" ||
      mode == "adaface") {
    return "five_point";
  }
  throw std::invalid_argument("--face-rec-align-mode must be one of: eye, five-point");
}

static std::string normalize_projection_mode(std::string mode) {
  std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  std::replace(mode.begin(), mode.end(), '-', '_');
  if (mode == "fisheye" || mode == "fisheye_panorama" || mode == "panorama") {
    return "fisheye_panorama";
  }
  if (mode == "wide" || mode == "wide_angle" || mode == "wideangle") {
    return "wide_angle";
  }
  throw std::invalid_argument("--projection-mode must be one of: fisheye-panorama, wide-angle");
}

static bool is_wide_angle_projection(const std::string& mode) {
  return normalize_projection_mode(mode) == "wide_angle";
}

static bool file_exists(const std::string& path) {
  std::ifstream f(path.c_str(), std::ios::binary);
  return static_cast<bool>(f);
}

static std::string dirname_of(const std::string& path) {
  const size_t pos = path.find_last_of('/');
  if (pos == std::string::npos) {
    return "";
  }
  if (pos == 0) {
    return "/";
  }
  return path.substr(0, pos);
}

static std::string stem_without_ext(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  const size_t begin = slash == std::string::npos ? 0 : slash + 1;
  const size_t dot = path.find_last_of('.');
  if (dot == std::string::npos || dot < begin) {
    return path;
  }
  return path.substr(0, dot);
}

static std::string read_text_file(const std::string& path) {
  std::ifstream f(path.c_str(), std::ios::in);
  if (!f) {
    return "";
  }
  std::ostringstream oss;
  oss << f.rdbuf();
  return trim(oss.str());
}

static bool read_first_number(const std::string& text, double* value) {
  if (value == nullptr) {
    return false;
  }
  const char* begin = text.c_str();
  char* end = nullptr;
  while (*begin != '\0') {
    if ((*begin >= '0' && *begin <= '9') || *begin == '-' || *begin == '+') {
      const double parsed = std::strtod(begin, &end);
      if (end != begin && std::isfinite(parsed)) {
        *value = parsed;
        return true;
      }
    }
    ++begin;
  }
  return false;
}

static std::vector<std::string> list_dir_paths(const std::string& dir) {
  std::vector<std::string> out;
  DIR* dp = opendir(dir.c_str());
  if (dp == nullptr) {
    return out;
  }
  while (dirent* ent = readdir(dp)) {
    const std::string name = ent->d_name;
    if (name == "." || name == "..") {
      continue;
    }
    out.push_back(join_path(dir, name));
  }
  closedir(dp);
  std::sort(out.begin(), out.end());
  return out;
}

static std::string json_escape(const std::string& value) {
  std::ostringstream oss;
  for (char ch : value) {
    switch (ch) {
      case '\\':
        oss << "\\\\";
        break;
      case '"':
        oss << "\\\"";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\r':
        oss << "\\r";
        break;
      case '\t':
        oss << "\\t";
        break;
      default:
        oss << ch;
        break;
    }
  }
  return oss.str();
}

struct CpuTimes {
  uint64_t user = 0;
  uint64_t nice = 0;
  uint64_t system = 0;
  uint64_t idle = 0;
  uint64_t iowait = 0;
  uint64_t irq = 0;
  uint64_t softirq = 0;
  uint64_t steal = 0;

  uint64_t idle_all() const { return idle + iowait; }
  uint64_t total() const { return user + nice + system + idle + iowait + irq + softirq + steal; }
};

struct SystemLoadSnapshot {
  double timestamp = 0.0;
  double time_s = 0.0;
  int sample_id = 0;
  bool cpu_valid = false;
  double cpu_percent = 0.0;
  std::vector<double> cpu_per_core_percent;
  bool memory_valid = false;
  double memory_percent = 0.0;
  double memory_available_mb = 0.0;
  bool gpu_valid = false;
  double gpu_percent = 0.0;
  std::string gpu_load_raw;
  int64_t gpu_freq_hz = -1;
  bool npu_valid = false;
  double npu_percent = 0.0;
  std::vector<double> npu_core_percent;
  std::string npu_load_raw;
  std::string npu_load_source;
  std::string npu_power_raw;
  int64_t npu_freq_hz = -1;
  std::vector<double> thermal_c;
  double thermal_max_c = 0.0;
};

static bool parse_cpu_line(const std::string& line, CpuTimes* out) {
  if (out == nullptr) {
    return false;
  }
  std::istringstream iss(line);
  std::string label;
  iss >> label;
  if (label.compare(0, 3, "cpu") != 0) {
    return false;
  }
  CpuTimes t;
  iss >> t.user >> t.nice >> t.system >> t.idle >> t.iowait >> t.irq >> t.softirq >> t.steal;
  if (!iss && t.total() == 0) {
    return false;
  }
  *out = t;
  return true;
}

static std::vector<CpuTimes> read_proc_stat_cpu_times() {
  std::vector<CpuTimes> times;
  std::ifstream f("/proc/stat");
  std::string line;
  while (std::getline(f, line)) {
    if (line.compare(0, 3, "cpu") != 0) {
      break;
    }
    CpuTimes t;
    if (parse_cpu_line(line, &t)) {
      times.push_back(t);
    }
  }
  return times;
}

static double cpu_delta_percent(const CpuTimes& prev, const CpuTimes& cur) {
  const uint64_t prev_total = prev.total();
  const uint64_t cur_total = cur.total();
  const uint64_t prev_idle = prev.idle_all();
  const uint64_t cur_idle = cur.idle_all();
  if (cur_total <= prev_total) {
    return 0.0;
  }
  const double total_delta = static_cast<double>(cur_total - prev_total);
  const double idle_delta = cur_idle >= prev_idle ? static_cast<double>(cur_idle - prev_idle) : 0.0;
  return std::max(0.0, std::min(100.0, 100.0 * (total_delta - idle_delta) / total_delta));
}

static bool read_meminfo(double* memory_percent, double* available_mb) {
  std::ifstream f("/proc/meminfo");
  if (!f) {
    return false;
  }
  std::string key;
  double value = 0.0;
  std::string unit;
  double total_kb = 0.0;
  double available_kb = 0.0;
  while (f >> key >> value >> unit) {
    if (key == "MemTotal:") {
      total_kb = value;
    } else if (key == "MemAvailable:") {
      available_kb = value;
    }
  }
  if (total_kb <= 0.0 || available_kb < 0.0) {
    return false;
  }
  if (memory_percent != nullptr) {
    *memory_percent = std::max(0.0, std::min(100.0, 100.0 * (1.0 - available_kb / total_kb)));
  }
  if (available_mb != nullptr) {
    *available_mb = available_kb / 1024.0;
  }
  return true;
}

static bool parse_load_percent(const std::string& text, double* percent) {
  if (percent == nullptr || text.empty()) {
    return false;
  }
  const size_t pct_pos = text.find('%');
  if (pct_pos != std::string::npos) {
    size_t begin = pct_pos;
    while (begin > 0) {
      const char ch = text[begin - 1];
      if (!((ch >= '0' && ch <= '9') || ch == '.')) {
        break;
      }
      --begin;
    }
    if (begin < pct_pos) {
      const double value = std::strtod(text.substr(begin, pct_pos - begin).c_str(), nullptr);
      if (std::isfinite(value)) {
        *percent = std::max(0.0, std::min(100.0, value));
        return true;
      }
    }
  }
  const size_t at_pos = text.find('@');
  if (at_pos != std::string::npos) {
    double value = 0.0;
    if (read_first_number(text.substr(0, at_pos), &value) && value >= 0.0 && value <= 100.0) {
      *percent = value;
      return true;
    }
  }
  double first = 0.0;
  if (read_first_number(text, &first) && first >= 0.0 && first <= 100.0) {
    *percent = first;
    return true;
  }
  return false;
}

static bool name_contains_any(const std::string& text, const std::vector<std::string>& needles) {
  std::string lower = text;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  for (const std::string& needle : needles) {
    if (lower.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

static std::vector<std::string> find_devfreq_dirs(const std::vector<std::string>& needles) {
  std::vector<std::string> dirs;
  for (const std::string& path : list_dir_paths("/sys/class/devfreq")) {
    if (name_contains_any(path, needles)) {
      dirs.push_back(path);
    }
  }
  return dirs;
}

static bool sample_devfreq_kind(const std::vector<std::string>& needles,
                                double* percent,
                                int64_t* freq_hz,
                                std::string* raw) {
  const std::vector<std::string> dirs = find_devfreq_dirs(needles);
  bool any = false;
  if (freq_hz != nullptr) {
    *freq_hz = -1;
  }
  for (const std::string& dir : dirs) {
    const std::string load_path = join_path(dir, "load");
    const std::string load_text = read_text_file(load_path);
    double load_percent = 0.0;
    if (!load_text.empty() && parse_load_percent(load_text, &load_percent)) {
      if (percent != nullptr) {
        *percent = load_percent;
      }
      if (raw != nullptr) {
        *raw = load_text;
      }
      any = true;
    }
    const std::string freq_text = read_text_file(join_path(dir, "cur_freq"));
    double freq = 0.0;
    if (!freq_text.empty() && read_first_number(freq_text, &freq) && freq_hz != nullptr) {
      *freq_hz = static_cast<int64_t>(freq);
    }
    if (any) {
      return true;
    }
  }
  return any;
}

static bool parse_rknpu_debug_core_loads(const std::string& text,
                                         std::vector<double>* cores) {
  if (cores == nullptr) {
    return false;
  }
  cores->clear();
  size_t pos = 0;
  while (true) {
    const size_t core_pos = text.find("Core", pos);
    if (core_pos == std::string::npos) {
      break;
    }
    const char* idx_begin = text.c_str() + core_pos + 4;
    char* idx_end = nullptr;
    const long core_idx = std::strtol(idx_begin, &idx_end, 10);
    if (idx_end == idx_begin || core_idx < 0 || core_idx > 31) {
      pos = core_pos + 4;
      continue;
    }
    const size_t colon_pos = text.find(':', static_cast<size_t>(idx_end - text.c_str()));
    if (colon_pos == std::string::npos) {
      pos = core_pos + 4;
      continue;
    }
    char* value_end = nullptr;
    const double value = std::strtod(text.c_str() + colon_pos + 1, &value_end);
    if (value_end == text.c_str() + colon_pos + 1 || !std::isfinite(value)) {
      pos = colon_pos + 1;
      continue;
    }
    const size_t dst = static_cast<size_t>(core_idx);
    if (cores->size() <= dst) {
      cores->resize(dst + 1, std::numeric_limits<double>::quiet_NaN());
    }
    (*cores)[dst] = std::max(0.0, std::min(100.0, value));
    pos = static_cast<size_t>(value_end - text.c_str());
  }

  bool any = false;
  for (double value : *cores) {
    if (std::isfinite(value)) {
      any = true;
      break;
    }
  }
  if (!any) {
    cores->clear();
  }
  return any;
}

static double average_valid_percent(const std::vector<double>& values) {
  double sum = 0.0;
  int count = 0;
  for (double value : values) {
    if (std::isfinite(value)) {
      sum += value;
      count += 1;
    }
  }
  return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

static bool sample_rknpu_debug(SystemLoadSnapshot* snap) {
  if (snap == nullptr) {
    return false;
  }
  const std::string base = "/sys/kernel/debug/rknpu";
  const std::string load_text = read_text_file(join_path(base, "load"));
  std::vector<double> cores;
  if (!parse_rknpu_debug_core_loads(load_text, &cores)) {
    return false;
  }
  snap->npu_valid = true;
  snap->npu_core_percent = cores;
  snap->npu_percent = average_valid_percent(cores);
  snap->npu_load_raw = load_text;
  snap->npu_load_source = "debugfs_rknpu";
  snap->npu_power_raw = read_text_file(join_path(base, "power"));
  double freq = 0.0;
  const std::string freq_text = read_text_file(join_path(base, "freq"));
  if (!freq_text.empty() && read_first_number(freq_text, &freq)) {
    snap->npu_freq_hz = static_cast<int64_t>(freq);
  }
  return true;
}

static std::vector<double> sample_thermal_c() {
  std::vector<double> values;
  for (const std::string& zone : list_dir_paths("/sys/class/thermal")) {
    if (zone.find("thermal_zone") == std::string::npos) {
      continue;
    }
    double raw = 0.0;
    if (!read_first_number(read_text_file(join_path(zone, "temp")), &raw)) {
      continue;
    }
    if (raw > 1000.0) {
      raw /= 1000.0;
    }
    if (raw > -80.0 && raw < 180.0) {
      values.push_back(raw);
    }
  }
  return values;
}

static void write_json_nullable(std::ostream& os, bool valid, double value, int precision = 2) {
  if (!valid || !std::isfinite(value)) {
    os << "null";
    return;
  }
  os << std::fixed << std::setprecision(precision) << value;
}

static std::string system_snapshot_json(const SystemLoadSnapshot& s) {
  std::ostringstream os;
  os << "{";
  os << "\"timestamp\":" << std::fixed << std::setprecision(3) << s.timestamp;
  os << ",\"time_s\":" << std::fixed << std::setprecision(3) << s.time_s;
  os << ",\"sample_id\":" << s.sample_id;
  os << ",\"cpu_percent\":";
  write_json_nullable(os, s.cpu_valid, s.cpu_percent);
  os << ",\"cpu_per_core_percent\":";
  if (s.cpu_per_core_percent.empty()) {
    os << "null";
  } else {
    os << "[";
    for (size_t i = 0; i < s.cpu_per_core_percent.size(); ++i) {
      if (i > 0) {
        os << ",";
      }
      os << std::fixed << std::setprecision(2) << s.cpu_per_core_percent[i];
    }
    os << "]";
  }
  os << ",\"memory_percent\":";
  write_json_nullable(os, s.memory_valid, s.memory_percent);
  os << ",\"memory_available_mb\":";
  write_json_nullable(os, s.memory_valid, s.memory_available_mb, 1);
  os << ",\"gpu_percent\":";
  write_json_nullable(os, s.gpu_valid, s.gpu_percent);
  os << ",\"gpu_freq_hz\":";
  if (s.gpu_freq_hz >= 0) {
    os << s.gpu_freq_hz;
  } else {
    os << "null";
  }
  os << ",\"gpu_load_raw\":";
  if (!s.gpu_load_raw.empty()) {
    os << "\"" << json_escape(s.gpu_load_raw) << "\"";
  } else {
    os << "null";
  }
  os << ",\"npu_percent\":";
  write_json_nullable(os, s.npu_valid, s.npu_percent);
  os << ",\"npu_core_percent\":";
  if (s.npu_core_percent.empty()) {
    os << "null";
  } else {
    os << "[";
    for (size_t i = 0; i < s.npu_core_percent.size(); ++i) {
      if (i > 0) {
        os << ",";
      }
      write_json_nullable(os, std::isfinite(s.npu_core_percent[i]), s.npu_core_percent[i]);
    }
    os << "]";
  }
  for (int i = 0; i < 3; ++i) {
    os << ",\"npu_core" << i << "_percent\":";
    const bool valid =
        static_cast<size_t>(i) < s.npu_core_percent.size() &&
        std::isfinite(s.npu_core_percent[static_cast<size_t>(i)]);
    write_json_nullable(os, valid, valid ? s.npu_core_percent[static_cast<size_t>(i)] : 0.0);
  }
  os << ",\"npu_freq_hz\":";
  if (s.npu_freq_hz >= 0) {
    os << s.npu_freq_hz;
  } else {
    os << "null";
  }
  os << ",\"npu_load_raw\":";
  if (!s.npu_load_raw.empty()) {
    os << "\"" << json_escape(s.npu_load_raw) << "\"";
  } else {
    os << "null";
  }
  os << ",\"npu_load_source\":";
  if (!s.npu_load_source.empty()) {
    os << "\"" << json_escape(s.npu_load_source) << "\"";
  } else {
    os << "null";
  }
  os << ",\"npu_power_raw\":";
  if (!s.npu_power_raw.empty()) {
    os << "\"" << json_escape(s.npu_power_raw) << "\"";
  } else {
    os << "null";
  }
  os << ",\"thermal_max_c\":";
  write_json_nullable(os, !s.thermal_c.empty(), s.thermal_max_c);
  os << ",\"thermal_c\":";
  if (s.thermal_c.empty()) {
    os << "null";
  } else {
    os << "[";
    for (size_t i = 0; i < s.thermal_c.size(); ++i) {
      if (i > 0) {
        os << ",";
      }
      os << std::fixed << std::setprecision(2) << s.thermal_c[i];
    }
    os << "]";
  }
  os << "}";
  return os.str();
}

class SystemLoadSampler {
 public:
  SystemLoadSampler(int interval_ms, const std::string& output_jsonl)
      : interval_ms_(std::max(50, interval_ms)), output_jsonl_(output_jsonl) {}

  void start() {
    if (running_) {
      return;
    }
    if (!output_jsonl_.empty()) {
      const std::string dir = dirname_of(output_jsonl_);
      if (!dir.empty()) {
        mkdir(dir.c_str(), 0775);
      }
      out_.open(output_jsonl_.c_str(), std::ios::out | std::ios::trunc);
      if (!out_) {
        throw std::runtime_error("cannot open system profile jsonl: " + output_jsonl_);
      }
    }
    run_start_wall_s_ = wall_time_seconds();
    prev_cpu_times_ = read_proc_stat_cpu_times();
    running_ = true;
    worker_ = std::thread(&SystemLoadSampler::run_loop, this);
  }

  void stop() {
    if (!running_) {
      return;
    }
    stop_.store(true);
    if (worker_.joinable()) {
      worker_.join();
    }
    running_ = false;
    if (out_) {
      out_.close();
    }
  }

  ~SystemLoadSampler() { stop(); }

  const std::string& output_jsonl() const { return output_jsonl_; }
  int samples() const { return sample_count_; }
  std::string latest_json() const {
    std::lock_guard<std::mutex> lock(latest_mutex_);
    return latest_json_;
  }

 private:
  void run_loop() {
    while (!stop_.load()) {
      const double t0 = steady_seconds();
      SystemLoadSnapshot snap = sample_once();
      const std::string payload = system_snapshot_json(snap);
      {
        std::lock_guard<std::mutex> lock(latest_mutex_);
        latest_json_ = payload;
      }
      if (out_) {
        out_ << payload << "\n";
      }
      const double elapsed_ms = (steady_seconds() - t0) * 1000.0;
      const int wait_ms = std::max(1, interval_ms_ - static_cast<int>(elapsed_ms));
      for (int slept = 0; slept < wait_ms && !stop_.load(); slept += 10) {
        std::this_thread::sleep_for(std::chrono::milliseconds(std::min(10, wait_ms - slept)));
      }
    }
  }

  SystemLoadSnapshot sample_once() {
    SystemLoadSnapshot snap;
    snap.timestamp = wall_time_seconds();
    snap.time_s = std::max(0.0, snap.timestamp - run_start_wall_s_);
    snap.sample_id = ++sample_count_;

    std::vector<CpuTimes> cur_times = read_proc_stat_cpu_times();
    if (!prev_cpu_times_.empty() && cur_times.size() == prev_cpu_times_.size()) {
      snap.cpu_valid = true;
      snap.cpu_percent = cpu_delta_percent(prev_cpu_times_[0], cur_times[0]);
      for (size_t i = 1; i < cur_times.size(); ++i) {
        snap.cpu_per_core_percent.push_back(cpu_delta_percent(prev_cpu_times_[i], cur_times[i]));
      }
    }
    if (!cur_times.empty()) {
      prev_cpu_times_ = cur_times;
    }

    snap.memory_valid = read_meminfo(&snap.memory_percent, &snap.memory_available_mb);
    snap.gpu_valid = sample_devfreq_kind({"gpu", "mali"}, &snap.gpu_percent, &snap.gpu_freq_hz, &snap.gpu_load_raw);
    sample_rknpu_debug(&snap);
    snap.thermal_c = sample_thermal_c();
    if (!snap.thermal_c.empty()) {
      snap.thermal_max_c = *std::max_element(snap.thermal_c.begin(), snap.thermal_c.end());
    }
    return snap;
  }

  int interval_ms_ = 200;
  std::string output_jsonl_;
  std::ofstream out_;
  std::thread worker_;
  std::atomic<bool> stop_{false};
  bool running_ = false;
  int sample_count_ = 0;
  double run_start_wall_s_ = 0.0;
  std::vector<CpuTimes> prev_cpu_times_;
  mutable std::mutex latest_mutex_;
  std::string latest_json_;
};
