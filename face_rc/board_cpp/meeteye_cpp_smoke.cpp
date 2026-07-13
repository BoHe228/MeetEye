#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <ifaddrs.h>
#include <iomanip>
#include <iostream>
#include <linux/videodev2.h>
#include <map>
#include <memory>
#include <mutex>
#include <netdb.h>
#include <net/if.h>
#include <netinet/in.h>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/time.h>
#include <thread>
#include <arpa/inet.h>
#include <unistd.h>
#include <utility>
#include <vector>

extern "C" {
using tjhandle = void*;
tjhandle tjInitDecompress(void);
tjhandle tjInitCompress(void);
int tjDestroy(tjhandle handle);
int tjDecompressHeader3(tjhandle handle,
                        const unsigned char* jpeg_buf,
                        unsigned long jpeg_size,
                        int* width,
                        int* height,
                        int* jpeg_subsamp,
                        int* jpeg_colorspace);
int tjDecompress2(tjhandle handle,
                  const unsigned char* jpeg_buf,
                  unsigned long jpeg_size,
                  unsigned char* dst_buf,
                  int width,
                  int pitch,
                  int height,
                  int pixel_format,
                  int flags);
int tjCompress2(tjhandle handle,
                const unsigned char* src_buf,
                int width,
                int pitch,
                int height,
                int pixel_format,
                unsigned char** jpeg_buf,
                unsigned long* jpeg_size,
                int jpeg_subsamp,
                int jpeg_quality,
                int flags);
void tjFree(unsigned char* buffer);
char* tjGetErrorStr2(tjhandle handle);

int ds_opencl_fused_create(const float* map_x,
                           const float* map_y,
                           const int* roi_rects,
                           int src_w,
                           int src_h,
                           int frame_stride,
                           int num_slices,
                           int roi_h,
                           int roi_w,
                           int imgsz,
                           void** out_handle,
                           char* err,
                           int err_len);
int ds_opencl_fused_run(void* handle,
                        const uint8_t* frame,
                        uint8_t* output,
                        double* timings,
                        char* err,
                        int err_len);
int ds_opencl_fused_run_split(void* handle,
                              const uint8_t* frame,
                              uint8_t** outputs,
                              int num_outputs,
                              double* timings,
                              char* err,
                              int err_len);
int ds_opencl_fused_import_output_fds(void* handle,
                                      const int* fds,
                                      const uint64_t* sizes,
                                      int num_outputs,
                                      char* err,
                                      int err_len);
int ds_opencl_fused_run_imported(void* handle,
                                 const uint8_t* frame,
                                 double* timings,
                                 char* err,
                                 int err_len);
void ds_opencl_fused_destroy(void* handle);

int face_rknn_parallel_create(const char* model_path, void** handle, char* err, int err_len);
void face_rknn_parallel_destroy(void* handle);
int face_rknn_parallel_get_shape(void* handle,
                                 int* input_h,
                                 int* input_w,
                                 int* input_c,
                                 int* channels,
                                 int* anchors);
int face_rknn_parallel_infer_merged(void* handle,
                                    const uint8_t* inputs,
                                    int num_inputs,
                                    int input_h,
                                    int input_w,
                                    int input_c,
                                    const int* slice_shapes,
                                    const float* gains,
                                    const int* pads,
                                    const float* slice_start_x,
                                    const int* slice_wrap_around,
                                    int num_slices,
                                    float original_width,
                                    float overlap_ratio,
                                    float merge_iou_threshold,
                                    float nms_iou_thresh,
                                    float conf_threshold,
                                    float decode_iou_threshold,
                                    int max_det,
                                    int max_nms,
                                    float* detections,
                                    int max_output_dets,
                                    int* detection_count,
                                    int* merge_stats,
                                    double* timings,
                                    char* err,
                                    int err_len);
int face_rknn_parallel_prepare_bound_inputs(void* handle, char* err, int err_len);
int face_rknn_parallel_get_bound_input_fds(void* handle,
                                           int* fds,
                                           uint64_t* sizes,
                                           int max_inputs,
                                           char* err,
                                           int err_len);
int face_rknn_parallel_get_bound_input_ptrs(void* handle,
                                            uint8_t** ptrs,
                                            uint64_t* sizes,
                                            int max_inputs,
                                            char* err,
                                            int err_len);
int face_rknn_parallel_infer_merged_bound(void* handle,
                                          int num_inputs,
                                          int input_h,
                                          int input_w,
                                          int input_c,
                                          const int* slice_shapes,
                                          const float* gains,
                                          const int* pads,
                                          const float* slice_start_x,
                                          const int* slice_wrap_around,
                                          int num_slices,
                                          float original_width,
                                          float overlap_ratio,
                                          float merge_iou_threshold,
                                          float nms_iou_thresh,
                                          float conf_threshold,
                                          float decode_iou_threshold,
                                          int max_det,
                                          int max_nms,
                                          float* detections,
                                          int max_output_dets,
                                          int* detection_count,
                                          int* merge_stats,
                                          double* timings,
                                          char* err,
                                          int err_len);
int face_rknn_parallel_infer_merged_bound_external(void* handle,
                                                   int num_inputs,
                                                   int input_h,
                                                   int input_w,
                                                   int input_c,
                                                   const int* slice_shapes,
                                                   const float* gains,
                                                   const int* pads,
                                                   const float* slice_start_x,
                                                   const int* slice_wrap_around,
                                                   int num_slices,
                                                   float original_width,
                                                   float overlap_ratio,
                                                   float merge_iou_threshold,
                                                   float nms_iou_thresh,
                                                   float conf_threshold,
                                                   float decode_iou_threshold,
                                                   int max_det,
                                                   int max_nms,
                                                   int external_device_input,
                                                   float* detections,
                                                   int max_output_dets,
                                                   int* detection_count,
                                                   int* merge_stats,
                                                   double* timings,
                                                   char* err,
                                                   int err_len);

int hybrid_sort_native_create(float det_thresh,
                              float low_thresh,
                              int max_age,
                              int min_hits,
                              float iou_threshold,
                              int delta_t,
                              float inertia,
                              int use_byte,
                              int tcm_first_step,
                              float tcm_first_step_weight,
                              int tcm_byte_step,
                              float tcm_byte_step_weight,
                              float new_track_thresh,
                              float new_track_overlap_thresh,
                              float lost_velocity_decay,
                              void** out_handle,
                              char* err,
                              int err_len);
int hybrid_sort_native_update(void* handle,
                              const float* input,
                              int n_input,
                              int det_stride,
                              float img_h,
                              float img_w,
                              float size_h,
                              float size_w,
                              float* out_rows,
                              int max_out_rows,
                              int* out_count,
                              int* stats,
                              int stats_len,
                              double* timings,
                              int timings_len,
                              char* err,
                              int err_len);
int hybrid_sort_native_get_tracks(void* handle,
                                  float* out_rows,
                                  int max_rows,
                                  int* out_count,
                                  char* err,
                                  int err_len);
int hybrid_sort_native_reset_ids(void* handle, char* err, int err_len);
void hybrid_sort_native_destroy(void* handle);
}

namespace {

constexpr int kTurboJpegPixelFormatBgr = 1;
constexpr int kTurboJpegSubsample420 = 2;
constexpr int kDetectionFields = 20;
constexpr int kDefaultMaxDet = 100;
constexpr int kDefaultMaxNms = 300;
constexpr int kDefaultMaxOutputDets = 300;
constexpr double kPi = 3.14159265358979323846;

static double steady_seconds();
static double wall_time_seconds();

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
  std::string model_path =
      "models/yolov8n-face-640-b1-int8-hybrid-split-kptconf-rk3588.rknn";
  std::string ws_host = "0.0.0.0";
  std::string ws_path = "/ws/inference";
  std::string webui_host = "0.0.0.0";
  bool no_stdout_json = false;
  bool stdout_debug_json = false;
  bool no_output_jsonl = false;
  bool ws_json = true;
  bool webui = false;
  bool json_debug_keypoints = false;
  bool sector_output = false;
  bool print_profile_summary = false;
  bool profile_system_load = false;
  bool decode_prefetch = false;
  bool camera_prefetch = true;
  bool bound_input = true;
  bool staging_copy_input = true;
  bool staging_pipeline = true;
  bool angle_vectorized = false;
  bool force_build = false;
  int camera_width = 1920;
  int camera_height = 1080;
  int camera_fps = 30;
  int camera_buffers = 4;
  int camera_timeout_ms = 3000;
  int max_frames = 0;
  int ws_port = 8001;
  int webui_port = 8080;
  int webui_jpeg_quality = 80;
  int num_sectors = 8;
  int profile_interval = 30;
  int system_load_interval_ms = 200;
  int fit_degree = 4;
  float conf_threshold = 0.1f;
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
  bool coast_hold = false;
  bool boundary_recover = true;
  bool filter_invalid_boxes = true;
  int track_buffer = 500;
  int coast_frames = 0;
  int tracker_frame_rate = 30;
  int tracker_delta_t = 3;
  int tracker_min_hits = 1;
  float tracker_high_thresh = 0.5f;
  float tracker_low_thresh = 0.1f;
  float tracker_match_thresh = 0.15f;
  float tracker_new_thresh = 0.5f;
  float new_track_overlap_thresh = 0.4f;
  float tracker_inertia = 0.1f;
  float lost_velocity_decay = 0.85f;
  float smooth_bbox_alpha = 0.6f;
  float max_width_ratio = 0.6f;
  float inherit_center_dist_thresh = 1.0f;
  float inherit_size_ratio_thresh = 0.5f;
  float inherit_ambiguity_margin = 0.25f;
  float boundary_margin = 0.04f;
  float boundary_size_ratio_thresh = 0.45f;
  float boundary_center_dist_thresh = 1.8f;
  int boundary_time_window = 90;
  bool final_boundary_dedup = true;
  float final_boundary_dedup_iou_thresh = 0.70f;
  float final_boundary_dedup_center_thresh = 0.80f;
  float final_boundary_dedup_size_ratio_thresh = 0.25f;
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
  std::vector<SliceMeta> slices;
};

struct Image {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> bgr;
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
  std::vector<int> track_ids;
  int raw_detection_count = 0;
  int detection_count = 0;
  int filtered_invalid = 0;
  int inherited_ids = 0;
  int boundary_recovered = 0;
  int coasting_added = 0;
  int final_dedup_removed = 0;
  bool tracker_enabled = false;
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
  FrameResult result;
};

struct FrameRateStats {
  double frame_ms = 0.0;
  double instant_fps = 0.0;
  double average_fps = 0.0;
  double elapsed_s = 0.0;
  int frames = 0;
};

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
  std::string npu_load_raw;
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
    snap.npu_valid = sample_devfreq_kind({"npu", "rknpu"}, &snap.npu_percent, &snap.npu_freq_hz, &snap.npu_load_raw);
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

static std::vector<uint8_t> read_bytes(const std::string& path) {
  std::ifstream f(path.c_str(), std::ios::binary);
  if (!f) {
    throw std::runtime_error("cannot open file: " + path);
  }
  f.seekg(0, std::ios::end);
  const std::streamoff size = f.tellg();
  if (size <= 0) {
    throw std::runtime_error("empty file: " + path);
  }
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(static_cast<size_t>(size));
  f.read(reinterpret_cast<char*>(data.data()), size);
  if (!f) {
    throw std::runtime_error("failed to read file: " + path);
  }
  return data;
}

static std::vector<float> read_floats(const std::string& path) {
  const std::vector<uint8_t> bytes = read_bytes(path);
  if (bytes.size() % sizeof(float) != 0) {
    throw std::runtime_error("float file size is not aligned: " + path);
  }
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

static std::vector<std::string> read_image_list(const std::string& path) {
  std::ifstream f(path.c_str());
  if (!f) {
    throw std::runtime_error("cannot open image list: " + path);
  }
  std::vector<std::string> paths;
  std::string line;
  while (std::getline(f, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    paths.push_back(line);
  }
  return paths;
}

static int xioctl(int fd, unsigned long request, void* arg) {
  int ret = 0;
  do {
    ret = ioctl(fd, request, arg);
  } while (ret == -1 && errno == EINTR);
  return ret;
}

static int get_int(const std::map<std::string, std::string>& meta,
                   const std::string& key,
                   int default_value = 0) {
  const auto it = meta.find(key);
  if (it == meta.end() || it->second.empty()) {
    return default_value;
  }
  return std::stoi(it->second);
}

static float get_float(const std::map<std::string, std::string>& meta,
                       const std::string& key,
                       float default_value = 0.0f) {
  const auto it = meta.find(key);
  if (it == meta.end() || it->second.empty()) {
    return default_value;
  }
  return std::stof(it->second);
}

static std::map<std::string, std::string> load_meta_kv(const std::string& path) {
  std::ifstream f(path.c_str());
  if (!f) {
    throw std::runtime_error("cannot open meta file: " + path);
  }
  std::map<std::string, std::string> meta;
  std::string line;
  while (std::getline(f, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const size_t eq = line.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    meta[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
  }
  return meta;
}

static MapMeta load_map_meta(const std::string& map_dir) {
  const auto kv = load_meta_kv(join_path(map_dir, "meta.txt"));
  MapMeta meta;
  meta.num_slices = get_int(kv, "num_slices");
  meta.roi_h = get_int(kv, "roi_h");
  meta.roi_w = get_int(kv, "roi_w");
  meta.imgsz = get_int(kv, "imgsz");
  meta.process_width = get_int(kv, "process_width");
  meta.process_height = get_int(kv, "process_height");
  meta.img_width = get_int(kv, "img_width");
  meta.img_height = get_int(kv, "img_height");
  meta.center_x = get_int(kv, "center_x");
  meta.center_y = get_int(kv, "center_y");
  meta.radius = get_int(kv, "radius");
  meta.base_map_width = get_int(kv, "base_map_width");
  meta.base_map_height = get_int(kv, "base_map_height");

  if (meta.num_slices <= 0 || meta.roi_h <= 0 || meta.roi_w <= 0 || meta.imgsz <= 0) {
    throw std::runtime_error("invalid map meta shape in " + join_path(map_dir, "meta.txt"));
  }
  meta.slices.resize(static_cast<size_t>(meta.num_slices));
  for (int i = 0; i < meta.num_slices; ++i) {
    const std::string prefix = "slice" + std::to_string(i) + ".";
    SliceMeta& s = meta.slices[static_cast<size_t>(i)];
    s.start_x = get_int(kv, prefix + "start_x");
    s.slice_width = get_int(kv, prefix + "slice_width");
    s.slice_height = get_int(kv, prefix + "slice_height");
    s.original_width = get_int(kv, prefix + "original_width");
    s.original_height = get_int(kv, prefix + "original_height");
    s.wrap_around = get_int(kv, prefix + "wrap_around");
    s.gain = get_float(kv, prefix + "gain", 1.0f);
    s.left = get_int(kv, prefix + "left");
    s.top = get_int(kv, prefix + "top");
    s.new_width = get_int(kv, prefix + "new_width", meta.imgsz);
    s.new_height = get_int(kv, prefix + "new_height", meta.imgsz);
    if (s.slice_width <= 0 || s.slice_height <= 0 || s.gain <= 0.0f) {
      throw std::runtime_error("invalid slice meta: " + prefix);
    }
  }
  return meta;
}

class TurboJpegDecoder {
 public:
  TurboJpegDecoder() {
    handle_ = tjInitDecompress();
    if (!handle_) {
      throw std::runtime_error("tjInitDecompress failed");
    }
  }

  ~TurboJpegDecoder() {
    if (handle_) {
      tjDestroy(handle_);
    }
  }

  Image decode_buffer(const uint8_t* jpeg_data, size_t jpeg_size) {
    Image image;
    int subsamp = 0;
    int colorspace = 0;
    if (tjDecompressHeader3(
            handle_,
            jpeg_data,
            static_cast<unsigned long>(jpeg_size),
            &image.width,
            &image.height,
            &subsamp,
            &colorspace) != 0) {
      throw std::runtime_error("tjDecompressHeader3 failed: " + error_string());
    }
    if (image.width <= 0 || image.height <= 0) {
      throw std::runtime_error("invalid jpeg shape");
    }
    image.bgr.resize(static_cast<size_t>(image.width) * image.height * 3);
    if (tjDecompress2(
            handle_,
            jpeg_data,
            static_cast<unsigned long>(jpeg_size),
            image.bgr.data(),
            image.width,
            0,
            image.height,
            kTurboJpegPixelFormatBgr,
            0) != 0) {
      throw std::runtime_error("tjDecompress2 failed: " + error_string());
    }
    return image;
  }

 private:
  std::string error_string() const {
    const char* err = tjGetErrorStr2(handle_);
    return err ? std::string(err) : std::string("unknown");
  }

  tjhandle handle_ = nullptr;
};

class TurboJpegEncoder {
 public:
  TurboJpegEncoder() {
    handle_ = tjInitCompress();
    if (!handle_) {
      throw std::runtime_error("tjInitCompress failed");
    }
  }

  ~TurboJpegEncoder() {
    if (handle_) {
      tjDestroy(handle_);
    }
  }

  std::vector<uint8_t> encode_bgr(const Image& image, int quality) {
    if (image.width <= 0 || image.height <= 0 ||
        image.bgr.size() != static_cast<size_t>(image.width) * image.height * 3) {
      throw std::runtime_error("invalid image for jpeg encode");
    }
    quality = std::max(1, std::min(100, quality));
    unsigned char* jpeg_buf = nullptr;
    unsigned long jpeg_size = 0;
    const int ret = tjCompress2(
        handle_,
        image.bgr.data(),
        image.width,
        0,
        image.height,
        kTurboJpegPixelFormatBgr,
        &jpeg_buf,
        &jpeg_size,
        kTurboJpegSubsample420,
        quality,
        0);
    if (ret != 0) {
      std::string err = error_string();
      if (jpeg_buf) {
        tjFree(jpeg_buf);
      }
      throw std::runtime_error("tjCompress2 failed: " + err);
    }
    std::vector<uint8_t> out(jpeg_buf, jpeg_buf + jpeg_size);
    tjFree(jpeg_buf);
    return out;
  }

 private:
  std::string error_string() const {
    const char* err = tjGetErrorStr2(handle_);
    return err ? std::string(err) : std::string("unknown");
  }

  tjhandle handle_ = nullptr;
};

struct DecodedFrame {
  size_t index = 0;
  std::string image_path;
  Image image;
  double file_read_ms = 0.0;
  double jpeg_decode_ms = 0.0;
  double camera_read_ms = 0.0;
};

class ImageDecodePrefetcher {
 public:
  explicit ImageDecodePrefetcher(const std::vector<std::string>& paths)
      : paths_(paths) {
    worker_ = std::thread(&ImageDecodePrefetcher::worker_loop, this);
  }

  ~ImageDecodePrefetcher() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  DecodedFrame pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&]() { return !queue_.empty() || done_ || error_ != nullptr; });
    if (error_ != nullptr) {
      std::rethrow_exception(error_);
    }
    if (queue_.empty()) {
      throw std::runtime_error("decode prefetch queue ended unexpectedly");
    }
    DecodedFrame frame = std::move(queue_.front());
    queue_.pop_front();
    cv_.notify_all();
    return frame;
  }

 private:
  void worker_loop() {
    try {
      TurboJpegDecoder decoder;
      for (size_t i = 0; i < paths_.size(); ++i) {
        {
          std::unique_lock<std::mutex> lock(mutex_);
          cv_.wait(lock, [&]() { return stop_ || queue_.size() < kMaxQueue; });
          if (stop_) {
            return;
          }
        }

        const std::string& image_path = paths_[i];
        double t0 = steady_seconds();
        const std::vector<uint8_t> jpeg = read_bytes(image_path);
        double t1 = steady_seconds();
        const double file_read_ms = (t1 - t0) * 1000.0;
        t0 = steady_seconds();
        Image image = decoder.decode_buffer(jpeg.data(), jpeg.size());
        t1 = steady_seconds();
        const double jpeg_decode_ms = (t1 - t0) * 1000.0;

        DecodedFrame frame;
        frame.index = i;
        frame.image_path = image_path;
        frame.image = std::move(image);
        frame.file_read_ms = file_read_ms;
        frame.jpeg_decode_ms = jpeg_decode_ms;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          queue_.push_back(std::move(frame));
        }
        cv_.notify_all();
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
      }
      cv_.notify_all();
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = std::current_exception();
        done_ = true;
      }
      cv_.notify_all();
    }
  }

  static constexpr size_t kMaxQueue = 2;
  const std::vector<std::string>& paths_;
  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<DecodedFrame> queue_;
  bool stop_ = false;
  bool done_ = false;
  std::exception_ptr error_;
};

class V4L2MjpegCamera {
 public:
  struct Frame {
    const uint8_t* data = nullptr;
    size_t size = 0;
    int index = -1;
  };

  V4L2MjpegCamera(const std::string& device,
                  int width,
                  int height,
                  int fps,
                  int buffer_count,
                  int timeout_ms)
      : device_(device),
        width_(width),
        height_(height),
        fps_(fps),
        buffer_count_(std::max(2, buffer_count)),
        timeout_ms_(timeout_ms) {}

  ~V4L2MjpegCamera() {
    try {
      close();
    } catch (...) {
    }
  }

  void open() {
    fd_ = ::open(device_.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (fd_ < 0) {
      throw std::runtime_error("cannot open camera " + device_ + ": " + std::strerror(errno));
    }

    v4l2_capability cap{};
    if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
      throw std::runtime_error("VIDIOC_QUERYCAP failed: " + std::string(std::strerror(errno)));
    }
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) || !(cap.capabilities & V4L2_CAP_STREAMING)) {
      throw std::runtime_error("camera must support VIDEO_CAPTURE and STREAMING");
    }

    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = static_cast<uint32_t>(width_);
    fmt.fmt.pix.height = static_cast<uint32_t>(height_);
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;
    if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
      throw std::runtime_error("VIDIOC_S_FMT MJPEG failed: " + std::string(std::strerror(errno)));
    }
    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG) {
      throw std::runtime_error("camera did not accept MJPEG format");
    }
    width_ = static_cast<int>(fmt.fmt.pix.width);
    height_ = static_cast<int>(fmt.fmt.pix.height);

    if (fps_ > 0) {
      v4l2_streamparm parm{};
      parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      parm.parm.capture.timeperframe.numerator = 1;
      parm.parm.capture.timeperframe.denominator = static_cast<uint32_t>(fps_);
      xioctl(fd_, VIDIOC_S_PARM, &parm);
    }

    v4l2_requestbuffers req{};
    req.count = static_cast<uint32_t>(buffer_count_);
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
      throw std::runtime_error("VIDIOC_REQBUFS failed: " + std::string(std::strerror(errno)));
    }
    if (req.count < 2) {
      throw std::runtime_error("camera returned too few buffers");
    }

    buffers_.resize(req.count);
    for (uint32_t i = 0; i < req.count; ++i) {
      v4l2_buffer buf{};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;
      if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
        throw std::runtime_error("VIDIOC_QUERYBUF failed: " + std::string(std::strerror(errno)));
      }
      void* start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
      if (start == MAP_FAILED) {
        throw std::runtime_error("mmap camera buffer failed: " + std::string(std::strerror(errno)));
      }
      buffers_[i].start = start;
      buffers_[i].length = buf.length;
    }

    for (uint32_t i = 0; i < buffers_.size(); ++i) {
      v4l2_buffer buf{};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;
      if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
        throw std::runtime_error("VIDIOC_QBUF failed: " + std::string(std::strerror(errno)));
      }
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
      throw std::runtime_error("VIDIOC_STREAMON failed: " + std::string(std::strerror(errno)));
    }
    streaming_ = true;
  }

  Frame read_frame() {
    for (;;) {
      fd_set fds;
      FD_ZERO(&fds);
      FD_SET(fd_, &fds);
      timeval tv{};
      tv.tv_sec = timeout_ms_ / 1000;
      tv.tv_usec = (timeout_ms_ % 1000) * 1000;
      const int sel = select(fd_ + 1, &fds, nullptr, nullptr, &tv);
      if (sel < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw std::runtime_error("select camera failed: " + std::string(std::strerror(errno)));
      }
      if (sel == 0) {
        throw std::runtime_error("camera read timeout");
      }

      v4l2_buffer buf{};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      if (xioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
        if (errno == EAGAIN) {
          continue;
        }
        throw std::runtime_error("VIDIOC_DQBUF failed: " + std::string(std::strerror(errno)));
      }
      if (buf.index >= buffers_.size()) {
        throw std::runtime_error("camera returned invalid buffer index");
      }
      Frame frame;
      frame.data = static_cast<const uint8_t*>(buffers_[buf.index].start);
      frame.size = buf.bytesused;
      frame.index = static_cast<int>(buf.index);
      return frame;
    }
  }

  void release_frame(const Frame& frame) {
    if (frame.index < 0) {
      return;
    }
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = static_cast<uint32_t>(frame.index);
    if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
      throw std::runtime_error("VIDIOC_QBUF release failed: " + std::string(std::strerror(errno)));
    }
  }

  void close() {
    if (fd_ < 0) {
      return;
    }
    if (streaming_) {
      v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      xioctl(fd_, VIDIOC_STREAMOFF, &type);
      streaming_ = false;
    }
    for (Buffer& buffer : buffers_) {
      if (buffer.start && buffer.start != MAP_FAILED) {
        munmap(buffer.start, buffer.length);
        buffer.start = nullptr;
        buffer.length = 0;
      }
    }
    buffers_.clear();
    ::close(fd_);
    fd_ = -1;
  }

 private:
  struct Buffer {
    void* start = nullptr;
    size_t length = 0;
  };

  std::string device_;
  int width_ = 0;
  int height_ = 0;
  int fps_ = 0;
  int buffer_count_ = 0;
  int timeout_ms_ = 0;
  int fd_ = -1;
  bool streaming_ = false;
  std::vector<Buffer> buffers_;
};

class CameraDecodePrefetcher {
 public:
  explicit CameraDecodePrefetcher(const Config& cfg)
      : cfg_(cfg) {
    worker_ = std::thread(&CameraDecodePrefetcher::worker_loop, this);
  }

  ~CameraDecodePrefetcher() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  DecodedFrame pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&]() { return !queue_.empty() || done_ || error_ != nullptr; });
    if (error_ != nullptr) {
      std::rethrow_exception(error_);
    }
    if (queue_.empty()) {
      throw std::runtime_error("camera decode prefetch queue ended unexpectedly");
    }
    DecodedFrame frame = std::move(queue_.front());
    queue_.pop_front();
    cv_.notify_all();
    return frame;
  }

 private:
  void worker_loop() {
    try {
      TurboJpegDecoder decoder;
      V4L2MjpegCamera camera(
          cfg_.camera_device,
          cfg_.camera_width,
          cfg_.camera_height,
          cfg_.camera_fps,
          cfg_.camera_buffers,
          cfg_.camera_timeout_ms);
      camera.open();

      for (int i = 0; cfg_.max_frames == 0 || i < cfg_.max_frames; ++i) {
        {
          std::unique_lock<std::mutex> lock(mutex_);
          cv_.wait(lock, [&]() { return stop_ || queue_.size() < kMaxQueue; });
          if (stop_) {
            return;
          }
        }

        V4L2MjpegCamera::Frame camera_frame;
        bool frame_acquired = false;
        double t0 = steady_seconds();
        camera_frame = camera.read_frame();
        double t1 = steady_seconds();
        frame_acquired = true;
        const double camera_read_ms = (t1 - t0) * 1000.0;

        Image image;
        double jpeg_decode_ms = 0.0;
        try {
          t0 = steady_seconds();
          image = decoder.decode_buffer(camera_frame.data, camera_frame.size);
          t1 = steady_seconds();
          jpeg_decode_ms = (t1 - t0) * 1000.0;
          camera.release_frame(camera_frame);
          frame_acquired = false;
        } catch (...) {
          if (frame_acquired) {
            try {
              camera.release_frame(camera_frame);
            } catch (...) {
            }
          }
          throw;
        }

        std::ostringstream frame_name;
        frame_name << cfg_.camera_device << "#" << i;
        DecodedFrame decoded;
        decoded.index = static_cast<size_t>(i);
        decoded.image_path = frame_name.str();
        decoded.image = std::move(image);
        decoded.camera_read_ms = camera_read_ms;
        decoded.jpeg_decode_ms = jpeg_decode_ms;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          queue_.push_back(std::move(decoded));
        }
        cv_.notify_all();
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
      }
      cv_.notify_all();
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = std::current_exception();
        done_ = true;
      }
      cv_.notify_all();
    }
  }

  static constexpr size_t kMaxQueue = 2;
  Config cfg_;
  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<DecodedFrame> queue_;
  bool stop_ = false;
  bool done_ = false;
  std::exception_ptr error_;
};

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
      << "  --sector-output       emit Python-compatible sector aggregate JSON\n"
      << "  --num-sectors N       sector count for --sector-output, default: 8\n"
      << "  --print-profile-summary  print average per-stage timings to stderr\n"
      << "  --profile-system-load write background CPU/GPU/NPU/memory load JSONL\n"
      << "  --system-load-interval-ms N  hardware load sample interval, default: 200\n"
      << "  --decode-prefetch     pre-decode next image-list frame in a worker thread\n"
      << "  --no-camera-prefetch  disable camera read/decode worker pipeline\n"
      << "  --no-bound-input      disable RKNN bound-input/OpenCL imported-output path\n"
      << "  --staging-copy-input  OpenCL writes staging buffers, then CPU copies to RKNN input\n"
      << "  --staging-pipeline    overlap next OpenCL staging with current blocking RKNN, default: on\n"
      << "  --no-staging-pipeline disable staging overlap pipeline\n"
      << "  --no-stdout-json       suppress JSON results on stdout\n"
      << "  --stdout-debug-json    print full debug detection JSON to stdout\n"
      << "  --map-dir DIR          exported map directory, default: map_export\n"
      << "  --model PATH           RKNN model path\n"
      << "  --calib-yaml PATH      accepted for Python CLI compatibility; C++ defaults to fit_4\n"
      << "  --conf VALUE           confidence threshold, default: 0.1\n"
      << "  --conf-threshold VALUE alias of --conf\n"
      << "  --iou-threshold VALUE  alias of --decode-iou\n"
      << "  --decode-iou VALUE     decode NMS IoU threshold, default: 0.99\n"
      << "  --merge-iou VALUE      cross-slice merge IoU threshold, default: 0.2\n"
      << "  --nms-iou VALUE        final NMS IoU threshold, default: 0.6\n"
      << "  --overlap VALUE        slice overlap ratio, default: 0.1\n"
      << "  --max-det N            per-slice max detections, default: 100\n"
      << "  --no-tracker           disable native HybridSORT tracking\n"
      << "  --track-buffer N       tracker lost buffer frames, default: 500\n"
      << "  --tracker-match-thresh VALUE  tracker association IoU threshold, default: 0.15\n"
      << "  --tracker-new-thresh VALUE    new track confidence threshold, default: 0.5\n"
      << "  --new-track-overlap-thresh VALUE  avoid duplicate new tracks, default: 0.4\n"
      << "  --lost-velocity-decay VALUE   multiply unmatched track velocity each frame, default: 0.85\n"
      << "  --inherit-center-dist-thresh VALUE  short occlusion ID inheritance gate, default: 1.0\n"
      << "  --inherit-size-ratio-thresh VALUE   short occlusion size gate, default: 0.5\n"
      << "  --inherit-ambiguity-margin VALUE    short occlusion unique-best gate, default: 0.25\n"
      << "  --no-smooth-bbox       disable tracker bbox EMA smoothing\n"
      << "  --smooth-bbox-alpha VALUE  tracker bbox EMA alpha, default: 0.5\n"
      << "  --coast-frames N       output lost tracks for N frames, default: 0\n"
      << "  --coast-hold           keep coasting bbox at last observation\n"
      << "  --kalman-bbox          use native Kalman state bbox for active tracks\n"
      << "  --no-boundary-recover  disable geometry boundary ID recovery\n"
      << "  --no-final-boundary-dedup  disable final boundary duplicate suppression\n"
      << "  --final-boundary-dedup-iou VALUE     final min-area IoU duplicate gate, default: 0.70\n"
      << "  --final-boundary-dedup-center VALUE  final spatial duplicate center gate, default: 0.80\n"
      << "  --final-boundary-dedup-size VALUE    final spatial duplicate area-ratio gate, default: 0.25\n"
      << "  --no-filter-invalid-boxes  disable Python-compatible invalid bbox filtering\n"
      << "  --angle-vectorized     accepted; C++ angle path is already vector-style\n"
      << "  --force-build          accepted as no-op; board_cpp is prebuilt\n";
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
    } else if (arg == "--sector-output") {
      cfg.sector_output = true;
    } else if (arg == "--num-sectors") {
      cfg.num_sectors = std::stoi(need_value(arg));
    } else if (arg == "--print-profile-summary") {
      cfg.print_profile_summary = true;
    } else if (arg == "--decode-prefetch") {
      cfg.decode_prefetch = true;
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
    } else if (arg == "--track-buffer") {
      cfg.track_buffer = std::stoi(need_value(arg));
    } else if (arg == "--tracker-match-thresh") {
      cfg.tracker_match_thresh = std::stof(need_value(arg));
    } else if (arg == "--tracker-new-thresh") {
      cfg.tracker_new_thresh = std::stof(need_value(arg));
    } else if (arg == "--new-track-overlap-thresh") {
      cfg.new_track_overlap_thresh = std::stof(need_value(arg));
    } else if (arg == "--lost-velocity-decay") {
      cfg.lost_velocity_decay = std::stof(need_value(arg));
    } else if (arg == "--inherit-center-dist-thresh") {
      cfg.inherit_center_dist_thresh = std::stof(need_value(arg));
    } else if (arg == "--inherit-size-ratio-thresh") {
      cfg.inherit_size_ratio_thresh = std::stof(need_value(arg));
    } else if (arg == "--inherit-ambiguity-margin") {
      cfg.inherit_ambiguity_margin = std::stof(need_value(arg));
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
    } else if (arg == "--coast-frames") {
      cfg.coast_frames = std::stoi(need_value(arg));
    } else if (arg == "--coast-hold") {
      cfg.coast_hold = true;
    } else if (arg == "--kalman-bbox") {
      cfg.kalman_bbox = true;
    } else if (arg == "--boundary-recover") {
      cfg.boundary_recover = true;
    } else if (arg == "--no-boundary-recover") {
      cfg.boundary_recover = false;
    } else if (arg == "--boundary-margin") {
      cfg.boundary_margin = std::stof(need_value(arg));
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
    } else if (arg == "--max-width-ratio") {
      cfg.max_width_ratio = std::stof(need_value(arg));
    } else if (arg == "--profile-interval") {
      cfg.profile_interval = std::stoi(need_value(arg));
    } else if (arg == "--profile-system-load" || arg == "--json-system-load") {
      cfg.profile_system_load = true;
    } else if (arg == "--system-load-interval-ms") {
      cfg.system_load_interval_ms = std::stoi(need_value(arg));
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
  if (cfg.camera_width <= 0 || cfg.camera_height <= 0) {
    throw std::runtime_error("camera width/height must be positive");
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
  if (cfg.max_frames < 0) {
    throw std::runtime_error("--max-frames must be non-negative");
  }
  if (cfg.num_sectors <= 0) {
    throw std::runtime_error("--num-sectors must be positive");
  }
  if (cfg.track_buffer < 0) {
    throw std::runtime_error("--track-buffer must be non-negative");
  }
  if (cfg.coast_frames < 0) {
    throw std::runtime_error("--coast-frames must be non-negative");
  }
  cfg.system_load_interval_ms = std::max(50, cfg.system_load_interval_ms);
  if (cfg.fit_degree != 4 && cfg.fit_degree != 5) {
    throw std::runtime_error("--fit-degree must be 4 or 5");
  }
  cfg.smooth_bbox_alpha = std::max(0.0f, std::min(cfg.smooth_bbox_alpha, 0.99f));
  cfg.lost_velocity_decay = std::max(0.0f, std::min(cfg.lost_velocity_decay, 1.0f));
  cfg.inherit_center_dist_thresh =
      std::max(0.0f, std::min(cfg.inherit_center_dist_thresh, 5.0f));
  cfg.inherit_size_ratio_thresh =
      std::max(0.0f, std::min(cfg.inherit_size_ratio_thresh, 1.0f));
  cfg.inherit_ambiguity_margin =
      std::max(0.0f, std::min(cfg.inherit_ambiguity_margin, 5.0f));
  cfg.boundary_margin = std::max(0.01f, std::min(cfg.boundary_margin, 0.4f));
  cfg.max_width_ratio = std::max(0.05f, std::min(cfg.max_width_ratio, 1.0f));
  cfg.final_boundary_dedup_iou_thresh =
      std::max(0.0f, std::min(cfg.final_boundary_dedup_iou_thresh, 1.0f));
  cfg.final_boundary_dedup_center_thresh =
      std::max(0.05f, std::min(cfg.final_boundary_dedup_center_thresh, 5.0f));
  cfg.final_boundary_dedup_size_ratio_thresh =
      std::max(0.0f, std::min(cfg.final_boundary_dedup_size_ratio_thresh, 1.0f));
  return cfg;
}

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
      "gpu_percent", "gpu_freq_hz", "npu_percent", "npu_freq_hz", "thermal_max_c"};
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
            << print_stat("npu_percent", "%") << " | "
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

struct AngleDistanceState {
  bool has_last_distance = false;
  double last_distance = 0.0;
};

struct TargetInfo {
  int id = -1;
  bool has_azimuth = false;
  bool has_elevation = false;
  bool has_eye_pixel_dist = false;
  bool has_distance = false;
  double azimuth = 0.0;
  double elevation = 0.0;
  double eye_pixel_dist = 0.0;
  double distance = 0.0;
};

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

static void draw_annotation(Image& image,
                            const Config& cfg,
                            const FrameResult& result,
                            const std::vector<TargetInfo>& targets,
                            const FrameRateStats& fps) {
  draw_text(image, 12, 12, "FPS:" + std::to_string(static_cast<int>(fps.average_fps + 0.5)),
            32, 255, 255, 3);
  draw_sector_overlay(image, cfg, targets);
  for (int i = 0; i < result.detection_count; ++i) {
    const float* det = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
    const int track_id =
        (i < static_cast<int>(result.track_ids.size()) && result.track_ids[static_cast<size_t>(i)] > 0)
            ? result.track_ids[static_cast<size_t>(i)]
            : i + 1;
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
    const uint8_t b = sector_target ? 0 : static_cast<uint8_t>((track_id * 53) % 180 + 60);
    const uint8_t g = sector_target ? 0 : static_cast<uint8_t>((track_id * 97) % 180 + 60);
    const uint8_t r = sector_target ? 255 : static_cast<uint8_t>((track_id * 131) % 180 + 60);
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
    label << "ID:" << track_id;
    const TargetInfo* target =
        i < static_cast<int>(targets.size()) ? &targets[static_cast<size_t>(i)] : nullptr;
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
      TargetInfo info;
      info.id = track_id;
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
      active_ids.push_back(target.id);
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
};

static std::string build_inference_json(const std::vector<TargetInfo>& targets,
                                        int frame_id,
                                        bool debug_keypoints,
                                        const FrameResult& result,
                                        const FrameRateStats& fps) {
  (void)debug_keypoints;
  (void)result;
  std::ostringstream os;
  os << "{\"timestamp\":" << std::fixed << std::setprecision(3) << wall_time_seconds()
     << ",\"frame_id\":" << frame_id
     << ",\"fps\":{\"instant\":" << fps.instant_fps
     << ",\"average\":" << fps.average_fps
     << ",\"frame_ms\":" << fps.frame_ms
     << ",\"elapsed_s\":" << fps.elapsed_s
     << ",\"frames\":" << fps.frames << "}"
     << ",\"timings_ms\":{\"frame_total\":" << fps.frame_ms << "}"
     << ",\"targets\":{";
  for (size_t i = 0; i < targets.size(); ++i) {
    if (i > 0) {
      os << ",";
    }
    const TargetInfo& target = targets[i];
    os << "\"" << target.id << "\":{\"id\":" << target.id << ",\"azimuth\":";
    print_json_nullable(os, target.has_azimuth, target.azimuth, 3);
    os << ",\"elevation\":";
    print_json_nullable(os, target.has_elevation, target.elevation, 3);
    os << ",\"eye_pixel_dist\":";
    print_json_nullable(os, target.has_eye_pixel_dist, target.eye_pixel_dist, 2);
    os << ",\"distance\":";
    print_json_nullable(os, target.has_distance, target.distance, 3);
    os << "}";
  }
  os << "}}";
  return os.str();
}

static std::string build_sector_json(const std::vector<TargetInfo>& targets,
                                     int frame_id,
                                     int num_sectors,
                                     const FrameResult& result,
                                     const FrameRateStats& fps) {
  const int sectors_count = std::max(1, num_sectors);
  const double sector_size = 360.0 / static_cast<double>(sectors_count);

  struct SectorBest {
    bool valid = false;
    double area = 0.0;
    double azimuth = 0.0;
    double elevation = 0.0;
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
    return build_sector_json(targets, frame_id, cfg.num_sectors, result, fps);
  }
  return build_inference_json(targets, frame_id, cfg.json_debug_keypoints, result, fps);
}

static void print_inference_jsonl(std::ostream& os,
                                  const std::vector<TargetInfo>& targets,
                                  int frame_id,
                                  bool debug_keypoints,
                                  const FrameResult& result) {
  FrameRateStats empty_fps;
  os << build_inference_json(targets, frame_id, debug_keypoints, result, empty_fps) << "\n";
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
  os << "  \"map\": {\"num_slices\": " << meta.num_slices << ", \"roi_w\": " << meta.roi_w
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
            << ", \"inherited\": " << result.inherited_ids
            << ", \"boundary_recovered\": " << result.boundary_recovered
            << ", \"coasting_added\": " << result.coasting_added
            << ", \"final_dedup_removed\": " << result.final_dedup_removed << "},\n";
  os << "  \"detection_count\": " << result.detection_count << ",\n";
  os << "  \"detections\": [\n";
  for (int i = 0; i < result.detection_count; ++i) {
    const float* det = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
    os << "    {\"bbox\": ";
    print_json_array(os, det, 4);
    if (i < static_cast<int>(result.track_ids.size()) && result.track_ids[static_cast<size_t>(i)] > 0) {
      os << ", \"track_id\": " << result.track_ids[static_cast<size_t>(i)];
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
                               const FrameRateStats& fps) {
  os << "{\"frame_index\":" << result.frame_index
     << ",\"image\":\"" << json_escape(result.image_path) << "\""
     << ",\"frame\":{\"width\":" << result.frame_width << ",\"height\":" << result.frame_height << "}"
     << ",\"map\":{\"num_slices\":" << meta.num_slices << ",\"roi_w\":" << meta.roi_w
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
     << ",\"inherited\":" << result.inherited_ids
     << ",\"boundary_recovered\":" << result.boundary_recovered
     << ",\"coasting_added\":" << result.coasting_added
     << ",\"final_dedup_removed\":" << result.final_dedup_removed << "}"
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
    os << ",\"score\":" << std::fixed << std::setprecision(6) << det[4] << ",\"keypoints\":[";
    for (int k = 0; k < 5; ++k) {
      if (k > 0) {
        os << ",";
      }
      print_json_array(os, det + 5 + k * 3, 3);
    }
    os << "]}";
  }
  os << "]}\n";
}

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
    const int max_age =
        static_cast<int>((static_cast<float>(cfg_.tracker_frame_rate) / 30.0f) * cfg_.track_buffer);
    max_age_ = std::max(0, max_age);
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
    const int ret = hybrid_sort_native_update(
        handle_.get(),
        tracker_input.data(),
        tracker_input_count,
        5,
        tracker_h,
        tracker_w,
        tracker_h,
        tracker_w,
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

    apply_short_occlusion_inheritance(result, raw_ids, snapshot_by_raw);
    apply_boundary_recovery(result, raw_ids);
    apply_public_ids(result, raw_ids);
    add_coasting_tracks(result, raw_ids, snapshot_by_raw);
    apply_final_boundary_dedup(result);
    update_public_seen_counts(result);
    update_lost_state(current_active_raw_ids, snapshot_by_raw);
  }

 private:
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
      int best_old_raw = -1;
      float best_dist = cfg_.inherit_center_dist_thresh;
      float second_best_dist = std::numeric_limits<float>::infinity();
      for (const auto& item : lost_confirmed) {
        const int old_raw = item.first;
        if (old_raw == raw_id || public_id_map_.find(old_raw) == public_id_map_.end()) {
          continue;
        }
        const NativeTrackSnapshot& old_track = item.second;
        const std::array<float, 4> obs_bbox = bbox_from_last_observation(old_track);
        float d = center_distance_norm(new_bbox, obs_bbox);
        float size_score = size_ratio_score(new_bbox, obs_bbox);
        if (finite_bbox(old_track.state)) {
          const float state_dist = center_distance_norm(new_bbox, old_track.state);
          if (state_dist < d) {
            d = state_dist;
            size_score = size_ratio_score(new_bbox, old_track.state);
          }
        }
        if (size_score < cfg_.inherit_size_ratio_thresh) {
          continue;
        }
        if (d < best_dist) {
          second_best_dist = best_dist;
          best_dist = d;
          best_old_raw = old_raw;
        } else if (d < second_best_dist) {
          second_best_dist = d;
        }
      }
      if (best_old_raw > 0) {
        if (std::isfinite(second_best_dist) &&
            (second_best_dist - best_dist) < cfg_.inherit_ambiguity_margin) {
          continue;
        }
        const int old_public = public_id_map_[best_old_raw];
        public_id_map_.erase(best_old_raw);
        public_id_map_[raw_id] = old_public;
        result.inherited_ids += 1;
      }
    }
  }

  void apply_boundary_recovery(FrameResult& result, const std::vector<int>& raw_ids) {
    if (!cfg_.boundary_recover || boundary_lost_.empty()) {
      return;
    }
    cleanup_boundary_lost();
    std::vector<int> claimed_old;
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
        public_id_map_[raw_id] = public_id_map_[best_old_raw];
        claimed_old.push_back(best_old_raw);
        remove_boundary_lost(best_old_raw);
        result.boundary_recovered += 1;
      }
    }
  }

  void apply_public_ids(FrameResult& result, const std::vector<int>& raw_ids) {
    result.track_ids.assign(static_cast<size_t>(result.detection_count), -1);
    for (int i = 0; i < result.detection_count; ++i) {
      const int raw_id = i < static_cast<int>(raw_ids.size()) ? raw_ids[static_cast<size_t>(i)] : -1;
      result.track_ids[static_cast<size_t>(i)] = raw_id > 0 ? public_id_for(raw_id) : -1;
    }
  }

  void add_coasting_tracks(FrameResult& result,
                           const std::vector<int>& active_raw_ids,
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
      result.track_ids.push_back(public_id_for(raw_id));
      result.detection_count += 1;
      result.coasting_added += 1;
    }
  }

  void apply_final_boundary_dedup(FrameResult& result) {
    if (!cfg_.final_boundary_dedup || result.detection_count <= 1) {
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
      const BoundarySide side_a = boundary_side_for_bbox(a);
      if (side_a == BoundarySide::None) {
        continue;
      }
      for (int j = i + 1; j < n; ++j) {
        if (!keep[static_cast<size_t>(j)]) {
          continue;
        }
        const std::array<float, 4> b =
            bbox4_from_ptr(result.detections.data() + static_cast<size_t>(j) * kDetectionFields);
        const BoundarySide side_b = boundary_side_for_bbox(b);
        if (!is_final_boundary_duplicate(a, side_a, b, side_b)) {
          continue;
        }
        const int drop = choose_final_boundary_duplicate_drop(result, i, j);
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

    std::vector<float> compact;
    std::vector<int> compact_ids;
    compact.reserve(static_cast<size_t>(n) * kDetectionFields);
    compact_ids.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      if (!keep[static_cast<size_t>(i)]) {
        continue;
      }
      const float* src = result.detections.data() + static_cast<size_t>(i) * kDetectionFields;
      compact.insert(compact.end(), src, src + kDetectionFields);
      compact_ids.push_back(i < static_cast<int>(result.track_ids.size())
                                ? result.track_ids[static_cast<size_t>(i)]
                                : -1);
    }
    result.detections.swap(compact);
    result.track_ids.swap(compact_ids);
    result.detection_count = static_cast<int>(result.track_ids.size());
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
    if (inter <= 0.0f) {
      return false;
    }
    if (min_iou > cfg_.final_boundary_dedup_iou_thresh) {
      return true;
    }
    const float dist = boundary_center_distance_norm(a, side_a, b, side_b);
    const float area_ratio =
        std::min(bbox_area4(a), bbox_area4(b)) / (std::max(bbox_area4(a), bbox_area4(b)) + 1.0e-6f);
    return dist < cfg_.final_boundary_dedup_center_thresh &&
           x_cover > 0.35f &&
           y_cover > 0.35f &&
           area_ratio > cfg_.final_boundary_dedup_size_ratio_thresh;
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

  int choose_final_boundary_duplicate_drop(const FrameResult& result, int i, int j) const {
    const int id_i =
        i < static_cast<int>(result.track_ids.size()) ? result.track_ids[static_cast<size_t>(i)] : -1;
    const int id_j =
        j < static_cast<int>(result.track_ids.size()) ? result.track_ids[static_cast<size_t>(j)] : -1;
    const int seen_i = public_seen_count(id_i);
    const int seen_j = public_seen_count(id_j);
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

  int public_seen_count(int public_id) const {
    const auto it = public_seen_count_.find(public_id);
    return it == public_seen_count_.end() ? 0 : it->second;
  }

  void update_public_seen_counts(const FrameResult& result) {
    for (int i = 0; i < result.detection_count; ++i) {
      if (i >= static_cast<int>(result.track_ids.size())) {
        continue;
      }
      const int public_id = result.track_ids[static_cast<size_t>(i)];
      if (public_id > 0) {
        public_seen_count_[public_id] += 1;
      }
    }
  }

  void update_lost_state(const std::vector<int>& current_active_raw_ids,
                         const std::map<int, NativeTrackSnapshot>& snapshot_by_raw) {
    std::vector<int> lost_ids;
    for (int prev_raw : prev_active_raw_ids_) {
      if (std::find(current_active_raw_ids.begin(), current_active_raw_ids.end(), prev_raw) ==
          current_active_raw_ids.end()) {
        lost_ids.push_back(prev_raw);
      }
    }
    for (int lost_raw : lost_ids) {
      bbox_cache_.erase(lost_raw);
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
  std::map<int, int> public_seen_count_;
  int public_id_counter_ = 0;
  int frame_id_ = 0;
  int max_age_ = 0;
  std::map<int, std::array<float, kDetectionFields>> meta_cache_;
  std::map<int, std::array<float, 4>> bbox_cache_;
  std::vector<int> prev_active_raw_ids_;
  std::map<int, std::array<float, 4>> prev_bbox_;
  std::map<int, std::array<float, 4>> prev_prev_bbox_;
  std::vector<BoundaryLostTarget> boundary_lost_;
};

class MeetEyeRuntime {
 public:
  explicit MeetEyeRuntime(const Config& cfg)
      : cfg_(cfg),
        meta_(load_map_meta(cfg.map_dir)),
        map_x_(read_floats(join_path(cfg.map_dir, "map_x.bin"))),
        map_y_(read_floats(join_path(cfg.map_dir, "map_y.bin"))) {
    if (meta_.num_slices != 3) {
      throw std::runtime_error("librknn_capi_parallel currently requires exactly 3 slices");
    }
    const size_t expected_map_values =
        static_cast<size_t>(meta_.num_slices) * meta_.roi_h * meta_.roi_w;
    if (map_x_.size() != expected_map_values || map_y_.size() != expected_map_values) {
      throw std::runtime_error("map_x/map_y size mismatch with meta.txt");
    }
    const std::string base_map_x_path = join_path(cfg.map_dir, "base_map_x.bin");
    const std::string base_map_y_path = join_path(cfg.map_dir, "base_map_y.bin");
    if (file_exists(base_map_x_path) || file_exists(base_map_y_path)) {
      if (!file_exists(base_map_x_path) || !file_exists(base_map_y_path)) {
        throw std::runtime_error("base_map_x.bin/base_map_y.bin must exist together");
      }
      if (meta_.base_map_width <= 0 || meta_.base_map_height <= 0) {
        throw std::runtime_error("base map files exist but base_map_width/base_map_height are missing in meta.txt");
      }
      base_map_x_ = read_floats(base_map_x_path);
      base_map_y_ = read_floats(base_map_y_path);
      const size_t expected_base_values =
          static_cast<size_t>(meta_.base_map_width) * static_cast<size_t>(meta_.base_map_height);
      if (base_map_x_.size() != expected_base_values || base_map_y_.size() != expected_base_values) {
        throw std::runtime_error("base_map_x/base_map_y size mismatch with meta.txt");
      }
    }

    roi_rects_.resize(static_cast<size_t>(meta_.num_slices) * 4);
    slice_shapes_.resize(static_cast<size_t>(meta_.num_slices) * 2);
    gains_.resize(static_cast<size_t>(meta_.num_slices));
    pads_.resize(static_cast<size_t>(meta_.num_slices) * 2);
    starts_.resize(static_cast<size_t>(meta_.num_slices));
    wraps_.resize(static_cast<size_t>(meta_.num_slices));

    for (int i = 0; i < meta_.num_slices; ++i) {
      const SliceMeta& s = meta_.slices[static_cast<size_t>(i)];
      roi_rects_[static_cast<size_t>(i) * 4 + 0] = s.left;
      roi_rects_[static_cast<size_t>(i) * 4 + 1] = s.top;
      roi_rects_[static_cast<size_t>(i) * 4 + 2] = s.new_width;
      roi_rects_[static_cast<size_t>(i) * 4 + 3] = s.new_height;
      slice_shapes_[static_cast<size_t>(i) * 2 + 0] = s.slice_height;
      slice_shapes_[static_cast<size_t>(i) * 2 + 1] = s.slice_width;
      gains_[static_cast<size_t>(i)] = s.gain;
      pads_[static_cast<size_t>(i) * 2 + 0] = s.left;
      pads_[static_cast<size_t>(i) * 2 + 1] = s.top;
      starts_[static_cast<size_t>(i)] = static_cast<float>(s.start_x);
      wraps_[static_cast<size_t>(i)] = s.wrap_around;
    }

    char err[4096] = {0};
    int ret = face_rknn_parallel_create(cfg_.model_path.c_str(), rknn_.out(), err, sizeof(err));
    if (ret != 0) {
      throw std::runtime_error(std::string("face_rknn_parallel_create failed: ") + err);
    }

    ret = face_rknn_parallel_get_shape(
        rknn_.get(), &input_h_, &input_w_, &input_c_, &channels_, &anchors_);
    if (ret != 0) {
      throw std::runtime_error("face_rknn_parallel_get_shape failed");
    }
    if (input_h_ != meta_.imgsz || input_w_ != meta_.imgsz || input_c_ != 3) {
      std::ostringstream oss;
      oss << "model input shape " << input_w_ << "x" << input_h_ << "x" << input_c_
          << " does not match map imgsz=" << meta_.imgsz;
      throw std::runtime_error(oss.str());
    }
  }

  const MapMeta& meta() const { return meta_; }
  int channels() const { return channels_; }
  int anchors() const { return anchors_; }
  const std::vector<float>* base_map_x() const {
    return base_map_x_.empty() ? nullptr : &base_map_x_;
  }
  const std::vector<float>* base_map_y() const {
    return base_map_y_.empty() ? nullptr : &base_map_y_;
  }

  PreparedStagingFrame prepare_staging_frame(PreparedStagingFrame frame) {
    const Image& image = frame.image;
    if (meta_.img_width > 0 && meta_.img_height > 0 &&
        (image.width != meta_.img_width || image.height != meta_.img_height)) {
      std::ostringstream oss;
      oss << "image shape " << image.width << "x" << image.height
          << " does not match exported map source " << meta_.img_width << "x" << meta_.img_height;
      throw std::runtime_error(oss.str());
    }

    frame.valid = true;
    frame.result = FrameResult();
    frame.result.frame_index = frame.frame_index;
    frame.result.image_path = frame.image_path;
    frame.result.frame_width = image.width;
    frame.result.frame_height = image.height;
    frame.result.detections.assign(
        static_cast<size_t>(cfg_.max_output_dets) * kDetectionFields, 0.0f);

    double t0 = steady_seconds();
    ensure_opencl(image);
    double t1 = steady_seconds();
    frame.result.profile_ms[kProfileOpenclEnsure] = (t1 - t0) * 1000.0;

    frame.original_width =
        meta_.slices.empty() ? static_cast<float>(meta_.process_width)
                             : static_cast<float>(meta_.slices[0].original_width);

    if (!prepare_bound_inputs(frame.result)) {
      throw std::runtime_error("staging pipeline requires RKNN bound input memory");
    }

    ensure_staging_buffers();
    char err[4096] = {0};
    t0 = steady_seconds();
    const int ret = ds_opencl_fused_run_split(
        opencl_.get(),
        image.bgr.data(),
        staging_ptrs_.data(),
        meta_.num_slices,
        frame.result.remap_timings,
        err,
        sizeof(err));
    t1 = steady_seconds();
    frame.result.profile_ms[kProfileOpenclRunOuter] = (t1 - t0) * 1000.0;
    if (ret != 0) {
      throw std::runtime_error(std::string("ds_opencl_fused_run_split(staging) failed: ") + err);
    }
    return frame;
  }

  std::future<PreparedStagingFrame> start_staging_inference(PreparedStagingFrame frame) {
    if (!frame.valid) {
      throw std::runtime_error("cannot infer invalid staging frame");
    }
    copy_staging_to_bound_inputs(frame.result);
    return std::async(std::launch::async, [this, frame = std::move(frame)]() mutable {
      run_bound_inference(frame.original_width, false, frame.result);
      frame.result.detection_count =
          std::max(0, std::min(frame.result.detection_count, cfg_.max_output_dets));
      return frame;
    });
  }

  FrameResult process(const Image& image, const std::string& image_path, int frame_index) {
    if (meta_.img_width > 0 && meta_.img_height > 0 &&
        (image.width != meta_.img_width || image.height != meta_.img_height)) {
      std::ostringstream oss;
      oss << "image shape " << image.width << "x" << image.height
          << " does not match exported map source " << meta_.img_width << "x" << meta_.img_height;
      throw std::runtime_error(oss.str());
    }

    FrameResult result;
    result.frame_index = frame_index;
    result.image_path = image_path;
    result.frame_width = image.width;
    result.frame_height = image.height;
    result.detections.assign(static_cast<size_t>(cfg_.max_output_dets) * kDetectionFields, 0.0f);

    double t0 = steady_seconds();
    ensure_opencl(image);
    double t1 = steady_seconds();
    result.profile_ms[kProfileOpenclEnsure] = (t1 - t0) * 1000.0;

    const float original_width =
        meta_.slices.empty() ? static_cast<float>(meta_.process_width)
                             : static_cast<float>(meta_.slices[0].original_width);

    char err[4096] = {0};
    int ret = 0;
    if (try_process_bound(image, original_width, result)) {
      result.detection_count = std::max(0, std::min(result.detection_count, cfg_.max_output_dets));
      return result;
    }

    t0 = steady_seconds();
    std::vector<uint8_t> rknn_inputs(
        static_cast<size_t>(meta_.num_slices) * meta_.imgsz * meta_.imgsz * 3,
        114);
    t1 = steady_seconds();
    result.profile_ms[kProfileRknnInputAlloc] = (t1 - t0) * 1000.0;

    t0 = steady_seconds();
    ret = ds_opencl_fused_run(
        opencl_.get(),
        image.bgr.data(),
        rknn_inputs.data(),
        result.remap_timings,
        err,
        sizeof(err));
    if (ret != 0) {
      throw std::runtime_error(std::string("ds_opencl_fused_run failed: ") + err);
    }
    t1 = steady_seconds();
    result.profile_ms[kProfileOpenclRunOuter] = (t1 - t0) * 1000.0;

    t0 = steady_seconds();
    ret = face_rknn_parallel_infer_merged(
        rknn_.get(),
        rknn_inputs.data(),
        meta_.num_slices,
        input_h_,
        input_w_,
        input_c_,
        slice_shapes_.data(),
        gains_.data(),
        pads_.data(),
        starts_.data(),
        wraps_.data(),
        meta_.num_slices,
        original_width,
        cfg_.overlap_ratio,
        cfg_.merge_iou_threshold,
        cfg_.nms_iou_threshold,
        cfg_.conf_threshold,
        cfg_.decode_iou_threshold,
        cfg_.max_det,
        cfg_.max_nms,
        result.detections.data(),
        cfg_.max_output_dets,
        &result.detection_count,
        result.merge_stats,
        result.rknn_timings,
        err,
        sizeof(err));
    if (ret != 0) {
      throw std::runtime_error(std::string("face_rknn_parallel_infer_merged failed: ") + err);
    }
    t1 = steady_seconds();
    result.profile_ms[kProfileRknnTotalOuter] = (t1 - t0) * 1000.0;

    result.detection_count = std::max(0, std::min(result.detection_count, cfg_.max_output_dets));
    return result;
  }

 private:
  void disable_bound_once(const std::string& reason) {
    bound_disabled_ = true;
    bound_imported_ = false;
    if (!bound_warned_) {
      std::cerr << "[board_cpp] bound input disabled, fallback to CPU input path: "
                << reason << "\n";
      bound_warned_ = true;
    }
  }

  void disable_zero_copy_once(const std::string& reason) {
    bound_import_failed_ = true;
    bound_imported_ = false;
    if (!bound_import_warned_) {
      std::cerr << "[board_cpp] RKNN/OpenCL zero-copy disabled, fallback to bound-copy path: "
                << reason << "\n";
      bound_import_warned_ = true;
    }
  }

  bool prepare_bound_inputs(FrameResult& result) {
    if (!cfg_.bound_input || bound_disabled_) {
      return false;
    }

    char err[4096] = {0};
    int ret = 0;
    if (!bound_prepared_) {
      const double t0 = steady_seconds();
      ret = face_rknn_parallel_prepare_bound_inputs(rknn_.get(), err, sizeof(err));
      const double t1 = steady_seconds();
      result.profile_ms[kProfileBoundPrepare] = (t1 - t0) * 1000.0;
      if (ret != 0) {
        disable_bound_once(std::string("face_rknn_parallel_prepare_bound_inputs failed: ") + err);
        return false;
      }
      bound_prepared_ = true;
    }

    if (!bound_ptrs_ready_) {
      std::fill(bound_ptrs_, bound_ptrs_ + 3, nullptr);
      std::fill(bound_ptr_sizes_, bound_ptr_sizes_ + 3, 0);
      err[0] = '\0';
      ret = face_rknn_parallel_get_bound_input_ptrs(
          rknn_.get(), bound_ptrs_, bound_ptr_sizes_, 3, err, sizeof(err));
      if (ret != meta_.num_slices) {
        std::ostringstream oss;
        oss << "face_rknn_parallel_get_bound_input_ptrs returned " << ret
            << ", expected " << meta_.num_slices;
        if (err[0] != '\0') {
          oss << ": " << err;
        }
        throw std::runtime_error(oss.str());
      }
      const uint64_t slice_bytes =
          static_cast<uint64_t>(meta_.imgsz) * static_cast<uint64_t>(meta_.imgsz) * 3ULL;
      for (int i = 0; i < meta_.num_slices; ++i) {
        if (bound_ptrs_[i] == nullptr || bound_ptr_sizes_[i] < slice_bytes) {
          std::ostringstream oss;
          oss << "invalid bound input ptr/size for slice " << i
              << ": ptr=" << static_cast<const void*>(bound_ptrs_[i])
              << " size=" << bound_ptr_sizes_[i];
          throw std::runtime_error(oss.str());
        }
      }
      bound_ptrs_ready_ = true;
    }

    if (bound_imported_ || bound_import_failed_) {
      return true;
    }

    if (cfg_.staging_copy_input) {
      bound_imported_ = false;
      bound_import_failed_ = true;
      return true;
    }

    std::fill(bound_fds_, bound_fds_ + 3, -1);
    std::fill(bound_sizes_, bound_sizes_ + 3, 0);
    const double t0 = steady_seconds();
    ret = face_rknn_parallel_get_bound_input_fds(
        rknn_.get(), bound_fds_, bound_sizes_, 3, err, sizeof(err));
    if (ret != meta_.num_slices) {
      const double t1 = steady_seconds();
      result.profile_ms[kProfileBoundImport] = (t1 - t0) * 1000.0;
      std::ostringstream oss;
      oss << "face_rknn_parallel_get_bound_input_fds returned " << ret
          << ", expected " << meta_.num_slices;
      if (err[0] != '\0') {
        oss << ": " << err;
      }
      disable_zero_copy_once(oss.str());
      return true;
    }
    for (int i = 0; i < meta_.num_slices; ++i) {
      if (bound_fds_[i] < 0 || bound_sizes_[i] == 0) {
        const double t1 = steady_seconds();
        result.profile_ms[kProfileBoundImport] = (t1 - t0) * 1000.0;
        std::ostringstream oss;
        oss << "invalid bound input fd/size for slice " << i
            << ": fd=" << bound_fds_[i] << " size=" << bound_sizes_[i];
        disable_zero_copy_once(oss.str());
        return true;
      }
    }

    err[0] = '\0';
    ret = ds_opencl_fused_import_output_fds(
        opencl_.get(), bound_fds_, bound_sizes_, meta_.num_slices, err, sizeof(err));
    const double t1 = steady_seconds();
    result.profile_ms[kProfileBoundImport] = (t1 - t0) * 1000.0;
    if (ret != 0) {
      disable_zero_copy_once(std::string("ds_opencl_fused_import_output_fds failed: ") + err);
      return true;
    }

    bound_imported_ = true;
    return true;
  }

  void ensure_staging_buffers() {
    const size_t slice_bytes =
        static_cast<size_t>(meta_.imgsz) * static_cast<size_t>(meta_.imgsz) * 3U;
    if (staging_buffers_.size() == static_cast<size_t>(meta_.num_slices) &&
        staging_ptrs_.size() == static_cast<size_t>(meta_.num_slices)) {
      bool ready = true;
      for (int i = 0; i < meta_.num_slices; ++i) {
        ready = ready &&
                staging_buffers_[static_cast<size_t>(i)].size() == slice_bytes &&
                staging_ptrs_[static_cast<size_t>(i)] ==
                    staging_buffers_[static_cast<size_t>(i)].data();
      }
      if (ready) {
        return;
      }
    }

    staging_buffers_.assign(static_cast<size_t>(meta_.num_slices), std::vector<uint8_t>());
    staging_ptrs_.assign(static_cast<size_t>(meta_.num_slices), nullptr);
    for (int i = 0; i < meta_.num_slices; ++i) {
      staging_buffers_[static_cast<size_t>(i)].assign(slice_bytes, 114);
      staging_ptrs_[static_cast<size_t>(i)] = staging_buffers_[static_cast<size_t>(i)].data();
    }
  }

  void copy_staging_to_bound_inputs(FrameResult& result) {
    const size_t slice_bytes =
        static_cast<size_t>(meta_.imgsz) * static_cast<size_t>(meta_.imgsz) * 3U;
    const double t0 = steady_seconds();
    for (int i = 0; i < meta_.num_slices; ++i) {
      if (bound_ptrs_[i] == nullptr || bound_ptr_sizes_[i] < slice_bytes ||
          staging_ptrs_[static_cast<size_t>(i)] == nullptr) {
        std::ostringstream oss;
        oss << "invalid staging/bound input ptr for slice " << i;
        throw std::runtime_error(oss.str());
      }
      std::memcpy(bound_ptrs_[i], staging_ptrs_[static_cast<size_t>(i)], slice_bytes);
    }
    const double t1 = steady_seconds();
    result.profile_ms[kProfileStagingCopy] = (t1 - t0) * 1000.0;
  }

  bool run_bound_inference(float original_width, bool external_device_input, FrameResult& result) {
    char err[4096] = {0};
    double t0 = steady_seconds();
    int ret = 0;
    if (external_device_input) {
      ret = face_rknn_parallel_infer_merged_bound_external(
          rknn_.get(),
          meta_.num_slices,
          input_h_,
          input_w_,
          input_c_,
          slice_shapes_.data(),
          gains_.data(),
          pads_.data(),
          starts_.data(),
          wraps_.data(),
          meta_.num_slices,
          original_width,
          cfg_.overlap_ratio,
          cfg_.merge_iou_threshold,
          cfg_.nms_iou_threshold,
          cfg_.conf_threshold,
          cfg_.decode_iou_threshold,
          cfg_.max_det,
          cfg_.max_nms,
          1,
          result.detections.data(),
          cfg_.max_output_dets,
          &result.detection_count,
          result.merge_stats,
          result.rknn_timings,
          err,
          sizeof(err));
    } else {
      ret = face_rknn_parallel_infer_merged_bound(
          rknn_.get(),
          meta_.num_slices,
          input_h_,
          input_w_,
          input_c_,
          slice_shapes_.data(),
          gains_.data(),
          pads_.data(),
          starts_.data(),
          wraps_.data(),
          meta_.num_slices,
          original_width,
          cfg_.overlap_ratio,
          cfg_.merge_iou_threshold,
          cfg_.nms_iou_threshold,
          cfg_.conf_threshold,
          cfg_.decode_iou_threshold,
          cfg_.max_det,
          cfg_.max_nms,
          result.detections.data(),
          cfg_.max_output_dets,
          &result.detection_count,
          result.merge_stats,
          result.rknn_timings,
          err,
          sizeof(err));
    }
    const double t1 = steady_seconds();
    result.profile_ms[kProfileRknnTotalOuter] = (t1 - t0) * 1000.0;
    if (ret != 0) {
      throw std::runtime_error(
          std::string(external_device_input
                          ? "face_rknn_parallel_infer_merged_bound_external failed: "
                          : "face_rknn_parallel_infer_merged_bound failed: ") +
          err);
    }
    return true;
  }

  bool try_process_bound(const Image& image, float original_width, FrameResult& result) {
    if (!prepare_bound_inputs(result)) {
      return false;
    }

    char err[4096] = {0};
    double t0 = steady_seconds();
    int ret = 0;
    if (cfg_.staging_copy_input) {
      ensure_staging_buffers();
      ret = ds_opencl_fused_run_split(
          opencl_.get(),
          image.bgr.data(),
          staging_ptrs_.data(),
          meta_.num_slices,
          result.remap_timings,
          err,
          sizeof(err));
      const double t1 = steady_seconds();
      result.profile_ms[kProfileOpenclRunOuter] = (t1 - t0) * 1000.0;
      if (ret != 0) {
        throw std::runtime_error(
            std::string("ds_opencl_fused_run_split(staging) failed: ") + err);
      }
      copy_staging_to_bound_inputs(result);
      return run_bound_inference(original_width, false, result);
    }

    if (bound_imported_) {
      ret = ds_opencl_fused_run_imported(
          opencl_.get(),
          image.bgr.data(),
          result.remap_timings,
          err,
          sizeof(err));
      if (ret != 0) {
        disable_zero_copy_once(std::string("ds_opencl_fused_run_imported failed: ") + err);
      }
    }
    if (!bound_imported_) {
      ret = ds_opencl_fused_run_split(
          opencl_.get(),
          image.bgr.data(),
          bound_ptrs_,
          meta_.num_slices,
          result.remap_timings,
          err,
          sizeof(err));
    } else {
      ret = 0;
    }
    double t1 = steady_seconds();
    result.profile_ms[kProfileOpenclRunOuter] = (t1 - t0) * 1000.0;
    if (ret != 0) {
      throw std::runtime_error(
          std::string("ds_opencl_fused_run_split(bound) failed: ") + err);
    }

    if (bound_imported_) {
      try {
        return run_bound_inference(original_width, true, result);
      } catch (const std::exception& exc) {
        disable_zero_copy_once(exc.what());
        t0 = steady_seconds();
        err[0] = '\0';
        ret = ds_opencl_fused_run_split(
            opencl_.get(),
            image.bgr.data(),
            bound_ptrs_,
            meta_.num_slices,
            result.remap_timings,
            err,
            sizeof(err));
        t1 = steady_seconds();
        result.profile_ms[kProfileOpenclRunOuter] += (t1 - t0) * 1000.0;
        if (ret != 0) {
          throw std::runtime_error(
              std::string("ds_opencl_fused_run_split(bound) failed after zero-copy fallback: ") +
              err);
        }
      }
    }
    return run_bound_inference(original_width, false, result);
  }

  void ensure_opencl(const Image& image) {
    if (opencl_.get() != nullptr && opencl_width_ == image.width && opencl_height_ == image.height) {
      return;
    }
    opencl_.reset();
    bound_imported_ = false;
    bound_import_failed_ = false;

    char err[4096] = {0};
    const int ret = ds_opencl_fused_create(
        map_x_.data(),
        map_y_.data(),
        roi_rects_.data(),
        image.width,
        image.height,
        image.width * 3,
        meta_.num_slices,
        meta_.roi_h,
        meta_.roi_w,
        meta_.imgsz,
        opencl_.out(),
        err,
        sizeof(err));
    if (ret != 0) {
      throw std::runtime_error(std::string("ds_opencl_fused_create failed: ") + err);
    }
    opencl_width_ = image.width;
    opencl_height_ = image.height;
  }

  Config cfg_;
  MapMeta meta_;
  std::vector<float> map_x_;
  std::vector<float> map_y_;
  std::vector<float> base_map_x_;
  std::vector<float> base_map_y_;
  std::vector<int> roi_rects_;
  std::vector<int> slice_shapes_;
  std::vector<float> gains_;
  std::vector<int> pads_;
  std::vector<float> starts_;
  std::vector<int> wraps_;
  RknnHandle rknn_;
  OpenCLHandle opencl_;
  int opencl_width_ = 0;
  int opencl_height_ = 0;
  int input_h_ = 0;
  int input_w_ = 0;
  int input_c_ = 0;
  int channels_ = 0;
  int anchors_ = 0;
  bool bound_disabled_ = false;
  bool bound_prepared_ = false;
  bool bound_imported_ = false;
  bool bound_import_failed_ = false;
  bool bound_import_warned_ = false;
  bool bound_ptrs_ready_ = false;
  bool bound_warned_ = false;
  int bound_fds_[3] = {-1, -1, -1};
  uint64_t bound_sizes_[3] = {0, 0, 0};
  uint8_t* bound_ptrs_[3] = {nullptr, nullptr, nullptr};
  uint64_t bound_ptr_sizes_[3] = {0, 0, 0};
  std::vector<std::vector<uint8_t>> staging_buffers_;
  std::vector<uint8_t*> staging_ptrs_;
};

static uint32_t sha1_rotl(uint32_t value, int bits) {
  return (value << bits) | (value >> (32 - bits));
}

static std::array<uint8_t, 20> sha1_digest(const std::string& input) {
  std::vector<uint8_t> data(input.begin(), input.end());
  const uint64_t bit_len = static_cast<uint64_t>(data.size()) * 8ULL;
  data.push_back(0x80);
  while ((data.size() % 64) != 56) {
    data.push_back(0x00);
  }
  for (int i = 7; i >= 0; --i) {
    data.push_back(static_cast<uint8_t>((bit_len >> (i * 8)) & 0xff));
  }

  uint32_t h0 = 0x67452301U;
  uint32_t h1 = 0xefcdab89U;
  uint32_t h2 = 0x98badcfeU;
  uint32_t h3 = 0x10325476U;
  uint32_t h4 = 0xc3d2e1f0U;

  for (size_t offset = 0; offset < data.size(); offset += 64) {
    uint32_t w[80] = {0};
    for (int i = 0; i < 16; ++i) {
      const size_t p = offset + static_cast<size_t>(i) * 4;
      w[i] = (static_cast<uint32_t>(data[p]) << 24) |
             (static_cast<uint32_t>(data[p + 1]) << 16) |
             (static_cast<uint32_t>(data[p + 2]) << 8) |
             static_cast<uint32_t>(data[p + 3]);
    }
    for (int i = 16; i < 80; ++i) {
      w[i] = sha1_rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }

    uint32_t a = h0;
    uint32_t b = h1;
    uint32_t c = h2;
    uint32_t d = h3;
    uint32_t e = h4;
    for (int i = 0; i < 80; ++i) {
      uint32_t f = 0;
      uint32_t k = 0;
      if (i < 20) {
        f = (b & c) | ((~b) & d);
        k = 0x5a827999U;
      } else if (i < 40) {
        f = b ^ c ^ d;
        k = 0x6ed9eba1U;
      } else if (i < 60) {
        f = (b & c) | (b & d) | (c & d);
        k = 0x8f1bbcdcU;
      } else {
        f = b ^ c ^ d;
        k = 0xca62c1d6U;
      }
      const uint32_t temp = sha1_rotl(a, 5) + f + e + k + w[i];
      e = d;
      d = c;
      c = sha1_rotl(b, 30);
      b = a;
      a = temp;
    }
    h0 += a;
    h1 += b;
    h2 += c;
    h3 += d;
    h4 += e;
  }

  std::array<uint8_t, 20> out{{0}};
  const uint32_t words[5] = {h0, h1, h2, h3, h4};
  for (int i = 0; i < 5; ++i) {
    out[static_cast<size_t>(i) * 4 + 0] = static_cast<uint8_t>((words[i] >> 24) & 0xff);
    out[static_cast<size_t>(i) * 4 + 1] = static_cast<uint8_t>((words[i] >> 16) & 0xff);
    out[static_cast<size_t>(i) * 4 + 2] = static_cast<uint8_t>((words[i] >> 8) & 0xff);
    out[static_cast<size_t>(i) * 4 + 3] = static_cast<uint8_t>(words[i] & 0xff);
  }
  return out;
}

static std::string base64_encode(const uint8_t* data, size_t size) {
  static const char* table =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve(((size + 2) / 3) * 4);
  for (size_t i = 0; i < size; i += 3) {
    const uint32_t b0 = data[i];
    const uint32_t b1 = i + 1 < size ? data[i + 1] : 0;
    const uint32_t b2 = i + 2 < size ? data[i + 2] : 0;
    const uint32_t triple = (b0 << 16) | (b1 << 8) | b2;
    out.push_back(table[(triple >> 18) & 0x3f]);
    out.push_back(table[(triple >> 12) & 0x3f]);
    out.push_back(i + 1 < size ? table[(triple >> 6) & 0x3f] : '=');
    out.push_back(i + 2 < size ? table[triple & 0x3f] : '=');
  }
  return out;
}

static std::string websocket_accept_key(const std::string& client_key) {
  const std::string guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
  const std::array<uint8_t, 20> digest = sha1_digest(client_key + guid);
  return base64_encode(digest.data(), digest.size());
}

static std::string lower_ascii(std::string value) {
  for (char& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

static std::string get_http_header(const std::string& request, const std::string& name) {
  const std::string wanted = lower_ascii(name);
  size_t pos = 0;
  while (pos < request.size()) {
    const size_t end = request.find("\r\n", pos);
    const std::string line =
        request.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
    const size_t colon = line.find(':');
    if (colon != std::string::npos) {
      std::string key = lower_ascii(trim(line.substr(0, colon)));
      if (key == wanted) {
        return trim(line.substr(colon + 1));
      }
    }
    if (end == std::string::npos) {
      break;
    }
    pos = end + 2;
  }
  return "";
}

static bool send_all(int fd, const uint8_t* data, size_t size) {
  size_t sent = 0;
  while (sent < size) {
    const ssize_t n = ::send(fd, data + sent, size - sent, MSG_NOSIGNAL);
    if (n <= 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    sent += static_cast<size_t>(n);
  }
  return true;
}

static bool send_all(int fd, const std::string& data) {
  return send_all(fd, reinterpret_cast<const uint8_t*>(data.data()), data.size());
}

static bool set_fd_nonblocking(int fd) {
  const int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) {
    return false;
  }
  return fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

static void set_socket_send_buffer(int fd, int bytes) {
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bytes, sizeof(bytes));
}

static void set_socket_recv_timeout(int fd, int timeout_ms) {
  timeval tv {};
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = static_cast<suseconds_t>((timeout_ms % 1000) * 1000);
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}

static bool send_all_nonblocking(int fd,
                                 const uint8_t* data,
                                 size_t size,
                                 int timeout_ms = 50) {
  size_t sent = 0;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (sent < size) {
    const ssize_t n = ::send(fd, data + sent, size - sent,
                             MSG_NOSIGNAL | MSG_DONTWAIT);
    if (n > 0) {
      sent += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        return false;
      }
      const auto remaining_us =
          std::chrono::duration_cast<std::chrono::microseconds>(deadline - now).count();
      timeval tv {};
      tv.tv_sec = 0;
      tv.tv_usec = static_cast<suseconds_t>(
          std::max<int64_t>(1000, std::min<int64_t>(remaining_us, 5000)));
      fd_set wfds;
      FD_ZERO(&wfds);
      FD_SET(fd, &wfds);
      const int ret = select(fd + 1, nullptr, &wfds, nullptr, &tv);
      if (ret > 0) {
        continue;
      }
      if (ret < 0 && errno == EINTR) {
        continue;
      }
      continue;
    }
    return false;
  }
  return true;
}

static bool send_all_nonblocking(int fd, const std::vector<uint8_t>& data) {
  return send_all_nonblocking(fd, data.data(), data.size());
}

static std::vector<uint8_t> make_websocket_frame(const uint8_t* data,
                                                 size_t size,
                                                 uint8_t opcode) {
  std::vector<uint8_t> frame;
  frame.reserve(size + 10);
  frame.push_back(static_cast<uint8_t>(0x80 | (opcode & 0x0f)));
  const uint64_t len = static_cast<uint64_t>(size);
  if (len <= 125) {
    frame.push_back(static_cast<uint8_t>(len));
  } else if (len <= 0xffff) {
    frame.push_back(126);
    frame.push_back(static_cast<uint8_t>((len >> 8) & 0xff));
    frame.push_back(static_cast<uint8_t>(len & 0xff));
  } else {
    frame.push_back(127);
    for (int i = 7; i >= 0; --i) {
      frame.push_back(static_cast<uint8_t>((len >> (i * 8)) & 0xff));
    }
  }
  frame.insert(frame.end(), data, data + size);
  return frame;
}

static int create_tcp_listener(const std::string& host, int port, int backlog) {
  struct addrinfo hints {};
  hints.ai_family = (host.empty() || host == "0.0.0.0") ? AF_INET : AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;
  struct addrinfo* result = nullptr;
  const std::string port_text = std::to_string(port);
  const char* node = (host.empty() || host == "0.0.0.0") ? nullptr : host.c_str();
  const int gai = getaddrinfo(node, port_text.c_str(), &hints, &result);
  if (gai != 0) {
    throw std::runtime_error(std::string("getaddrinfo failed for bind: ") + gai_strerror(gai));
  }

  int fd = -1;
  for (struct addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
    fd = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd < 0) {
      continue;
    }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (::bind(fd, rp->ai_addr, rp->ai_addrlen) == 0 && ::listen(fd, backlog) == 0) {
      break;
    }
    ::close(fd);
    fd = -1;
  }
  freeaddrinfo(result);
  if (fd < 0) {
    throw std::runtime_error("cannot bind socket on " + host + ":" + std::to_string(port));
  }
  return fd;
}

static std::vector<std::string> local_ipv4_addresses() {
  std::vector<std::string> addrs;
  struct ifaddrs* ifaddr = nullptr;
  if (getifaddrs(&ifaddr) != 0) {
    return addrs;
  }
  for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr || ifa->ifa_addr->sa_family != AF_INET) {
      continue;
    }
    if ((ifa->ifa_flags & IFF_LOOPBACK) != 0) {
      continue;
    }
    char host[INET_ADDRSTRLEN] = {0};
    const sockaddr_in* addr = reinterpret_cast<const sockaddr_in*>(ifa->ifa_addr);
    if (inet_ntop(AF_INET, &addr->sin_addr, host, sizeof(host)) != nullptr) {
      addrs.push_back(host);
    }
  }
  freeifaddrs(ifaddr);
  std::sort(addrs.begin(), addrs.end());
  addrs.erase(std::unique(addrs.begin(), addrs.end()), addrs.end());
  return addrs;
}

class BoardWebUiServer {
 public:
  BoardWebUiServer(const std::string& host, int port)
      : host_(host), port_(port) {}

  ~BoardWebUiServer() { stop(); }

  void start() {
    if (running_.load()) {
      return;
    }
    listen_fd_ = create_tcp_listener(host_, port_, 64);
    running_.store(true);
    sender_thread_ = std::thread(&BoardWebUiServer::sender_loop, this);
    accept_thread_ = std::thread(&BoardWebUiServer::accept_loop, this);
    std::cerr << "[board_cpp-webui] listening: http://" << host_ << ":" << port_ << "/\n";
    if (host_.empty() || host_ == "0.0.0.0") {
      const std::vector<std::string> addrs = local_ipv4_addresses();
      if (addrs.empty()) {
        std::cerr << "[board_cpp-webui] open: http://<board-ip>:" << port_ << "/\n";
      } else {
        for (const std::string& addr : addrs) {
          std::cerr << "[board_cpp-webui] open: http://" << addr << ":" << port_ << "/\n";
        }
      }
    } else {
      std::cerr << "[board_cpp-webui] open: http://" << host_ << ":" << port_ << "/\n";
    }
  }

  void stop() {
    if (!running_.load() && listen_fd_ < 0) {
      return;
    }
    running_.store(false);
    if (listen_fd_ >= 0) {
      ::shutdown(listen_fd_, SHUT_RDWR);
      ::close(listen_fd_);
      listen_fd_ = -1;
    }
    sender_cv_.notify_all();
    if (accept_thread_.joinable()) {
      accept_thread_.join();
    }
    if (sender_thread_.joinable()) {
      sender_thread_.join();
    }
    std::vector<int> json_clients;
    std::vector<int> system_clients;
    std::vector<int> frame_clients;
    std::vector<int> mjpeg_clients;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      json_clients.swap(json_clients_);
      system_clients.swap(system_clients_);
      frame_clients.swap(frame_clients_);
      mjpeg_clients.swap(mjpeg_clients_);
    }
    close_all(json_clients);
    close_all(system_clients);
    close_all(frame_clients);
    close_all(mjpeg_clients);
  }

  bool has_frame_clients() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !frame_clients_.empty() || !mjpeg_clients_.empty();
  }

  void publish_json(const std::string& payload) {
    if (!running_.load()) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_json_ = payload;
      json_dirty_ = true;
    }
    sender_cv_.notify_one();
  }

  void publish_system(const std::string& payload) {
    if (!running_.load() || payload.empty()) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_system_ = payload;
      system_dirty_ = true;
    }
    sender_cv_.notify_one();
  }

  void publish_jpeg(const std::vector<uint8_t>& jpeg) {
    if (!running_.load() || jpeg.empty()) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_jpeg_ = jpeg;
      jpeg_dirty_ = true;
    }
    sender_cv_.notify_one();
  }

 private:
  static void close_all(const std::vector<int>& fds) {
    for (int fd : fds) {
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
    }
  }

  void remove_failed(std::vector<int>& owned,
                     const std::vector<int>& candidates,
                     const std::vector<uint8_t>& payload) {
    std::vector<int> failed;
    for (int fd : candidates) {
      if (!send_all_nonblocking(fd, payload)) {
        failed.push_back(fd);
      }
    }
    if (failed.empty()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int> kept;
    kept.reserve(owned.size());
    for (int fd : owned) {
      if (std::find(failed.begin(), failed.end(), fd) == failed.end()) {
        kept.push_back(fd);
      } else {
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
      }
    }
    owned.swap(kept);
  }

  void sender_loop() {
    while (true) {
      std::string json_payload;
      std::string system_payload;
      std::vector<uint8_t> jpeg_payload;
      std::vector<int> json_clients;
      std::vector<int> system_clients;
      std::vector<int> frame_clients;
      std::vector<int> mjpeg_clients;
      bool send_json = false;
      bool send_system = false;
      bool send_jpeg = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        sender_cv_.wait(lock, [&] {
          return !running_.load() || json_dirty_ || system_dirty_ || jpeg_dirty_;
        });
        if (!running_.load() && !json_dirty_ && !system_dirty_ && !jpeg_dirty_) {
          break;
        }
        if (json_dirty_) {
          json_payload = latest_json_;
          json_clients = json_clients_;
          json_dirty_ = false;
          send_json = !json_payload.empty() && !json_clients.empty();
        }
        if (system_dirty_) {
          system_payload = latest_system_;
          system_clients = system_clients_;
          system_dirty_ = false;
          send_system = !system_payload.empty() && !system_clients.empty();
        }
        if (jpeg_dirty_) {
          jpeg_payload = latest_jpeg_;
          frame_clients = frame_clients_;
          mjpeg_clients = mjpeg_clients_;
          jpeg_dirty_ = false;
          send_jpeg = !jpeg_payload.empty() &&
                      (!frame_clients.empty() || !mjpeg_clients.empty());
        }
      }

      if (send_json) {
        std::vector<uint8_t> frame = make_websocket_frame(
            reinterpret_cast<const uint8_t*>(json_payload.data()), json_payload.size(), 0x2);
        remove_failed(json_clients_, json_clients, frame);
      }
      if (send_system) {
        std::vector<uint8_t> frame = make_websocket_frame(
            reinterpret_cast<const uint8_t*>(system_payload.data()), system_payload.size(), 0x2);
        remove_failed(system_clients_, system_clients, frame);
      }
      if (send_jpeg) {
        if (!frame_clients.empty()) {
          std::vector<uint8_t> frame = make_websocket_frame(
              jpeg_payload.data(), jpeg_payload.size(), 0x2);
          remove_failed(frame_clients_, frame_clients, frame);
        }
        if (!mjpeg_clients.empty()) {
          std::ostringstream header;
          header << "--frame\r\n"
                 << "Content-Type: image/jpeg\r\n"
                 << "Content-Length: " << jpeg_payload.size() << "\r\n\r\n";
          const std::string header_text = header.str();
          std::vector<uint8_t> chunk(header_text.begin(), header_text.end());
          chunk.insert(chunk.end(), jpeg_payload.begin(), jpeg_payload.end());
          chunk.push_back('\r');
          chunk.push_back('\n');
          remove_failed(mjpeg_clients_, mjpeg_clients, chunk);
        }
      }
    }
  }

  void accept_loop() {
    while (running_.load()) {
      sockaddr_storage addr {};
      socklen_t addr_len = sizeof(addr);
      const int fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &addr_len);
      if (fd < 0) {
        if (running_.load() && errno != EINTR) {
          std::cerr << "[board_cpp-webui] accept failed: " << std::strerror(errno) << "\n";
        }
        continue;
      }
      set_socket_recv_timeout(fd, 1000);
      handle_client(fd);
    }
  }

  void handle_client(int fd) {
    std::string request;
    char buffer[1024];
    while (request.find("\r\n\r\n") == std::string::npos && request.size() < 8192) {
      const ssize_t n = ::recv(fd, buffer, sizeof(buffer), 0);
      if (n <= 0) {
        ::close(fd);
        return;
      }
      request.append(buffer, buffer + n);
    }

    const size_t first_line_end = request.find("\r\n");
    const std::string first_line =
        first_line_end == std::string::npos ? request : request.substr(0, first_line_end);
    std::istringstream iss(first_line);
    std::string method;
    std::string path;
    std::string version;
    iss >> method >> path >> version;
    if (method != "GET") {
      send_simple_response(fd, "405 Method Not Allowed", "text/plain", "method not allowed");
      ::close(fd);
      return;
    }
    const size_t query_pos = path.find('?');
    if (query_pos != std::string::npos) {
      path = path.substr(0, query_pos);
    }
    const std::string upgrade = get_http_header(request, "Upgrade");
    if (!upgrade.empty()) {
      if (path == "/ws/inference") {
        std::string latest;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          latest = latest_json_;
        }
        accept_websocket(fd, request, json_clients_, latest);
        return;
      }
      if (path == "/ws/system") {
        std::string latest;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          latest = latest_system_;
        }
        accept_websocket(fd, request, system_clients_, latest);
        return;
      }
      if (path == "/ws/frame") {
        accept_websocket(fd, request, frame_clients_, std::string());
        return;
      }
      send_simple_response(fd, "404 Not Found", "text/plain", "not found");
      ::close(fd);
      return;
    }
    if (path == "/" || path == "/index.html") {
      send_simple_response(fd, "200 OK", "text/html; charset=utf-8", index_html());
      ::close(fd);
      return;
    }
    if (path == "/inference/latest") {
      std::string payload;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        payload = latest_json_;
      }
      if (payload.empty()) {
        send_simple_response(fd, "503 Service Unavailable", "application/json",
                             "{\"error\":\"no inference result yet\"}");
      } else {
        send_simple_response(fd, "200 OK", "application/json", payload);
      }
      ::close(fd);
      return;
    }
    if (path == "/system/latest") {
      std::string payload;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        payload = latest_system_;
      }
      if (payload.empty()) {
        send_simple_response(fd, "503 Service Unavailable", "application/json",
                             "{\"error\":\"no system load sample yet\"}");
      } else {
        send_simple_response(fd, "200 OK", "application/json", payload);
      }
      ::close(fd);
      return;
    }
    if (path == "/video/infer") {
      const std::string header =
          "HTTP/1.1 200 OK\r\n"
          "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
          "Cache-Control: no-store\r\n"
          "Connection: close\r\n\r\n";
      if (!send_all(fd, header)) {
        ::close(fd);
        return;
      }
      if (!set_fd_nonblocking(fd)) {
        ::close(fd);
        return;
      }
      set_socket_send_buffer(fd, 1024 * 1024);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        mjpeg_clients_.push_back(fd);
      }
      return;
    }
    send_simple_response(fd, "404 Not Found", "text/plain", "not found");
    ::close(fd);
  }

  bool accept_websocket(int fd,
                        const std::string& request,
                        std::vector<int>& client_list,
                        const std::string& latest_payload) {
    const std::string key = get_http_header(request, "Sec-WebSocket-Key");
    if (key.empty()) {
      send_simple_response(fd, "400 Bad Request", "text/plain", "missing websocket key");
      ::close(fd);
      return false;
    }
    std::ostringstream response;
    response << "HTTP/1.1 101 Switching Protocols\r\n"
             << "Upgrade: websocket\r\n"
             << "Connection: Upgrade\r\n"
             << "Sec-WebSocket-Accept: " << websocket_accept_key(key) << "\r\n"
             << "\r\n";
    if (!send_all(fd, response.str())) {
      ::close(fd);
      return false;
    }
    if (!set_fd_nonblocking(fd)) {
      ::close(fd);
      return false;
    }
    set_socket_send_buffer(fd, 1024 * 1024);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      client_list.push_back(fd);
    }
    if (!latest_payload.empty()) {
      std::vector<uint8_t> frame = make_websocket_frame(
          reinterpret_cast<const uint8_t*>(latest_payload.data()), latest_payload.size(), 0x2);
      if (!send_all_nonblocking(fd, frame)) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          client_list.erase(std::remove(client_list.begin(), client_list.end(), fd),
                            client_list.end());
        }
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
      }
    }
    return true;
  }

  static void send_simple_response(int fd,
                                   const std::string& status,
                                   const std::string& content_type,
                                   const std::string& body) {
    std::ostringstream os;
    os << "HTTP/1.1 " << status << "\r\n"
       << "Content-Type: " << content_type << "\r\n"
       << "Content-Length: " << body.size() << "\r\n"
       << "Cache-Control: no-store\r\n"
       << "Connection: close\r\n\r\n"
       << body;
    send_all(fd, os.str());
  }

  static std::string index_html() {
    return R"HTML(<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MeetEye C++ WebUI</title>
  <style>
    body{margin:0;background:#101214;color:#e5e7eb;font-family:Arial,"Microsoft YaHei",sans-serif;}
    header{height:48px;display:flex;align-items:center;gap:18px;padding:0 18px;background:#171b20;border-bottom:1px solid #2a313a;}
    h1{font-size:18px;margin:0;font-weight:600;}
    .status{font-size:13px;color:#9ca3af;}
    main{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:8px;padding:8px;}
    .view{background:#050608;border:1px solid #29313a;min-height:60vh;display:flex;align-items:center;justify-content:center;}
    img{width:100%;height:auto;display:block;object-fit:contain;}
    aside{background:#171b20;border:1px solid #29313a;padding:6px;overflow:auto;max-height:calc(100vh - 64px);}
    h2{font-size:12px;margin:6px 0 4px;color:#f3f4f6;}
    .metric-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:3px;margin-bottom:6px;}
    .metric{background:#0f1318;border:1px solid #29313a;padding:4px;min-height:30px;font-size:10px;}
    .metric span{display:block;color:#9ca3af;white-space:nowrap;margin-bottom:2px;}
    .metric b{display:block;font-size:13px;color:#f9fafb;white-space:nowrap;}
    .grid{display:grid;grid-template-columns:repeat(6,1fr);gap:3px;margin-bottom:5px;}
    .card{background:#0f1318;border:1px solid #29313a;padding:4px;min-height:32px;}
    .label{font-size:9px;color:#9ca3af;margin-bottom:2px;white-space:nowrap;}
    .value{font-size:13px;font-weight:700;color:#f9fafb;white-space:nowrap;}
    .bar{height:3px;background:#28313b;margin-top:3px;overflow:hidden;}
    .fill{height:100%;width:0;background:#38bdf8;}
    .warn .fill{background:#f59e0b;}
    .hot .fill{background:#ef4444;}
    .cores{display:grid;grid-template-columns:repeat(6,1fr);gap:3px;margin-top:4px;}
    .core{font-size:9px;color:#d1d5db;background:#0b0e12;border:1px solid #26303a;padding:2px;text-align:center;}
    .sector-grid{display:grid;grid-template-columns:repeat(8,1fr);gap:3px;margin:4px 0 6px;}
    .sector{font-size:9px;line-height:1.25;color:#9ca3af;background:#0b0e12;border:1px solid #26303a;padding:3px;text-align:center;}
    .sector.on{color:#fee2e2;border-color:#ef4444;background:#3a1010;box-shadow:inset 0 0 0 1px #ef4444;}
    pre{white-space:pre-wrap;word-break:break-word;font-size:10px;line-height:1.3;color:#d1d5db;}
    @media(max-width:900px){main{grid-template-columns:1fr;}aside{max-height:none;}}
  </style>
</head>
<body>
  <header>
    <h1>MeetEye C++ WebUI</h1>
    <div class="status" id="status">连接中...</div>
  </header>
  <main>
    <section class="view"><img id="infer" src="/video/infer" alt="等待推理画面"></section>
    <aside>
      <h2>运行状态</h2>
      <div class="metric-grid">
        <div class="metric"><span>帧号</span><b id="frame">-</b></div>
        <div class="metric"><span>目标数</span><b id="count">-</b></div>
        <div class="metric"><span>FPS</span><b id="fps">-</b></div>
        <div class="metric"><span>JSON WS</span><b id="ws">-</b></div>
        <div class="metric"><span>硬件 WS</span><b id="sysws">-</b></div>
      </div>
      <h2>硬件负载</h2>
      <div class="grid">
        <div class="card"><div class="label">CPU 总占用</div><div class="value" id="cpu">-</div><div class="bar"><div id="cpu-bar" class="fill"></div></div></div>
        <div class="card"><div class="label">NPU 占用</div><div class="value" id="npu">-</div><div class="bar"><div id="npu-bar" class="fill"></div></div></div>
        <div class="card"><div class="label">GPU 占用</div><div class="value" id="gpu">-</div><div class="bar"><div id="gpu-bar" class="fill"></div></div></div>
        <div class="card"><div class="label">内存占用</div><div class="value" id="mem">-</div><div class="bar"><div id="mem-bar" class="fill"></div></div></div>
        <div class="card"><div class="label">最高温度</div><div class="value" id="thermal">-</div><div class="bar"><div id="thermal-bar" class="fill"></div></div></div>
        <div class="card"><div class="label">可用内存</div><div class="value" id="memfree">-</div></div>
      </div>
      <div class="cores" id="cores"></div>
      <h2>扇区</h2>
      <div class="sector-grid" id="sectorGrid"></div>
      <h2>推理 JSON</h2>
      <pre id="json">{}</pre>
    </aside>
  </main>
  <script>
    const statusEl=document.getElementById('status');
    const wsEl=document.getElementById('ws');
    const frameEl=document.getElementById('frame');
    const countEl=document.getElementById('count');
    const fpsEl=document.getElementById('fps');
    const sysWsEl=document.getElementById('sysws');
    const jsonEl=document.getElementById('json');
    const coresEl=document.getElementById('cores');
    const sectorGridEl=document.getElementById('sectorGrid');
    function fmtPct(v){return Number.isFinite(v)?`${v.toFixed(1)}%`:'-';}
    function fmtFreq(v){return Number.isFinite(v)&&v>0?`${(v/1000000).toFixed(0)}MHz`:'';}
    function fmtDeg(v){
      if(v===null||v===undefined)return '-';
      const n=Number(v);
      return Number.isFinite(n)?`${n.toFixed(0)}°`:'-';
    }
    function setMetric(id,barId,value,suffix='%',extra=''){
      const el=document.getElementById(id);
      const bar=document.getElementById(barId);
      if(!Number.isFinite(value)){el.textContent='-'; if(bar)bar.style.width='0%'; return;}
      el.textContent=`${value.toFixed(1)}${suffix}${extra}`;
      if(bar)bar.style.width=`${Math.max(0,Math.min(100,value))}%`;
    }
    function connect(){
      const proto=location.protocol==='https:'?'wss':'ws';
      const ws=new WebSocket(`${proto}://${location.host}/ws/inference`);
      ws.binaryType='arraybuffer';
      ws.onopen=()=>{statusEl.textContent='已连接';wsEl.textContent='online';};
      ws.onclose=()=>{statusEl.textContent='已断开，重连中...';wsEl.textContent='offline';setTimeout(connect,1000);};
      ws.onerror=()=>{statusEl.textContent='连接异常';};
      ws.onmessage=(ev)=>{
        const decode=(data)=> data instanceof ArrayBuffer ? new TextDecoder().decode(data) : data;
        const text=decode(ev.data);
        try{
          const obj=JSON.parse(text);
          frameEl.textContent=obj.frame_id ?? '-';
          if(obj.fps && Number.isFinite(obj.fps.average)){fpsEl.textContent=obj.fps.average.toFixed(1);}
          if(obj.targets){
            const entries=Object.entries(obj.targets).sort((a,b)=>Number(a[0])-Number(b[0]));
            countEl.textContent=entries.length;
            sectorGridEl.style.gridTemplateColumns=`repeat(${Math.max(1,Math.min(4,entries.length||1))},1fr)`;
            sectorGridEl.innerHTML=entries.map(([k,t])=>{
              const az=fmtDeg(t&&t.azimuth);
              const el=fmtDeg(t&&t.elevation);
              return `<div class="sector on">ID${k}<br><b>A ${az}</b><br><b>E ${el}</b></div>`;
            }).join('');
          }
          else if(obj.sectors){
            const keys=Object.keys(obj.sectors).sort((a,b)=>Number(a)-Number(b));
            countEl.textContent=Object.values(obj.sectors).filter(s=>s.has_target).length;
            sectorGridEl.style.gridTemplateColumns=`repeat(${Math.max(1,keys.length)},1fr)`;
            sectorGridEl.innerHTML=keys.map(k=>{
              const s=obj.sectors[k]||{};
              const az=fmtDeg(s.azimuth);
              const el=fmtDeg(s.elevation);
              return `<div class="sector ${s.has_target?'on':''}">S${k}<br><b>A ${az}</b><br><b>E ${el}</b></div>`;
            }).join('');
          }
          jsonEl.textContent=JSON.stringify(obj,null,2);
        }catch(e){jsonEl.textContent=text;}
      };
    }
    function connectSystem(){
      const proto=location.protocol==='https:'?'wss':'ws';
      const ws=new WebSocket(`${proto}://${location.host}/ws/system`);
      ws.binaryType='arraybuffer';
      ws.onopen=()=>{sysWsEl.textContent='online';};
      ws.onclose=()=>{sysWsEl.textContent='offline';setTimeout(connectSystem,1000);};
      ws.onerror=()=>{sysWsEl.textContent='error';};
      ws.onmessage=(ev)=>{
        const decode=(data)=> data instanceof ArrayBuffer ? new TextDecoder().decode(data) : data;
        try{
          const obj=JSON.parse(decode(ev.data));
          setMetric('cpu','cpu-bar',obj.cpu_percent);
          setMetric('npu','npu-bar',obj.npu_percent,'%',fmtFreq(obj.npu_freq_hz)?` ${fmtFreq(obj.npu_freq_hz)}`:'');
          setMetric('gpu','gpu-bar',obj.gpu_percent,'%',fmtFreq(obj.gpu_freq_hz)?` ${fmtFreq(obj.gpu_freq_hz)}`:'');
          setMetric('mem','mem-bar',obj.memory_percent);
          setMetric('thermal','thermal-bar',obj.thermal_max_c,'°C');
          document.getElementById('memfree').textContent=Number.isFinite(obj.memory_available_mb)?`${obj.memory_available_mb.toFixed(0)}MB`:'-';
          if(Array.isArray(obj.cpu_per_core_percent)){
            coresEl.innerHTML=obj.cpu_per_core_percent.map((v,i)=>`<div class="core">C${i} <b>${fmtPct(v)}</b></div>`).join('');
          }
        }catch(e){}
      };
    }
    connect();
    connectSystem();
  </script>
</body>
</html>)HTML";
  }

  std::string host_;
  int port_ = 0;
  int listen_fd_ = -1;
  std::atomic<bool> running_{false};
  std::thread accept_thread_;
  std::thread sender_thread_;
  mutable std::mutex mutex_;
  std::condition_variable sender_cv_;
  std::vector<int> json_clients_;
  std::vector<int> system_clients_;
  std::vector<int> frame_clients_;
  std::vector<int> mjpeg_clients_;
  std::string latest_json_;
  std::string latest_system_;
  std::vector<uint8_t> latest_jpeg_;
  bool json_dirty_ = false;
  bool system_dirty_ = false;
  bool jpeg_dirty_ = false;
};

class AsyncWebUiFramePublisher {
 public:
  AsyncWebUiFramePublisher(const Config& cfg,
                           const MeetEyeRuntime& runtime,
                           BoardWebUiServer* webui_server)
      : cfg_(cfg), runtime_(runtime), webui_server_(webui_server) {}

  ~AsyncWebUiFramePublisher() { stop(); }

  void start() {
    if (running_.load()) {
      return;
    }
    running_.store(true);
    worker_ = std::thread(&AsyncWebUiFramePublisher::run_loop, this);
  }

  void stop() {
    if (!running_.load()) {
      return;
    }
    running_.store(false);
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  bool submit(const Image& source,
              const FrameResult& result,
              const std::vector<TargetInfo>& targets,
              const FrameRateStats& fps) {
    if (!running_.load() || webui_server_ == nullptr || !webui_server_->has_frame_clients()) {
      return false;
    }
    std::unique_ptr<Job> job(new Job);
    job->source = source;
    job->result = result;
    job->targets = targets;
    job->fps = fps;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_job_ = std::move(job);
    }
    cv_.notify_one();
    return true;
  }

 private:
  struct Job {
    Image source;
    FrameResult result;
    std::vector<TargetInfo> targets;
    FrameRateStats fps;
  };

  void run_loop() {
    TurboJpegEncoder encoder;
    while (true) {
      std::unique_ptr<Job> job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] {
          return !running_.load() || latest_job_ != nullptr;
        });
        if (!running_.load() && latest_job_ == nullptr) {
          break;
        }
        job = std::move(latest_job_);
      }
      if (!job || webui_server_ == nullptr || !webui_server_->has_frame_clients()) {
        continue;
      }
      Image annotated = remap_panorama_cpu(
          job->source, runtime_.meta(), runtime_.base_map_x(), runtime_.base_map_y());
      draw_annotation(annotated, cfg_, job->result, job->targets, job->fps);
      const std::vector<uint8_t> jpeg =
          encoder.encode_bgr(annotated, cfg_.webui_jpeg_quality);
      webui_server_->publish_jpeg(jpeg);
    }
  }

  const Config& cfg_;
  const MeetEyeRuntime& runtime_;
  BoardWebUiServer* webui_server_ = nullptr;
  std::atomic<bool> running_{false};
  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::unique_ptr<Job> latest_job_;
};

class WebSocketServer {
 public:
  WebSocketServer(const std::string& host, int port, const std::string& path)
      : host_(host), port_(port), path_(path.empty() ? "/ws/inference" : path) {}

  ~WebSocketServer() { stop(); }

  void start() {
    if (running_.load()) {
      return;
    }
    listen_fd_ = create_listen_socket();
    running_.store(true);
    sender_thread_ = std::thread(&WebSocketServer::sender_loop, this);
    accept_thread_ = std::thread(&WebSocketServer::accept_loop, this);
    std::cerr << "[board_cpp-ws] JSON WebSocket listening: ws://" << host_ << ":"
              << port_ << path_ << "\n";
  }

  void stop() {
    if (!running_.load() && listen_fd_ < 0) {
      return;
    }
    running_.store(false);
    if (listen_fd_ >= 0) {
      ::shutdown(listen_fd_, SHUT_RDWR);
      ::close(listen_fd_);
      listen_fd_ = -1;
    }
    sender_cv_.notify_all();
    if (accept_thread_.joinable()) {
      accept_thread_.join();
    }
    if (sender_thread_.joinable()) {
      sender_thread_.join();
    }
    std::vector<int> clients;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      clients.swap(clients_);
    }
    for (int fd : clients) {
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
    }
  }

  void publish(const std::string& payload) {
    if (!running_.load()) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_payload_ = payload;
      payload_dirty_ = true;
    }
    sender_cv_.notify_one();
  }

 private:
  int create_listen_socket() const {
    struct addrinfo hints {};
    hints.ai_family = (host_.empty() || host_ == "0.0.0.0") ? AF_INET : AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    struct addrinfo* result = nullptr;
    const std::string port = std::to_string(port_);
    const char* node = (host_.empty() || host_ == "0.0.0.0") ? nullptr : host_.c_str();
    const int gai = getaddrinfo(node, port.c_str(), &hints, &result);
    if (gai != 0) {
      throw std::runtime_error(std::string("getaddrinfo failed for websocket bind: ") +
                               gai_strerror(gai));
    }

    int fd = -1;
    for (struct addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
      fd = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
      if (fd < 0) {
        continue;
      }
      int one = 1;
      setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
      if (::bind(fd, rp->ai_addr, rp->ai_addrlen) == 0 && ::listen(fd, 16) == 0) {
        break;
      }
      ::close(fd);
      fd = -1;
    }
    freeaddrinfo(result);
    if (fd < 0) {
      throw std::runtime_error("cannot bind websocket server on " + host_ + ":" +
                               std::to_string(port_));
    }
    return fd;
  }

  void accept_loop() {
    while (running_.load()) {
      sockaddr_storage addr {};
      socklen_t addr_len = sizeof(addr);
      const int fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &addr_len);
      if (fd < 0) {
        if (running_.load() && errno != EINTR) {
          std::cerr << "[board_cpp-ws] accept failed: " << std::strerror(errno) << "\n";
        }
        continue;
      }
      set_socket_recv_timeout(fd, 1000);
      if (!handle_handshake(fd)) {
        ::close(fd);
        continue;
      }
      std::string latest;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        clients_.push_back(fd);
        latest = latest_payload_;
      }
      if (!set_fd_nonblocking(fd)) {
        remove_clients(std::vector<int>{fd});
        continue;
      }
      set_socket_send_buffer(fd, 1024 * 1024);
      if (!latest.empty()) {
        std::vector<uint8_t> frame = make_frame(latest);
        if (!send_all_nonblocking(fd, frame)) {
          remove_clients(std::vector<int>{fd});
        }
      }
    }
  }

  void sender_loop() {
    while (true) {
      std::string payload;
      std::vector<int> clients;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        sender_cv_.wait(lock, [&] {
          return !running_.load() || payload_dirty_;
        });
        if (!running_.load() && !payload_dirty_) {
          break;
        }
        payload = latest_payload_;
        clients = clients_;
        payload_dirty_ = false;
      }
      if (payload.empty() || clients.empty()) {
        continue;
      }
      const std::vector<uint8_t> frame = make_frame(payload);
      std::vector<int> failed;
      for (int fd : clients) {
        if (!send_all_nonblocking(fd, frame)) {
          failed.push_back(fd);
        }
      }
      if (!failed.empty()) {
        remove_clients(failed);
      }
    }
  }

  bool handle_handshake(int fd) const {
    std::string request;
    char buffer[1024];
    while (request.find("\r\n\r\n") == std::string::npos && request.size() < 8192) {
      const ssize_t n = ::recv(fd, buffer, sizeof(buffer), 0);
      if (n <= 0) {
        return false;
      }
      request.append(buffer, buffer + n);
    }
    const size_t first_line_end = request.find("\r\n");
    const std::string first_line =
        first_line_end == std::string::npos ? request : request.substr(0, first_line_end);
    const std::string expected = "GET " + path_ + " ";
    if (first_line.find(expected) != 0) {
      const std::string not_found =
          "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
      send_all(fd, not_found);
      return false;
    }
    const std::string key = get_http_header(request, "Sec-WebSocket-Key");
    if (key.empty()) {
      const std::string bad =
          "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
      send_all(fd, bad);
      return false;
    }
    std::ostringstream response;
    response << "HTTP/1.1 101 Switching Protocols\r\n"
             << "Upgrade: websocket\r\n"
             << "Connection: Upgrade\r\n"
             << "Sec-WebSocket-Accept: " << websocket_accept_key(key) << "\r\n"
             << "\r\n";
    return send_all(fd, response.str());
  }

  std::vector<uint8_t> make_frame(const std::string& payload) const {
    std::vector<uint8_t> frame;
    frame.reserve(payload.size() + 10);
    frame.push_back(0x82);  // FIN + binary frame, matching Python send_bytes().
    const uint64_t len = static_cast<uint64_t>(payload.size());
    if (len <= 125) {
      frame.push_back(static_cast<uint8_t>(len));
    } else if (len <= 0xffff) {
      frame.push_back(126);
      frame.push_back(static_cast<uint8_t>((len >> 8) & 0xff));
      frame.push_back(static_cast<uint8_t>(len & 0xff));
    } else {
      frame.push_back(127);
      for (int i = 7; i >= 0; --i) {
        frame.push_back(static_cast<uint8_t>((len >> (i * 8)) & 0xff));
      }
    }
    frame.insert(frame.end(), payload.begin(), payload.end());
    return frame;
  }

  void remove_clients(const std::vector<int>& failed) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int> kept;
    kept.reserve(clients_.size());
    for (int fd : clients_) {
      if (std::find(failed.begin(), failed.end(), fd) == failed.end()) {
        kept.push_back(fd);
      } else {
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
      }
    }
    clients_.swap(kept);
  }

  std::string host_;
  int port_ = 0;
  std::string path_;
  int listen_fd_ = -1;
  std::atomic<bool> running_{false};
  std::thread accept_thread_;
  std::thread sender_thread_;
  std::mutex mutex_;
  std::condition_variable sender_cv_;
  std::vector<int> clients_;
  std::string latest_payload_;
  bool payload_dirty_ = false;
};

static void write_outputs(const Config& cfg,
                          const MeetEyeRuntime& runtime,
                          AngleAndDistanceRuntime& angle_runtime,
                          std::ofstream& jsonl,
                          std::ofstream& debug_jsonl,
                          WebSocketServer* ws_server,
                          BoardWebUiServer* webui_server,
                          AsyncWebUiFramePublisher* webui_frame_publisher,
                          SystemLoadSampler* system_sampler,
                          const Image* source_image,
                          FrameResult& result,
                          const FrameRateStats& fps,
                          int frame_id,
                          bool first_stdout) {
  const double write_start_s = steady_seconds();
  double t0 = steady_seconds();
  const std::vector<TargetInfo> targets = angle_runtime.build_targets(result);
  double t1 = steady_seconds();
  result.profile_ms[kProfileAngleTargets] = (t1 - t0) * 1000.0;

  t0 = steady_seconds();
  const std::string payload = build_output_json(cfg, targets, frame_id, result, fps);
  t1 = steady_seconds();
  result.profile_ms[kProfileBuildPayload] = (t1 - t0) * 1000.0;

  if (!cfg.no_stdout_json) {
    t0 = steady_seconds();
    if (!first_stdout) {
      std::cout << "\n";
    }
    if (cfg.stdout_debug_json) {
      print_result_json(std::cout, runtime.meta(), runtime.channels(), runtime.anchors(), result, fps);
    } else {
      std::cout << payload << "\n";
    }
    t1 = steady_seconds();
    result.profile_ms[kProfileStdout] = (t1 - t0) * 1000.0;
  }
  if (jsonl) {
    t0 = steady_seconds();
    jsonl << payload << "\n";
    t1 = steady_seconds();
    result.profile_ms[kProfileJsonl] = (t1 - t0) * 1000.0;
  }
  if (ws_server != nullptr) {
    t0 = steady_seconds();
    ws_server->publish(payload);
    t1 = steady_seconds();
    result.profile_ms[kProfileWebSocket] = (t1 - t0) * 1000.0;
  }
  if (webui_server != nullptr) {
    t0 = steady_seconds();
    webui_server->publish_json(payload);
    if (system_sampler != nullptr) {
      webui_server->publish_system(system_sampler->latest_json());
    }
    if (webui_frame_publisher != nullptr && source_image != nullptr) {
      webui_frame_publisher->submit(*source_image, result, targets, fps);
    }
    t1 = steady_seconds();
    result.profile_ms[kProfileWebSocket] += (t1 - t0) * 1000.0;
  }
  if (debug_jsonl) {
    t0 = steady_seconds();
    print_result_jsonl(debug_jsonl, runtime.meta(), runtime.channels(), runtime.anchors(), result, fps);
    t1 = steady_seconds();
    result.profile_ms[kProfileDebugJsonl] = (t1 - t0) * 1000.0;
  }
  result.profile_ms[kProfileWriteOutputs] = (steady_seconds() - write_start_s) * 1000.0;
}

static void finish_and_write_frame(const Config& cfg,
                                   const MeetEyeRuntime& runtime,
                                   AngleAndDistanceRuntime& angle_runtime,
                                   NativeHybridSortTracker* tracker,
                                   std::ofstream& jsonl,
                                   std::ofstream& debug_jsonl,
                                   WebSocketServer* ws_server,
                                   BoardWebUiServer* webui_server,
                                   AsyncWebUiFramePublisher* webui_frame_publisher,
                                   SystemLoadSampler* system_sampler,
                                   PreparedStagingFrame& frame,
                                   int& frame_id,
                                   double run_start_s,
                                   ProfileSummary& profile_summary,
                                   bool& first_stdout) {
  FrameResult& result = frame.result;
  result.profile_ms[kProfileFileRead] = frame.file_read_ms;
  result.profile_ms[kProfileCameraRead] = frame.camera_read_ms;
  result.profile_ms[kProfileJpegDecode] = frame.jpeg_decode_ms;
  result.profile_ms[kProfileDecodeWait] = frame.decode_wait_ms;
  result.profile_ms[kProfileDecode] =
      frame.file_read_ms + frame.camera_read_ms + frame.jpeg_decode_ms;

  double t0 = steady_seconds();
  if (tracker != nullptr) {
    tracker->update(result);
    const double t1 = steady_seconds();
    result.profile_ms[kProfileTrackerOuter] = (t1 - t0) * 1000.0;
  } else {
    result.raw_detection_count = result.detection_count;
    result.track_ids.assign(static_cast<size_t>(result.detection_count), -1);
  }

  frame_id += 1;
  const double now_s = steady_seconds();
  FrameRateStats fps;
  fps.frame_ms = (now_s - frame.frame_start_s) * 1000.0;
  fps.instant_fps = fps.frame_ms > 0.0 ? 1000.0 / fps.frame_ms : 0.0;
  fps.elapsed_s = std::max(0.0, now_s - run_start_s);
  fps.frames = frame_id;
  fps.average_fps = fps.elapsed_s > 0.0 ? static_cast<double>(fps.frames) / fps.elapsed_s : 0.0;
  result.profile_ms[kProfileFrameTotal] = fps.frame_ms;

  if (cfg.print_profile_summary) {
    profile_summary.add(result, fps);
    if (cfg.profile_interval > 0 && frame_id % cfg.profile_interval == 0) {
      profile_summary.print(std::cerr, "running");
    }
  }

  write_outputs(
      cfg,
      runtime,
      angle_runtime,
      jsonl,
      debug_jsonl,
      ws_server,
      webui_server,
      webui_frame_publisher,
      system_sampler,
      &frame.image,
      result,
      fps,
      frame_id,
      first_stdout);
  first_stdout = false;
}

static int run(Config cfg) {
  if (cfg.output_jsonl_path.empty() && !cfg.no_output_jsonl) {
    cfg.output_jsonl_path = default_output_jsonl_path();
  }

  MeetEyeRuntime runtime(cfg);
  AngleAndDistanceRuntime angle_runtime(cfg, runtime.meta(), runtime.base_map_x(), runtime.base_map_y());
  std::unique_ptr<NativeHybridSortTracker> tracker;
  if (cfg.tracker_enabled) {
    tracker.reset(new NativeHybridSortTracker(cfg, runtime.meta()));
  }

  std::ofstream jsonl;
  if (!cfg.output_jsonl_path.empty()) {
    jsonl.open(cfg.output_jsonl_path.c_str(), std::ios::out | std::ios::trunc);
    if (!jsonl) {
      throw std::runtime_error("cannot open output jsonl: " + cfg.output_jsonl_path);
    }
  }
  std::ofstream debug_jsonl;
  if (!cfg.debug_jsonl_path.empty()) {
    debug_jsonl.open(cfg.debug_jsonl_path.c_str(), std::ios::out | std::ios::trunc);
    if (!debug_jsonl) {
      throw std::runtime_error("cannot open debug jsonl: " + cfg.debug_jsonl_path);
    }
  }
  std::unique_ptr<WebSocketServer> ws_server;
  if (cfg.ws_json) {
    ws_server.reset(new WebSocketServer(cfg.ws_host, cfg.ws_port, cfg.ws_path));
    ws_server->start();
  }
  std::unique_ptr<BoardWebUiServer> webui_server;
  std::unique_ptr<AsyncWebUiFramePublisher> webui_frame_publisher;
  if (cfg.webui) {
    webui_server.reset(new BoardWebUiServer(cfg.webui_host, cfg.webui_port));
    webui_server->start();
    webui_frame_publisher.reset(
        new AsyncWebUiFramePublisher(cfg, runtime, webui_server.get()));
    webui_frame_publisher->start();
  }

  std::string system_profile_path;
  std::unique_ptr<SystemLoadSampler> system_sampler;
  if (cfg.profile_system_load || cfg.webui) {
    if (cfg.profile_system_load) {
      system_profile_path = system_profile_path_for(cfg);
    }
    system_sampler.reset(new SystemLoadSampler(cfg.system_load_interval_ms, system_profile_path));
    system_sampler->start();
    std::cerr << "[system-load] background sampler enabled: interval="
              << cfg.system_load_interval_ms << "ms\n";
    if (!system_profile_path.empty()) {
      std::cerr << "[system-load] profile JSONL: " << system_profile_path << "\n";
    }
  }

  bool first_stdout = true;
  int frame_id = 0;
  const double run_start_s = steady_seconds();
  ProfileSummary profile_summary;
  TurboJpegDecoder jpeg_decoder;
  std::unique_ptr<ImageDecodePrefetcher> image_prefetcher;
  if (cfg.decode_prefetch && !cfg.image_paths.empty()) {
    image_prefetcher.reset(new ImageDecodePrefetcher(cfg.image_paths));
  }
  if (cfg.staging_pipeline && !cfg.image_paths.empty()) {
    auto load_prepared_image = [&](size_t i) -> PreparedStagingFrame {
      PreparedStagingFrame frame;
      frame.frame_start_s = steady_seconds();
      double t0 = steady_seconds();
      double t1 = t0;
      if (image_prefetcher) {
        t0 = steady_seconds();
        DecodedFrame decoded = image_prefetcher->pop();
        t1 = steady_seconds();
        frame.decode_wait_ms = (t1 - t0) * 1000.0;
        frame.image_path = decoded.image_path;
        frame.image = std::move(decoded.image);
        frame.file_read_ms = decoded.file_read_ms;
        frame.jpeg_decode_ms = decoded.jpeg_decode_ms;
        frame.frame_index = static_cast<int>(i);
      } else {
        frame.image_path = cfg.image_paths[i];
        frame.frame_index = static_cast<int>(i);
        t0 = steady_seconds();
        const std::vector<uint8_t> jpeg = read_bytes(frame.image_path);
        t1 = steady_seconds();
        frame.file_read_ms = (t1 - t0) * 1000.0;
        t0 = steady_seconds();
        frame.image = jpeg_decoder.decode_buffer(jpeg.data(), jpeg.size());
        t1 = steady_seconds();
        frame.jpeg_decode_ms = (t1 - t0) * 1000.0;
      }
      return runtime.prepare_staging_frame(std::move(frame));
    };

    PreparedStagingFrame pending = load_prepared_image(0);
    for (size_t i = 1; i < cfg.image_paths.size(); ++i) {
      std::future<PreparedStagingFrame> current_future =
          runtime.start_staging_inference(std::move(pending));
      PreparedStagingFrame next = load_prepared_image(i);
      PreparedStagingFrame done = current_future.get();
      finish_and_write_frame(
          cfg,
          runtime,
          angle_runtime,
          tracker.get(),
          jsonl,
          debug_jsonl,
          ws_server.get(),
          webui_server.get(),
          webui_frame_publisher.get(),
          system_sampler.get(),
          done,
          frame_id,
          run_start_s,
          profile_summary,
          first_stdout);
      pending = std::move(next);
    }
    std::future<PreparedStagingFrame> current_future =
        runtime.start_staging_inference(std::move(pending));
    PreparedStagingFrame done = current_future.get();
    finish_and_write_frame(
        cfg,
        runtime,
        angle_runtime,
        tracker.get(),
        jsonl,
        debug_jsonl,
        ws_server.get(),
        webui_server.get(),
        webui_frame_publisher.get(),
        system_sampler.get(),
        done,
        frame_id,
        run_start_s,
        profile_summary,
        first_stdout);
  } else {
  for (size_t i = 0; i < cfg.image_paths.size(); ++i) {
    const double frame_start_s = steady_seconds();
    std::string image_path;
    Image image;
    double file_read_ms = 0.0;
    double jpeg_decode_ms = 0.0;
    double decode_wait_ms = 0.0;
    double t0 = steady_seconds();
    double t1 = t0;
    if (image_prefetcher) {
      t0 = steady_seconds();
      DecodedFrame decoded = image_prefetcher->pop();
      t1 = steady_seconds();
      decode_wait_ms = (t1 - t0) * 1000.0;
      image_path = decoded.image_path;
      image = std::move(decoded.image);
      file_read_ms = decoded.file_read_ms;
      jpeg_decode_ms = decoded.jpeg_decode_ms;
    } else {
      image_path = cfg.image_paths[i];
      t0 = steady_seconds();
      const std::vector<uint8_t> jpeg = read_bytes(image_path);
      t1 = steady_seconds();
      file_read_ms = (t1 - t0) * 1000.0;
      t0 = steady_seconds();
      image = jpeg_decoder.decode_buffer(jpeg.data(), jpeg.size());
      t1 = steady_seconds();
      jpeg_decode_ms = (t1 - t0) * 1000.0;
    }
    FrameResult result = runtime.process(image, image_path, static_cast<int>(i));
    result.profile_ms[kProfileFileRead] = file_read_ms;
    result.profile_ms[kProfileJpegDecode] = jpeg_decode_ms;
    result.profile_ms[kProfileDecodeWait] = decode_wait_ms;
    result.profile_ms[kProfileDecode] = file_read_ms + jpeg_decode_ms;
    if (tracker) {
      t0 = steady_seconds();
      tracker->update(result);
      t1 = steady_seconds();
      result.profile_ms[kProfileTrackerOuter] = (t1 - t0) * 1000.0;
    } else {
      result.raw_detection_count = result.detection_count;
      result.track_ids.assign(static_cast<size_t>(result.detection_count), -1);
    }

    frame_id += 1;
    const double now_s = steady_seconds();
    FrameRateStats fps;
    fps.frame_ms = (now_s - frame_start_s) * 1000.0;
    fps.instant_fps = fps.frame_ms > 0.0 ? 1000.0 / fps.frame_ms : 0.0;
    fps.elapsed_s = std::max(0.0, now_s - run_start_s);
    fps.frames = frame_id;
    fps.average_fps = fps.elapsed_s > 0.0 ? static_cast<double>(fps.frames) / fps.elapsed_s : 0.0;
    result.profile_ms[kProfileFrameTotal] = fps.frame_ms;
    if (cfg.print_profile_summary) {
      profile_summary.add(result, fps);
      if (cfg.profile_interval > 0 && frame_id % cfg.profile_interval == 0) {
        profile_summary.print(std::cerr, "running");
      }
    }
    write_outputs(
        cfg,
        runtime,
        angle_runtime,
        jsonl,
        debug_jsonl,
        ws_server.get(),
        webui_server.get(),
          webui_frame_publisher.get(),
        system_sampler.get(),
        &image,
        result,
        fps,
        frame_id,
        first_stdout);
    first_stdout = false;
  }
  }

  if (!cfg.camera_device.empty()) {
    if (cfg.camera_prefetch) {
      CameraDecodePrefetcher camera_prefetcher(cfg);
      if (cfg.staging_pipeline) {
        auto load_prepared_camera = [&]() -> PreparedStagingFrame {
          PreparedStagingFrame frame;
          frame.frame_start_s = steady_seconds();
          const double t0 = steady_seconds();
          DecodedFrame decoded = camera_prefetcher.pop();
          const double t1 = steady_seconds();
          frame.decode_wait_ms = (t1 - t0) * 1000.0;
          frame.image_path = decoded.image_path;
          frame.image = std::move(decoded.image);
          frame.frame_index = static_cast<int>(decoded.index);
          frame.camera_read_ms = decoded.camera_read_ms;
          frame.jpeg_decode_ms = decoded.jpeg_decode_ms;
          return runtime.prepare_staging_frame(std::move(frame));
        };

        PreparedStagingFrame pending = load_prepared_camera();
        for (int i = 1; cfg.max_frames == 0 || i < cfg.max_frames; ++i) {
          std::future<PreparedStagingFrame> current_future =
              runtime.start_staging_inference(std::move(pending));
          PreparedStagingFrame next = load_prepared_camera();
          PreparedStagingFrame done = current_future.get();
          finish_and_write_frame(
              cfg,
              runtime,
              angle_runtime,
              tracker.get(),
              jsonl,
              debug_jsonl,
              ws_server.get(),
              webui_server.get(),
              webui_frame_publisher.get(),
              system_sampler.get(),
              done,
              frame_id,
              run_start_s,
              profile_summary,
              first_stdout);
          pending = std::move(next);
        }
        if (cfg.max_frames > 0) {
          std::future<PreparedStagingFrame> current_future =
              runtime.start_staging_inference(std::move(pending));
          PreparedStagingFrame done = current_future.get();
          finish_and_write_frame(
              cfg,
              runtime,
              angle_runtime,
              tracker.get(),
              jsonl,
              debug_jsonl,
              ws_server.get(),
              webui_server.get(),
              webui_frame_publisher.get(),
              system_sampler.get(),
              done,
              frame_id,
              run_start_s,
              profile_summary,
              first_stdout);
        }
      } else {
      for (int i = 0; cfg.max_frames == 0 || i < cfg.max_frames; ++i) {
        const double frame_start_s = steady_seconds();
        double t0 = steady_seconds();
        DecodedFrame decoded = camera_prefetcher.pop();
        double t1 = steady_seconds();
        const double decode_wait_ms = (t1 - t0) * 1000.0;

        FrameResult result =
            runtime.process(decoded.image, decoded.image_path, static_cast<int>(decoded.index));
        result.profile_ms[kProfileCameraRead] = decoded.camera_read_ms;
        result.profile_ms[kProfileJpegDecode] = decoded.jpeg_decode_ms;
        result.profile_ms[kProfileDecodeWait] = decode_wait_ms;
        result.profile_ms[kProfileDecode] = decoded.camera_read_ms + decoded.jpeg_decode_ms;
        if (tracker) {
          t0 = steady_seconds();
          tracker->update(result);
          t1 = steady_seconds();
          result.profile_ms[kProfileTrackerOuter] = (t1 - t0) * 1000.0;
        } else {
          result.raw_detection_count = result.detection_count;
          result.track_ids.assign(static_cast<size_t>(result.detection_count), -1);
        }

        frame_id += 1;
        const double now_s = steady_seconds();
        FrameRateStats fps;
        fps.frame_ms = (now_s - frame_start_s) * 1000.0;
        fps.instant_fps = fps.frame_ms > 0.0 ? 1000.0 / fps.frame_ms : 0.0;
        fps.elapsed_s = std::max(0.0, now_s - run_start_s);
        fps.frames = frame_id;
        fps.average_fps =
            fps.elapsed_s > 0.0 ? static_cast<double>(fps.frames) / fps.elapsed_s : 0.0;
        result.profile_ms[kProfileFrameTotal] = fps.frame_ms;
        if (cfg.print_profile_summary) {
          profile_summary.add(result, fps);
          if (cfg.profile_interval > 0 && frame_id % cfg.profile_interval == 0) {
            profile_summary.print(std::cerr, "running");
          }
        }
        write_outputs(
            cfg,
            runtime,
            angle_runtime,
            jsonl,
            debug_jsonl,
            ws_server.get(),
            webui_server.get(),
            webui_frame_publisher.get(),
            system_sampler.get(),
            &decoded.image,
            result,
            fps,
            frame_id,
            first_stdout);
        first_stdout = false;
      }
      }
    } else {
      V4L2MjpegCamera camera(
          cfg.camera_device,
          cfg.camera_width,
          cfg.camera_height,
          cfg.camera_fps,
          cfg.camera_buffers,
          cfg.camera_timeout_ms);
      camera.open();

      for (int i = 0; cfg.max_frames == 0 || i < cfg.max_frames; ++i) {
        const double frame_start_s = steady_seconds();
        double t0 = steady_seconds();
        V4L2MjpegCamera::Frame frame = camera.read_frame();
        double t1 = steady_seconds();
        const double camera_read_ms = (t1 - t0) * 1000.0;
        Image image;
        double jpeg_decode_ms = 0.0;
        try {
          t0 = steady_seconds();
          image = jpeg_decoder.decode_buffer(frame.data, frame.size);
          t1 = steady_seconds();
          jpeg_decode_ms = (t1 - t0) * 1000.0;
        } catch (...) {
          camera.release_frame(frame);
          throw;
        }
        camera.release_frame(frame);

        std::ostringstream frame_name;
        frame_name << cfg.camera_device << "#" << i;
        FrameResult result = runtime.process(image, frame_name.str(), i);
        result.profile_ms[kProfileCameraRead] = camera_read_ms;
        result.profile_ms[kProfileJpegDecode] = jpeg_decode_ms;
        result.profile_ms[kProfileDecode] = camera_read_ms + jpeg_decode_ms;
        if (tracker) {
          t0 = steady_seconds();
          tracker->update(result);
          t1 = steady_seconds();
          result.profile_ms[kProfileTrackerOuter] = (t1 - t0) * 1000.0;
        } else {
          result.raw_detection_count = result.detection_count;
          result.track_ids.assign(static_cast<size_t>(result.detection_count), -1);
        }

        frame_id += 1;
        const double now_s = steady_seconds();
        FrameRateStats fps;
        fps.frame_ms = (now_s - frame_start_s) * 1000.0;
        fps.instant_fps = fps.frame_ms > 0.0 ? 1000.0 / fps.frame_ms : 0.0;
        fps.elapsed_s = std::max(0.0, now_s - run_start_s);
        fps.frames = frame_id;
        fps.average_fps =
            fps.elapsed_s > 0.0 ? static_cast<double>(fps.frames) / fps.elapsed_s : 0.0;
        result.profile_ms[kProfileFrameTotal] = fps.frame_ms;
        if (cfg.print_profile_summary) {
          profile_summary.add(result, fps);
          if (cfg.profile_interval > 0 && frame_id % cfg.profile_interval == 0) {
            profile_summary.print(std::cerr, "running");
          }
        }
        write_outputs(
            cfg,
            runtime,
            angle_runtime,
            jsonl,
            debug_jsonl,
            ws_server.get(),
            webui_server.get(),
            webui_frame_publisher.get(),
            system_sampler.get(),
            &image,
            result,
            fps,
            frame_id,
            first_stdout);
        first_stdout = false;
      }
    }
  }

  if (cfg.print_profile_summary) {
    profile_summary.print(std::cerr, "final");
  }
  if (system_sampler) {
    system_sampler->stop();
    const double elapsed_s = std::max(0.0, steady_seconds() - run_start_s);
    write_system_load_summary(
        system_profile_path,
        !cfg.debug_jsonl_path.empty() ? cfg.debug_jsonl_path : cfg.output_jsonl_path,
        elapsed_s,
        frame_id);
  }

  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config cfg = parse_args(argc, argv);
    return run(cfg);
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n\n";
    print_usage(argv[0]);
    return 1;
  }
}
