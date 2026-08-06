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
#include <functional>
#include <future>
#include <ifaddrs.h>
#include <iomanip>
#include <iostream>
#include <linux/videodev2.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <netdb.h>
#include <net/if.h>
#include <netinet/in.h>
#include <ostream>
#include <set>
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
int face_rknn_parallel_infer_merged_single(void* handle,
                                           const uint8_t* input,
                                           int input_h,
                                           int input_w,
                                           int input_c,
                                           int image_h,
                                           int image_w,
                                           float gain,
                                           int pad_left,
                                           int pad_top,
                                           float nms_iou_thresh,
                                           float conf_threshold,
                                           int max_det,
                                           int max_nms,
                                           float* detections,
                                           int max_output_dets,
                                           int* detection_count,
                                           int* merge_stats,
                                           double* timings,
                                           char* err,
                                           int err_len);
int face_rknn_parallel_infer_merged_single_core(void* handle,
                                                int core_index,
                                                const uint8_t* input,
                                                int input_h,
                                                int input_w,
                                                int input_c,
                                                int image_h,
                                                int image_w,
                                                float gain,
                                                int pad_left,
                                                int pad_top,
                                                float nms_iou_thresh,
                                                float conf_threshold,
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
                              float new_track_center_suppress_thresh,
                              float new_track_suppress_max_side,
                              float byte_residual_size_ratio_thresh,
                              float byte_residual_max_side,
                              float lost_track_reconnect_center_thresh,
                              float lost_track_reconnect_multi_target_center_thresh,
                              float lost_track_reconnect_size_ratio_thresh,
                              float lost_track_reconnect_ambiguity_margin,
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
int hybrid_sort_native_retire_track(void* handle, int raw_id, char* err, int err_len);
void hybrid_sort_native_destroy(void* handle);

int adaface_rknn_create(const char* model_path, void** handle, char* err, int err_len);
int adaface_rknn_create_with_core_mask(const char* model_path,
                                       int core_mask,
                                       void** handle,
                                       char* err,
                                       int err_len);
void adaface_rknn_destroy(void* handle);
int adaface_rknn_get_shape(void* handle,
                           int* input_h,
                           int* input_w,
                           int* input_c,
                           int* feature_dim,
                           char* err,
                           int err_len);
int adaface_rknn_infer(void* handle,
                       const float* input_nchw,
                       int input_count,
                       float* feature,
                       int feature_cap,
                       char* err,
                       int err_len);
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


// Implementation blocks are split for maintainability but included into
// this translation unit to keep board_cpp behavior and build output unchanged.
#include "src/meeteye_types.inc"
#include "src/meeteye_profile_system.inc"
#include "src/meeteye_frame_sources.inc"
#include "src/meeteye_cli.inc"
#include "src/meeteye_drawing_json.inc"
#include "src/meeteye_face_recognition.inc"
#include "src/meeteye_output_payloads.inc"
#include "src/meeteye_native_tracker.inc"
#include "src/meeteye_runtime_pipeline.inc"
#include "src/meeteye_webui.inc"
#include "src/meeteye_runner.inc"

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
