// C API backend for three parallel RKNN slice inference.
//
// Python passes three NHWC uint8 RGB inputs. This library keeps three RKNN
// contexts bound to core0/core1/core2, runs them in native threads, reads
// dequantized float outputs, and packs each split-output face model result into
// [20, anchors]: bbox(4), score(1), keypoints(15). The model can expose the
// keypoints either as one 15-channel tensor or as xy(5x2) and conf(5x1) tensors.

#include "rknn_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kNumContexts = 3;
constexpr int kCombinedChannels = 20;
constexpr int kDetectionFields = 20;  // xyxy, score, 5 face keypoints x/y/conf
constexpr int kRknnMemorySyncToDeviceCompat = 0x1;
constexpr int kRknnMemorySyncFromDeviceCompat = 0x2;

using RknnMemSyncFn = int (*)(rknn_context, rknn_tensor_mem*, int);

struct ContextState {
  rknn_context ctx = 0;
  rknn_tensor_attr input_attr{};
  rknn_tensor_attr output_attrs[8]{};
  uint32_t n_output = 0;
  int input_h = 0;
  int input_w = 0;
  int input_c = 0;
  size_t input_bytes = 0;
  int anchors = 0;
  bool split_keypoint_conf = false;
  rknn_tensor_attr bound_input_attr{};
  rknn_tensor_mem* input_mem = nullptr;
  bool input_mem_bound = false;
};

struct ParallelRunner {
  ContextState states[kNumContexts];
};

static int64_t now_us()
{
  using clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::microseconds>(clock::now().time_since_epoch()).count();
}

static void set_error(char* err, int err_len, const std::string& msg)
{
  if (err == nullptr || err_len <= 0) {
    return;
  }
  std::snprintf(err, static_cast<size_t>(err_len), "%s", msg.c_str());
}

static void release_bound_input_mem(ContextState* state)
{
  if (state == nullptr || state->ctx == 0 || state->input_mem == nullptr) {
    return;
  }
  rknn_destroy_mem(state->ctx, state->input_mem);
  state->input_mem = nullptr;
  state->input_mem_bound = false;
}

static RknnMemSyncFn resolve_rknn_mem_sync()
{
  static bool resolved = false;
  static RknnMemSyncFn fn = nullptr;
  if (!resolved) {
    resolved = true;
    fn = reinterpret_cast<RknnMemSyncFn>(dlsym(RTLD_DEFAULT, "rknn_mem_sync"));
  }
  return fn;
}

static int channel_offset(int channels)
{
  if (channels == 4) {
    return 0;
  }
  if (channels == 1) {
    return 4;
  }
  if (channels == 15) {
    return 5;
  }
  return -1;
}

struct OutputLayout {
  int channels = 0;
  int anchors = 0;
};

static OutputLayout output_layout(const rknn_tensor_attr& attr)
{
  OutputLayout layout;
  if (attr.n_dims == 4) {
    layout.channels = static_cast<int>(attr.dims[1]) * static_cast<int>(attr.dims[2]);
    layout.anchors = static_cast<int>(attr.dims[3]);
  } else if (attr.n_dims == 3) {
    layout.channels = static_cast<int>(attr.dims[1]);
    layout.anchors = static_cast<int>(attr.dims[2]);
  } else if (attr.n_dims == 2) {
    layout.channels = static_cast<int>(attr.dims[0]);
    layout.anchors = static_cast<int>(attr.dims[1]);
  }
  return layout;
}

static void copy_output_to_combined(const rknn_tensor_attr& attr,
                                    const rknn_output& out,
                                    int anchors,
                                    float* output)
{
  if (out.buf == nullptr || output == nullptr || anchors <= 0) {
    return;
  }
  const OutputLayout layout = output_layout(attr);
  if (layout.anchors != anchors || layout.channels <= 0) {
    return;
  }
  const float* buf = static_cast<const float*>(out.buf);
  const int offset = channel_offset(layout.channels);
  if (offset >= 0) {
    std::memcpy(
        output + static_cast<size_t>(offset) * anchors,
        buf,
        static_cast<size_t>(layout.channels) * anchors * sizeof(float));
    return;
  }
  if (layout.channels == 10) {
    for (int k = 0; k < 5; ++k) {
      for (int a = 0; a < anchors; ++a) {
        output[static_cast<size_t>(5 + k * 3 + 0) * anchors + a] =
            buf[static_cast<size_t>(k * 2 + 0) * anchors + a];
        output[static_cast<size_t>(5 + k * 3 + 1) * anchors + a] =
            buf[static_cast<size_t>(k * 2 + 1) * anchors + a];
      }
    }
  } else if (layout.channels == 5) {
    for (int k = 0; k < 5; ++k) {
      for (int a = 0; a < anchors; ++a) {
        output[static_cast<size_t>(5 + k * 3 + 2) * anchors + a] =
            buf[static_cast<size_t>(k) * anchors + a];
      }
    }
  }
}

struct DecodeMeta {
  int slice_h = 0;
  int slice_w = 0;
  float gain = 1.0f;
  int pad_left = 0;
  int pad_top = 0;
};

struct Candidate {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int anchor = 0;
};

static float sigmoid_f(float value)
{
  if (value < -80.0f) {
    value = -80.0f;
  } else if (value > 80.0f) {
    value = 80.0f;
  }
  return 1.0f / (1.0f + std::exp(-value));
}

static float probability_f(float value, bool use_sigmoid)
{
  return use_sigmoid ? sigmoid_f(value) : value;
}

static float clip_f(float value, float low, float high)
{
  return std::max(low, std::min(value, high));
}

static float iou_one(const Candidate& a, const Candidate& b)
{
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);
  const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  return inter / (area_a + area_b - inter + 1e-6f);
}

struct MergeBox {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
};

struct MergeDetection {
  MergeBox box;
  float score = 0.0f;
  int label = 0;
  int slice_idx = 0;
  float keypoints[15]{};
  float area = 0.0f;
};

static float merge_box_iou(const MergeBox& a, const MergeBox& b)
{
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);
  const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  return inter / (area_a + area_b - inter + 1e-6f);
}

static void merge_intersection_stats(const MergeBox& a,
                                     const MergeBox& b,
                                     float* inter,
                                     float* min_area,
                                     float* iw,
                                     float* ih,
                                     float* min_w,
                                     float* min_h,
                                     float* area_a,
                                     float* area_b)
{
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);
  *iw = std::max(0.0f, x2 - x1);
  *ih = std::max(0.0f, y2 - y1);
  *inter = (*iw) * (*ih);
  const float w1 = std::max(0.0f, a.x2 - a.x1);
  const float h1 = std::max(0.0f, a.y2 - a.y1);
  const float w2 = std::max(0.0f, b.x2 - b.x1);
  const float h2 = std::max(0.0f, b.y2 - b.y1);
  *area_a = w1 * h1;
  *area_b = w2 * h2;
  *min_area = std::min(*area_a, *area_b);
  *min_w = std::min(w1, w2);
  *min_h = std::min(h1, h2);
}

static float merge_center_distance_norm(const MergeBox& a, const MergeBox& b)
{
  const float cx1 = (a.x1 + a.x2) * 0.5f;
  const float cy1 = (a.y1 + a.y2) * 0.5f;
  const float cx2 = (b.x1 + b.x2) * 0.5f;
  const float cy2 = (b.y1 + b.y2) * 0.5f;
  const float h1 = std::max(a.y2 - a.y1, 1.0f);
  const float h2 = std::max(b.y2 - b.y1, 1.0f);
  const float avg_h = (h1 + h2) * 0.5f;
  const float dx = cx1 - cx2;
  const float dy = cy1 - cy2;
  return std::sqrt(dx * dx + dy * dy) / avg_h;
}

static bool merge_old_spatial_duplicate(const MergeBox& a, const MergeBox& b)
{
  float inter = 0.0f;
  float min_area = 0.0f;
  float iw = 0.0f;
  float ih = 0.0f;
  float min_w = 0.0f;
  float min_h = 0.0f;
  float area_a = 0.0f;
  float area_b = 0.0f;
  merge_intersection_stats(a, b, &inter, &min_area, &iw, &ih, &min_w, &min_h, &area_a, &area_b);
  if (min_area <= 0.0f) {
    return false;
  }

  const float min_iou = inter / (min_area + 1e-6f);
  if (min_iou > 0.7f) {
    return true;
  }

  const float cd_norm = merge_center_distance_norm(a, b);
  const float x_cover = min_w > 0.0f ? iw / (min_w + 1e-6f) : 0.0f;
  const float y_cover = min_h > 0.0f ? ih / (min_h + 1e-6f) : 0.0f;
  const float area_ratio = std::min(area_a, area_b) / (std::max(area_a, area_b) + 1e-6f);
  return cd_norm < 0.8f && x_cover > 0.35f && y_cover > 0.35f && area_ratio > 0.25f;
}

static bool merge_wrap_spatial_duplicate(const MergeBox& a,
                                         const MergeBox& b,
                                         float min_iou_thresh)
{
  float inter = 0.0f;
  float min_area = 0.0f;
  float iw = 0.0f;
  float ih = 0.0f;
  float min_w = 0.0f;
  float min_h = 0.0f;
  float area_a = 0.0f;
  float area_b = 0.0f;
  merge_intersection_stats(a, b, &inter, &min_area, &iw, &ih, &min_w, &min_h, &area_a, &area_b);
  if (min_area <= 0.0f) {
    return false;
  }

  const float min_iou = inter / (min_area + 1e-6f);
  if (min_iou > min_iou_thresh) {
    return true;
  }

  const float dist = merge_center_distance_norm(a, b);
  const float y_cover = min_h > 0.0f ? ih / (min_h + 1e-6f) : 0.0f;
  const float area_ratio = std::min(area_a, area_b) / (std::max(area_a, area_b) + 1e-6f);
  return dist < 0.80f && y_cover > 0.45f && area_ratio > 0.25f;
}

static MergeBox merge_convert_box_to_original(const MergeBox& local,
                                              float start_x,
                                              int wrap_around,
                                              float original_width)
{
  MergeBox out = local;
  if (!wrap_around) {
    out.x1 = local.x1 + start_x;
    out.x2 = local.x2 + start_x;
  } else if (start_x < 0.0f) {
    const float right_width = -start_x;
    if (local.x2 <= right_width) {
      out.x1 = local.x1 + (original_width + start_x);
      out.x2 = local.x2 + (original_width + start_x);
    } else if (local.x1 >= right_width) {
      out.x1 = local.x1 - right_width;
      out.x2 = local.x2 - right_width;
    } else {
      out.x1 = local.x1 - right_width;
      out.x2 = local.x2 - right_width;
      if (out.x2 > original_width) {
        out.x2 = original_width;
      }
    }
  } else {
    const float left_width = original_width - start_x;
    if (local.x2 <= left_width) {
      out.x1 = local.x1 + start_x;
      out.x2 = local.x2 + start_x;
    } else if (local.x1 >= left_width) {
      out.x1 = local.x1 - left_width;
      out.x2 = local.x2 - left_width;
    } else {
      out.x1 = local.x1 + start_x;
      out.x2 = local.x2 + start_x;
      if (out.x2 > original_width) {
        out.x2 = original_width;
      }
    }
  }

  out.x1 = clip_f(out.x1, 0.0f, original_width - 1.0f);
  out.x2 = clip_f(out.x2, 0.0f, original_width - 1.0f);
  if (out.x1 > out.x2) {
    std::swap(out.x1, out.x2);
  }
  return out;
}

static float merge_convert_x_to_original(float x,
                                         float start_x,
                                         int wrap_around,
                                         float original_width)
{
  float out = x;
  if (!wrap_around) {
    out = x + start_x;
  } else if (start_x < 0.0f) {
    const float right_width = -start_x;
    out = x <= right_width ? x + (original_width + start_x) : x - right_width;
  } else {
    const float left_width = original_width - start_x;
    out = x <= left_width ? x + start_x : x - left_width;
  }
  return clip_f(out, 0.0f, original_width - 1.0f);
}

static std::vector<int> merge_area_weighted_nms(const std::vector<MergeDetection>& dets,
                                                const std::vector<int>& indices,
                                                float nms_iou_thresh)
{
  if (indices.size() <= 1) {
    return indices;
  }

  float max_area = 0.0f;
  for (int idx : indices) {
    max_area = std::max(max_area, dets[static_cast<size_t>(idx)].area);
  }

  std::vector<int> sorted = indices;
  std::sort(sorted.begin(), sorted.end(), [&](int lhs, int rhs) {
    const MergeDetection& a = dets[static_cast<size_t>(lhs)];
    const MergeDetection& b = dets[static_cast<size_t>(rhs)];
    const float wa = a.score * (0.6f + 0.4f * (a.area / (max_area + 1e-6f)));
    const float wb = b.score * (0.6f + 0.4f * (b.area / (max_area + 1e-6f)));
    return wa > wb;
  });

  std::vector<int> keep;
  std::vector<uint8_t> removed(sorted.size(), 0);
  for (size_t i = 0; i < sorted.size(); ++i) {
    if (removed[i]) {
      continue;
    }
    keep.push_back(sorted[i]);
    const MergeBox& cur = dets[static_cast<size_t>(sorted[i])].box;
    for (size_t j = i + 1; j < sorted.size(); ++j) {
      if (!removed[j] && merge_box_iou(cur, dets[static_cast<size_t>(sorted[j])].box) >= nms_iou_thresh) {
        removed[j] = 1;
      }
    }
  }
  return keep;
}

static std::vector<int> merge_final_spatial_dedup(const std::vector<MergeDetection>& dets,
                                                  const std::vector<int>& keep_indices)
{
  if (keep_indices.size() <= 1) {
    return keep_indices;
  }

  std::set<int> labels;
  for (int idx : keep_indices) {
    labels.insert(dets[static_cast<size_t>(idx)].label);
  }

  std::set<int> selected;
  for (int label : labels) {
    std::vector<int> label_indices;
    for (int idx : keep_indices) {
      if (dets[static_cast<size_t>(idx)].label == label) {
        label_indices.push_back(idx);
      }
    }
    if (label_indices.size() <= 1) {
      selected.insert(label_indices.begin(), label_indices.end());
      continue;
    }

    float max_area = 0.0f;
    for (int idx : label_indices) {
      max_area = std::max(max_area, dets[static_cast<size_t>(idx)].area);
    }
    std::sort(label_indices.begin(), label_indices.end(), [&](int lhs, int rhs) {
      const MergeDetection& a = dets[static_cast<size_t>(lhs)];
      const MergeDetection& b = dets[static_cast<size_t>(rhs)];
      const float wa = a.score * (0.6f + 0.4f * (a.area / (max_area + 1e-6f)));
      const float wb = b.score * (0.6f + 0.4f * (b.area / (max_area + 1e-6f)));
      return wa > wb;
    });

    std::vector<int> kept_for_label;
    for (int idx : label_indices) {
      bool duplicate = false;
      for (int kept_idx : kept_for_label) {
        if (merge_old_spatial_duplicate(dets[static_cast<size_t>(idx)].box,
                                        dets[static_cast<size_t>(kept_idx)].box)) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        kept_for_label.push_back(idx);
      }
    }
    selected.insert(kept_for_label.begin(), kept_for_label.end());
  }

  std::vector<int> final_keep;
  for (int idx : keep_indices) {
    if (selected.count(idx)) {
      final_keep.push_back(idx);
    }
  }
  return final_keep;
}

static int merge_decoded_slices(const float* decoded,
                                const int* decoded_counts,
                                int max_det,
                                const float* slice_start_x,
                                const int* slice_wrap_around,
                                int num_slices,
                                float original_width,
                                float overlap_ratio,
                                float iou_threshold,
                                float nms_iou_thresh,
                                float* out_detections,
                                int max_output_dets,
                                int* out_count,
                                int* stats,
                                double* timings,
                                char* err,
                                int err_len)
{
  if (out_count != nullptr) {
    *out_count = 0;
  }
  if (decoded == nullptr || decoded_counts == nullptr || slice_start_x == nullptr ||
      slice_wrap_around == nullptr || out_detections == nullptr || out_count == nullptr) {
    set_error(err, err_len, "invalid null argument");
    return -1;
  }
  if (max_det <= 0 || max_output_dets <= 0 || num_slices <= 0 || original_width <= 1.0f) {
    set_error(err, err_len, "invalid merge sizes");
    return -1;
  }

  int raw_count = 0;
  for (int i = 0; i < num_slices; ++i) {
    raw_count += std::max(0, decoded_counts[i]);
  }

  int64_t t0 = now_us();
  std::vector<MergeDetection> dets;
  dets.reserve(static_cast<size_t>(raw_count));
  for (int slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
    const int count = std::min(std::max(0, decoded_counts[slice_idx]), max_det);
    for (int det_idx = 0; det_idx < count; ++det_idx) {
      const float* row = decoded + (static_cast<size_t>(slice_idx) * max_det + det_idx) * kDetectionFields;
      MergeBox local;
      local.x1 = row[0];
      local.y1 = row[1];
      local.x2 = row[2];
      local.y2 = row[3];

      MergeDetection det;
      det.box = merge_convert_box_to_original(
          local,
          slice_start_x[slice_idx],
          slice_wrap_around[slice_idx],
          original_width);
      det.score = row[4];
      det.label = 0;
      det.slice_idx = slice_idx;
      det.area = std::max(0.0f, det.box.x2 - det.box.x1) * std::max(0.0f, det.box.y2 - det.box.y1);
      for (int k = 0; k < 5; ++k) {
        const int base = k * 3;
        det.keypoints[base + 0] = merge_convert_x_to_original(
            row[5 + base + 0],
            slice_start_x[slice_idx],
            slice_wrap_around[slice_idx],
            original_width);
        det.keypoints[base + 1] = row[5 + base + 1];
        det.keypoints[base + 2] = row[5 + base + 2];
      }
      dets.push_back(det);
    }
  }
  int64_t t_coords = now_us();

  std::vector<uint8_t> suppressed(dets.size(), 0);
  const float panorama_width = original_width;
  const int last_slice_idx = num_slices - 1;
  const float slice_width = panorama_width / static_cast<float>(num_slices);
  const float max_dist = 3.0f * slice_width * overlap_ratio;

  for (size_t i = 0; i < dets.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    for (size_t j = i + 1; j < dets.size(); ++j) {
      if (suppressed[j]) {
        continue;
      }

      const MergeDetection& di = dets[i];
      const MergeDetection& dj = dets[j];
      bool is_same_target = false;

      if (di.slice_idx == dj.slice_idx) {
        float inter = 0.0f;
        float min_area = 0.0f;
        float iw = 0.0f;
        float ih = 0.0f;
        float min_w = 0.0f;
        float min_h = 0.0f;
        float area_i = 0.0f;
        float area_j = 0.0f;
        merge_intersection_stats(di.box, dj.box, &inter, &min_area, &iw, &ih, &min_w, &min_h, &area_i, &area_j);
        const float intra_min_iou = inter / (min_area + 1e-6f);
        is_same_target = intra_min_iou > 0.7f;
      } else {
        const bool is_wrap_pair =
            num_slices >= 3 &&
            ((di.slice_idx == 0 && dj.slice_idx == last_slice_idx) ||
             (dj.slice_idx == 0 && di.slice_idx == last_slice_idx));
        const bool is_adjacent_pair = std::abs(di.slice_idx - dj.slice_idx) == 1;

        if (is_wrap_pair) {
          MergeBox bi = di.box;
          MergeBox bj = dj.box;
          const float cxi = (bi.x1 + bi.x2) * 0.5f;
          const float cxj = (bj.x1 + bj.x2) * 0.5f;
          if (cxj > cxi) {
            bj.x1 -= panorama_width;
            bj.x2 -= panorama_width;
          } else {
            bi.x1 -= panorama_width;
            bi.x2 -= panorama_width;
          }

          is_same_target = merge_wrap_spatial_duplicate(bi, bj, iou_threshold);
        } else if (is_adjacent_pair) {
          const float cxi = (di.box.x1 + di.box.x2) * 0.5f;
          const float cxj = (dj.box.x1 + dj.box.x2) * 0.5f;
          const float boundary_x = static_cast<float>(std::max(di.slice_idx, dj.slice_idx)) * slice_width;
          if (std::fabs(cxi - boundary_x) <= max_dist && std::fabs(cxj - boundary_x) <= max_dist) {
            float inter = 0.0f;
            float min_area = 0.0f;
            float iw = 0.0f;
            float ih = 0.0f;
            float min_w = 0.0f;
            float min_h = 0.0f;
            float area_i = 0.0f;
            float area_j = 0.0f;
            merge_intersection_stats(di.box, dj.box, &inter, &min_area, &iw, &ih, &min_w, &min_h, &area_i, &area_j);
            const float min_iou = inter / (min_area + 1e-6f);
            if (min_iou > iou_threshold) {
              is_same_target = true;
            } else {
              const float cyi = (di.box.y1 + di.box.y2) * 0.5f;
              const float cyj = (dj.box.y1 + dj.box.y2) * 0.5f;
              const float hi = di.box.y2 - di.box.y1;
              const float hj = dj.box.y2 - dj.box.y1;
              const float avg_h = std::max((hi + hj) * 0.5f, 1.0f);
              const float dx = cxi - cxj;
              const float dy = cyi - cyj;
              const float cd_norm = std::sqrt(dx * dx + dy * dy) / avg_h;
              is_same_target = cd_norm < 0.8f;
            }
          }
        }
      }

      if (is_same_target) {
        if (di.score >= dj.score) {
          suppressed[j] = 1;
        } else {
          suppressed[i] = 1;
          break;
        }
      }
    }
  }
  int64_t t_dedup = now_us();

  std::set<int> labels_set;
  for (size_t i = 0; i < dets.size(); ++i) {
    if (!suppressed[i]) {
      labels_set.insert(dets[i].label);
    }
  }
  std::vector<int> keep_indices;
  for (int label : labels_set) {
    std::vector<int> label_indices;
    for (size_t i = 0; i < dets.size(); ++i) {
      if (!suppressed[i] && dets[i].label == label) {
        label_indices.push_back(static_cast<int>(i));
      }
    }
    std::vector<int> kept = merge_area_weighted_nms(dets, label_indices, nms_iou_thresh);
    keep_indices.insert(keep_indices.end(), kept.begin(), kept.end());
  }
  int64_t t_nms = now_us();

  const int nms_keep_count = static_cast<int>(keep_indices.size());
  keep_indices = merge_final_spatial_dedup(dets, keep_indices);
  int64_t t_final = now_us();

  int count = 0;
  for (int idx : keep_indices) {
    if (count >= max_output_dets) {
      break;
    }
    const MergeDetection& det = dets[static_cast<size_t>(idx)];
    float* row = out_detections + static_cast<size_t>(count) * kDetectionFields;
    row[0] = det.box.x1;
    row[1] = det.box.y1;
    row[2] = det.box.x2;
    row[3] = det.box.y2;
    row[4] = det.score;
    for (int k = 0; k < 15; ++k) {
      row[5 + k] = det.keypoints[k];
    }
    ++count;
  }
  *out_count = count;
  int64_t t_build = now_us();

  if (stats != nullptr) {
    stats[0] = static_cast<int>(dets.size());
    stats[1] = nms_keep_count;
    stats[2] = count;
  }
  if (timings != nullptr) {
    timings[0] = (t_coords - t0) / 1000.0;
    timings[1] = (t_dedup - t_coords) / 1000.0;
    timings[2] = (t_nms - t_dedup) / 1000.0;
    timings[3] = (t_final - t_nms) / 1000.0;
    timings[4] = (t_build - t_final) / 1000.0;
  }
  return 0;
}

static int decode_slice_outputs(const ContextState* state,
                                const rknn_output* outs,
                                const DecodeMeta& meta,
                                float conf_threshold,
                                float iou_threshold,
                                int max_det,
                                int max_nms,
                                float* decoded)
{
  const float* box_ptr = nullptr;
  const float* score_ptr = nullptr;
  const float* kpt_ptr = nullptr;
  const float* kpt_xy_ptr = nullptr;
  const float* kpt_conf_ptr = nullptr;

  for (uint32_t i = 0; i < state->n_output; ++i) {
    const auto& attr = state->output_attrs[i];
    const OutputLayout layout = output_layout(attr);
    if (layout.anchors != state->anchors || outs[i].buf == nullptr) {
      continue;
    }
    const float* buf = static_cast<const float*>(outs[i].buf);
    if (layout.channels == 4) {
      box_ptr = buf;
    } else if (layout.channels == 1) {
      score_ptr = buf;
    } else if (layout.channels == 15) {
      kpt_ptr = buf;
    } else if (layout.channels == 10) {
      kpt_xy_ptr = buf;
    } else if (layout.channels == 5) {
      kpt_conf_ptr = buf;
    }
  }
  if (box_ptr == nullptr || score_ptr == nullptr ||
      (kpt_ptr == nullptr && (kpt_xy_ptr == nullptr || kpt_conf_ptr == nullptr)) ||
      decoded == nullptr || max_det <= 0 || max_nms <= 0 ||
      meta.slice_h <= 0 || meta.slice_w <= 0 || meta.gain <= 0.0f) {
    return 0;
  }

  bool score_use_sigmoid = false;
  for (int a = 0; a < state->anchors; ++a) {
    const float v = score_ptr[a];
    if (v < 0.0f || v > 1.0f) {
      score_use_sigmoid = true;
      break;
    }
  }

  struct ScoredAnchor {
    float score;
    int anchor;
  };
  std::vector<ScoredAnchor> scored;
  scored.reserve(static_cast<size_t>(std::min(state->anchors, max_nms)));
  for (int a = 0; a < state->anchors; ++a) {
    const float score = probability_f(score_ptr[a], score_use_sigmoid);
    if (score >= conf_threshold) {
      scored.push_back({score, a});
    }
  }
  if (scored.empty()) {
    return 0;
  }

  auto score_desc = [](const ScoredAnchor& lhs, const ScoredAnchor& rhs) {
    return lhs.score > rhs.score;
  };
  if (static_cast<int>(scored.size()) > max_nms) {
    std::partial_sort(scored.begin(), scored.begin() + max_nms, scored.end(), score_desc);
    scored.resize(static_cast<size_t>(max_nms));
  }

  const float max_x = static_cast<float>(meta.slice_w - 1);
  const float max_y = static_cast<float>(meta.slice_h - 1);
  std::vector<Candidate> candidates;
  candidates.reserve(scored.size());
  for (const auto& item : scored) {
    const int a = item.anchor;
    const float cx = box_ptr[0 * state->anchors + a];
    const float cy = box_ptr[1 * state->anchors + a];
    const float bw = box_ptr[2 * state->anchors + a];
    const float bh = box_ptr[3 * state->anchors + a];

    Candidate cand;
    cand.x1 = (cx - bw * 0.5f - static_cast<float>(meta.pad_left)) / meta.gain;
    cand.y1 = (cy - bh * 0.5f - static_cast<float>(meta.pad_top)) / meta.gain;
    cand.x2 = (cx + bw * 0.5f - static_cast<float>(meta.pad_left)) / meta.gain;
    cand.y2 = (cy + bh * 0.5f - static_cast<float>(meta.pad_top)) / meta.gain;
    cand.x1 = clip_f(cand.x1, 0.0f, max_x);
    cand.x2 = clip_f(cand.x2, 0.0f, max_x);
    cand.y1 = clip_f(cand.y1, 0.0f, max_y);
    cand.y2 = clip_f(cand.y2, 0.0f, max_y);
    if ((cand.x2 - cand.x1) <= 2.0f || (cand.y2 - cand.y1) <= 2.0f) {
      continue;
    }
    cand.score = item.score;
    cand.anchor = a;
    candidates.push_back(cand);
  }
  if (candidates.empty()) {
    return 0;
  }

  std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
    return lhs.score > rhs.score;
  });

  std::vector<int> keep;
  keep.reserve(static_cast<size_t>(std::min(max_det, static_cast<int>(candidates.size()))));
  std::vector<uint8_t> suppressed(candidates.size(), 0);
  for (size_t i = 0; i < candidates.size() && static_cast<int>(keep.size()) < max_det; ++i) {
    if (suppressed[i]) {
      continue;
    }
    keep.push_back(static_cast<int>(i));
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      if (!suppressed[j] && iou_one(candidates[i], candidates[j]) > iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }
  if (keep.empty()) {
    return 0;
  }

  bool keypoint_use_sigmoid = false;
  for (int kept_idx : keep) {
    const int anchor = candidates[static_cast<size_t>(kept_idx)].anchor;
    for (int k = 0; k < 5; ++k) {
      const float v = kpt_conf_ptr != nullptr
          ? kpt_conf_ptr[k * state->anchors + anchor]
          : kpt_ptr[(k * 3 + 2) * state->anchors + anchor];
      if (v < 0.0f || v > 1.0f) {
        keypoint_use_sigmoid = true;
        break;
      }
    }
    if (keypoint_use_sigmoid) {
      break;
    }
  }

  int out_count = 0;
  for (int kept_idx : keep) {
    const Candidate& cand = candidates[static_cast<size_t>(kept_idx)];
    float* row = decoded + static_cast<size_t>(out_count) * kDetectionFields;
    row[0] = cand.x1;
    row[1] = cand.y1;
    row[2] = cand.x2;
    row[3] = cand.y2;
    row[4] = cand.score;
    for (int k = 0; k < 5; ++k) {
      const int base = k * 3;
      const int anchor = cand.anchor;
      float x = 0.0f;
      float y = 0.0f;
      float conf = 0.0f;
      if (kpt_xy_ptr != nullptr && kpt_conf_ptr != nullptr) {
        x = kpt_xy_ptr[(k * 2 + 0) * state->anchors + anchor];
        y = kpt_xy_ptr[(k * 2 + 1) * state->anchors + anchor];
        conf = kpt_conf_ptr[k * state->anchors + anchor];
      } else {
        x = kpt_ptr[(base + 0) * state->anchors + anchor];
        y = kpt_ptr[(base + 1) * state->anchors + anchor];
        conf = kpt_ptr[(base + 2) * state->anchors + anchor];
      }
      x = (x - static_cast<float>(meta.pad_left)) / meta.gain;
      y = (y - static_cast<float>(meta.pad_top)) / meta.gain;
      row[5 + base + 0] = clip_f(x, 0.0f, max_x);
      row[5 + base + 1] = clip_f(y, 0.0f, max_y);
      row[5 + base + 2] = probability_f(conf, keypoint_use_sigmoid);
    }
    ++out_count;
  }
  return out_count;
}

static int get_dim(const rknn_tensor_attr& attr, uint32_t idx)
{
  return idx < attr.n_dims ? attr.dims[idx] : 0;
}

static int infer_shape_from_attr(ContextState* state, char* err, int err_len)
{
  const auto& in = state->input_attr;
  if (in.fmt == RKNN_TENSOR_NHWC) {
    state->input_h = get_dim(in, 1);
    state->input_w = get_dim(in, 2);
    state->input_c = get_dim(in, 3);
  } else if (in.fmt == RKNN_TENSOR_NCHW) {
    state->input_h = get_dim(in, 2);
    state->input_w = get_dim(in, 3);
    state->input_c = get_dim(in, 1);
  } else {
    set_error(err, err_len, "unsupported input tensor format; expected NHWC/NCHW");
    return -1;
  }
  if (state->input_h <= 0 || state->input_w <= 0 || state->input_c <= 0) {
    set_error(err, err_len, "invalid input tensor shape");
    return -1;
  }
  state->input_bytes = static_cast<size_t>(state->input_h) * state->input_w * state->input_c;

  int anchors = 0;
  int seen = 0;
  bool has_box = false;
  bool has_score = false;
  bool has_kpt15 = false;
  bool has_kpt_xy = false;
  bool has_kpt_conf = false;
  for (uint32_t i = 0; i < state->n_output; ++i) {
    const auto& out = state->output_attrs[i];
    const OutputLayout layout = output_layout(out);
    if (layout.channels <= 0 || layout.anchors <= 0) {
      continue;
    }
    if (channel_offset(layout.channels) < 0 && layout.channels != 10 && layout.channels != 5) {
      continue;
    }
    if (anchors == 0) {
      anchors = layout.anchors;
    }
    if (anchors != layout.anchors) {
      set_error(err, err_len, "split output anchors mismatch");
      return -1;
    }
    seen += layout.channels;
    if (layout.channels == 4) {
      has_box = true;
    } else if (layout.channels == 1) {
      has_score = true;
    } else if (layout.channels == 15) {
      has_kpt15 = true;
    } else if (layout.channels == 10) {
      has_kpt_xy = true;
    } else if (layout.channels == 5) {
      has_kpt_conf = true;
    }
  }
  const bool old_kpt_layout = has_box && has_score && has_kpt15 && !has_kpt_xy && !has_kpt_conf;
  const bool split_kpt_layout = has_box && has_score && !has_kpt15 && has_kpt_xy && has_kpt_conf;
  if (anchors <= 0 || seen != kCombinedChannels || (!old_kpt_layout && !split_kpt_layout)) {
    set_error(err, err_len, "expected split outputs with channels 4,1,15 or 4,1,10,5");
    return -1;
  }
  state->anchors = anchors;
  state->split_keypoint_conf = split_kpt_layout;
  return 0;
}

static int init_one(ContextState* state, const char* model_path, rknn_core_mask core_mask, char* err, int err_len)
{
  int ret = rknn_init(&state->ctx, const_cast<char*>(model_path), 0, 0, nullptr);
  if (ret < 0) {
    set_error(err, err_len, "rknn_init failed: " + std::to_string(ret));
    return ret;
  }

  ret = rknn_set_core_mask(state->ctx, core_mask);
  if (ret < 0) {
    set_error(err, err_len, "rknn_set_core_mask failed: " + std::to_string(ret));
    return ret;
  }

  rknn_input_output_num io_num{};
  ret = rknn_query(state->ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    set_error(err, err_len, "RKNN_QUERY_IN_OUT_NUM failed: " + std::to_string(ret));
    return ret;
  }
  if (io_num.n_input != 1 || io_num.n_output > 8) {
    set_error(err, err_len, "expected 1 input and <=8 outputs");
    return -1;
  }
  state->n_output = io_num.n_output;

  std::memset(&state->input_attr, 0, sizeof(state->input_attr));
  state->input_attr.index = 0;
  ret = rknn_query(state->ctx, RKNN_QUERY_INPUT_ATTR, &state->input_attr, sizeof(state->input_attr));
  if (ret != RKNN_SUCC) {
    set_error(err, err_len, "RKNN_QUERY_INPUT_ATTR failed: " + std::to_string(ret));
    return ret;
  }

  for (uint32_t i = 0; i < state->n_output; ++i) {
    std::memset(&state->output_attrs[i], 0, sizeof(rknn_tensor_attr));
    state->output_attrs[i].index = i;
    ret = rknn_query(state->ctx, RKNN_QUERY_OUTPUT_ATTR, &state->output_attrs[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      set_error(err, err_len, "RKNN_QUERY_OUTPUT_ATTR failed: " + std::to_string(ret));
      return ret;
    }
  }

  return infer_shape_from_attr(state, err, err_len);
}

static int prepare_bound_input_mem_one(ContextState* state, char* err, int err_len)
{
  if (state == nullptr || state->ctx == 0) {
    set_error(err, err_len, "invalid context");
    return -1;
  }
  if (state->input_mem_bound && state->input_mem != nullptr) {
    return 0;
  }

  rknn_tensor_attr attr{};
  attr.index = 0;
  int ret = rknn_query(state->ctx, RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR, &attr, sizeof(attr));
  if (ret != RKNN_SUCC) {
    std::memset(&attr, 0, sizeof(attr));
    attr.index = 0;
    ret = rknn_query(state->ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &attr, sizeof(attr));
  }
  if (ret != RKNN_SUCC) {
    set_error(err, err_len, "RKNN_QUERY_NATIVE_INPUT_ATTR failed: " + std::to_string(ret));
    return ret;
  }

  attr.index = 0;
  attr.type = RKNN_TENSOR_UINT8;
  attr.fmt = RKNN_TENSOR_NHWC;
  attr.pass_through = 0;
  attr.h_stride = 0;

  const uint32_t expected_bytes = static_cast<uint32_t>(state->input_bytes);
  const uint32_t stride_bytes = attr.size_with_stride > 0 ? attr.size_with_stride : attr.size;
  if (stride_bytes != expected_bytes) {
    set_error(
        err,
        err_len,
        "native input stride is not contiguous; size_with_stride=" +
            std::to_string(stride_bytes) + " expected=" + std::to_string(expected_bytes));
    return -1;
  }
  if (attr.w_stride != 0 && static_cast<int>(attr.w_stride) != state->input_w) {
    set_error(
        err,
        err_len,
        "native input width stride is not contiguous; w_stride=" +
            std::to_string(attr.w_stride) + " expected=" + std::to_string(state->input_w));
    return -1;
  }

  state->input_mem = rknn_create_mem(state->ctx, stride_bytes);
  if (state->input_mem == nullptr || state->input_mem->virt_addr == nullptr) {
    set_error(err, err_len, "rknn_create_mem(input) failed");
    state->input_mem = nullptr;
    return -1;
  }
  ret = rknn_set_io_mem(state->ctx, state->input_mem, &attr);
  if (ret != RKNN_SUCC) {
    release_bound_input_mem(state);
    set_error(err, err_len, "rknn_set_io_mem(input) failed: " + std::to_string(ret));
    return ret;
  }
  state->bound_input_attr = attr;
  state->input_mem_bound = true;
  return 0;
}

static int run_one(ContextState* state,
                   const uint8_t* input,
                   float* output,
                   double* run_ms,
                   double* outputs_ms,
                   std::string* error)
{
  rknn_input rknn_in{};
  rknn_in.index = 0;
  rknn_in.pass_through = 0;
  rknn_in.type = RKNN_TENSOR_UINT8;
  rknn_in.fmt = RKNN_TENSOR_NHWC;
  rknn_in.size = static_cast<uint32_t>(state->input_bytes);
  rknn_in.buf = const_cast<uint8_t*>(input);

  int ret = rknn_inputs_set(state->ctx, 1, &rknn_in);
  if (ret < 0) {
    *error = "rknn_inputs_set failed: " + std::to_string(ret);
    return ret;
  }

  int64_t t0 = now_us();
  ret = rknn_run(state->ctx, nullptr);
  int64_t t1 = now_us();
  *run_ms = (t1 - t0) / 1000.0;
  if (ret < 0) {
    *error = "rknn_run failed: " + std::to_string(ret);
    return ret;
  }

  rknn_output outs[8]{};
  for (uint32_t i = 0; i < state->n_output; ++i) {
    outs[i].want_float = 1;
    outs[i].index = i;
    outs[i].is_prealloc = 0;
  }
  int64_t t2 = now_us();
  ret = rknn_outputs_get(state->ctx, state->n_output, outs, nullptr);
  int64_t t3 = now_us();
  *outputs_ms = (t3 - t2) / 1000.0;
  if (ret < 0) {
    *error = "rknn_outputs_get failed: " + std::to_string(ret);
    return ret;
  }

  for (uint32_t i = 0; i < state->n_output; ++i) {
    copy_output_to_combined(state->output_attrs[i], outs[i], state->anchors, output);
  }

  rknn_outputs_release(state->ctx, state->n_output, outs);
  return 0;
}

static int run_one_outputs(ContextState* state,
                           rknn_output* outs,
                           double* run_ms,
                           double* outputs_ms,
                           std::string* error)
{
  int64_t t0 = now_us();
  int ret = rknn_run(state->ctx, nullptr);
  int64_t t1 = now_us();
  *run_ms = (t1 - t0) / 1000.0;
  if (ret < 0) {
    *error = "rknn_run failed: " + std::to_string(ret);
    return ret;
  }

  for (uint32_t i = 0; i < state->n_output; ++i) {
    outs[i].want_float = 1;
    outs[i].index = i;
    outs[i].is_prealloc = 0;
  }
  int64_t t2 = now_us();
  ret = rknn_outputs_get(state->ctx, state->n_output, outs, nullptr);
  int64_t t3 = now_us();
  *outputs_ms = (t3 - t2) / 1000.0;
  if (ret < 0) {
    *error = "rknn_outputs_get failed: " + std::to_string(ret);
    return ret;
  }
  return 0;
}

static int run_one_decoded(ContextState* state,
                           const uint8_t* input,
                           const DecodeMeta& meta,
                           float conf_threshold,
                           float iou_threshold,
                           int max_det,
                           int max_nms,
                           float* decoded,
                           int* decoded_count,
                           double* run_ms,
                           double* outputs_ms,
                           double* decode_ms,
                           std::string* error)
{
  if (decoded_count != nullptr) {
    *decoded_count = 0;
  }

  rknn_input rknn_in{};
  rknn_in.index = 0;
  rknn_in.pass_through = 0;
  rknn_in.type = RKNN_TENSOR_UINT8;
  rknn_in.fmt = RKNN_TENSOR_NHWC;
  rknn_in.size = static_cast<uint32_t>(state->input_bytes);
  rknn_in.buf = const_cast<uint8_t*>(input);

  int ret = rknn_inputs_set(state->ctx, 1, &rknn_in);
  if (ret < 0) {
    *error = "rknn_inputs_set failed: " + std::to_string(ret);
    return ret;
  }

  int64_t t0 = now_us();
  ret = rknn_run(state->ctx, nullptr);
  int64_t t1 = now_us();
  *run_ms = (t1 - t0) / 1000.0;
  if (ret < 0) {
    *error = "rknn_run failed: " + std::to_string(ret);
    return ret;
  }

  rknn_output outs[8]{};
  for (uint32_t i = 0; i < state->n_output; ++i) {
    outs[i].want_float = 1;
    outs[i].index = i;
    outs[i].is_prealloc = 0;
  }
  int64_t t2 = now_us();
  ret = rknn_outputs_get(state->ctx, state->n_output, outs, nullptr);
  int64_t t3 = now_us();
  *outputs_ms = (t3 - t2) / 1000.0;
  if (ret < 0) {
    *error = "rknn_outputs_get failed: " + std::to_string(ret);
    return ret;
  }

  int64_t t4 = now_us();
  const int count = decode_slice_outputs(
      state,
      outs,
      meta,
      conf_threshold,
      iou_threshold,
      max_det,
      max_nms,
      decoded);
  int64_t t5 = now_us();
  *decode_ms = (t5 - t4) / 1000.0;
  if (decoded_count != nullptr) {
    *decoded_count = count;
  }

  rknn_outputs_release(state->ctx, state->n_output, outs);
  return 0;
}

static int run_one_decoded_bound(ContextState* state,
                                 const DecodeMeta& meta,
                                 float conf_threshold,
                                 float iou_threshold,
                                 int max_det,
                                 int max_nms,
                                 bool external_device_input,
                                 float* decoded,
                                 int* decoded_count,
                                 double* sync_ms,
                                 double* run_ms,
                                 double* outputs_ms,
                                 double* decode_ms,
                                 std::string* error)
{
  if (decoded_count != nullptr) {
    *decoded_count = 0;
  }
  if (state == nullptr || !state->input_mem_bound || state->input_mem == nullptr) {
    *error = "bound input mem is not prepared";
    return -1;
  }

  int ret = 0;
  int64_t ts0 = now_us();
  RknnMemSyncFn mem_sync = resolve_rknn_mem_sync();
  if (mem_sync != nullptr) {
    const int sync_mode = external_device_input
        ? kRknnMemorySyncFromDeviceCompat
        : kRknnMemorySyncToDeviceCompat;
    ret = mem_sync(state->ctx, state->input_mem, sync_mode);
    if (ret != RKNN_SUCC) {
      *error = std::string("rknn_mem_sync(input ") +
               (external_device_input ? "FROM_DEVICE" : "TO_DEVICE") +
               ") failed: " + std::to_string(ret);
      return ret;
    }
  }
  int64_t ts1 = now_us();
  *sync_ms = (ts1 - ts0) / 1000.0;

  rknn_output outs[8]{};
  ret = run_one_outputs(state, outs, run_ms, outputs_ms, error);
  if (ret != 0) {
    return ret;
  }

  int64_t t4 = now_us();
  const int count = decode_slice_outputs(
      state,
      outs,
      meta,
      conf_threshold,
      iou_threshold,
      max_det,
      max_nms,
      decoded);
  int64_t t5 = now_us();
  *decode_ms = (t5 - t4) / 1000.0;
  if (decoded_count != nullptr) {
    *decoded_count = count;
  }

  rknn_outputs_release(state->ctx, state->n_output, outs);
  return 0;
}

}  // namespace

extern "C" {

void face_rknn_parallel_destroy(void* handle);
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

int face_rknn_parallel_create(const char* model_path, void** handle, char* err, int err_len)
{
  if (model_path == nullptr || handle == nullptr) {
    set_error(err, err_len, "invalid null argument");
    return -1;
  }
  auto* runner = new ParallelRunner();
  const rknn_core_mask masks[kNumContexts] = {
      RKNN_NPU_CORE_0,
      RKNN_NPU_CORE_1,
      RKNN_NPU_CORE_2,
  };
  for (int i = 0; i < kNumContexts; ++i) {
    int ret = init_one(&runner->states[i], model_path, masks[i], err, err_len);
    if (ret != 0) {
      for (int j = 0; j <= i; ++j) {
        if (runner->states[j].ctx != 0) {
          rknn_destroy(runner->states[j].ctx);
        }
      }
      delete runner;
      return ret;
    }
  }

  const auto& first = runner->states[0];
  for (int i = 1; i < kNumContexts; ++i) {
    const auto& cur = runner->states[i];
    if (cur.input_h != first.input_h || cur.input_w != first.input_w ||
        cur.input_c != first.input_c || cur.anchors != first.anchors) {
      set_error(err, err_len, "parallel contexts have mismatched shapes");
      face_rknn_parallel_destroy(runner);
      return -1;
    }
  }

  *handle = runner;
  return 0;
}

void face_rknn_parallel_destroy(void* handle)
{
  auto* runner = static_cast<ParallelRunner*>(handle);
  if (runner == nullptr) {
    return;
  }
  for (int i = 0; i < kNumContexts; ++i) {
    release_bound_input_mem(&runner->states[i]);
    if (runner->states[i].ctx != 0) {
      rknn_destroy(runner->states[i].ctx);
      runner->states[i].ctx = 0;
    }
  }
  delete runner;
}

int face_rknn_parallel_get_shape(void* handle, int* input_h, int* input_w, int* input_c, int* channels, int* anchors)
{
  auto* runner = static_cast<ParallelRunner*>(handle);
  if (runner == nullptr) {
    return -1;
  }
  const auto& state = runner->states[0];
  if (input_h) {
    *input_h = state.input_h;
  }
  if (input_w) {
    *input_w = state.input_w;
  }
  if (input_c) {
    *input_c = state.input_c;
  }
  if (channels) {
    *channels = kCombinedChannels;
  }
  if (anchors) {
    *anchors = state.anchors;
  }
  return 0;
}

int face_rknn_parallel_prepare_bound_inputs(void* handle, char* err, int err_len)
{
  auto* runner = static_cast<ParallelRunner*>(handle);
  if (runner == nullptr) {
    set_error(err, err_len, "invalid null handle");
    return -1;
  }
  for (int i = 0; i < kNumContexts; ++i) {
    int ret = prepare_bound_input_mem_one(&runner->states[i], err, err_len);
    if (ret != 0) {
      return ret;
    }
  }
  return 0;
}

int face_rknn_parallel_get_bound_input_ptrs(void* handle,
                                            uint8_t** ptrs,
                                            uint64_t* sizes,
                                            int max_inputs,
                                            char* err,
                                            int err_len)
{
  auto* runner = static_cast<ParallelRunner*>(handle);
  if (runner == nullptr || ptrs == nullptr || sizes == nullptr) {
    set_error(err, err_len, "invalid null argument");
    return -1;
  }
  if (max_inputs < kNumContexts) {
    set_error(err, err_len, "max_inputs must be >= 3");
    return -1;
  }
  for (int i = 0; i < kNumContexts; ++i) {
    ContextState& state = runner->states[i];
    if (!state.input_mem_bound || state.input_mem == nullptr || state.input_mem->virt_addr == nullptr) {
      set_error(err, err_len, "bound input mem is not prepared");
      return -1;
    }
    ptrs[i] = static_cast<uint8_t*>(state.input_mem->virt_addr);
    sizes[i] = static_cast<uint64_t>(state.input_mem->size);
  }
  return kNumContexts;
}

int face_rknn_parallel_get_bound_input_fds(void* handle,
                                           int* fds,
                                           uint64_t* sizes,
                                           int max_inputs,
                                           char* err,
                                           int err_len)
{
  auto* runner = static_cast<ParallelRunner*>(handle);
  if (runner == nullptr || fds == nullptr || sizes == nullptr) {
    set_error(err, err_len, "invalid null argument");
    return -1;
  }
  if (max_inputs < kNumContexts) {
    set_error(err, err_len, "max_inputs must be >= 3");
    return -1;
  }
  for (int i = 0; i < kNumContexts; ++i) {
    ContextState& state = runner->states[i];
    if (!state.input_mem_bound || state.input_mem == nullptr) {
      set_error(err, err_len, "bound input mem is not prepared");
      return -1;
    }
    if (state.input_mem->fd < 0) {
      set_error(err, err_len, "bound input mem has no fd");
      return -1;
    }
    fds[i] = state.input_mem->fd;
    sizes[i] = static_cast<uint64_t>(state.input_mem->size);
  }
  return kNumContexts;
}

int face_rknn_parallel_sync_bound_inputs_from_device(void* handle,
                                                     char* err,
                                                     int err_len)
{
  auto* runner = static_cast<ParallelRunner*>(handle);
  if (runner == nullptr) {
    set_error(err, err_len, "invalid null handle");
    return -1;
  }
  RknnMemSyncFn mem_sync = resolve_rknn_mem_sync();
  if (mem_sync == nullptr) {
    set_error(err, err_len, "rknn_mem_sync is unavailable");
    return -1;
  }
  for (int i = 0; i < kNumContexts; ++i) {
    ContextState& state = runner->states[i];
    if (!state.input_mem_bound || state.input_mem == nullptr) {
      set_error(err, err_len, "bound input mem is not prepared");
      return -1;
    }
    int ret = mem_sync(state.ctx, state.input_mem, kRknnMemorySyncFromDeviceCompat);
    if (ret != RKNN_SUCC) {
      set_error(err, err_len, "rknn_mem_sync(FROM_DEVICE) failed: " + std::to_string(ret));
      return ret;
    }
  }
  return 0;
}

int face_rknn_parallel_infer(void* handle,
                             const uint8_t* inputs,
                             int num_inputs,
                             int input_h,
                             int input_w,
                             int input_c,
                             float* outputs,
                             double* timings,
                             char* err,
                             int err_len)
{
  auto* runner = static_cast<ParallelRunner*>(handle);
  if (runner == nullptr || inputs == nullptr || outputs == nullptr) {
    set_error(err, err_len, "invalid null argument");
    return -1;
  }
  if (num_inputs != kNumContexts) {
    set_error(err, err_len, "num_inputs must be 3");
    return -1;
  }
  const auto& first = runner->states[0];
  if (input_h != first.input_h || input_w != first.input_w || input_c != first.input_c) {
    set_error(err, err_len, "input shape mismatch");
    return -1;
  }

  const size_t one_input_bytes = first.input_bytes;
  const size_t one_output_floats = static_cast<size_t>(kCombinedChannels) * first.anchors;
  double run_ms[kNumContexts] = {0.0, 0.0, 0.0};
  double outputs_ms[kNumContexts] = {0.0, 0.0, 0.0};
  int rets[kNumContexts] = {0, 0, 0};
  std::string errors[kNumContexts];

  int64_t wall0 = now_us();
  std::thread workers[kNumContexts];
  for (int i = 0; i < kNumContexts; ++i) {
    workers[i] = std::thread([&, i]() {
      const uint8_t* in = inputs + static_cast<size_t>(i) * one_input_bytes;
      float* out = outputs + static_cast<size_t>(i) * one_output_floats;
      rets[i] = run_one(&runner->states[i], in, out, &run_ms[i], &outputs_ms[i], &errors[i]);
    });
  }
  for (int i = 0; i < kNumContexts; ++i) {
    workers[i].join();
  }
  int64_t wall1 = now_us();

  for (int i = 0; i < kNumContexts; ++i) {
    if (rets[i] != 0) {
      set_error(err, err_len, "slice " + std::to_string(i) + ": " + errors[i]);
      return rets[i];
    }
  }

  if (timings != nullptr) {
    timings[0] = (wall1 - wall0) / 1000.0;
    timings[1] = std::max({run_ms[0], run_ms[1], run_ms[2]});
    timings[2] = std::max({outputs_ms[0], outputs_ms[1], outputs_ms[2]});
    timings[3] = run_ms[0];
    timings[4] = run_ms[1];
    timings[5] = run_ms[2];
    timings[6] = outputs_ms[0];
    timings[7] = outputs_ms[1];
    timings[8] = outputs_ms[2];
  }
  return 0;
}

int face_rknn_parallel_infer_decoded(void* handle,
                                     const uint8_t* inputs,
                                     int num_inputs,
                                     int input_h,
                                     int input_w,
                                     int input_c,
                                     const int* slice_shapes,
                                     const float* gains,
                                     const int* pads,
                                     float conf_threshold,
                                     float iou_threshold,
                                     int max_det,
                                     int max_nms,
                                     float* detections,
                                     int* detection_counts,
                                     double* timings,
                                     char* err,
                                     int err_len)
{
  auto* runner = static_cast<ParallelRunner*>(handle);
  if (runner == nullptr || inputs == nullptr || detections == nullptr ||
      detection_counts == nullptr || slice_shapes == nullptr ||
      gains == nullptr || pads == nullptr) {
    set_error(err, err_len, "invalid null argument");
    return -1;
  }
  if (num_inputs != kNumContexts) {
    set_error(err, err_len, "num_inputs must be 3");
    return -1;
  }
  if (max_det <= 0) {
    set_error(err, err_len, "max_det must be > 0");
    return -1;
  }
  if (max_nms <= 0) {
    max_nms = 300;
  }
  const auto& first = runner->states[0];
  if (input_h != first.input_h || input_w != first.input_w || input_c != first.input_c) {
    set_error(err, err_len, "input shape mismatch");
    return -1;
  }

  DecodeMeta metas[kNumContexts];
  for (int i = 0; i < kNumContexts; ++i) {
    metas[i].slice_h = slice_shapes[i * 2 + 0];
    metas[i].slice_w = slice_shapes[i * 2 + 1];
    metas[i].gain = gains[i];
    metas[i].pad_left = pads[i * 2 + 0];
    metas[i].pad_top = pads[i * 2 + 1];
  }

  const size_t one_input_bytes = first.input_bytes;
  const size_t one_detection_floats = static_cast<size_t>(max_det) * kDetectionFields;
  double run_ms[kNumContexts] = {0.0, 0.0, 0.0};
  double outputs_ms[kNumContexts] = {0.0, 0.0, 0.0};
  double decode_ms[kNumContexts] = {0.0, 0.0, 0.0};
  int rets[kNumContexts] = {0, 0, 0};
  std::string errors[kNumContexts];

  for (int i = 0; i < kNumContexts; ++i) {
    detection_counts[i] = 0;
  }

  int64_t wall0 = now_us();
  std::thread workers[kNumContexts];
  for (int i = 0; i < kNumContexts; ++i) {
    workers[i] = std::thread([&, i]() {
      const uint8_t* in = inputs + static_cast<size_t>(i) * one_input_bytes;
      float* out = detections + static_cast<size_t>(i) * one_detection_floats;
      rets[i] = run_one_decoded(
          &runner->states[i],
          in,
          metas[i],
          conf_threshold,
          iou_threshold,
          max_det,
          max_nms,
          out,
          &detection_counts[i],
          &run_ms[i],
          &outputs_ms[i],
          &decode_ms[i],
          &errors[i]);
    });
  }
  for (int i = 0; i < kNumContexts; ++i) {
    workers[i].join();
  }
  int64_t wall1 = now_us();

  for (int i = 0; i < kNumContexts; ++i) {
    if (rets[i] != 0) {
      set_error(err, err_len, "slice " + std::to_string(i) + ": " + errors[i]);
      return rets[i];
    }
  }

  if (timings != nullptr) {
    timings[0] = (wall1 - wall0) / 1000.0;
    timings[1] = std::max({run_ms[0], run_ms[1], run_ms[2]});
    timings[2] = std::max({outputs_ms[0], outputs_ms[1], outputs_ms[2]});
    timings[3] = std::max({decode_ms[0], decode_ms[1], decode_ms[2]});
    timings[4] = run_ms[0];
    timings[5] = run_ms[1];
    timings[6] = run_ms[2];
    timings[7] = outputs_ms[0];
    timings[8] = outputs_ms[1];
    timings[9] = outputs_ms[2];
    timings[10] = decode_ms[0];
    timings[11] = decode_ms[1];
    timings[12] = decode_ms[2];
  }
  return 0;
}

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
                                    int err_len)
{
  auto* runner = static_cast<ParallelRunner*>(handle);
  if (runner == nullptr || inputs == nullptr || detections == nullptr ||
      detection_count == nullptr || slice_shapes == nullptr || gains == nullptr ||
      pads == nullptr || slice_start_x == nullptr || slice_wrap_around == nullptr) {
    set_error(err, err_len, "invalid null argument");
    return -1;
  }
  if (num_inputs != kNumContexts || num_slices != kNumContexts) {
    set_error(err, err_len, "num_inputs and num_slices must be 3");
    return -1;
  }
  if (max_det <= 0 || max_output_dets <= 0) {
    set_error(err, err_len, "max_det and max_output_dets must be > 0");
    return -1;
  }
  if (max_nms <= 0) {
    max_nms = 300;
  }
  const auto& first = runner->states[0];
  if (input_h != first.input_h || input_w != first.input_w || input_c != first.input_c) {
    set_error(err, err_len, "input shape mismatch");
    return -1;
  }

  DecodeMeta metas[kNumContexts];
  for (int i = 0; i < kNumContexts; ++i) {
    metas[i].slice_h = slice_shapes[i * 2 + 0];
    metas[i].slice_w = slice_shapes[i * 2 + 1];
    metas[i].gain = gains[i];
    metas[i].pad_left = pads[i * 2 + 0];
    metas[i].pad_top = pads[i * 2 + 1];
  }

  const size_t one_input_bytes = first.input_bytes;
  const size_t one_detection_floats = static_cast<size_t>(max_det) * kDetectionFields;
  std::vector<float> decoded(static_cast<size_t>(kNumContexts) * one_detection_floats, 0.0f);
  int decoded_counts[kNumContexts] = {0, 0, 0};
  double run_ms[kNumContexts] = {0.0, 0.0, 0.0};
  double outputs_ms[kNumContexts] = {0.0, 0.0, 0.0};
  double decode_ms[kNumContexts] = {0.0, 0.0, 0.0};
  int rets[kNumContexts] = {0, 0, 0};
  std::string errors[kNumContexts];

  int64_t wall0 = now_us();
  std::thread workers[kNumContexts];
  for (int i = 0; i < kNumContexts; ++i) {
    workers[i] = std::thread([&, i]() {
      const uint8_t* in = inputs + static_cast<size_t>(i) * one_input_bytes;
      float* out = decoded.data() + static_cast<size_t>(i) * one_detection_floats;
      rets[i] = run_one_decoded(
          &runner->states[i],
          in,
          metas[i],
          conf_threshold,
          decode_iou_threshold,
          max_det,
          max_nms,
          out,
          &decoded_counts[i],
          &run_ms[i],
          &outputs_ms[i],
          &decode_ms[i],
          &errors[i]);
    });
  }
  for (int i = 0; i < kNumContexts; ++i) {
    workers[i].join();
  }
  int64_t wall1 = now_us();

  for (int i = 0; i < kNumContexts; ++i) {
    if (rets[i] != 0) {
      set_error(err, err_len, "slice " + std::to_string(i) + ": " + errors[i]);
      return rets[i];
    }
  }

  double merge_timing[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  int64_t merge0 = now_us();
  const int merge_ret = merge_decoded_slices(
      decoded.data(),
      decoded_counts,
      max_det,
      slice_start_x,
      slice_wrap_around,
      num_slices,
      original_width,
      overlap_ratio,
      merge_iou_threshold,
      nms_iou_thresh,
      detections,
      max_output_dets,
      detection_count,
      merge_stats,
      merge_timing,
      err,
      err_len);
  int64_t merge1 = now_us();
  if (merge_ret != 0) {
    return merge_ret;
  }

  if (timings != nullptr) {
    timings[0] = (wall1 - wall0) / 1000.0;
    timings[1] = std::max({run_ms[0], run_ms[1], run_ms[2]});
    timings[2] = std::max({outputs_ms[0], outputs_ms[1], outputs_ms[2]});
    timings[3] = std::max({decode_ms[0], decode_ms[1], decode_ms[2]});
    timings[4] = (merge1 - merge0) / 1000.0;
    timings[5] = merge_timing[0];
    timings[6] = merge_timing[1];
    timings[7] = merge_timing[2];
    timings[8] = merge_timing[3];
    timings[9] = merge_timing[4];
    timings[10] = run_ms[0];
    timings[11] = run_ms[1];
    timings[12] = run_ms[2];
    timings[13] = outputs_ms[0];
    timings[14] = outputs_ms[1];
    timings[15] = outputs_ms[2];
    timings[16] = decode_ms[0];
    timings[17] = decode_ms[1];
    timings[18] = decode_ms[2];
  }
  return 0;
}

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
                                                int err_len)
{
  auto* runner = static_cast<ParallelRunner*>(handle);
  if (runner == nullptr || input == nullptr || detections == nullptr ||
      detection_count == nullptr) {
    set_error(err, err_len, "invalid null argument");
    return -1;
  }
  if (image_h <= 0 || image_w <= 0 || gain <= 0.0f ||
      max_det <= 0 || max_output_dets <= 0) {
    set_error(err, err_len, "invalid single-image shape");
    return -1;
  }
  if (max_nms <= 0) {
    max_nms = 300;
  }
  if (core_index < 0 || core_index >= kNumContexts) {
    set_error(err, err_len, "invalid single-image core index");
    return -1;
  }
  ContextState& state = runner->states[core_index];
  if (input_h != state.input_h || input_w != state.input_w || input_c != state.input_c) {
    set_error(err, err_len, "input shape mismatch");
    return -1;
  }

  DecodeMeta meta;
  meta.slice_h = image_h;
  meta.slice_w = image_w;
  meta.gain = gain;
  meta.pad_left = pad_left;
  meta.pad_top = pad_top;

  const int decode_max_det = std::min(max_det, max_output_dets);
  int decoded_count = 0;
  double run_ms = 0.0;
  double outputs_ms = 0.0;
  double decode_ms = 0.0;
  std::string error;

  const int64_t wall0 = now_us();
  const int ret = run_one_decoded(
      &state,
      input,
      meta,
      conf_threshold,
      nms_iou_thresh,
      decode_max_det,
      max_nms,
      detections,
      &decoded_count,
      &run_ms,
      &outputs_ms,
      &decode_ms,
      &error);
  const int64_t wall1 = now_us();
  if (ret != 0) {
    set_error(err, err_len, error);
    return ret;
  }

  *detection_count = std::max(0, std::min(decoded_count, max_output_dets));
  if (merge_stats != nullptr) {
    merge_stats[0] = *detection_count;
    merge_stats[1] = *detection_count;
    merge_stats[2] = *detection_count;
  }
  if (timings != nullptr) {
    timings[0] = (wall1 - wall0) / 1000.0;
    timings[1] = run_ms;
    timings[2] = outputs_ms;
    timings[3] = decode_ms;
    timings[4] = 0.0;
    timings[10] = run_ms;
    timings[13] = outputs_ms;
    timings[16] = decode_ms;
  }
  return 0;
}

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
                                           int err_len)
{
  return face_rknn_parallel_infer_merged_single_core(
      handle,
      0,
      input,
      input_h,
      input_w,
      input_c,
      image_h,
      image_w,
      gain,
      pad_left,
      pad_top,
      nms_iou_thresh,
      conf_threshold,
      max_det,
      max_nms,
      detections,
      max_output_dets,
      detection_count,
      merge_stats,
      timings,
      err,
      err_len);
}

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
                                          int err_len)
{
  return face_rknn_parallel_infer_merged_bound_external(
      handle,
      num_inputs,
      input_h,
      input_w,
      input_c,
      slice_shapes,
      gains,
      pads,
      slice_start_x,
      slice_wrap_around,
      num_slices,
      original_width,
      overlap_ratio,
      merge_iou_threshold,
      nms_iou_thresh,
      conf_threshold,
      decode_iou_threshold,
      max_det,
      max_nms,
      0,
      detections,
      max_output_dets,
      detection_count,
      merge_stats,
      timings,
      err,
      err_len);
}

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
                                                   int err_len)
{
  auto* runner = static_cast<ParallelRunner*>(handle);
  if (runner == nullptr || detections == nullptr ||
      detection_count == nullptr || slice_shapes == nullptr || gains == nullptr ||
      pads == nullptr || slice_start_x == nullptr || slice_wrap_around == nullptr) {
    set_error(err, err_len, "invalid null argument");
    return -1;
  }
  if (num_inputs != kNumContexts || num_slices != kNumContexts) {
    set_error(err, err_len, "num_inputs and num_slices must be 3");
    return -1;
  }
  if (max_det <= 0 || max_output_dets <= 0) {
    set_error(err, err_len, "max_det and max_output_dets must be > 0");
    return -1;
  }
  if (max_nms <= 0) {
    max_nms = 300;
  }
  const auto& first = runner->states[0];
  if (input_h != first.input_h || input_w != first.input_w || input_c != first.input_c) {
    set_error(err, err_len, "input shape mismatch");
    return -1;
  }
  for (int i = 0; i < kNumContexts; ++i) {
    if (!runner->states[i].input_mem_bound || runner->states[i].input_mem == nullptr) {
      set_error(err, err_len, "bound input mem is not prepared");
      return -1;
    }
  }

  DecodeMeta metas[kNumContexts];
  for (int i = 0; i < kNumContexts; ++i) {
    metas[i].slice_h = slice_shapes[i * 2 + 0];
    metas[i].slice_w = slice_shapes[i * 2 + 1];
    metas[i].gain = gains[i];
    metas[i].pad_left = pads[i * 2 + 0];
    metas[i].pad_top = pads[i * 2 + 1];
  }

  const size_t one_detection_floats = static_cast<size_t>(max_det) * kDetectionFields;
  std::vector<float> decoded(static_cast<size_t>(kNumContexts) * one_detection_floats, 0.0f);
  int decoded_counts[kNumContexts] = {0, 0, 0};
  double sync_ms[kNumContexts] = {0.0, 0.0, 0.0};
  double run_ms[kNumContexts] = {0.0, 0.0, 0.0};
  double outputs_ms[kNumContexts] = {0.0, 0.0, 0.0};
  double decode_ms[kNumContexts] = {0.0, 0.0, 0.0};
  int rets[kNumContexts] = {0, 0, 0};
  std::string errors[kNumContexts];

  int64_t wall0 = now_us();
  std::thread workers[kNumContexts];
  for (int i = 0; i < kNumContexts; ++i) {
    workers[i] = std::thread([&, i]() {
      float* out = decoded.data() + static_cast<size_t>(i) * one_detection_floats;
      rets[i] = run_one_decoded_bound(
          &runner->states[i],
          metas[i],
          conf_threshold,
          decode_iou_threshold,
          max_det,
          max_nms,
          external_device_input != 0,
          out,
          &decoded_counts[i],
          &sync_ms[i],
          &run_ms[i],
          &outputs_ms[i],
          &decode_ms[i],
          &errors[i]);
    });
  }
  for (int i = 0; i < kNumContexts; ++i) {
    workers[i].join();
  }
  int64_t wall1 = now_us();

  for (int i = 0; i < kNumContexts; ++i) {
    if (rets[i] != 0) {
      set_error(err, err_len, "slice " + std::to_string(i) + ": " + errors[i]);
      return rets[i];
    }
  }

  double merge_timing[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  int64_t merge0 = now_us();
  const int merge_ret = merge_decoded_slices(
      decoded.data(),
      decoded_counts,
      max_det,
      slice_start_x,
      slice_wrap_around,
      num_slices,
      original_width,
      overlap_ratio,
      merge_iou_threshold,
      nms_iou_thresh,
      detections,
      max_output_dets,
      detection_count,
      merge_stats,
      merge_timing,
      err,
      err_len);
  int64_t merge1 = now_us();
  if (merge_ret != 0) {
    return merge_ret;
  }

  if (timings != nullptr) {
    timings[0] = (wall1 - wall0) / 1000.0;
    timings[1] = std::max({run_ms[0], run_ms[1], run_ms[2]});
    timings[2] = std::max({outputs_ms[0], outputs_ms[1], outputs_ms[2]});
    timings[3] = std::max({decode_ms[0], decode_ms[1], decode_ms[2]});
    timings[4] = (merge1 - merge0) / 1000.0;
    timings[5] = merge_timing[0];
    timings[6] = merge_timing[1];
    timings[7] = merge_timing[2];
    timings[8] = merge_timing[3];
    timings[9] = merge_timing[4];
    timings[10] = run_ms[0];
    timings[11] = run_ms[1];
    timings[12] = run_ms[2];
    timings[13] = outputs_ms[0];
    timings[14] = outputs_ms[1];
    timings[15] = outputs_ms[2];
    timings[16] = decode_ms[0];
    timings[17] = decode_ms[1];
    timings[18] = decode_ms[2];
    timings[19] = std::max({sync_ms[0], sync_ms[1], sync_ms[2]});
    timings[20] = sync_ms[0];
    timings[21] = sync_ms[1];
    timings[22] = sync_ms[2];
  }
  return 0;
}

}  // extern "C"
